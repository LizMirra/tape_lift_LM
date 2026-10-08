"""
Detect particles on a tape-lift image by contrast

This replaced an earlier version that used a single global median brightness
and Otsu's method. That broke down on real photos with uneven illumination
(soft out-of-focus blotches, vignetting): Otsu assumes the pixel-deviation
histogram is clearly two-humped (background vs particle), but real photos
like these are mostly continuous low-contrast texture with a few sharp
particles buried in it, so Otsu picked a threshold sitting almost entirely
inside the noise floor and flagged 30-45% of the image as "particles."

Approach now:
1. Grayscale + light Gaussian blur (blur_sigma) to suppress sensor/JPEG
   noise before anything else.
2. Estimate the local background brightness at every pixel.
   - Default (background=None): a heavy Gaussian blur of the image itself
     (local_bg_sigma, default 120px) standing in for "what this patch of
     tape would look like with no dust on it." Because the blur radius is
     large relative to real dust (which is small and sharp), genuine
     particles survive as a residual after subtracting this blurred
     version, while gradual lighting changes and big soft out-of-focus
     blobs get absorbed into the estimate and subtracted away cleanly.
     This needs no second photo and adapts per-image.
   - Optional (background=<array>): a separate clean reference photo
     (see load_background()), used directly as the per-pixel background
     instead of self-blurring. Only better than the default if that
     reference is both pixel-aligned and genuinely free of debris --
     otherwise the self-blur default is more robust.
3. Build a signed deviation image: smoothed - background_estimate.
4. Threshold deviation using a robust statistic instead of Otsu: the
   median absolute deviation (MAD) of the deviation image, scaled to
   behave like a standard deviation (x1.4826) and then multiplied by
   `sensitivity` (default 6 -- i.e. "6 robust-sigma away from zero counts
   as a particle"). MAD is far less thrown off by a having a few genuinely
   large particles or a non-bimodal histogram than Otsu is, which is why
   it holds up on real noisy data where Otsu didn't.
5. Small speckle noise is removed and small holes inside particles are
   filled, then connected components are labeled and measured.

This is intentionally simple and fast. If you later plug in a pretrained
SAM model for segmentation, it would slot in here: SAM would replace
segment_particles() and hand back a similar boolean/label mask, while
measure_particles() and everything downstream (conversion, labeling,
histogram) stays the same.
"""
import numpy as np
import pandas as pd
from PIL import Image
from skimage import filters, measure, morphology
from scipy import ndimage as ndi


def load_grayscale(image_path):
    """Load an image as a single-channel grayscale numpy array."""
    img = Image.open(image_path).convert('L')
    return np.array(img)


def load_background(image_path, blur_sigma=1.0):
    """
    Load a clean reference (background-only, no dust) image and smooth it
    the same way segment_particles smooths the image being analyzed, so
    the two are directly comparable pixel-for-pixel.

    Returns a 2D float array the same shape as the reference image.

    Note: this only helps if the reference photo is itself free of debris
    and pixel-aligned with the image being analyzed (same camera position,
    same resolution). If you're not sure your reference is clean, leave
    `background` as None in segment_particles and let it estimate the
    local background from the image itself instead -- that's usually more
    robust in practice.
    """
    gray = load_grayscale(image_path)
    return filters.gaussian(gray, sigma=blur_sigma, preserve_range=True)


def segment_particles(gray, min_size=4, blur_sigma=1.0, background=None,
                       local_bg_sigma=120, sensitivity=6.0):
    """
    Returns a boolean mask the same shape as `gray`, True where a particle
    was detected. Catches particles both darker AND lighter than the
    background in a single pass -- see module docstring for why.

    min_size: particles smaller than this many pixels are discarded as
              noise -- raise this if you're getting speckle false positives,
              lower it if you're missing genuinely small dust.
    blur_sigma: Gaussian blur strength applied before anything else, to
              suppress sensor/JPEG noise. Default 1.0.
    background: optional 2D array, same shape as `gray` -- a per-pixel
              background estimate, typically the output of
              load_background() on a clean reference photo. When given,
              it's used as-is instead of estimating the background from
              `gray` itself. When None (the default), the background is
              estimated by heavily blurring `gray` -- see module docstring.
    local_bg_sigma: only used when background is None. How large a blur
              radius to use for the self-estimated background, in pixels.
              Must be large relative to real particle size so particles
              aren't blurred into the "background" themselves, but small
              enough to track genuine large-scale lighting changes across
              the frame. Default 120 -- raise it if large particles are
              being partly absorbed into the background estimate (visible
              as a faint ring at the particle's edge instead of a solid
              outline); lower it if lighting changes faster than that
              across the frame.
    sensitivity: how many robust standard deviations away from zero a
              pixel's deviation must be to count as a particle. Higher =
              stricter (fewer false positives, may miss faint dust).
              Lower = more sensitive (catches fainter dust, more false
              positives from residual noise/texture). Default 6.0.
    """
    smoothed = filters.gaussian(gray, sigma=blur_sigma, preserve_range=True)

    if background is None:
        bg_estimate = filters.gaussian(smoothed, sigma=local_bg_sigma, preserve_range=True)
    else:
        if background.shape != smoothed.shape:
            raise ValueError(
                f"Background reference image shape {background.shape} doesn't match "
                f"the analyzed image's cropped shape {smoothed.shape} -- they need to "
                f"be the same size (crop both the same way, from the same camera position)."
            )
        bg_estimate = background

    deviation = smoothed - bg_estimate  # signed: negative = darker than background, positive = lighter

    robust_std = np.median(np.abs(deviation - np.median(deviation))) * 1.4826
    if robust_std == 0:
        # Degenerate case (perfectly flat image) -- avoid a zero threshold
        # flagging every pixel with any noise at all.
        robust_std = 1e-6
    dev_thresh = sensitivity * robust_std

    particle_mask = np.abs(deviation) > dev_thresh

    particle_mask = morphology.remove_small_objects(particle_mask, min_size=min_size)
    particle_mask = ndi.binary_fill_holes(particle_mask)

    return particle_mask


def measure_particles(particle_mask, intensity_image=None):
    """
    Label connected components in particle_mask and measure their size.

    Returns a DataFrame with columns matching what the rest of the
    pipeline (ParticleSizeConverter, ParticleOverlay) expects:
      Area       -- particle area, px^2
      Major      -- major axis length of best-fit ellipse, px
      Minor      -- minor axis length of best-fit ellipse, px
      Feret      -- maximum caliper (Feret) diameter, px
      MinFeret   -- approximated as the minor axis length (skimage doesn't
                    compute a true minimum-caliper diameter)
      CentroidX, CentroidY -- particle center, px (used for label placement)
    """
    labeled = measure.label(particle_mask)
    props = measure.regionprops(labeled, intensity_image=intensity_image)

    rows = []
    for p in props:
        y, x = p.centroid
        major_len = getattr(p, 'axis_major_length', None)
        if major_len is None:
            major_len = p.major_axis_length  # fallback for older skimage
        minor_len = getattr(p, 'axis_minor_length', None)
        if minor_len is None:
            minor_len = p.minor_axis_length  # fallback for older skimage
        feret_max = getattr(p, 'feret_diameter_max', None)
        if feret_max is None:
            feret_max = major_len  # fallback for very old skimage
        rows.append({
            'Area': p.area,
            'Major': major_len,
            'Minor': minor_len,
            'Feret': feret_max,
            'MinFeret': minor_len,
            'CentroidX': x,
            'CentroidY': y,
        })

    df = pd.DataFrame(rows, columns=['Area', 'Major', 'Minor', 'Feret', 'MinFeret', 'CentroidX', 'CentroidY'])
    df.index = range(1, len(df) + 1)  # 1-indexed, matching the old ImageJ Results table
    df.index.name = ''
    return df