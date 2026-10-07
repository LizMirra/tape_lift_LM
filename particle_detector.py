"""
Detect particles on a tape-lift image by contrast, automatically -- no need
to know in advance whether the background is light or dark 

Approach:
1. Grayscale + light Gaussian blur (reduces sensor noise so it isn't
   picked up as fake particles).
2. Estimate the background brightness as the image's median gray value.
   The median is robust to outliers, so as long as dust covers well under
   half the image, it lands on the tape's own brightness rather than
   being skewed by the dust.
3. Build a "deviation" image: how far each pixel's brightness is from
   that background level, regardless of direction. This is the key
   difference from a plain Otsu split -- it treats a particle darker than
   the tape and a particle lighter than the tape as the *same kind of
   anomaly*, so both are caught in one pass instead of needing separate
   dark/light thresholds like the old "painted" surface case did.
4. Otsu's method is applied to that deviation image to automatically pick
   how much deviation counts as "particle" vs "background noise/texture".
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


def segment_particles(gray, min_size=4, blur_sigma=1.0):
    """
    Returns a boolean mask the same shape as `gray`, True where a particle
    was detected. Catches particles both darker AND lighter than the
    background in a single pass -- see module docstring for why.

    min_size: particles smaller than this many pixels are discarded as
              noise -- raise this if you're getting speckle false positives,
              lower it if you're missing genuinely small dust.
    blur_sigma: Gaussian blur strength applied before thresholding.
    """
    smoothed = filters.gaussian(gray, sigma=blur_sigma, preserve_range=True)

    background_level = np.median(smoothed)
    deviation = np.abs(smoothed - background_level)

    dev_thresh = filters.threshold_otsu(deviation)
    particle_mask = deviation > dev_thresh

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
