"""
Dust-on-tape particle analysis for microscope JPGs that don't carry
embedded scale metadata.

Unlike tape_lift.py, this does NOT ask you to specify a surface type
(metal/painted).

Workflow per image:
1. Read the JPG's pixel dimensions.
2. Crop off the ruler strip (and anything else) from the top/bottom.
3. Compute micrometers-per-pixel from a manually supplied ruler
   measurement, or use a value you already know.
4. Detect particles automatically (contrast-deviation threshold -- see
   particle_detector.py for details). By default, "background" means this
   image's own median brightness. You can instead point -background-img
   at a clean reference photo of the same tape (no dust), taken from the
   same camera position, and each pixel gets compared against that
   photo's brightness at the same spot -- more precise if the lighting
   isn't perfectly even across the frame.
5. Convert pixel measurements to micrometers.
6. Draw a labeled overlay image.
7. Bin particles by size and compare against specification.csv (same
   file/format used by tape_lift.py), plotting a pass/fail histogram.

Example:
    python dust_lift.py -img sample1.jpg sample2.jpg -save results/ \\
        -crop-top 60 -ruler 5.0 812

    (crops 60px off the top for the ruler strip, and calibrates using a
    5.0mm ruler span that measured 812px wide in the image)

Example with a background reference photo:
    python dust_lift.py -img sample1.jpg -save results/ \\
        -crop-top 60 -ruler 5.0 812 -background-img clean_tape.jpg
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

from scale_calibration import get_image_dimensions, compute_scale, compute_scale_from_full_width
from particle_detector import load_grayscale, load_background, segment_particles, measure_particles
from particle_size_converter import ParticleSizeConverter
from overlay_and_label import ParticleOverlay


def crop_ruler(image_path, crop_top=0, crop_bottom=0):
    """Crop the ruler strip (and anything else) off the image; returns temp path + new size."""
    img = Image.open(image_path)
    width, height = img.size
    box = (0, crop_top, width, height - crop_bottom)
    cropped = img.crop(box)
    root, ext = os.path.splitext(image_path)
    cropped_path = f"{root}_cropped_temp{ext}"
    cropped.save(cropped_path)
    return cropped_path, cropped.size


def analyze_image(image_path, save_location, scale_mode, scale_value, crop_top, crop_bottom,
                   spec_path='specification.csv', min_particle_size=4, background_image_path=None):

    file_name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(save_location, file_name)
    os.makedirs(out_dir, exist_ok=True)

    width_px, height_px = get_image_dimensions(image_path)
    print(f"Original image size: {width_px} x {height_px} px")

    if scale_mode == 'full_width':
        # scale_value here is the real-world width (mm) the whole photo spans.
        scale_um_per_pixel = compute_scale(scale_value, width_px)
        print(f"Scale for this image: {scale_um_per_pixel:.4f} um/pixel "
              f"(from full image width = {scale_value} mm across {width_px}px)")
    else:
        scale_um_per_pixel = scale_value

    cropped_path, (cropped_w, cropped_h) = crop_ruler(image_path, crop_top, crop_bottom)
    print(f"Cropped image size (ruler removed): {cropped_w} x {cropped_h} px")

    # --- Optional background reference photo (cropped the same way, so
    #     it lines up pixel-for-pixel with the image being analyzed) ---
    background = None
    cropped_bg_path = None
    if background_image_path is not None:
        cropped_bg_path, (bg_w, bg_h) = crop_ruler(background_image_path, crop_top, crop_bottom)
        if (bg_w, bg_h) != (cropped_w, cropped_h):
            if os.path.exists(cropped_bg_path):
                os.remove(cropped_bg_path)
            if os.path.exists(cropped_path):
                os.remove(cropped_path)
            raise ValueError(
                f"Background image '{background_image_path}' is {bg_w}x{bg_h} after cropping, "
                f"but '{image_path}' is {cropped_w}x{cropped_h}. They need to match -- use a "
                f"background photo taken at the same resolution and camera position."
            )
        background = load_background(cropped_bg_path, blur_sigma=1.0)
        print(f"Using background reference: {background_image_path}")

    # --- Particle detection (automatic contrast, no surface type) ---
    gray = load_grayscale(cropped_path)
    particle_mask = segment_particles(gray, min_size=min_particle_size, background=background)
    df = measure_particles(particle_mask, intensity_image=gray)

    if len(df) == 0:
        print("No particles detected in this image.")
    elif len(df) > 1000:
        print("#" * 60)
        print(f"Warning: Detected {len(df)} particles -- check for noise or a bad threshold.")
        print("Try raising -min-size, or inspect the overlay image.")
        print("#" * 60)
    else:
        print(f"Detected {len(df)} particles.")

    raw_csv_path = os.path.join(out_dir, f"{file_name}_particle_count.csv")
    df.to_csv(raw_csv_path)

    # --- Convert to real-world units ---
    converter = ParticleSizeConverter(scale_um_per_pixel)
    df_sizes = converter.convert(df)
    sized_csv_path = os.path.join(out_dir, f"{file_name}_particle_count_with_sizes.csv")
    df_sizes.to_csv(sized_csv_path)
    print(f"Saved sized particle data to: {sized_csv_path}")

    # --- Overlay + labels ---
    overlay_path = os.path.join(out_dir, f"{file_name}_overlay_labeled.png")
    overlay = ParticleOverlay()
    overlay.draw(cropped_path, particle_mask, df_sizes, overlay_path, scale_um_per_pixel)
    print(f"Saved labeled overlay to: {overlay_path}")

    if os.path.exists(cropped_path):
        os.remove(cropped_path)
    if cropped_bg_path is not None and os.path.exists(cropped_bg_path):
        os.remove(cropped_bg_path)

    # --- Bin by size and compare to specification ---
    df_spec = pd.read_csv(spec_path)
    area_bins = df_spec['Area(um^2)'].tolist()
    diameters = df_spec['Diameter(um)'].tolist()

    particle_areas = df_sizes['Area_um2'].tolist()
    bin_indices = np.digitize(particle_areas, area_bins, right=False)
    bin_counts = np.bincount(bin_indices, minlength=len(area_bins) + 1)

    bin_labels = []
    for i in range(len(area_bins)):
        if i == 0:
            bin_labels.append(f"<= {diameters[i]} um")
        else:
            bin_labels.append(f"{diameters[i - 1]} - {diameters[i]} um")
    bin_labels.append(f"> {diameters[-1]} um")

    image_area_cm2 = converter.image_area_cm2(cropped_w, cropped_h)
    counts_per_cm2 = np.array(bin_counts) / image_area_cm2

    spec_counts = np.array(df_spec["Allowed_particle_count(counts/cm^2 for 500ng/cm^2)"])
    spec_counts = np.append(spec_counts, 0)  # no limit specified for the largest bin's overflow

    pass_fail = counts_per_cm2 <= spec_counts
    colors = ['green' if pf else 'red' for pf in pass_fail]

    for counts, spec, pf in zip(counts_per_cm2, spec_counts, pass_fail):
        status = "within" if pf else "EXCEEDS"
        print(f"Count {counts:.3f}/cm^2 {status} specification {spec}")

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(bin_counts)), bin_counts, color=colors, edgecolor='black', alpha=0.8)
    plt.xlabel('Particle Size Range')
    plt.ylabel('Count')
    plt.title('Equivalent Spherical Particle Diameter Distribution')
    plt.xticks(range(len(bin_counts)), bin_labels, rotation=45, ha='right')
    plt.legend(handles=[Patch(facecolor='green', edgecolor='black', label='Pass'),
                         Patch(facecolor='red', edgecolor='black', label='Fail')])
    plt.tight_layout()
    hist_path = os.path.join(out_dir, "particle_size_distribution.png")
    plt.savefig(hist_path)
    plt.close()
    print(f"Saved histogram to: {hist_path}")
    print("Done.\n")


def main():
    parser = argparse.ArgumentParser(
        prog='DustLiftAnalysis',
        description='Detect and size dust particles on tape-lift microscope JPGs '
                    '(automatic contrast-based detection, no surface type needed)')
    parser.add_argument('-img', '--Images', nargs='+', required=True, help='Path(s) to JPG images')
    parser.add_argument('-save', '--SaveLocation', required=True, help='Directory to save results')
    parser.add_argument('-spec', '--Specification', default='specification.csv',
                        help='Path to specification CSV (default: specification.csv in cwd)')
    parser.add_argument('-crop-top', type=int, default=0, help='Pixels to crop off the top (ruler strip)')
    parser.add_argument('-crop-bottom', type=int, default=0, help='Pixels to crop off the bottom')
    parser.add_argument('-min-size', type=int, default=4,
                        help='Minimum particle size in pixels; raise this if noise is being counted as particles')
    parser.add_argument('-background-img', default=None,
                        help='Optional path to a clean reference photo of the same tape with no dust '
                             'on it, taken from the same camera position. When given, particles are '
                             'detected by comparing each pixel against this photo\'s brightness at the '
                             'same position, instead of this image\'s own median brightness. The '
                             'reference is cropped with the same -crop-top/-crop-bottom as the image '
                             'being analyzed, so it must match in resolution and framing. If omitted '
                             '(default), detection works exactly as before, using the analyzed image '
                             'itself to estimate the background.')

    scale_group = parser.add_mutually_exclusive_group(required=True)
    scale_group.add_argument('-scale-um-per-px', type=float,
                             help='Directly specify micrometers-per-pixel, if already known')
    scale_group.add_argument('-image-width-mm', type=float,
                             help='Simplest option: the real-world horizontal length the whole '
                                  'photo spans (e.g. your ruler runs edge-to-edge across the top '
                                  'and shows 5mm total). The script reads the image\'s pixel width '
                                  'itself and works out the scale -- no pixel measuring needed. '
                                  'Example: -image-width-mm 5.0')
    scale_group.add_argument('-ruler', nargs=2, type=float, metavar=('REAL_MM', 'PIXEL_LENGTH'),
                             help='Calibrate from a specific pixel span instead of the full width: '
                                  'real-world distance in mm, then the pixel distance it spans in '
                                  'the image. Example: -ruler 5.0 812 means a 5.0mm ruler span '
                                  'measured 812 pixels wide. Use -image-width-mm instead if your '
                                  'ruler spans the entire image width -- it\'s simpler.')

    args = parser.parse_args()

    if args.scale_um_per_px is not None:
        scale_mode, scale_value = 'direct', args.scale_um_per_px
        print(f"Using scale: {scale_value:.4f} um/pixel")
    elif args.image_width_mm is not None:
        scale_mode, scale_value = 'full_width', args.image_width_mm
        print(f"Calibrating from full image width = {scale_value} mm (computed per image below)")
    else:
        real_mm, pixel_len = args.ruler
        scale_mode, scale_value = 'direct', compute_scale(real_mm, pixel_len)
        print(f"Using scale: {scale_value:.4f} um/pixel (from {real_mm}mm across {pixel_len}px)")

    os.makedirs(args.SaveLocation, exist_ok=True)

    for path in args.Images:
        if not path.lower().endswith(('.jpg', '.jpeg')):
            print(f"Skipping {path}: must be a JPG")
            continue
        analyze_image(path, args.SaveLocation, scale_mode, scale_value, args.crop_top, args.crop_bottom,
                     spec_path=args.Specification, min_particle_size=args.min_size,
                     background_image_path=args.background_img)


if __name__ == '__main__':
    main()