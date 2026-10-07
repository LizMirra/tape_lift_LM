"""
Determine real-world scale (micrometers per pixel) for microscope images
that don't carry embedded metadata

Two ways to get the scale:
1. Manual (default, no display needed): you already know how many pixels
   correspond to a known real-world distance (e.g. you measured it in an
   image viewer, or the ruler spans the image's full width at a known
   length). Use compute_scale().
2. Interactive (needs a display, run this on your own machine, not headless):
   click two points on the ruler in a popup window, then type in the
   real-world distance between them. Use interactive_calibration().
"""
from PIL import Image


def get_image_dimensions(image_path):
    """
    Return (width_px, height_px) for the image.
    This reads only the file header via PIL, so it's fast even for large JPGs.
    """
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def compute_scale(real_length_mm, pixel_length):
    """
    Compute micrometers-per-pixel.

    real_length_mm: real-world distance in millimeters that the ruler
                    (or a known reference) shows.
    pixel_length:   number of pixels that distance spans in the image.
    """
    if pixel_length <= 0:
        raise ValueError("pixel_length must be > 0")
    if real_length_mm <= 0:
        raise ValueError("real_length_mm must be > 0")
    real_length_um = real_length_mm * 1000
    return real_length_um / pixel_length


def compute_scale_from_full_width(image_path, real_width_mm):
    """
    Calibrate using the *entire horizontal width* of the photo -- e.g. your
    ruler spans edge-to-edge across the top of the image and you know that
    whole span is, say, 5mm. No pixel measuring required: this reads the
    image's pixel width itself and does the division for you.
    """
    width_px, _height_px = get_image_dimensions(image_path)
    return compute_scale(real_width_mm, width_px)


def interactive_calibration(image_path):
    """
    Opens the image in a window. Click two points on the ruler (e.g. the
    0mm and 5mm tick marks), then press any key. You'll be asked for the
    real-world distance between the two points in millimeters, and this
    returns the resulting micrometers-per-pixel scale.

    Requires a display and opencv-python (pip install opencv-python).
    Run this on your own machine -- it will not work in a headless
    environment.
    """
    import cv2

    points = []
    window_name = "Click two ruler points, then press any key"

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not open {image_path}")

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_click)

    while True:
        display = img.copy()
        for p in points:
            cv2.circle(display, p, 5, (0, 0, 255), -1)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20)
        if len(points) == 2 or key != -1:
            break

    cv2.destroyAllWindows()

    if len(points) < 2:
        raise RuntimeError("Need two points to calibrate -- try again.")

    (x1, y1), (x2, y2) = points
    pixel_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    real_length_mm = float(input(
        f"Pixel distance measured: {pixel_length:.1f}px. "
        f"Enter the real-world distance between those two points (mm): "
    ))
    return compute_scale(real_length_mm, pixel_length)