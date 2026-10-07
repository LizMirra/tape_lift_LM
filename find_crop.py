"""
Usage:

    # Step 1: just see pixel-row gridlines, no crop line yet
    python find_crop.py photo.jpg

    # Step 2: once you've picked a candidate value, check it
    python find_crop.py photo.jpg --top 60

    # You can check a bottom crop too
    python find_crop.py photo.jpg --top 60 --bottom 40

Output is saved as <name>_crop_preview.jpg next to the original -- open
that file and read off which gridline lines up with the bottom edge of
your ruler strip. Re-run with that number as --top to confirm, then use
the same number with dust_lift.py's -crop-top flag.
"""
import argparse
import os
from PIL import Image, ImageDraw, ImageFont


def _load_font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
        except Exception:
            return ImageFont.load_default()


def draw_crop_preview(image_path, crop_top=None, crop_bottom=None, gridline_spacing=20):
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    width, height = img.size

    font = _load_font(14)

    # Draw a horizontal gridline + row-number label every `gridline_spacing`
    # pixels, only near the top and bottom where crops usually happen.
    band = 250  # how far down/up from each edge to draw gridlines
    for y in range(0, min(band, height), gridline_spacing):
        draw.line([(0, y), (60, y)], fill='lime', width=1)
        draw.text((64, y - 6), str(y), fill='lime', font=font)

    for y in range(max(height - band, 0), height, gridline_spacing):
        draw.line([(0, y), (60, y)], fill='cyan', width=1)
        draw.text((64, y - 6), str(y), fill='cyan', font=font)

    # Draw the candidate crop line(s) in red, full-width, if given
    if crop_top:
        draw.line([(0, crop_top), (width, crop_top)], fill='red', width=2)
        draw.text((width // 2 - 60, crop_top + 4), f"crop-top = {crop_top}", fill='red', font=_load_font(20))

    if crop_bottom:
        y = height - crop_bottom
        draw.line([(0, y), (width, y)], fill='red', width=2)
        draw.text((width // 2 - 60, y - 26), f"crop-bottom = {crop_bottom}", fill='red', font=_load_font(20))

    root, ext = os.path.splitext(image_path)
    out_path = f"{root}_crop_preview{ext}"
    img.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Preview crop lines to find the right -crop-top/-crop-bottom for dust_lift.py")
    parser.add_argument('image', help='Path to the image')
    parser.add_argument('--top', type=int, default=None, help='Candidate crop-top value to draw and check')
    parser.add_argument('--bottom', type=int, default=None, help='Candidate crop-bottom value to draw and check')
    parser.add_argument('--spacing', type=int, default=20, help='Gridline spacing in pixels (default 20)')
    args = parser.parse_args()

    out_path = draw_crop_preview(args.image, args.top, args.bottom, args.spacing)
    print(f"Saved preview to: {out_path}")
    print("Green gridlines = rows from the top, cyan = rows from the bottom.")
    print("Find the gridline that sits right at the boundary between the ruler")
    print("strip and the tape image -- that number is your -crop-top value.")


if __name__ == '__main__':
    main()