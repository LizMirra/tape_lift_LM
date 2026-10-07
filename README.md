# dust_lift

## Files

| File | Purpose |
|---|---|
| `dust_lift.py` | Main script — run this |
| `scale_calibration.py` | Gets image pixel size, computes µm/pixel from a ruler measurement |
| `particle_detector.py` | Automatic contrast-based particle detection |
| `particle_size_converter.py` | Pixel → µm conversion |
| `overlay_and_label.py` | Draws particle outlines + numbers on the image (like the old `ParticleLabeler`) |
| `specification.csv` | Same spec file used by `tape_lift.py` |

## Install

```
pip install pillow numpy pandas matplotlib scikit-image scipy
```

(`opencv-python` is optional — only needed if you want the interactive
click-to-calibrate ruler option, see below.)

## Basic usage

The simplest way to calibrate: if your ruler runs edge-to-edge across the
top of the photo and shows a known total length (say, 5mm across the full
width), just tell the script that -- it reads the image's own pixel width
and works out the scale itself, no pixel measuring involved:

```
python dust_lift.py -img photo1.jpg photo2.jpg -save results/ -crop-top 60 -image-width-mm 5.0
```

- `-img`: one or more JPGs to analyze
- `-save`: output directory (one subfolder per image gets created)
- `-crop-top` / `-crop-bottom`: pixels to crop off (removes your ruler strip,
  timestamp, etc. — same idea as the old 30px/40px crop in `tape_lift.py`).
  Use `find_crop.py` (see below) if you're not sure what number to use.
- `-image-width-mm`: the real-world length the *entire width* of the photo
  spans. This is computed fresh per image, so it's fine even if different
  photos have slightly different pixel resolutions.
- `-min-size`: minimum particle size in pixels, to filter out sensor noise
  (default 4 — raise it if tiny speckle noise is being counted as particles)

### Other ways to calibrate

If your ruler only covers *part* of the image width rather than edge-to-edge,
you can instead give a specific pixel span (measured by hovering in an image
viewer over two known tick marks and subtracting their x-coordinates):

```
python dust_lift.py -img photo1.jpg -save results/ -crop-top 60 -ruler 5.0 812
```

(`-ruler 5.0 812` = a 5.0mm span measured 812 pixels wide.)

If you already know the µm/pixel scale outright (e.g. from the microscope's
spec sheet, or a prior calibration), skip both of the above:

```
python dust_lift.py -img photo1.jpg -save results/ -crop-top 60 -scale-um-per-px 6.15
```

These three options (`-image-width-mm`, `-ruler`, `-scale-um-per-px`) are
mutually exclusive — use exactly one per run.

For an even more precise pixel-span measurement, `scale_calibration.py` also
has `interactive_calibration(image_path)`: it opens the image, you click two
points on the ruler, type in the real-world distance, and it computes the
scale for you. Requires `opencv-python` and a display — run it from a
Python shell on your own machine, not headless:

```python
from scale_calibration import interactive_calibration
scale = interactive_calibration("photo1.jpg")
print(scale)  # µm/pixel — pass this to -scale-um-per-px
```

## Finding the right -crop-top value

Run `find_crop.py` on a sample photo -- it draws a labeled pixel-row ruler
near the top and bottom of the image so you can read off the crop line
directly instead of guessing:

```
python find_crop.py photo1.jpg
```

This saves `photo1_crop_preview.jpg` with green gridlines (every 20px)
labeled by row number. Find the line that sits right at the boundary
between your ruler strip and the tape/dust area, then confirm it:

```
python find_crop.py photo1.jpg --top 60
```

That draws a solid red line at row 60 across the full image so you can
visually check it lands exactly where you want before using it as
`-crop-top` in `dust_lift.py`.

## How detection works (no surface type needed)

Old code: pick a fixed brightness window per surface type (`metal` used one
window; `painted` needed *two* passes — one for dark particles, one for
light particles — because either could appear).

New code (`particle_detector.py`):
1. Blur slightly to suppress sensor noise.
2. Take the image's **median** brightness as the background level (robust to
   outliers as long as dust covers well under half the image).
3. Build a "deviation" image: `abs(pixel - background)`, regardless of
   direction.
4. Auto-threshold that deviation image (Otsu's method) to decide how much
   deviation counts as "particle."

This is what lets one pass catch dust that's *either* darker or lighter than
the tape, without you telling it which — the two-pass dark/light split from
the `painted` surface case is no longer needed.

## Path to SAM

If you later want to swap in a pretrained Segment Anything Model, it slots
in as a drop-in replacement for `segment_particles()` in
`particle_detector.py` — SAM would hand back a similar boolean/label mask,
and everything downstream (`measure_particles`, conversion, labeling,
histogram) stays as-is.

## Known limitations (carried over from tape_lift.py, same idea)

- The "minority pixel count = particle" logic in `segment_particles`
  assumes dust covers a small fraction of the image. If a photo is mostly
  dust/debris, the background-median estimate will be wrong. This mirrors
  the old code's own assumption baked into per-surface thresholds.
- `MinFeret` is approximated as the minor axis length of the best-fit
  ellipse — scikit-image doesn't compute a true minimum-caliper diameter
  the way ImageJ's Feret measurement does. Close enough for binning, but
  not identical to the ImageJ numbers from `tape_lift.py`.
- If a zero-particle image comes through (e.g. a clean control), the script
  handles it fine (unlike the original bug you hit with `tape_lift.py`'s
  painted/light channel) — it'll print "No particles detected" and continue.