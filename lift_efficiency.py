"""
Compare successive tape lifts and quantify how much each one improves.

Give it the *_particle_count_with_sizes.csv files written by dust_lift.py,
IN LIFT ORDER (lift 1 first):

    python lift_efficiency.py \\
        results/IMG00185/IMG00185_particle_count_with_sizes.csv \\
        results/IMG00186/IMG00186_particle_count_with_sizes.csv \\
        results/IMG00187/IMG00187_particle_count_with_sizes.csv \\
        -labels "Lift 1" "Lift 2" "Lift 3" -out efficiency/

dust_lift.py writes the run info (analysed area, scale, sensitivity, min size,
background image...) as '# key,value' lines at the top of each CSV; this script
reads the analysed area from there. For older CSVs without those lines, lifts are
compared PER IMAGE (assumes the same photo size, crop and scale for every lift),
or pass either
  -area-cm2 2.56                              (one value, or one per lift)
  -scale-um-per-px 8.2721 -image-px W H       (scale and cropped size in px)
To read these CSVs yourself in pandas:  pd.read_csv(path, comment='#', index_col=0)

Outputs (in -out):
  lift_efficiency_summary.csv   one row per lift: counts, density, area, reductions
  lift_efficiency_by_bin.csv    particles/cm^2 per size bin per lift + reductions
  lift_efficiency.png           figure

Definitions
  density            particles per cm^2 of tape analysed
  coverage           total particle area / analysed area (%)
  reduction vs prev  (1 - density_this / density_previous) x 100
  reduction vs first (1 - density_this / density_first) x 100
  share of total     this lift's density / sum of all lifts' densities x 100
                     (how much of all the dust collected came off in this lift)
  95% CI (-ci)       optional. Poisson counting uncertainty on the density ratio
                     (log method); off by default.
"""
import os
import sys
import csv
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_DIAMETERS = [0.5, 1, 2, 5, 10, 20, 50, 100]
SUFFIX = '_particle_count_with_sizes.csv'


def load_bins(spec_path):
    if spec_path and os.path.exists(spec_path):
        spec = pd.read_csv(spec_path, encoding='utf-8-sig')
        return spec['Area(um^2)'].to_numpy(float), spec['Diameter(um)'].tolist()
    d = DEFAULT_DIAMETERS
    return np.pi / 4 * np.array(d, float) ** 2, d


def bin_labels(diameters):
    labels = [f"<= {diameters[0]} um"]
    labels += [f"{diameters[i - 1]} - {diameters[i]} um" for i in range(1, len(diameters))]
    labels.append(f"> {diameters[-1]} um")
    return labels


def read_meta(csv_path):
    """Run info from the leading '# key,value' lines (empty dict for older CSVs)."""
    meta = {}
    with open(csv_path, newline='') as fh:
        for line in fh:
            if not line.startswith('#'):
                break
            row = next(csv.reader([line.rstrip('\r\n')]))
            if len(row) >= 2:
                meta[row[0].lstrip('# ').strip()] = row[1].strip()
    return meta


def find_area(csv_path, override, scale, image_px, meta=None):
    """Analysed area in cm^2, or None if it can't be determined (the CSV itself doesn't store it)."""
    if override is not None:
        return float(override), 'from -area-cm2'
    if scale is not None and image_px is not None:
        return image_px[0] * image_px[1] * scale ** 2 / 1e8, 'from -scale-um-per-px and -image-px'
    if meta and 'analyzed_area_cm2' in meta:
        return float(meta['analyzed_area_cm2']), 'from CSV header'
    return None, 'not given'


def reduction(n_prev, a_prev, n_cur, a_cur):
    """% reduction in density and its 95% Poisson CI. Returns (pct, lo, hi) or NaNs."""
    if n_prev == 0:
        return np.nan, np.nan, np.nan
    d_prev, d_cur = n_prev / a_prev, n_cur / a_cur
    ratio = d_cur / d_prev
    if n_cur == 0:
        hi_ratio = (3.0 / a_cur) / d_prev          # one-sided-ish upper bound for 0 events
        return 100.0, 100 * (1 - hi_ratio), 100.0
    se = np.sqrt(1 / n_prev + 1 / n_cur)
    lo_r, hi_r = ratio * np.exp(-1.96 * se), ratio * np.exp(1.96 * se)
    return 100 * (1 - ratio), 100 * (1 - hi_r), 100 * (1 - lo_r)


def main():
    ap = argparse.ArgumentParser(description='Efficiency of successive tape lifts.')
    ap.add_argument('csvs', nargs='+', help='*_particle_count_with_sizes.csv files, in lift order')
    ap.add_argument('-labels', nargs='+', help='Names for the lifts (default: Lift 1, Lift 2, ...)')
    ap.add_argument('-out', default='lift_efficiency', help='Output directory')
    ap.add_argument('-ci', action='store_true',
                    help='Also report 95%% Poisson confidence intervals on the %% reductions '
                         '(off by default; mostly useful when particle counts are small).')
    ap.add_argument('-spec', default='specification.csv', help='Specification CSV (for size bins)')
    ap.add_argument('-area-cm2', nargs='+', type=float,
                    help='Analysed area in cm^2: one value for all lifts, or one per lift')
    ap.add_argument('-scale-um-per-px', type=float,
                    help='Alternative to -area-cm2: scale used in dust_lift.py (with -image-px)')
    ap.add_argument('-image-px', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'),
                    help='Cropped image size in px used in dust_lift.py (with -scale-um-per-px)')
    args = ap.parse_args()

    n = len(args.csvs)
    if n < 2:
        sys.exit('Need at least two CSVs (one per lift).')
    labels = args.labels or [f"Lift {i + 1}" for i in range(n)]
    if len(labels) != n:
        sys.exit('-labels must have one name per CSV.')
    areas_in = args.area_cm2 or [None]
    if len(areas_in) not in (1, n):
        sys.exit('-area-cm2 takes 1 value or one per CSV.')
    if len(areas_in) == 1:
        areas_in = areas_in * n

    area_bins, diameters = load_bins(args.spec)
    blabels = bin_labels(diameters)
    os.makedirs(args.out, exist_ok=True)

    lifts = []
    for path, lab, a_over in zip(args.csvs, labels, areas_in):
        df = pd.read_csv(path, comment='#', index_col=0)
        meta = read_meta(path)
        area_cm2, src = find_area(path, a_over, args.scale_um_per_px, args.image_px, meta)
        areas_um2 = df['Area_um2'].to_numpy(float) if 'Area_um2' in df and len(df) else np.array([])
        diam = df['Diameter_um'].to_numpy(float) if 'Diameter_um' in df and len(df) else np.array([])
        idx = np.digitize(areas_um2, area_bins, right=False)
        counts = np.bincount(idx, minlength=len(area_bins) + 1)
        lifts.append(dict(label=lab, path=path, meta=meta, area_cm2=area_cm2, n=len(df), counts=counts,
                          total_area_um2=float(areas_um2.sum()),
                          median_d=float(np.median(diam)) if diam.size else np.nan,
                          max_d=float(diam.max()) if diam.size else np.nan))
        print(f"{lab}: {len(df)} particles, analysed area {area_cm2 if area_cm2 else 'unknown'}"
              f"{' cm^2' if area_cm2 else ''} ({src})")

    # Warn if the lifts were analysed with different settings (that would skew the comparison).
    for key in ('scale_um_per_px', 'analyzed_width_px', 'analyzed_height_px', 'min_size_px',
                'sensitivity', 'bg_sigma_px', 'background_image'):
        vals = [l['meta'].get(key) for l in lifts]
        if any(v is not None for v in vals) and len(set(vals)) > 1:
            print(f"WARNING: '{key}' differs between lifts: " + ', '.join(f"{l['label']}={v}" for l, v in zip(lifts, vals)))

    # Older CSVs don't store the image area. Without it, compare per image
    # (valid when every lift was photographed with the same field of view, crop and scale).
    per_image = any(l['area_cm2'] is None for l in lifts)
    if per_image:
        if not all(l['area_cm2'] is None for l in lifts):
            sys.exit('Area known for some lifts but not others -- give -area-cm2 for all of them.')
        print("\nNo area given: comparing particles PER IMAGE, assuming every lift has the same "
              "field of view/crop/scale.\n(Add -area-cm2 or -scale-um-per-px with -image-px for per-cm^2 values.)")
        for l in lifts:
            l['area_cm2'] = 1.0
    unit = 'image' if per_image else 'cm^2'

    if not per_image and len({round(l['area_cm2'], 3) for l in lifts}) > 1:
        print("Note: analysed areas differ between lifts; all comparisons use per-cm^2 densities.")

    dens = [l['n'] / l['area_cm2'] for l in lifts]
    total_dens = sum(dens)
    rows = []
    for i, l in enumerate(lifts):
        row = {
            'lift': l['label'], 'n_particles': l['n'],
            **({} if per_image else {'area_cm2': round(l['area_cm2'], 4)}),
            ('particles_per_image' if per_image else 'particles_per_cm2'): dens[i],
            **({} if per_image else {'coverage_pct': 100 * l['total_area_um2'] / (l['area_cm2'] * 1e8)}),
            'median_diameter_um': l['median_d'], 'max_diameter_um': l['max_d'],
            'share_of_total_pct': 100 * dens[i] / total_dens if total_dens else np.nan,
        }
        if i > 0:
            p = lifts[i - 1]
            r, lo, hi = reduction(p['n'], p['area_cm2'], l['n'], l['area_cm2'])
            row['reduction_vs_prev_pct'] = r
            if args.ci:
                row.update(vs_prev_ci95_low=lo, vs_prev_ci95_high=hi)
            f = lifts[0]
            r, lo, hi = reduction(f['n'], f['area_cm2'], l['n'], l['area_cm2'])
            row['reduction_vs_first_pct'] = r
            if args.ci:
                row.update(vs_first_ci95_low=lo, vs_first_ci95_high=hi)
            pa = lifts[i - 1]['total_area_um2'] / lifts[i - 1]['area_cm2']
            ca = l['total_area_um2'] / l['area_cm2']
            row['area_reduction_vs_prev_pct'] = 100 * (1 - ca / pa) if pa else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)

    # per-bin table
    brows = []
    for b, name in enumerate(blabels):
        row = {'size_bin': name}
        for i, l in enumerate(lifts):
            row[f"{l['label']} count"] = int(l['counts'][b])
            row[f"{l['label']} per_{'image' if per_image else 'cm2'}"] = l['counts'][b] / l['area_cm2']
        for i in range(1, n):
            p, c = lifts[i - 1], lifts[i]
            r, _, _ = reduction(int(p['counts'][b]), p['area_cm2'], int(c['counts'][b]), c['area_cm2'])
            row[f"{c['label']} reduction_vs_prev_pct"] = r
        brows.append(row)
    by_bin = pd.DataFrame(brows)

    summary.to_csv(os.path.join(args.out, 'lift_efficiency_summary.csv'), index=False)
    by_bin.to_csv(os.path.join(args.out, 'lift_efficiency_by_bin.csv'), index=False)

    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 30)
    print('\n=== Per lift ===')
    print(summary.round(2).to_string(index=False))
    print('\n=== Per size bin (only bins with particles) ===')
    show = by_bin[[c for c in by_bin.columns if c.endswith(' count') or c == 'size_bin']]
    show = show[show.drop(columns='size_bin').sum(axis=1) > 0]
    print(show.to_string(index=False))

    # ---- figure ----
    used = [b for b in range(len(blabels)) if any(l['counts'][b] for l in lifts)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [1.4, 1]})
    cmap = plt.get_cmap('viridis')
    w = 0.8 / n
    for i, l in enumerate(lifts):
        xs = np.arange(len(used)) + i * w - 0.4 + w / 2
        ax1.bar(xs, [l['counts'][b] / l['area_cm2'] for b in used], w, label=l['label'],
                color=cmap(i / max(n - 1, 1) * 0.85), edgecolor='black')
    ax1.set_xticks(np.arange(len(used)))
    ax1.set_xticklabels([blabels[b] for b in used], rotation=30, ha='right')
    ax1.set_ylabel(f"Particles per {'image' if per_image else 'cm$^2$'} ")
    ax1.set_title('Particles by size bin, per lift')
    ax1.legend()

    xs = np.arange(n)
    ax2.bar(xs, dens, color=[cmap(i / max(n - 1, 1) * 0.85) for i in range(n)], edgecolor='black')
    top = max(dens) if max(dens) > 0 else 1
    for i in range(n):
        txt = f"{dens[i]:.1f}"
        if i > 0 and not np.isnan(rows[i]['reduction_vs_prev_pct']):
            txt += f"\n{rows[i]['reduction_vs_prev_pct']:+.0f}% vs prev".replace('+', '-') \
                if rows[i]['reduction_vs_prev_pct'] >= 0 else \
                f"\n+{-rows[i]['reduction_vs_prev_pct']:.0f}% vs prev"
        ax2.text(i, dens[i] + 0.01 * top, txt, ha='center', va='bottom', fontsize=9)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, top * 1.25)
    ax2.set_ylabel(f"Particles per {'image' if per_image else 'cm$^2$'}")
    ax2.set_title('Total particle density per lift')
    fig.tight_layout()
    fig_path = os.path.join(args.out, 'lift_efficiency.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {args.out}/lift_efficiency_summary.csv, lift_efficiency_by_bin.csv, lift_efficiency.png")


if __name__ == '__main__':
    main()