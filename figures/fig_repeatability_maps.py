"""Scan-rescan GluCEST maps for the 10 repeatability subjects (as in the ISMRM abstract, restyled for the paper).

Columns: subjects ordered young -> old (age on top). Rows: scan 1, scan 2 (matched slice from REPRO_SLICES).
GluCEST on a common 0-12% inferno scale, brain only on white; per-subject GM / WM CoV under each column.

The scan-2 maps live only on the raw data drive (F:). First run with the drive mounted writes
_cache/retest_maps.npz; after that it runs offline.   Run:  python fig_repeatability_maps.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import copy

import figdata as D
import nature_style as S

S.apply()
HERE = Path(__file__).resolve().parent
import nature_style as _S  # noqa: E402
CACHE = _S.CACHE / 'retest_maps.npz'
VMIN, VMAX = 0, 12
INF = copy(plt.get_cmap('inferno')); INF.set_bad(alpha=0)


def extract():
    """matched slice of scan A and scan B for each repeat subject -> _cache/retest_maps.npz (needs F:)"""
    sys.path.insert(0, str(HERE.parent))
    from utils_main import SCANS, REPRO_SLICES, DATA_DIR, scan_key, load_scan, SCAN_FOLDERS
    from extract_subregions import seg_for, LOCAL_SCAN_FOLDERS
    ## the data drive mounts under different letters (F:, E:, D:); use whichever has the data folder
    found = [p for p in [Path(DATA_DIR)] + [Path(f'{L}:/') / Path(DATA_DIR).name for L in 'FEDG'] if p.exists()]
    if not found:
        raise FileNotFoundError(f'{Path(DATA_DIR).name} not found on any drive: mount the data drive once to cache the scan-2 maps')
    data_dir = found[0]
    print('  data from', data_dir)
    out = {}
    for ini, (nA, zA, nB, zB) in REPRO_SLICES.items():
        if ini in D.EXCLUDE:
            continue
        for tag, n, z in (('A', nA, zA), ('B', nB, zB)):
            o = load_scan(SCANS[scan_key(ini, n)], data_dir)
            out[f'{ini}_{tag}_cest'] = np.asarray(o.CESTmap, float)[:, :, z]
            out[f'{ini}_{tag}_seg'] = seg_for(o, {**SCAN_FOLDERS, **LOCAL_SCAN_FOLDERS}[(ini, n)]).astype(int)[:, :, z]
        print('  cached', ini)
    np.savez_compressed(CACHE, **out)


def load(preview=False):
    """{subject: [(cest, seg) scan 1, (cest, seg) scan 2]}; preview=True fills both rows with the cached scan-1 slice"""
    if preview:
        sys.path.insert(0, str(HERE.parent))
        from utils_main import REPRO_SLICES
        return {s: [(D.vol(s)['CESTmap'][:, :, z], D.vol(s)['seg'][:, :, z])] * 2
                for s, (_, z, _, _) in REPRO_SLICES.items() if s not in D.EXCLUDE}
    if not CACHE.exists():
        extract()
    z = np.load(CACHE)
    subs = sorted({k.split('_')[0] for k in z.files})
    return {s: [(z[f'{s}_{t}_cest'], z[f'{s}_{t}_seg']) for t in 'P13'] for s in subs}


def cov_table():
    """per-subject scan-rescan CoV (%) = |A - B| / sqrt(2) / mean, as in Table 1"""
    r = D.retest().set_index('Subject')
    return {t: (100 * (r[f'{t}_A'] - r[f'{t}_B']).abs() / np.sqrt(2) / r[[f'{t}_A', f'{t}_B']].mean(axis=1)) for t in ['GM', 'WM']}


def make(preview=False):
    maps = load(preview)
    ages = dict(D.subjects_by_age())
    order = sorted(maps, key=lambda s: ages[s])
    cov = cov_table()
    n = len(order)
    x0, cw, gap, ch = 9, 15.2, 0.6, 17.5          ## mm: left margin, cell width, gap, cell height
    F = S.MMFigure(S.DOUBLE, 50)
    for j, s in enumerate(order):
        segs = [m[1] for m in maps[s]]
        ys, xs = np.where((segs[0] > 0) | (segs[1] > 0))
        sl = (slice(ys.min() - 2, ys.max() + 3), slice(xs.min() - 2, xs.max() + 3))
        for i, (c, sg) in enumerate(maps[s]):
            ax = F.ax(x0 + j * (cw + gap), 6 + i * (ch + 1), cw, ch); S.img_ax(ax)
            ok = (sg > 0) & np.isfinite(c) & (c > -10) & (c < 30) & (c != 0)
            im = ax.imshow(np.ma.masked_where(~ok, c)[sl], cmap=INF, vmin=VMIN, vmax=VMAX, interpolation='nearest')
        F.text(x0 + j * (cw + gap) + cw / 2, 3.5, f'{int(ages[s])} y', fontsize=6, fontweight='bold')
        if s in cov['GM'].index:
            F.text(x0 + j * (cw + gap) + cw / 2, 6 + 2 * ch + 1 + 2.6,
                   f'GM {cov["GM"][s]:.1f}\nWM {cov["WM"][s]:.1f}', fontsize=4.8, linespacing=1.15, va='top')
    for i, lab in enumerate(['Scan 1', 'Scan 2']):
        F.text(x0 - 2.2, 6 + i * (ch + 1) + ch / 2, lab, fontsize=6.5, fontweight='bold', rotation=90)
    F.text(x0 - 2.2, 6 + 2 * ch + 1 + 4.2, 'CoV (%)', fontsize=4.8, rotation=90)
    xc = x0 + n * (cw + gap) + 1
    S.colorbar_inset(F.fig, im, F.ax(xc, 8, 1.3, 2 * ch - 3), 'GluCEST (%)', ticks=[0, 3, 6, 9, 12], orientation='vertical')
    F.save('Fig_repeatability_maps' + ('_PREVIEW' if preview else ''))


if __name__ == '__main__':
    make(preview='--preview' in sys.argv)
