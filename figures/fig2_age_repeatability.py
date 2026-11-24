"""Figure 2 (merged): the 10 rescanned participants ordered young -> old.

a  repeatability: MP2RAGE, GluCEST scan 1, GluCEST scan 2 (matched slices), per-subject GM / WM CoV underneath
b  age: GluCEST in gray matter and white matter (scan 1, on the MP2RAGE)
One plasma 0-12 % scale. Scan-2 maps come from fig_repeatability_maps (cached once from the F: drive).

Run:  python fig2_age_repeatability.py            (python fig2_age_repeatability.py --preview: scan 1 in both rows)
"""
import sys

import numpy as np

import figdata as D
import nature_style as S
from nature_style import TISSUE_COL as TC
import fig_repeatability_maps as R

S.apply()
LIM = (0, 12)


def make(preview=False, extra=0, after=None, out=None):
    # extra / after: room and a callback for more panels underneath (used by fig2_merged.py)
    cmap = S.mpl.colormaps['plasma'].copy(); cmap.set_bad(alpha=0)   ## plasma: saturated yellow top stays visible on white
    grey = S.mpl.colormaps['gray'].copy(); grey.set_bad('white')
    maps = R.load(preview)                                   ## {subject: [(cest, seg) scan 1, (cest, seg) scan 2]}
    sys.path.insert(0, str(R.HERE.parent))
    from utils_main import REPRO_SLICES
    zA = {s: REPRO_SLICES[s][1] for s in maps}                ## scan-1 slice of the matched pair
    strip = [(s, a) for s, a in D.subjects_by_age() if s in maps]
    cov = R.cov_table()
    crop, H, W = D.common_crop([s for s, _ in strip], pad=2)
    x0, g, n = 16, 0.8, len(strip)
    mw = (S.DOUBLE - x0 - 12 - (n - 1) * g) / n
    mh = mw * H / W
    rg, cov_h, gap_b = 1.0, 6.5, 4.0
    ya = 13                                                    ## top of block a
    yb = ya + 3 * (mh + rg) + cov_h + gap_b                    ## top of block b
    FH = yb + 2 * (mh + rg) + 2 + extra
    F = S.MMFigure(S.DOUBLE, FH)

    ## young -> old arrow and ages
    F.fig.add_artist(S.mpl.patches.FancyArrowPatch((x0 / F.W, 1 - 4 / F.H), ((S.DOUBLE - 12) / F.W, 1 - 4 / F.H),
                                                   transform=F.fig.transFigure, arrowstyle='-|>', mutation_scale=6,
                                                   lw=0.6, color='k'))
    F.text(x0, 2.6, 'Younger', ha='left', va='bottom', fontsize=6, fontweight='bold')
    F.text(S.DOUBLE - 14, 2.6, 'Older', ha='right', va='bottom', fontsize=6, fontweight='bold')
    for i, (s, age) in enumerate(strip):
        F.text(x0 + i * (mw + g) + mw / 2, 9, f'{age} y', fontsize=6, fontweight='bold')

    def cell(i, y):
        ax = F.ax(x0 + i * (mw + g), y, mw, mh); S.img_ax(ax)
        return ax

    def valid(c, sg, codes):
        return (np.isin(sg, codes)) & np.isfinite(c) & (c > -10) & (c < 30) & (c != 0)

    ## a: MP2RAGE, scan 1, scan 2, CoV
    F.letter('a', 1, ya - 3)
    rows_a = [('MP2RAGE', 'k'), ('Scan 1', 'k'), ('Scan 2', 'k')]
    for r, (name, col) in enumerate(rows_a):
        y = ya + r * (mh + rg)
        F.text(x0 - 2, y + mh / 2, name, rotation=90, fontsize=6, fontweight='bold', color=col)
        for i, (s, _) in enumerate(strip):
            ax = cell(i, y)
            if r == 0:
                ax.imshow(crop(D.underlay(s, zA[s], 'UNI'), s), cmap=grey, vmin=0, vmax=1.35)
            else:
                c, sg = maps[s][r - 1]
                im = ax.imshow(crop(np.ma.masked_where(~valid(c, sg, (1, 2, 3)), c), s), cmap=cmap,
                               vmin=LIM[0], vmax=LIM[1], interpolation='nearest')
    yc = ya + 3 * (mh + rg) + 0.8
    F.text(x0 - 2, yc + 2.2, 'CoV (%)', rotation=90, fontsize=4.8)
    for i, (s, _) in enumerate(strip):
        F.text(x0 + i * (mw + g) + mw / 2, yc, f'GM {cov["GM"][s]:.1f}\nWM {cov["WM"][s]:.1f}', fontsize=5,
               linespacing=1.15, va='top')

    ## b: gray matter and white matter (scan 1) on the MP2RAGE
    F.letter('b', 1, yb - 3)
    for r, (name, codes, col) in enumerate([('Gray matter', (2,), TC['GM']), ('White matter', (3,), TC['WM'])]):
        y = yb + r * (mh + rg)
        F.text(x0 - 2, y + mh / 2, name, rotation=90, fontsize=6, fontweight='bold', color=col)
        for i, (s, _) in enumerate(strip):
            ax = cell(i, y)
            c, sg = D.vol(s)['CESTmap'][:, :, zA[s]], D.vol(s)['seg'][:, :, zA[s]]   ## the paper's scan (same session as the MP2RAGE)
            ax.imshow(crop(D.underlay(s, zA[s], 'UNI'), s), cmap=grey, vmin=0, vmax=1.35)
            ax.imshow(crop(np.ma.masked_where(~valid(c, sg, codes), c), s), cmap=cmap, vmin=LIM[0], vmax=LIM[1],
                      interpolation='nearest')

    ## one colour bar for every GluCEST row (a and b share the 0-12 % scale)
    y1, y2 = ya + (mh + rg) + 2, yb + 2 * (mh + rg) - 3
    S.colorbar_inset(F.fig, im, F.ax(S.DOUBLE - 9, y1, 1.4, y2 - y1), 'GluCEST (%)',
                     ticks=[0, 3, 6, 9, 12], orientation='vertical')
    if preview:   ## make the placeholder impossible to miss: scan 2 is a copy of scan 1 until the F: drive is cached
        F.fig.text(0.5, 1 - (ya + 2 * (mh + rg) + mh / 2) / F.H,
                   'PREVIEW: Scan 2 row duplicates Scan 1 (scan-2 maps not yet loaded)', ha='center', va='center',
                   fontsize=9, fontweight='bold', color='red',
                   bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='red', lw=1, alpha=0.9))
    if after:
        after(F, yb + 2 * (mh + rg) + 2)
    F.save(out or 'Fig2_age_repeatability' + ('_PREVIEW' if preview else ''))


if __name__ == '__main__':
    make(preview='--preview' in sys.argv)
