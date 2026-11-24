"""Supplemental figures for the GluCEST WM ageing paper.

supp_t1()      Supp. Fig. 4: GluCEST vs T1 in GM and WM (runs offline from _cache, same numbers as the analysis notebook)
field_maps()   Supp. Figs. 2-3: B0 / B1 maps, scan 1 vs scan 2 (needs the raw data drive, uses ../utils_main.py)
"""
import sys
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import pearsonr

import figdata as D
import nature_style as S

TC = S.TISSUE_COL
TIS = ['GM', 'WM']
NAME = {'GM': 'Gray matter', 'WM': 'White matter'}


def rp(r, p):
    return S.minus(f'r = {r:.2f}, P = {p:.2f}' if p >= 0.01 else f'r = {r:.2f}, P = {p:.3f}')


def fit_ci(ax, x, y, col):
    """scatter + least-squares line + 95% CI, returns pearson r, p"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    S.reg_plot(ax, x, y, col, s=10)
    return pearsonr(x, y)


def supp_t1(young='P01', old='P17', z=6):
    S.apply()
    tm = D.tissue_means()
    d = {t: tm[tm.Tissue == t].reset_index(drop=True) for t in TIS}

    ## stats: does GluCEST explain age beyond T1?
    st = {}
    for t in TIS:
        g = d[t]
        full = smf.ols('Age ~ T1 + GluCEST', g).fit()
        st[t] = dict(d_glu=full.rsquared - smf.ols('Age ~ T1', g).fit().rsquared,
                     d_t1=full.rsquared - smf.ols('Age ~ GluCEST', g).fit().rsquared,
                     p_glu=full.pvalues['GluCEST'], p_t1=full.pvalues['T1'],
                     rx=smf.ols('Age ~ T1', g).fit().resid, ry=smf.ols('GluCEST ~ T1', g).fit().resid)

    F = S.MMFigure(S.DOUBLE, 130)
    inf = copy(plt.get_cmap('inferno')); inf.set_bad('black')
    gry = copy(plt.get_cmap('gray')); gry.set_bad('black')
    crop, H, W = D.common_crop([young, old])
    ih = 36; iw = ih * W / H

    ## a: GluCEST and T1 maps, youngest vs oldest
    F.letter('a', 1, 1)
    ages = dict(D.subjects_by_age())
    for i, s in enumerate([young, old]):
        v = D.vol(s); seg = v['seg'][:, :, z]
        xg = 8 + i * (2 * iw + 10)
        F.text(xg + iw, 4, f'Age {ages[s]}', fontsize=6.5, fontweight='bold')
        for j, (arr, cm, lim, lab) in enumerate([(v['CESTmap'][:, :, z], inf, (0, 13), 'GluCEST'),
                                                 (v['T1img'][:, :, z], gry, (900, 2200), 'T1')]):
            ax = F.ax(xg + j * (iw + 1.5), 9, iw, ih); S.img_ax(ax)
            im = ax.imshow(crop(np.ma.masked_where(seg == 0, arr), s), cmap=cm, vmin=lim[0], vmax=lim[1])
            ax.set_title(lab, fontsize=6, pad=1.5)
            if i == 1:
                S.colorbar_inset(F.fig, im, F.ax(xg + 2 * (iw + 1.5) + 4 + j * 13, 12, 1.4, ih - 6),
                                 'GluCEST (%)' if j == 0 else 'T1 (ms)', orientation='vertical')

    ## b-d: GluCEST vs age, T1 vs age, GluCEST vs T1
    y2, h2 = 56, 28
    panels = [('Age', 'GluCEST', 'Age (years)', 'GluCEST (%)', 'b'),
              ('Age', 'T1', 'Age (years)', 'T1 (ms)', 'c'),
              ('T1', 'GluCEST', 'T1 (ms)', 'GluCEST (%)', 'd')]
    for k, (xcol, ycol, xl, yl, let) in enumerate(panels):
        F.letter(let, 1 + k * 61, y2 - 5)
        for m, t in enumerate(TIS):
            x_ = 14 + k * 61
            ax = F.ax(x_, y2 + m * (h2 / 2 + 3), 42, h2 / 2 - 1.5)
            r, p = fit_ci(ax, d[t][xcol], d[t][ycol], TC[t])
            ax.text(0.02, 0.98, f'{t}  {rp(r, p)}', transform=ax.transAxes, va='top', fontsize=5.3, color=TC[t])
            ax.margins(y=0.3)
            if m == 1:
                ax.set_xlabel(xl)
            else:
                ax.tick_params(labelbottom=False)
            S.despine(ax)
        F.text(x_ - 9, y2 + h2 / 2, yl, rotation=90, fontsize=6)

    ## e: added-variable plots (T1 removed from both age and GluCEST)
    y3 = 100
    F.letter('e', 1, y3 - 5)
    for k, t in enumerate(TIS):
        ax = F.ax(14 + k * 46, y3, 36, 24)
        r, _ = fit_ci(ax, st[t]['rx'], st[t]['ry'], TC[t])
        ax.axhline(0, color='0.7', lw=0.4); ax.axvline(0, color='0.7', lw=0.4)
        ax.set_title(f'{NAME[t]}\n{rp(r, st[t]["p_glu"])}', fontsize=5.8, color=TC[t], pad=2)
        ax.set_xlabel('Age | T1 (years)'); ax.set_ylabel('GluCEST | T1 (%)' if k == 0 else '')
        S.despine(ax)

    ## f: unique variance in age explained
    F.letter('f', 106, y3 - 5)
    ax = F.ax(118, y3, 60, 24)
    for k, t in enumerate(TIS):
        bars = [(st[t]['d_glu'], st[t]['p_glu'], dict(color='0.25', label='GluCEST beyond T1')),
                (st[t]['d_t1'], st[t]['p_t1'], dict(facecolor='white', edgecolor='0.25', hatch='//////', lw=0.5,
                                                    label='T1 beyond GluCEST'))]
        for j, (val, p, kw) in enumerate(bars):
            x = k + (j - 0.5) * 0.2
            ax.bar(x, val, 0.18, **{**kw, 'label': kw['label'] if k == 0 else '_nolegend_'})
            ax.text(x, val + 0.01, f'P = {p:.2f}' if p >= 0.01 else f'P = {p:.3f}', ha='center', va='bottom',
                    fontsize=5)
    ax.set_xticks([0, 1], TIS); ax.set_ylim(0, 0.6); ax.set_ylabel('ΔR² (age)')
    ax.legend(fontsize=5.3, loc='upper right'); S.despine(ax)
    F.save('Supp_Fig_T1_vs_GluCEST')


def field_maps(kind='B0'):
    """B0 or B1 map, scan 1 vs scan 2 for the 10 repeat subjects (needs the raw .mat data drive)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils_main import SCANS, REPRO_SLICES, AGE_MAP, DATA_DIR, scan_key, load_scan
    if not Path(DATA_DIR).exists():   ## check the drive before making an empty figure
        raise FileNotFoundError(f'{DATA_DIR} not found')
    field, cmap, lim = {'B0': ('B0map', 'RdBu_r', (-0.2, 0.2)), 'B1': ('B1map', 'turbo', (0, 1.2))}[kind]
    order = sorted(REPRO_SLICES, key=lambda i: AGE_MAP[scan_key(i, 1)])   ## young -> old
    fig, axes = plt.subplots(2, len(order), figsize=(len(order) * 1.2, 3.1),
                             gridspec_kw=dict(hspace=0.06, wspace=0.04))
    for col, ini in enumerate(order):
        nA, zA, nB, zB = REPRO_SLICES[ini][:4]
        for row, (n, z) in enumerate([(nA, zA), (nB, zB)]):
            out = load_scan(SCANS[scan_key(ini, n)])
            seg = np.asarray(out.segmask, int)[:, :, z]
            m = np.asarray(getattr(out, field), float)[:, :, z]
            im = axes[row, col].imshow(np.ma.masked_where(seg == 0, m), cmap=cmap, vmin=lim[0], vmax=lim[1])
            axes[row, col].set_axis_off()
        axes[0, col].set_title(f'Subject {col + 1}', fontsize=8)
    for row, lab in enumerate(['Scan 1', 'Scan 2']):
        axes[row, 0].text(-0.1, 0.5, lab, transform=axes[row, 0].transAxes, rotation=90, ha='right', va='center')
    fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01, label=kind)
    fig.suptitle(f'{kind} map, Scan 1 vs Scan 2', fontweight='bold', fontsize=9)
    plt.savefig(f'Supp_Fig_{kind}_maps.png', bbox_inches='tight', dpi=300)
    plt.show()
