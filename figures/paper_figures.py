"""Final figure set for the 7T GluCEST white-matter paper (P18 excluded, corrected atlas orientation).

Reads only _cache/ (written by extract_data.py / extract_subregions.py from F:), so it runs offline.
Colormaps: inferno for GluCEST and for B1.

  Fig1_maps            a anatomy + GM/WM masks, GluCEST, B1 (representative subject, 6 slices)
                       b GluCEST in all 17 participants, ordered by age
  Fig2_repeatability   a tissue CoV  b ICC  c-e scan vs rescan  f-h Bland-Altman  i subregion CoV
  Fig3_age_trends      GluCEST vs age in GM, WM, CSF
  Fig4_slope_difference  CR vs CC + cingulum: family trends, within-subject difference, robustness
  Table1_tissue_CoV_ICC, Table2_subregion_CoV   (rendered tables + CSV)
"""
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

import figdata as D
import nature_style as S

S.apply()
TIS = ['GM', 'WM', 'CSF']
TC = S.TISSUE_COL
PAPER_COV = {'GM': 1.63, 'WM': 2.97, 'CSF': 3.94}      # draft Table 1 means
CC, CR = 'CC + cingulum', 'Corona radiata'
FC = {CC: S.FAMILY_COL[CC], CR: S.FAMILY_COL[CR]}
INF = copy(plt.get_cmap('inferno')); INF.set_bad(alpha=0)
GREYMAP = plt.get_cmap('gray')
REP, SLICES, Z_MID = 'P05', [3, 4, 5, 6, 7, 8], 6
GLU_LIM = (2, 14)
SUBV = 'segWM'   # subregion repeatability/maps: atlas ∩ WM segmentation (CSF and GM excluded)
FIG4V = 'raw'
ABS_FAMILY_COV = {   # per-participant CoV (%), young -> old, from Abstract_CoV_tables.png (ISMRM 2026 abstract)
    'Corpus callosum': [0.38, 3.05, 4.00, 1.54, 7.99, 9.99, 1.01, 2.56, 10.07, 3.27],
    'Corona radiata': [3.74, 2.01, 1.88, 1.16, 10.28, 0.79, 5.43, 0.74, 7.37, 2.19],
    'Cingulum': [0.36, 1.99, 1.10, 1.51, 7.29, 7.44, 9.14, 10.29, 5.84, 4.88],
    'Longitudinal fasc.': [4.85, 2.17, 0.91, 1.52, 10.21, 2.05, 3.35, 2.64, 3.75, 3.31],
    'Fronto-occipital fasc.': [18.75, 9.02, 0.21, 0.75, 9.98, 0.08, 6.25, 6.78, 10.63, 0.33],
    'CC + cingulum': [0.44, 2.93, 3.59, 0.86, 7.88, 5.26, 1.25, 4.82, 9.12, 3.58]}
ABS_FAMILY_COL = {'Corpus callosum': '#CC79A7', 'Corona radiata': '#009E73', 'Cingulum': '#D55E00',
                  'Longitudinal fasc.': '#E69F00', 'Fronto-occipital fasc.': '#F0E442', 'CC + cingulum': '#8E3F75'}
ABS_FAMILY_MEAN = {'Corpus callosum': 4.38, 'Corona radiata': 3.56, 'Cingulum': 4.98, 'Longitudinal fasc.': 3.48,
                   'Fronto-occipital fasc.': 6.28, 'CC + cingulum': 3.97}   ## means as printed in the abstract table
WB = 'Whole slab'   # abstract definition: unweighted mean of the CSF, GM and WM means    # Fig 4 trends: all atlas voxels pooled per region (original notebook method), corrected atlas


def pct_cov(a, b):
    """Paper formula: |a-b|/sqrt(2) over the mean of |a|,|b|, in %."""
    return np.abs(a - b) / np.sqrt(2) / ((np.abs(a) + np.abs(b)) / 2) * 100


def icc_a1(x, y):
    """ICC(A,1): two-way random, absolute agreement, single measure."""
    Y = np.column_stack([x, y]).astype(float)
    n, k = Y.shape
    gm = Y.mean()
    msr = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((Y.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((Y - Y.mean(1, keepdims=True) - Y.mean(0) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def boot(f, *cols, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    cols = [np.asarray(c, float) for c in cols]
    m = len(cols[0])
    v = [f(*(c[i] for c in cols)) for i in (rng.integers(0, m, m) for _ in range(n))]
    return np.nanpercentile(v, [2.5, 97.5])


# ── data ─────────────────────────────────────────────────────────────────────
rt = D.retest()
tissue_rt = pd.concat([pd.DataFrame({'Subject': rt.Subject, 'Tissue': t, 'Scan1': rt[f'{t}_A'],
                                     'Scan2': rt[f'{t}_B']}) for t in TIS], ignore_index=True)
tissue_rt['CoV'] = pct_cov(tissue_rt.Scan1, tissue_rt.Scan2)

sr = D.subregion_retest(SUBV)
sub_rt = sr.pivot_table(index=['Subject', 'Subregion'], columns='scan', values='GluCEST').dropna().reset_index()
sub_rt['CoV'] = pct_cov(sub_rt.A, sub_rt.B)

tm = D.tissue_means()


# ── Fig 1: maps ──────────────────────────────────────────────────────────────
def fig1():
    crop, H, W = D.common_crop([REP])
    x0, gap, n = 14, 0.6, len(SLICES)
    iw = 22.0
    ih = iw * H / W
    subs = D.subjects_by_age()
    crop2, H2, W2 = D.common_crop([s for s, _ in subs])
    m = 9
    iw2 = (S.DOUBLE - x0 - 14 - (m - 1) * gap) / m
    ih2 = iw2 * H2 / W2
    yb = 5 + 3 * (ih + gap) + 5
    F = S.MMFigure(S.DOUBLE, yb + 4 + 2 * (ih2 + 4) + 2)
    v = D.vol(REP)
    brain = v['roi'] > 0
    b1 = np.where(brain, v['B1map'], np.nan)
    b1_lim = tuple(np.nanpercentile(b1[:, :, SLICES], [1, 99]))
    F.letter('a', 0.5, 1)
    F.text(x0, 2.5, f'Representative participant ({REP}, {int(tm[tm.Subject == REP].Age.iloc[0])} y), '
                    'axial slices inferior → superior', ha='left', fontsize=6)
    rows = [('MP2RAGE\n+ GM/WM', 'anat'), ('GluCEST', 'glu'), ('B1', 'b1')]
    for r, (lab, kind) in enumerate(rows):
        y = 5 + r * (ih + gap)
        F.text(x0 - 1.5, y + ih / 2, lab, ha='right', fontsize=6, fontweight='bold')
        for j, z in enumerate(SLICES):
            ax = F.ax(x0 + j * (iw + gap), y, iw, ih); S.img_ax(ax)
            ax.imshow(crop(D.underlay(REP, z, 'UNI'), REP), cmap=GREYMAP, vmin=0, vmax=1)
            if kind == 'anat':
                seg = crop(v['seg'][:, :, z], REP)
                ax.contour(seg == 2, levels=[0.5], colors=[TC['GM']], linewidths=0.3)
                ax.contour(seg == 3, levels=[0.5], colors=[TC['WM']], linewidths=0.3)
            elif kind == 'glu':
                im_g = ax.imshow(crop(D.cest_masked(REP, z), REP), cmap=INF, vmin=GLU_LIM[0], vmax=GLU_LIM[1],
                                 alpha=0.8)
            else:
                im_b = ax.imshow(crop(np.ma.masked_invalid(b1[:, :, z]), REP), cmap=INF, vmin=b1_lim[0],
                                 vmax=b1_lim[1], alpha=0.85)
    cx = x0 + n * (iw + gap) + 1.5
    for r, (im, lab) in enumerate([(im_g, 'GluCEST (%)'), (im_b, 'B1 (rel.)')]):
        y = 5 + (r + 1) * (ih + gap)
        S.colorbar_inset(F.fig, im, F.ax(cx, y + 2, 1.4, ih - 4), lab, orientation='vertical')
    F.text(x0, 5 + 3 * (ih + gap) + 1.5, '— GM   ', ha='left', fontsize=5.5, color=TC['GM'])
    F.text(x0 + 8, 5 + 3 * (ih + gap) + 1.5, '— WM', ha='left', fontsize=5.5, color=TC['WM'])

    # b: cohort
    F.letter('b', 0.5, yb)
    F.text(x0, yb + 1.5, f'GluCEST (GM + WM) in all participants (n = {len(subs)}), mid-slab slice, ordered by age',
           ha='left', fontsize=6)
    for i, (s, age) in enumerate(subs):
        r, j = divmod(i, m)
        y = yb + 4 + r * (ih2 + 4)
        ax = F.ax(x0 + j * (iw2 + gap), y, iw2, ih2); S.img_ax(ax)
        ax.imshow(crop2(D.underlay(s, Z_MID, 'UNI'), s), cmap=GREYMAP, vmin=0, vmax=1)
        im = ax.imshow(crop2(D.cest_masked(s, Z_MID), s), cmap=INF, vmin=GLU_LIM[0], vmax=GLU_LIM[1], alpha=0.8)
        F.text(x0 + j * (iw2 + gap) + iw2 / 2, y + ih2 + 1.3, f'{age} y', fontsize=5.5)
    S.colorbar_inset(F.fig, im, F.ax(S.DOUBLE - 10.5, yb + 6, 1.4, 2 * ih2), 'GluCEST (%)', orientation='vertical')
    F.save('Fig1_maps')


# ── Fig 2: repeatability ─────────────────────────────────────────────────────
def strip(ax, groups, colors, ref=None):
    rng = np.random.default_rng(2)
    for y, (name, vals) in enumerate(groups.items()):
        v = np.asarray(vals, float); y = len(groups) - 1 - y
        ax.scatter(v, y + rng.uniform(-0.17, 0.17, len(v)), s=5, color=colors[name], alpha=0.6, lw=0, zorder=2)
        lo, hi = boot(np.mean, v)
        ax.plot([lo, hi], [y, y], color='k', lw=0.8, zorder=3)
        ax.scatter(v.mean(), y, s=16, color=colors[name], edgecolor='k', lw=0.5, zorder=4)
        if ref and name in ref:
            ax.scatter(ref[name], y + 0.33, marker='v', s=9, fc='white', ec='k', lw=0.5, zorder=4)
    ax.set_yticks(range(len(groups))); ax.set_yticklabels(list(groups)[::-1])
    ax.set_ylim(-0.6, len(groups) - 0.4); ax.set_xlim(left=0)


def fig2():
    """Repeatability: a tissue CoV, b white-matter subregion CoV (panels for ICC / Bland-Altman removed)."""
    F = S.MMFigure(S.DOUBLE, 46)
    ## a: CoV per tissue
    ax = F.ax(14, 6, 40, 32)
    strip(ax, {t: tissue_rt[tissue_rt.Tissue == t].CoV for t in TIS}, TC)
    ax.set_xlabel('Scan–rescan CoV (%)')
    for y, t in enumerate(TIS[::-1]):
        v = tissue_rt[tissue_rt.Tissue == t].CoV
        ax.text(1.0, y, f'{v.mean():.2f}%', transform=ax.get_yaxis_transform(), va='center', ha='left', fontsize=5.5)
    F.letter('a', 0.5, 1)
    ## b: white-matter subregion CoV (tract families, as in the abstract table)
    ax = F.ax(76, 6, S.DOUBLE - 80, 32)
    rng = np.random.default_rng(3)
    for i, (fam, v) in enumerate(ABS_FAMILY_COV.items()):
        v = np.asarray(v, float)
        col = ABS_FAMILY_COL[fam]
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=4, color=col, alpha=0.6, lw=0)
        lo, hi = boot(np.mean, v)
        ax.plot([i, i], [lo, hi], color='k', lw=0.7)
        ax.scatter(i, v.mean(), s=14, color=col, edgecolor='k', lw=0.5, zorder=3)
        ax.text(i + 0.22, v.mean(), f'{ABS_FAMILY_MEAN[fam]:.2f}%', va='center', fontsize=5)
    for t in TIS:
        ax.axhline(tissue_rt[tissue_rt.Tissue == t].CoV.mean(), color=TC[t], lw=0.5, ls='--')
    ax.set_xticks(range(len(ABS_FAMILY_COV))); ax.set_xticklabels([f.replace(' ', '\n', 1) for f in ABS_FAMILY_COV], fontsize=5.5)
    ax.set_ylabel('Scan–rescan CoV (%)'); ax.set_ylim(0, 20); ax.set_xlim(-0.6, len(ABS_FAMILY_COV) - 0.4)
    F.letter('b', 64, 1)
    F.save('Fig2_repeatability')


# ── Fig 3: GM / WM / CSF vs age ──────────────────────────────────────────────
def fig3():
    """GluCEST vs age, one row in the style of Fig. 4b: a GM, b WM, c whole slab (each with the ROI as an inset on the
    MP2RAGE), d GM vs WM, e WM - GM slope difference. Abstract text sizes; CSF not reported."""
    F = S.MMFigure(S.DOUBLE, 50)
    wb = tm.groupby(['Subject', 'Age']).GluCEST.mean().reset_index().assign(Tissue=WB)   ## whole slab as in the abstract
    data = pd.concat([tm[['Subject', 'Age', 'Tissue', 'GluCEST']], wb], ignore_index=True)
    cols = {**TC, WB: '#7B4FA6'}
    X0, DX, PW, Y0, PH = 10, 35.5, 27, 13, 27
    seg = D.vol(REP)['seg'][:, :, Z_MID]
    crop, H, W = D.common_crop([REP])
    ins_h = 10; ins_w = ins_h * W / H
    anat = crop(D.underlay(REP, Z_MID, 'UNI'), REP)
    minus_fmt = plt.FuncFormatter(lambda v, _: f'{v:.1f}'.replace('-', '−'))

    def inset(x, colours):
        """ROI(s) on the MP2RAGE at the top right of a plot; colours = {seg codes: colour}"""
        axi = F.ax(x + PW - ins_w, Y0, ins_w, ins_h); S.img_ax(axi)
        axi.imshow(anat, cmap=GREYMAP, vmin=0, vmax=1.2)
        lab = np.full(seg.shape, np.nan)
        for k, (codes_, _) in enumerate(colours):
            lab[np.isin(seg, codes_)] = k
        axi.imshow(crop(np.ma.masked_invalid(lab), REP), interpolation='nearest', alpha=0.45 if len(colours[0][0]) > 1 else 0.9,
                   cmap=S.mpl.colors.ListedColormap([c for _, c in colours]), vmin=-0.5, vmax=len(colours) - 0.5)

    def style(ax, ylabel=''):
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.55 * (hi - lo))
        ax.set_xlim(18, 77); ax.set_xticks([20, 40, 60]); ax.tick_params(labelsize=5.5)
        ax.yaxis.set_major_formatter(minus_fmt)
        ax.set_xlabel('Age (years)', fontsize=7.5); ax.set_ylabel(ylabel, fontsize=7.5)

    rows = []
    roi = {'GM': [([2], cols['GM'])], 'WM': [([3], cols['WM'])], WB: [([1, 2, 3], cols[WB])]}
    for j, t in enumerate(['GM', 'WM', WB]):
        d = data[data.Tissue == t]
        x = X0 + j * DX
        ax = F.ax(x, Y0, PW, PH)
        r, p = S.reg_plot(ax, d.Age, d.GluCEST, cols[t], s=10)
        b = np.polyfit(d.Age, d.GluCEST, 1)[0]
        style(ax, 'GluCEST (%)' if j == 0 else '')
        ax.set_title(S.minus(f'{t}\n$\\mathbfit{{r}}$ = {r:.2f},  $\\mathbfit{{P}}$ = {p:.3f}'), fontsize=7.5, fontweight='bold', pad=4)
        inset(x, roi[t])
        F.letter('abc'[j], x - 9, 2)
        rows.append(dict(Tissue=t, n=len(d), r=r, P=p, slope_per_decade=10 * b))

    ## d: GM vs WM on one plot (as Fig. 4b)
    w = tm.pivot_table(index=['Subject', 'Age'], columns='Tissue', values='GluCEST').reset_index()
    age = w.Age.values.astype(float)
    x = X0 + 3 * DX
    ax = F.ax(x, Y0, PW, PH)
    for t in ['GM', 'WM']:
        sr = stats.linregress(age, w[t])
        ax.scatter(age, w[t], s=8, color=cols[t], edgecolor='white', lw=0.3, zorder=3)
        xs = np.array([age.min(), age.max()])
        ax.plot(xs, sr.intercept + sr.slope * xs, color=cols[t], lw=1.4, zorder=4,
                label=S.minus(f'{t}: {10 * sr.slope:+.2f}% per decade'))
    lo, hi = ax.get_ylim(); ax.set_ylim(lo - 0.45 * (hi - lo), hi)
    ax.legend(fontsize=4.8, loc='lower left', frameon=False, handlelength=1.2, borderaxespad=0.2)
    style(ax); ax.set_ylim(4.0, 14.8); ax.set_yticks([6, 8, 10, 12])
    ax.set_title('GM vs WM\n', fontsize=7.5, fontweight='bold', pad=4)
    inset(x, roi['GM'] + roi['WM'])
    F.letter('d', x - 9, 2)

    ## e: slope difference, within-subject WM - GM against age
    dif = w.WM - w.GM
    sd = stats.linregress(age, dif)
    x = X0 + 4 * DX
    ax = F.ax(x, Y0, PW, PH)
    S.reg_plot(ax, age, dif, '0.2', s=10)
    ax.set_xlim(18, 77); ax.set_xticks([20, 40, 60]); ax.tick_params(labelsize=5.5)
    ax.yaxis.set_major_formatter(minus_fmt)
    ax.set_xlabel('Age (years)', fontsize=7.5); ax.set_ylabel(S.minus('WM − GM (%)'), fontsize=7.5, labelpad=1)
    ax.set_title(S.minus(f'Slope difference\n{10 * sd.slope:+.2f}%/decade, ') + f'$\\mathbfit{{P}}$ = {sd.pvalue:.3f}',
                 fontsize=7.5, fontweight='bold', pad=4)
    F.letter('e', x - 9, 2)
    rows.append(dict(Tissue='WM - GM', n=len(dif), r=sd.rvalue, P=sd.pvalue, slope_per_decade=10 * sd.slope))
    pd.DataFrame(rows).to_csv(S.OUT / 'stats_Fig3_age_trends.csv', index=False)
    F.save('Fig3_age_trends')


# ── Fig 4: CR vs CC + cingulum slope difference ──────────────────────────────
def fam_wide(variant):
    f = D.family_means(variant)
    w = f.pivot_table(index=['Subject', 'Age'], columns='Family', values='GluCEST').dropna().reset_index()
    w['D'] = w[CR] - w[CC]
    return w, f[f.Family.isin([CC, CR])].copy()


def mixed(f, cov=()):
    f = f.copy()
    f['Fam'] = (f.Family == CR).astype(int)
    for c in cov:
        f[c] = f[c] - f.groupby('Family')[c].transform('mean')
    m = smf.mixedlm('GluCEST ~ Age * Fam' + ''.join(f' + {c}' for c in cov), f, groups=f.Subject).fit(reml=True)
    lo, hi = m.conf_int().loc['Age:Fam']
    return 10 * m.params['Age:Fam'], 10 * lo, 10 * hi, m.pvalues['Age:Fam']


def fig4():
    F = S.MMFigure(S.DOUBLE, 62)
    w, f = fam_wide(FIG4V)
    ax = F.ax(14, 8, 44, 44)
    for fam in (CC, CR):
        r, p = S.reg_plot(ax, w.Age, w[fam], FC[fam])
        b = np.polyfit(w.Age, w[fam], 1)[0]
        ax.plot([], [], color=FC[fam], marker='o', ms=3, label=S.minus(f'{fam}: {10 * b:+.2f}%/decade'))
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.35 * (hi - lo))
    ax.legend(loc='upper left', fontsize=5.3, handlelength=1, borderaxespad=0.2)
    ax.set(xlabel='Age (years)', ylabel='Mean GluCEST (%)')
    F.letter('a', 2, 2)

    ax = F.ax(74, 8, 44, 44)
    ax.axhline(0, color=S.LIGHT, lw=0.5)
    S.reg_plot(ax, w.Age, w.D, '#333333')
    res = stats.linregress(w.Age, w.D)
    t = stats.t.ppf(0.975, len(w) - 2)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.45 * (hi - lo))
    ax.text(0.03, 0.97, S.minus(f'Slope difference {10 * res.slope:+.2f}%/decade\n'
                                f'95% CI [{10 * (res.slope - t * res.stderr):+.2f}, {10 * (res.slope + t * res.stderr):+.2f}]\n')
            + S.fmt_r(res.rvalue, res.pvalue) + f', n = {len(w)}', transform=ax.transAxes, va='top', fontsize=5.5)
    ax.set(xlabel='Age (years)', ylabel=f'{CR} − ({CC}) (%)')
    F.letter('b', 62, 2)

    _, f_seg = fam_wide('segWM'); _, f_pv = fam_wide('pvWM90')
    specs = [('All atlas voxels (primary)', mixed(f)), ('+ T1', mixed(f, ('T1',))), ('+ WM fraction', mixed(f, ('WMfrac_pv',))),
             ('+ T1 + WM fraction', mixed(f, ('T1', 'WMfrac_pv'))), ('WM voxels only', mixed(f_seg)),
             ('Voxels ≥ 90% WM', mixed(f_pv))]
    ax = F.ax(146, 8, 26, 44)
    for i, (lab, (b, lo_, hi_, p)) in enumerate(specs):
        y = len(specs) - 1 - i
        col = '#333333' if p < 0.05 else S.GREY
        ax.plot([lo_, hi_], [y, y], color=col, lw=1)
        ax.scatter(b, y, s=14, color=col, zorder=3, edgecolor='white', lw=0.3)
        ax.text(1.03, y, S.fmt_p(p, prefix=False), transform=ax.get_yaxis_transform(), va='center', fontsize=5.2,
                color=col)
    ax.text(1.03, len(specs) - 0.35, r'$\it{P}$', transform=ax.get_yaxis_transform(), fontsize=5.5)
    ax.axvline(0, color=S.GREY, lw=0.5, ls=':')
    ax.set_yticks(range(len(specs))); ax.set_yticklabels([s for s, _ in specs][::-1], fontsize=5.5)
    ax.set_ylim(-0.6, len(specs) - 0.2)
    ax.set_xlabel('Region × age interaction\n(%/decade, 95% CI)')
    F.letter('c', 122, 2)
    pd.DataFrame([dict(spec=s, beta_per_decade=b, ci_lo=l, ci_hi=h, P=p) for s, (b, l, h, p) in specs]
                 + [dict(spec=f'difference score ({FIG4V})', beta_per_decade=10 * res.slope, P=res.pvalue)]
                 ).to_csv(S.OUT / 'stats_Fig4_slope_difference.csv', index=False)
    F.save('Fig4_slope_difference')


# ── Tables ───────────────────────────────────────────────────────────────────
def heat_table(tab, name, title, vmax=10, bold_cols=()):
    fig, ax = plt.subplots(figsize=(min(183, 12 * tab.shape[1] + 28) * S.MM, (4.3 * len(tab) + 14) * S.MM))
    vals = tab.values.astype(float)
    cm = copy(plt.get_cmap('YlOrRd')); cm.set_bad('0.93')
    ax.imshow(np.ma.masked_invalid(vals), cmap=cm, vmin=0, vmax=vmax, aspect='auto')
    for (i, j), v in np.ndenumerate(vals):
        ax.text(j, i, '–' if np.isnan(v) else f'{v:.2f}', ha='center', va='center', fontsize=5,
                color='white' if v > 0.65 * vmax else 'black',
                fontweight='bold' if tab.columns[j] in bold_cols else 'normal')
    ax.set_xticks(range(tab.shape[1])); ax.set_xticklabels(tab.columns, fontsize=5.5)
    ax.set_yticks(range(len(tab))); ax.set_yticklabels(tab.index, fontsize=5.5)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(title, loc='left', pad=12, fontsize=6.5, fontweight='bold')
    fig.tight_layout(pad=0.4)
    S.save(fig, name)
    tab.round(3).to_csv(S.OUT / f'{name}.csv')


def tables():
    t1 = tissue_rt.pivot(index='Tissue', columns='Subject', values='CoV').loc[TIS]
    t1['Mean'] = t1.mean(1); t1['SD'] = t1.iloc[:, :-1].std(1)
    t1['Paper'] = [PAPER_COV[t] for t in TIS]
    t1['ICC'] = [icc_a1(*tissue_rt[tissue_rt.Tissue == t][['Scan1', 'Scan2']].values.T) for t in TIS]
    heat_table(t1, 'Table1_tissue_CoV_ICC', f'Table 1 | Scan–rescan CoV (%) by tissue, n = {rt.Subject.nunique()} pairs '
               '(ICC column: ICC(A,1))', bold_cols=('Mean', 'ICC'))
    t2 = sub_rt.pivot(index='Subregion', columns='Subject', values='CoV')
    t2 = t2.loc[t2.mean(1).sort_values().index]
    t2['Mean'] = t2.mean(1)
    t2['ICC'] = [icc_a1(*sub_rt[sub_rt.Subregion == s][['A', 'B']].values.T) for s in t2.index]
    heat_table(t2, 'Table2_subregion_CoV', f'Table 2 | Scan–rescan CoV (%) by WM subregion (atlas ∩ WM segmentation; '
               f'CSF/GM excluded), n = {sub_rt.Subject.nunique()} pairs (P02: no usable scan-4 atlas)',
               bold_cols=('Mean', 'ICC'))


# ── Fig 5: subregion analysis ────────────────────────────────────────────────
def fig5():
    sm = D.subregions(SUBV)
    subs = [s for s in D.SUB_ORDER if s in set(sm.Subregion)]
    F = S.MMFigure(S.DOUBLE, 112)
    crop, H, W = D.common_crop([REP])
    zs = [4, 6, 8]
    iw = 24; ih = iw * H / W
    v = D.vol(REP)
    gmean = sm.groupby('Subregion').GluCEST.mean().to_dict()
    from matplotlib.colors import ListedColormap
    lab_cmap = ListedColormap([S.SUB_COL.get(s, S.GREY) for s in D.SUB_ORDER])
    F.letter('a', 0.5, 1); F.letter('b', 0.5, 6 + ih + 2)
    for j, z in enumerate(zs):
        wm = v['seg'][:, :, z] == 3
        for r in range(2):
            ax = F.ax(10 + j * (iw + 0.6), 4 + r * (ih + 4), iw, ih); S.img_ax(ax)
            ax.imshow(crop(D.underlay(REP, z, 'UNI'), REP), cmap=GREYMAP, vmin=0, vmax=1)
            if r == 0:
                idx = np.where(wm, D.subregion_index(REP, z), 0)
                ax.imshow(crop(np.ma.masked_equal(idx, 0), REP), cmap=lab_cmap, vmin=0.5,
                          vmax=len(D.SUB_ORDER) + 0.5, interpolation='nearest')
            else:
                pm = D.paint(REP, z, gmean)
                pm = np.ma.masked_where(~wm | np.ma.getmaskarray(pm), pm)
                im = ax.imshow(crop(pm, REP), cmap=INF, vmin=5, vmax=9, interpolation='nearest')
    S.colorbar_inset(F.fig, im, F.ax(10, 4 + 2 * (ih + 4) - 1.5, 3 * iw, 1.3), 'Group-mean GluCEST (%)',
                     label_side='below')
    hs = [plt.Rectangle((0, 0), 1, 1, fc=S.SUB_COL.get(s, S.GREY)) for s in subs]
    F.fig.legend(hs, subs, loc='upper left', bbox_to_anchor=(10 / S.DOUBLE, 1 - (4 + 2 * (ih + 4) + 4) / 112),
                 ncol=4, fontsize=5.2, handlelength=0.9, columnspacing=0.8, handletextpad=0.3)

    # c: per-subregion GluCEST, subjects + mean ± CI
    x0 = 92
    ax = F.ax(x0 + 20, 4, 30, 48)
    rng = np.random.default_rng(4)
    order = sm.groupby('Subregion').GluCEST.mean().reindex(subs).sort_values().index
    for i, s in enumerate(order):
        vv = sm[sm.Subregion == s].GluCEST.values
        ax.scatter(vv, i + rng.uniform(-0.18, 0.18, len(vv)), s=4, color=S.SUB_COL.get(s, S.GREY), alpha=0.6, lw=0)
        m, lo, hi = S.mean_ci(vv)
        ax.plot([lo, hi], [i, i], color='k', lw=0.7)
        ax.scatter(m, i, s=13, color=S.SUB_COL.get(s, S.GREY), edgecolor='k', lw=0.5, zorder=3)
    gm_wm = tm.groupby('Tissue').GluCEST.mean()
    for t in ('GM', 'WM'):
        ax.axvline(gm_wm[t], color=TC[t], lw=0.5, ls='--')
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=5.5)
    ax.set_xlabel('GluCEST (%)'); ax.set_title('Subregion GluCEST (n = 17)', fontsize=6, pad=3)
    F.letter('c', x0, 1)

    # d: age slope per subregion; e: ICC per subregion
    ax_d = F.ax(x0 + 62, 4, 26, 48)
    rows = []
    for i, s in enumerate(order):
        d = sm[sm.Subregion == s]
        res = stats.linregress(d.Age, d.GluCEST)
        t = stats.t.ppf(0.975, len(d) - 2)
        rows.append(dict(Subregion=s, n=len(d), slope=10 * res.slope, lo=10 * (res.slope - t * res.stderr),
                         hi=10 * (res.slope + t * res.stderr), P=res.pvalue))
    sl = pd.DataFrame(rows)
    from statsmodels.stats.multitest import multipletests
    sl['q'] = multipletests(sl.P, method='fdr_bh')[1]
    for i, r in sl.iterrows():
        col = S.SUB_COL.get(r.Subregion, S.GREY)
        ax_d.plot([r.lo, r.hi], [i, i], color=col, lw=0.9)
        ax_d.scatter(r.slope, i, s=13, color=col, edgecolor='k', lw=0.5, zorder=3)
    ax_d.axvline(0, color=S.GREY, lw=0.5, ls=':')
    ax_d.set_yticks(range(len(order))); ax_d.set_yticklabels([])
    ax_d.set_xlabel('Age slope (%/decade)'); ax_d.set_title('Age slope, 95% CI', fontsize=6, pad=3)
    ax_d.text(0.98, 0.01, f'all FDR q ≥ {sl.q.min():.2f}', transform=ax_d.transAxes, ha='right', fontsize=5,
              color=S.GREY)
    F.letter('d', x0 + 57, 1)
    sl.to_csv(S.OUT / 'stats_Fig5_subregion_age_slopes.csv', index=False)

    ax = F.ax(x0 + 20, 64, 70, 36)
    st = sub_rt.groupby('Subregion').agg(n=('CoV', 'size'), CoV=('CoV', 'mean')).reindex(order)
    st['ICC'] = [icc_a1(*sub_rt[sub_rt.Subregion == s][['A', 'B']].values.T) if st.n[s] >= 4 else np.nan
                 for s in order]
    ok = st.n >= 4
    ax.scatter(st.CoV[ok], st.ICC[ok], s=18, c=[S.SUB_COL.get(s, S.GREY) for s in st.index[ok]], edgecolor='k', lw=0.5)
    for s, r in st[ok].iterrows():
        ax.annotate(s, (r.CoV, r.ICC), xytext=(3, 0), textcoords='offset points', fontsize=5, va='center')
    ax.axhline(0.75, color=S.GREY, lw=0.5, ls=':')
    ax.set(xlabel='Scan–rescan CoV (%)', ylabel='ICC(A,1)', ylim=(min(0.4, st.ICC.min() - 0.05), 1.02),
           xlim=(st.CoV.min() - 0.4, st.CoV.max() + 1.6))
    ax.set_title(f'Subregion repeatability (n = {sub_rt.Subject.nunique()} pairs; ≥ 4 pairs shown)', fontsize=6, pad=3)
    F.letter('e', x0, 60)
    st.to_csv(S.OUT / 'stats_Fig5_subregion_repeatability.csv')
    F.save('Fig5_subregions')



# ── Fig 0: hero figure (front of paper) ──────────────────────────────────────
CC_SUBS = {'CC genu', 'CC body', 'CC splenium', 'Cingulum'}
CR_SUBS = {'ACR', 'SCR', 'PCR'}


def family_mask(s, z, subs):
    at = D.sub(s)['wm_atlas'][:, :, z]
    ks = [k for k, n in D.subregion_label_map(s).items() if n in subs]
    return np.isin(at, ks) & (D.vol(s)['roi'][:, :, z] > 0)


def hero():
    """Front-of-paper overview, read left to right: measurement -> regions -> repeatability -> finding -> caveat.
    Nature spec: 183 mm wide, Arial 5-7 pt, colour-blind-safe (Okabe-Ito magenta/teal), panels on one grid."""
    REPS = 'P05'
    crop, H, W = D.common_crop([REPS])
    asp = W / H                                    # width / height of the brain crop
    F = S.MMFigure(S.DOUBLE, 108)
    R1, R2 = 2, 60                                 # header tops of row 1 and row 2

    def header(x, y, w, letter, title, sub):
        F.letter(letter, x, y)
        F.text(x + 4.2, y + 3.6, title, ha='left', va='bottom', fontsize=7, fontweight='bold')
        F.hline(x, x + w, y + 4.4, color='k', lw=0.6)
        F.text(x, y + 5.2, sub, ha='left', va='top', fontsize=5.5, color=S.GREY)

    # ---------------- a: 3D slab as a stack of real slices ----------------
    header(1, R1, 56, 'a', 'Measurement', '3D GluCEST at 7 T · 12 × 2 mm slab · n = 17, 23–73 y')
    ih = 32; iw = ih * asp
    zs = [9, 8, 7, 6, 5, 4]                        # superior (back) -> inferior (front)
    for k, z in enumerate(zs):
        ax = F.ax(3 + k * 3.6, 11 + k * 1.9, iw, ih); S.img_ax(ax)
        ax.imshow(crop(D.underlay(REPS, z, 'UNI'), REPS), cmap=GREYMAP, vmin=0, vmax=1)
        im = ax.imshow(crop(D.cest_masked(REPS, z), REPS), cmap=INF, vmin=GLU_LIM[0], vmax=GLU_LIM[1], alpha=0.9)
    xb = 3 + 5 * 3.6 + iw + 3
    S.colorbar_inset(F.fig, im, F.ax(xb, 20, 1.3, 22), 'GluCEST (%)', ticks=[2, 8, 14], orientation='vertical')

    # ---------------- b: the two tract regions ----------------
    header(64, R1, 52, 'b', 'Tract regions', "JHU atlas warped to each participant")
    ih = 40; iw = ih * asp
    ax = F.ax(64 + (52 - iw) / 2 - 6, 11, iw, ih); S.img_ax(ax)
    ax.imshow(crop(D.underlay(REPS, Z_MID, 'UNI'), REPS), cmap=GREYMAP, vmin=0, vmax=1)
    for subs_, col in [(CC_SUBS, FC[CC]), (CR_SUBS, FC[CR])]:
        m = crop(family_mask(REPS, Z_MID, subs_), REPS)
        ax.imshow(np.ma.masked_where(~m, m), cmap=S.mpl.colors.ListedColormap([col]), alpha=0.9,
                  interpolation='nearest')
    xl = 64 + (52 - iw) / 2 - 6 + iw + 2
    for k, (lab, col) in enumerate([('Corpus\ncallosum\n+ cingulum', FC[CC]), ('Corona\nradiata', FC[CR])]):
        F.fig.add_artist(S.mpl.patches.Rectangle(((xl) / S.DOUBLE, 1 - (22 + k * 13) / 108), 2.2 / S.DOUBLE,
                                                 2.2 / 108, color=col, transform=F.fig.transFigure))
        F.text(xl + 3.2, 21 + k * 13, lab, ha='left', va='top', fontsize=5.8, color=col, fontweight='bold')

    # ---------------- c: repeatability ----------------
    header(124, R1, 58, 'c', 'Repeatability', 'Scan–rescan CoV, 10 pairs, 7–14 days apart')
    rows = [('Gray matter', tissue_rt[tissue_rt.Tissue == 'GM'].CoV.values, TC['GM']),
            ('White matter', tissue_rt[tissue_rt.Tissue == 'WM'].CoV.values, TC['WM']),
            ('CSF', tissue_rt[tissue_rt.Tissue == 'CSF'].CoV.values, TC['CSF']),
            ('CC + cingulum', np.array(ABS_FAMILY_COV['CC + cingulum']), FC[CC]),
            ('Corona radiata', np.array(ABS_FAMILY_COV['Corona radiata']), FC[CR])]
    ax = F.ax(145, 13, 28, 34)
    rng = np.random.default_rng(1)
    for y, (lab, v, col) in enumerate(rows[::-1]):
        ax.scatter(v, y + rng.uniform(-0.15, 0.15, len(v)), s=3, color=col, alpha=0.45, lw=0)
        lo, hi = boot(np.mean, v)
        ax.plot([lo, hi], [y, y], color='k', lw=0.7)
        ax.scatter(v.mean(), y, s=15, color=col, edgecolor='k', lw=0.5, zorder=3)
        ax.text(1.03, y, f'{v.mean():.1f}%', transform=ax.get_yaxis_transform(), va='center', ha='left', fontsize=5.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=5.8)
    ax.axhline(1.5, color=S.LIGHT, lw=0.5)
    ax.set_xlim(0, 11); ax.set_xticks([0, 5, 10]); ax.set_xlabel('CoV (%)', labelpad=1)
    S.despine(ax)

    # ---------------- d: the finding ----------------
    header(1, R2, 92, 'd', 'Finding', 'Corona radiata GluCEST rises relative to corpus callosum + cingulum with age')
    w, f = fam_wide(FIG4V)
    ax = F.ax(12, 72, 34, 26)
    for fam in (CC, CR):
        d = f[f.Family == fam]
        S.reg_plot(ax, d.Age, d.GluCEST, FC[fam], s=7, lw=1.0)
    ax.set(xlabel='Age (years)', ylabel='Regional GluCEST (%)', xlim=(20, 76), xticks=[20, 40, 60], ylim=(5.2, 8.6))
    ax.text(0.03, 0.99, 'CC + cingulum', color=FC[CC], transform=ax.transAxes, fontsize=5.8, va='top', fontweight='bold')
    ax.text(0.97, 0.03, 'Corona radiata', color=FC[CR], transform=ax.transAxes, fontsize=5.8, ha='right',
            fontweight='bold')
    ax2 = F.ax(60, 72, 32, 26)
    res = stats.linregress(w.Age, w.D)
    S.reg_plot(ax2, w.Age, w.D, 'k', s=7, lw=1.0)
    ax2.axhline(0, color=S.GREY, lw=0.5, ls=':')
    ax2.set(xlabel='Age (years)', ylabel='CR − (CC + cing.) (%)', xlim=(20, 76), xticks=[20, 40, 60], ylim=(-1.4, 1.1))
    ax2.text(0.03, 0.99, S.minus(f'+{10 * res.slope:.2f} pp per decade') + ', ' + rf'$\it{{P}}$ = {res.pvalue:.3f}',
             transform=ax2.transAxes, va='top', fontsize=5.8, fontweight='bold')
    for a in (ax, ax2): S.despine(a)

    # ---------------- e: robustness / caveat ----------------
    header(102, R2, 80, 'e', 'Robustness', 'Independent of T1; attenuated in nearly pure white matter')
    st = pd.read_csv(S.OUT / 'stats_Fig4_slope_difference.csv')
    keep = {'All atlas voxels (primary)': 'All atlas voxels', '+ T1': '+ T1',
            '+ T1 + WM fraction': '+ T1 + WM fraction', 'WM voxels only': 'Segmented WM voxels',
            'Voxels ≥ 90% WM': 'Voxels ≥ 90% WM'}
    st = st[st.spec.isin(keep)].set_index('spec').loc[list(keep)].reset_index()
    ax = F.ax(130, 72, 36, 26)
    for i, r in st.iterrows():
        y = len(st) - 1 - i; ok = r.P < 0.05
        col = 'k' if ok else S.GREY
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=col, lw=0.8)
        ax.scatter(r.beta_per_decade, y, s=14, facecolor=col if ok else 'white', edgecolor=col, lw=0.8, zorder=3)
        ax.text(1.03, y, rf'$\it{{P}}$ = {r.P:.3f}', transform=ax.get_yaxis_transform(), va='center', fontsize=5.3, color=col)
    ax.axvline(0, color=S.GREY, lw=0.5, ls=':')
    ax.set_yticks(range(len(st))); ax.set_yticklabels([keep[x] for x in st.spec[::-1]], fontsize=5.8)
    ax.set_ylim(-0.6, len(st) - 0.4)
    ax.set_xlim(-0.08, 0.32); ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.set_xlabel('Region × age interaction\n(pp per decade, 95% CI)', labelpad=1)
    S.despine(ax)
    F.save('Fig0_hero')

if __name__ == '__main__':
    tables()
    fig1(); fig2(); fig3(); fig4(); fig5(); hero()
    print(tissue_rt.groupby('Tissue').CoV.agg(['mean', 'std', 'count']).round(2))
    print(pd.read_csv(S.OUT / 'stats_Fig4_slope_difference.csv').round(4).to_string(index=False))
