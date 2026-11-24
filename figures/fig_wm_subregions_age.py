"""WM regions vs age: corpus callosum (genu + body + splenium), cingulum and corona radiata (anterior + superior + posterior).

Subregions are pooled into whole regions by voxel count in each subject (all valid atlas voxels of the region averaged
together), the same as the primary analysis and the ISMRM abstract.

a  regions on the MP2RAGE   b  CC + cingulum (blue) vs CR (orange) fits, inset map of the two regions
c  slope difference: within-subject CR - (CC + cingulum) against age
d  GluCEST vs age: CC, cingulum, CC + cingulum, CR
e  mixed model (age x region, random intercept per subject): both regions, participant intercept removed
f  same, one plot per region

Runs from _cache (P18 excluded, corrected atlas). Run:  python fig_wm_subregions_age.py
"""
from copy import copy

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

import figdata as D
import nature_style as S
import matplotlib.patheffects  # noqa: F401  (S.mpl.patheffects)

S.apply()
PARTS = {'Corpus callosum': ['CC genu', 'CC body', 'CC splenium'],
         'Cingulum': ['Cingulum'],
         'Corona radiata': ['ACR', 'SCR', 'PCR']}
PARTS_CCC = {**PARTS, 'CC + cingulum': PARTS['Corpus callosum'] + PARTS['Cingulum']}
COL = {'Corpus callosum': '#CC79A7', 'Cingulum': '#D55E00', 'Corona radiata': '#009E73', 'CC + cingulum': '#8E3F75'}
REP, ZS = 'P05', [5, 7]


def part_means(sub, weights=None):
    """per-subject region value. weights=None: voxel-pooled (each subregion weighted by its voxels in that subject);
    otherwise a fixed weight per subregion (cohort-average voxels, or 1 for equal weights)"""
    G = sub.pivot_table(index=['Subject', 'Age'], columns='Subregion', values='GluCEST')
    N = sub.pivot_table(index=['Subject', 'Age'], columns='Subregion', values='n_vox')
    out = {}
    for part, subs in PARTS_CCC.items():
        vals = G[subs]
        w = N[subs] if weights is None else vals.notna() * weights[subs]
        w = w.where(vals.notna(), 0)
        out[part] = (vals.fillna(0) * w).sum(1) / w.sum(1)
    return pd.DataFrame(out).reset_index()


def slope_stats(age, y):
    res = stats.linregress(age, y)
    t = stats.t.ppf(0.975, len(y) - 2)
    return dict(slope=10 * res.slope, lo=10 * (res.slope - t * res.stderr), hi=10 * (res.slope + t * res.stderr),
                r=res.rvalue, P=res.pvalue)


def make():
    sub = D.subregions('raw')
    subs_all = sum(PARTS.values(), [])
    sub = sub[sub.Subregion.isin(subs_all)]
    pm = part_means(sub)                                  ## voxel-pooled, as in the primary analysis
    age = pm.Age.values.astype(float)
    st = {p: slope_stats(age, pm[p]) for p in PARTS_CCC}
    st['CR − (CC + cingulum)'] = slope_stats(age, pm['Corona radiata'] - pm['CC + cingulum'])

    ## same divergence as a linear mixed model: GluCEST ~ Age * Region + (1 | subject), voxel-pooled values
    lg = pm.melt(id_vars=['Subject', 'Age'], value_vars=['CC + cingulum', 'Corona radiata'], var_name='Region',
                 value_name='G')
    lg['CR'] = (lg.Region == 'Corona radiata').astype(int)
    mm = smf.mixedlm('G ~ Age * CR', lg, groups='Subject').fit(reml=True)
    b, se = mm.params['Age:CR'], mm.bse['Age:CR']
    mst = dict(slope=10 * b, lo=10 * (b - 1.96 * se), hi=10 * (b + 1.96 * se), r=np.nan, P=mm.pvalues['Age:CR'])
    ## each participant's random intercept, removed from their values for plotting (panels e, f)
    ri = {k: v.iloc[0] for k, v in mm.random_effects.items()}
    lg['Gadj'] = lg.G - lg.Subject.map(ri)
    fe = mm.fe_params
    mline = {'CC + cingulum': (fe['Intercept'], fe['Age']),
             'Corona radiata': (fe['Intercept'] + fe['CR'], fe['Age'] + fe['Age:CR'])}
    pd.concat({'region': pd.DataFrame(st).T,
               'mixed model': pd.DataFrame({'Age x region': mst}).T}).to_csv(S.OUT / 'stats_WM_subregions_age.csv')

    F = S.MMFigure(S.DOUBLE, 148)
    grey = copy(S.mpl.colormaps['gray']); grey.set_bad('white')
    crop, H, W = D.common_crop([REP])
    ih = 34; iw = ih * W / H
    z = ZS[0]
    names = list(PARTS)
    lab_of = {s: names.index(p) + 1 for p, subs in PARTS.items() for s in subs}

    def roi_index(s_, z_):
        at = D.sub(s_)['wm_atlas'][:, :, z_]
        idx = np.zeros(at.shape)
        for key, name in D.subregion_label_map(s_).items():
            if name in lab_of:
                idx[at == key] = lab_of[name]
        return idx

    xa = 5; xb = xa + iw + 9; xc = 141
    ## a: regions on the MP2RAGE
    F.letter('a', xa - 4, 1)
    ax = F.ax(xa, 5, iw, ih); S.img_ax(ax)
    ax.imshow(crop(D.underlay(REP, z, 'UNI'), REP), cmap=grey, vmin=0, vmax=1.2)
    ax.imshow(crop(np.ma.masked_equal(roi_index(REP, z), 0), REP),
              cmap=S.mpl.colors.ListedColormap([COL[p] for p in names]), vmin=0.5, vmax=len(names) + 0.5,
              interpolation='nearest', alpha=0.95)
    ax.set_title('Regions', fontsize=6, fontweight='bold', pad=2)
    for k, p in enumerate(names):
        y = 5 + ih + 2.5 + k * 3.3
        F.fig.add_artist(S.mpl.patches.Rectangle((xa / F.W, 1 - (y + 1) / F.H), 2 / F.W, 2 / F.H, color=COL[p],
                                                 transform=F.fig.transFigure))
        F.text(xa + 2.8, y, p, ha='left', fontsize=5.3)

    ## b: CC + cingulum vs CR fits (Okabe-Ito blue / orange), with the two regions shown as an inset map
    F.letter('b', xb - 4, 1)
    BC = {'CC + cingulum': '#0072B2', 'Corona radiata': '#E69F00'}
    pwb = xc - 22 - xb
    ax = F.ax(xb + 6, 9, pwb, 34)
    for q, lab in [('CC + cingulum', 'CC + cing.'), ('Corona radiata', 'CR')]:
        ax.scatter(age, pm[q], s=11, color=BC[q], edgecolor='white', lw=0.35, zorder=3)
        b, a0 = np.polyfit(age, pm[q], 1)
        xs = np.array([age.min(), age.max()])
        ax.plot(xs, a0 + b * xs, color=BC[q], lw=1.6, zorder=4,
                label=S.minus(f'{lab}: {st[q]["slope"]:+.2f}% per decade ($P$ = {st[q]["P"]:.2f})'))
    ax.set_title('CC + cingulum vs CR', fontsize=7.5, fontweight='bold', pad=4)
    ax.legend(fontsize=5, loc='lower left', frameon=False, handlelength=1.6)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo - 0.35 * (hi - lo), hi + 0.45 * (hi - lo))
    ax.set_xlim(18, 77); ax.set_xticks([20, 40, 60]); ax.tick_params(labelsize=5.5)
    ax.set_xlabel('Age (years)', fontsize=7.5); ax.set_ylabel('GluCEST (%)', fontsize=7.5)
    S.despine(ax)
    ## inset: the two regions on the MP2RAGE, top right of the plot
    reg = {s_: 0 for s_ in PARTS_CCC['CC + cingulum']}
    reg.update({s_: 1 for s_ in PARTS['Corona radiata']})
    ins_h = 15; ins_w = ins_h * W / H
    axi = F.ax(xb + 6 + pwb - ins_w, 9, ins_w, ins_h); S.img_ax(axi)
    axi.imshow(crop(D.underlay(REP, z, 'UNI'), REP), cmap=grey, vmin=0, vmax=1.2)
    axi.imshow(crop(D.paint(REP, z, reg, mask_valid=False), REP), interpolation='nearest',
               cmap=S.mpl.colors.ListedColormap([BC['CC + cingulum'], BC['Corona radiata']]), vmin=-0.5, vmax=1.5)

    ## c: slope difference: within-subject CR − (CC + cingulum) against age
    F.letter('c', xc - 9, 1)
    d = st['CR − (CC + cingulum)']
    ax = F.ax(xc, 9, 40, 34)
    S.reg_plot(ax, age, pm['Corona radiata'] - pm['CC + cingulum'], '0.2', s=11)
    ax.axhline(0, color=S.GREY, lw=0.5, ls=':')
    ax.set_title(S.minus(f'Slope difference\n{d["slope"]:+.2f}% per decade, ') + f'$\\mathbfit{{P}}$ = {d["P"]:.3f}',
                 fontsize=7.5, fontweight='bold', pad=4)
    ax.set_xticks([20, 40, 60]); ax.tick_params(labelsize=5.5)
    ax.set_xlabel('Age (years)', fontsize=7.5); ax.set_ylabel(S.minus('CR − (CC + cing.) (%)'), fontsize=7.5)
    S.despine(ax)

    ## d: GluCEST vs age per region
    F.letter('d', 0.5, 60)
    pw, ph, gx = 35, 24, 8
    for i, p in enumerate(['Corpus callosum', 'Cingulum', 'CC + cingulum', 'Corona radiata']):
        ax = F.ax(13 + i * (pw + gx), 65, pw, ph)
        S.reg_plot(ax, age, pm[p], COL[p], s=8)
        q = st[p]
        ax.set_title(p, fontsize=6, color=COL[p], fontweight='bold', pad=2)
        ax.text(0.03, 0.97, S.minus(f'{q["slope"]:+.2f}% per decade') + '\n' + S.fmt_r(q['r'], q['P']),
                transform=ax.transAxes, va='top', fontsize=5)
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.35 * (hi - lo))
        ax.set_xlabel('Age (years)'); ax.set_ylabel('GluCEST (%)' if i == 0 else '')
        S.despine(ax)

    ## e: mixed model, both regions (like b): values minus each participant's random intercept, fixed-effect lines
    BC = {'CC + cingulum': '#0072B2', 'Corona radiata': '#E69F00'}
    xs = np.array([age.min(), age.max()])
    F.letter('e', 0.5, 101)
    ax = F.ax(13, 107, 70, 30)
    for q, lab in [('CC + cingulum', 'CC + cing.'), ('Corona radiata', 'CR')]:
        g = lg[lg.Region == q]
        ax.scatter(g.Age, g.Gadj, s=11, color=BC[q], edgecolor='white', lw=0.35, zorder=3)
        a0, b0 = mline[q]
        ax.plot(xs, a0 + b0 * xs, color=BC[q], lw=1.6, zorder=4,
                label=S.minus(f'{lab}: {10 * b0:+.2f}% per decade'))
    ax.set_title(S.minus(f'Mixed model (age × region): {mst["slope"]:+.2f}% per decade, ') + f'$\\mathbfit{{P}}$ = {mst["P"]:.3f}',
                 fontsize=7, fontweight='bold', pad=4)
    ax.legend(fontsize=5, loc='lower left', frameon=False, handlelength=1.6)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo - 0.3 * (hi - lo), hi)
    ax.set_xlim(18, 77); ax.set_xticks([20, 40, 60]); ax.tick_params(labelsize=5.5)
    ax.set_xlabel('Age (years)', fontsize=7)
    ax.set_ylabel('GluCEST, participant\nintercept removed (%)', fontsize=6.5)
    S.despine(ax)

    ## f: mixed model, one plot per region (like d)
    F.letter('f', 92, 101)
    for i, q in enumerate(['CC + cingulum', 'Corona radiata']):
        ax = F.ax(106 + i * 40, 107, 31, 30)
        g = lg[lg.Region == q]
        ax.scatter(g.Age, g.Gadj, s=8, color=BC[q], edgecolor='white', lw=0.3, zorder=3)
        a0, b0 = mline[q]
        ax.plot(xs, a0 + b0 * xs, color=BC[q], lw=1.2, zorder=4)
        ax.set_title(q, fontsize=6, color=BC[q], fontweight='bold', pad=2)
        ax.text(0.03, 0.97, S.minus(f'{10 * b0:+.2f}% per decade'), transform=ax.transAxes, va='top', fontsize=5)
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.25 * (hi - lo))
        ax.set_xlabel('Age (years)'); ax.set_ylabel('GluCEST, intercept\nremoved (%)' if i == 0 else '', fontsize=6)
        S.despine(ax)
    F.save('Fig_WM_subregions_age')


if __name__ == '__main__':
    make()
