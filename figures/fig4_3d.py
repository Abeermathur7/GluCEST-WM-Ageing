"""Figure 4, 3D version (white, styled like the WM overview hero on slide 5 of the feedback deck).

a  the two compared regions in 3D: CC + cingulum (blue) and corona radiata (orange) inside a grey glass brain,
   other slab WM ROIs light grey (one participant's anatomy, P04, as in the hero figures)
b  exploded 3D stack (CR below, CC + cingulum above) with each region's cohort age slope
c  GluCEST vs age per region (CC, cingulum, CC + cingulum, CR), each with a 3D thumbnail of the region
d  GluCEST vs age, both regions with linear fits
e  slope difference: within-participant CR - (CC + cingulum) against age
f  mixed model (GluCEST ~ age x region, random intercept per participant), intercepts removed
g  mixed model per region (as f, one plot each, 3D thumbnails) + their slope difference (the f interaction)

Numbers: subregion_means.csv ('raw' atlas voxels, voxel-pooled per participant), identical to Fig. 4 /
fig_wm_subregions_age.py.  Run:  python fig4_3d.py   (--recompose reuses the saved renders)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from PIL import Image

import figdata as D
import fig_hero_ab as AB
import fig_hero_cover as H
import fig_hero_gm as G
import fig_3d_wholebrain as WB
import nature_style as S
import render3d as R
from fig_wm_subregions_age import COL, PARTS, part_means, slope_stats
from scipy import ndimage as ndi

MODE = 'white'
BC = {'CC + cingulum': '#0072B2', 'Corona radiata': '#E69F00'}       # same as Fig. 4 (blue / yellow)
OTHER = '#C9CDD2'
CCL, CING, CR = [3, 4, 5], [35, 36], [23, 24, 25, 26, 27, 28]
CC_CING = CCL + CING
# individual-region panels: region -> (atlas labels, colour, render key)
IND = {'Corpus callosum': (CCL, COL['Corpus callosum'], 'ccl'), 'Cingulum': (CING, COL['Cingulum'], 'cing'),
       'CC + cingulum': (CC_CING, COL['CC + cingulum'], 'cc'), 'Corona radiata': (CR, COL['Corona radiata'], 'cr')}
KB = {'CC + cingulum': 'cc', 'Corona radiata': 'cr'}                 # thumbnails in the two-region colours
VIEW, DIST = (0.80, 0.55, 0.42), 520                                 # hero overview camera


def stats():
    sub = D.subregions('raw')
    sub = sub[sub.Subregion.isin(sum(PARTS.values(), []))]
    pm = part_means(sub)
    age = pm.Age.values.astype(float)
    st = {p: slope_stats(age, pm[p]) for p in IND}
    st['diff'] = slope_stats(age, pm['Corona radiata'] - pm['CC + cingulum'])
    lg = pm.melt(id_vars=['Subject', 'Age'], value_vars=list(BC), var_name='Region', value_name='G')
    lg['CR'] = (lg.Region == 'Corona radiata').astype(int)
    mm = smf.mixedlm('G ~ Age * CR', lg, groups='Subject').fit(reml=True)
    ri = {k: v.iloc[0] for k, v in mm.random_effects.items()}
    lg['Gadj'] = lg.G - lg.Subject.map(ri)
    fe = mm.fe_params
    mline = {'CC + cingulum': (fe['Intercept'], fe['Age']),
             'Corona radiata': (fe['Intercept'] + fe['CR'], fe['Age'] + fe['Age:CR'])}
    def ct(c):                                     # contrast on the fixed effects -> slope per decade, CI, P
        names = list(mm.fe_params.index)            # MixedLM t_test wants a numeric row over the fixed effects
        r = np.zeros((1, len(names)))
        for term in c.split(' = ')[0].split(' + '):
            r[0, names.index(term)] = 1
        t = mm.t_test(r)
        lo, hi = np.asarray(t.conf_int()).ravel()
        return dict(slope=10 * float(np.ravel(t.effect)[0]), lo=10 * lo, hi=10 * hi, P=float(np.ravel(t.pvalue)[0]))
    ms = {'CC + cingulum': ct('Age = 0'), 'Corona radiata': ct('Age + Age:CR = 0'), 'diff': ct('Age:CR = 0')}
    # fixed-effect line for CR - (CC + cing.): CR + Age:CR * age, with its covariance for a CI band
    ms['dline'] = (fe['CR'], fe['Age:CR'], mm.cov_params().loc[['CR', 'Age:CR'], ['CR', 'Age:CR']].values)
    return pm, age, st, lg, mline, (10 * mm.params['Age:CR'], mm.pvalues['Age:CR']), ms


def meshes():
    X = G.load()
    d = X['d']
    sl, slab = d['sl'], X['wm_slab']
    hull = ndi.binary_fill_holes(ndi.binary_closing(X['seg_wb'] >= 2, iterations=1))
    M = dict(hull=WB.surf_world(hull, X['seg_aff'], sigma=1.3),
             **{k: G.slab_mesh(X, slab & np.isin(sl, lab), sl) for lab, _, k in IND.values()},
             other=G.slab_mesh(X, slab & ~np.isin(sl, CC_CING + CR) & (sl > 0), sl))
    return M


def renders(M, st):
    solid = G._solid
    centre = np.asarray(M['hull'].center)
    a = H.shot(G.scene(MODE, (2000, 1700), centre, VIEW, DIST, [
        lambda p: G.grey_glass(p, M['hull'], MODE),
        lambda p: p.add_mesh(solid(M['cc'], BC['CC + cingulum']), scalars='rgb', rgb=True, **H.MESH),
        lambda p: p.add_mesh(solid(M['cr'], BC['Corona radiata']), scalars='rgb', rgb=True, **H.MESH)]), 'fig4_3d_a')
    # b: exploded stack, CR at the bottom, CC + cingulum on top, coloured by slope
    lift = 46.0
    p = AB.plotter((1500, 1900), MODE)
    cols = {'cr': BC['Corona radiata'], 'cc': BC['CC + cingulum']}
    for i, k in enumerate(('cr', 'cc')):
        m = M[k].translate((0, 0, i * lift), inplace=False)
        p.add_mesh(solid(m, cols[k]), scalars='rgb', rgb=True, **H.MESH)
    H.cam(p, np.asarray(M['cr'].center) + (0, 0, lift / 2), AB.STACK_DIR, 560)
    p.enable_parallel_projection()
    p.reset_camera()
    b = H.shot(p, 'fig4_3d_b')
    # thumbnails for the individual-region panels: the region inside the glass brain, hero camera
    th = {}
    for reg, (_, col, k) in IND.items():
        th[reg] = H.shot(G.scene(MODE, (1100, 935), centre, VIEW, DIST, [
            lambda p: G.grey_glass(p, M['hull'], MODE),
            lambda p, k=k, col=col: p.add_mesh(solid(M[k], col), scalars='rgb', rgb=True, **H.MESH)]), f'fig4_3d_t_{k}')
    for reg, k in KB.items():
        th['b ' + reg] = H.shot(G.scene(MODE, (1100, 935), centre, VIEW, DIST, [
            lambda p: G.grey_glass(p, M['hull'], MODE),
            lambda p, k=k, col=BC[reg]: p.add_mesh(solid(M[k], col), scalars='rgb', rgb=True, **H.MESH)]),
            f'fig4_3d_tb_{k}')
    return a, b, th


def crop(im, pad=12):
    return AB.cut(im, AB.crop(im, MODE, pad))


def compose(a, b, th, pm, age, st, lg, mline, mm, ms):
    S.apply()                                      # light Nature style (render3d switches to dark on import)
    S.mpl.rcParams.update({k: ('white' if 'face' in k else 'k') for k in S.DARK_RC})   # undo S.dark()
    F = S.MMFigure(S.DOUBLE, 244)
    a, b = crop(a), crop(b)
    xs = np.array([age.min(), age.max()])
    # a: the two compared regions in the glass brain
    F.letter('a', 0.5, 1)
    ha = 58
    wa = ha * a.shape[1] / a.shape[0]
    ax = F.ax(4, 5, wa, ha); ax.imshow(a); ax.axis('off')
    for k2, (lab, col) in enumerate([('Corpus callosum + cingulum', BC['CC + cingulum']),
                                     ('Corona radiata', BC['Corona radiata'])]):
        y = 5 + ha + 1.5 + k2 * 4.0
        F.fig.add_artist(S.mpl.patches.Rectangle((6 / F.W, 1 - (y + 1.4) / F.H), 2.6 / F.W, 2.6 / F.H, color=col,
                                                 transform=F.fig.transFigure))
        F.text(9.5, y, lab, ha='left', fontsize=6.5)
    # b: exploded stack with each region's age slope
    xb = 4 + wa + 6
    F.letter('b', xb - 3, 1)
    hb = 62
    wb = hb * b.shape[1] / b.shape[0]
    ax = F.ax(xb, 7, wb, hb); ax.imshow(b); ax.axis('off')
    tx = xb + wb + 1
    for reg, y in [('CC + cingulum', 20), ('Corona radiata', 50)]:
        s_ = st[reg]
        F.text(tx, y, reg, ha='left', fontsize=7, fontweight='bold', color=BC[reg])
        F.text(tx, y + 4.2, S.minus(f'{s_["slope"]:+.2f}% per decade'), ha='left', fontsize=6.5)
        F.text(tx, y + 8, f'$P$ = {s_["P"]:.2f}', ha='left', fontsize=6.5)
    # c: GluCEST vs age per region, each with its 3D thumbnail
    yc, pw, ph, gx = 86, 36, 30, 8.3
    F.letter('c', 0.5, yc - 7)
    for i, (reg, (_, col, k)) in enumerate(IND.items()):
        ax = F.ax(13 + i * (pw + gx), yc, pw, ph)
        S.reg_plot(ax, age, pm[reg], col, s=8)
        q = st[reg]
        ax.set_title(reg, fontsize=6.5, color=col, fontweight='bold', pad=2)
        ax.text(0.03, 0.97, S.minus(f'{q["slope"]:+.2f}% per decade') + '\n' + f'$P$ = {q["P"]:.2f}',
                transform=ax.transAxes, va='top', fontsize=5.5)
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.55 * (hi - lo))
        t = crop(th[reg], 6)
        axi = ax.inset_axes([0.58, 0.54, 0.44, 0.46 * (pw / ph) * t.shape[0] / t.shape[1]])
        axi.imshow(t); axi.axis('off')
        ax.set_xticks([20, 40, 60])
        ax.set_xlabel('Age (years)'); ax.set_ylabel('GluCEST (%)' if i == 0 else '')
        S.despine(ax)
    # d, e, f
    y0, h = 136, 46
    F.letter('d', 0.5, y0 - 8)
    ax = F.ax(13, y0, 46, h)
    for reg, lab in [('CC + cingulum', 'CC + cing.'), ('Corona radiata', 'CR')]:
        ax.scatter(age, pm[reg], s=12, color=BC[reg], edgecolor='white', lw=0.35, zorder=3)
        k, c0 = np.polyfit(age, pm[reg], 1)
        ax.plot(xs, c0 + k * xs, color=BC[reg], lw=1.6, zorder=4,
                label=S.minus(f'{lab}: {st[reg]["slope"]:+.2f}% per decade'))
    ax.set_xlabel('Age (years)'); ax.set_ylabel('GluCEST (%)')
    ax.legend(fontsize=5.5, loc='lower left', frameon=False, handlelength=1.4)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo - 0.3 * (hi - lo), hi)
    ax.set_xticks([20, 40, 60]); S.despine(ax)
    ax.set_title('CC + cingulum vs CR', fontsize=7, fontweight='bold', pad=4)

    F.letter('e', 64, y0 - 8)
    dd = st['diff']
    ax = F.ax(76, y0, 44, h)
    S.reg_plot(ax, age, pm['Corona radiata'] - pm['CC + cingulum'], '0.2', s=12)
    ax.axhline(0, color=S.GREY, lw=0.5, ls=':')
    ax.set_xlabel('Age (years)'); ax.set_ylabel(S.minus('CR − (CC + cing.) (%)'))
    ax.set_xticks([20, 40, 60]); S.despine(ax)
    ax.set_title(S.minus(f'Slope difference\n{dd["slope"]:+.2f}% per decade, ') + f'$P$ = {dd["P"]:.3f}', fontsize=7,
                 fontweight='bold', pad=4)

    F.letter('f', 125, y0 - 8)
    ax = F.ax(137, y0, 44, h)
    for reg in BC:
        g = lg[lg.Region == reg]
        ax.scatter(g.Age, g.Gadj, s=12, color=BC[reg], edgecolor='white', lw=0.35, zorder=3)
        a0, b0 = mline[reg]
        ax.plot(xs, a0 + b0 * xs, color=BC[reg], lw=1.6, zorder=4)
    ax.set_xlabel('Age (years)'); ax.set_ylabel('GluCEST, intercept removed (%)')
    ax.set_xticks([20, 40, 60]); S.despine(ax)
    ax.set_title(S.minus(f'Mixed model (age × region)\n{mm[0]:+.2f}% per decade, ') + f'$P$ = {mm[1]:.3f}',
                 fontsize=7, fontweight='bold', pad=4)
    q = ms['diff']
    ax.text(0.97, 0.03, S.minus(f'Slope difference (CR − CC + cing.)\n{q["slope"]:+.2f}% per decade\n'
                                f'95% CI {q["lo"]:.2f} to {q["hi"]:.2f}, ') + f'$P$ = {q["P"]:.3f}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=5.5)
    # g: mixed model, one plot per region, with 3D thumbnails; slope difference = age x region interaction
    yg, pw, ph = 200, 46, 32
    F.letter('g', 0.5, yg - 7)
    for i, reg in enumerate(BC):
        ax = F.ax(13 + i * (pw + 15), yg, pw, ph)
        g = lg[lg.Region == reg]
        ax.scatter(g.Age, g.Gadj, s=8, color=BC[reg], edgecolor='white', lw=0.3, zorder=3)
        a0, b0 = mline[reg]
        ax.plot(xs, a0 + b0 * xs, color=BC[reg], lw=1.3, zorder=4)
        q = ms[reg]
        ax.set_title(reg + ' (mixed model)', fontsize=6.5, color=BC[reg], fontweight='bold', y=1.04)
        ax.text(0.03, 0.97, S.minus(f'{q["slope"]:+.2f}% per decade') + '\n' + f'$P$ = {q["P"]:.2f}',
                transform=ax.transAxes, va='top', fontsize=5.5)
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.95 * (hi - lo))
        t = crop(th['b ' + reg], 6)
        axi = ax.inset_axes([0.64, 0.50, 0.36, 0.36 * (pw / ph) * t.shape[0] / t.shape[1]])
        axi.imshow(t); axi.axis('off')
        ax.set_xticks([20, 40, 60]); ax.set_xlabel('Age (years)')
        ax.set_ylabel('GluCEST, intercept\nremoved (%)' if i == 0 else '', fontsize=6)
        S.despine(ax)
    # g (right): mixed-model slope difference; points = per-participant CR - (CC + cing.) (intercept cancels)
    ax = F.ax(13 + 2 * (pw + 15), yg, pw, ph)
    wide = lg.pivot(index='Subject', columns='Region', values='Gadj')
    sa = pm.set_index('Subject').Age.loc[wide.index].values
    ax.scatter(sa, wide['Corona radiata'] - wide['CC + cingulum'], s=8, color='0.2', edgecolor='white', lw=0.3, zorder=3)
    c0, c1, V = ms['dline']
    xg = np.linspace(age.min(), age.max(), 100)
    X2 = np.c_[np.ones_like(xg), xg]
    se = np.sqrt(np.einsum('ij,jk,ik->i', X2, V, X2))
    ax.fill_between(xg, c0 + c1 * xg - 1.96 * se, c0 + c1 * xg + 1.96 * se, color='0.2', alpha=0.18, lw=0)
    ax.plot(xg, c0 + c1 * xg, color='0.2', lw=1.3, zorder=4)
    ax.axhline(0, color=S.GREY, lw=0.5, ls=':')
    q = ms['diff']
    ax.set_title('Slope difference (mixed model)', fontsize=6.5, fontweight='bold', y=1.04)
    ax.text(0.03, 0.97, S.minus(f'{q["slope"]:+.2f}% per decade') + '\n' + f'$P$ = {q["P"]:.3f}',
            transform=ax.transAxes, va='top', fontsize=5.5)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.3 * (hi - lo))
    ax.set_xticks([20, 40, 60]); ax.set_xlabel('Age (years)'); ax.set_ylabel(S.minus('CR − (CC + cing.) (%)'), fontsize=6)
    S.despine(ax)
    F.save('Fig4_3D_WM_divergence')


def main(recompose=False):
    # recompose=True reuses the saved 3D renders (no data drive needed)
    pm, age, st, lg, mline, mm, ms = stats()
    print({k: round(v['slope'], 3) for k, v in st.items()}, 'diff P', round(st['diff']['P'], 4), 'mixed', mm)
    if recompose:
        ld = lambda n: np.asarray(Image.open(H.ASSETS / f'hero_{n}.png').convert('RGB'))
        a, b = ld('fig4_3d_a'), ld('fig4_3d_b')
        th = {reg: ld(f'fig4_3d_t_{k}') for reg, (_, _, k) in IND.items()}
        th.update({'b ' + reg: ld(f'fig4_3d_tb_{k}') for reg, k in KB.items()})
    else:
        a, b, th = renders(meshes(), st)
    compose(a, b, th, pm, age, st, lg, mline, mm, ms)
    return st, mm, ms


if __name__ == '__main__':
    main('--recompose' in sys.argv)
