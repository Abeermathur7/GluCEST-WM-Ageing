"""Cover-style hero figure (dark): whole-brain WM atlas -> GluCEST slab -> subregions -> divergent ageing.

a  Whole-brain JHU WM atlas as translucent glass (tract-family colours), the GluCEST slab white matter
   rendered solid with GluCEST inside it (P04, 26 y; only participant with a whole-brain native atlas).
b  Exploded segmentation: slab WM ROIs -> CC + cingulum -> corona radiata (same participant, lifted in z).
c  23 y vs 63 y (P01, P16): GluCEST on CC + cingulum and corona radiata, region outlines in family colour.
d  The result: regional GluCEST vs age with OLS fits; slope difference CR - (CC + cingulum) from the
   within-participant difference regression (numbers computed from subregion_means.csv, as in the paper).

Colour: inferno with its black end trimmed (12-100 %) so low values stay visible on black; perceptually
uniform and colour-blind safe (rainbow/jet avoided per Nature artwork guidance). Scale 0-12 %.
Also writes Cover_KJ.png (2:3 portrait, no text) and assets_3d/hero_*.png.

Run:  python fig_hero_cover.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyvista as pv
import statsmodels.formula.api as smf
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy import ndimage as ndi

import figdata as D
import nature_style as S
import render3d as R
import fig_3d_wholebrain as WB
import fig_3d_overview as OV

C = S.SUB_COL
PINK, TEAL = C['CC body'], C['SCR']
BG = 'black'
FG = '#EDEDED'
CMAP = ListedColormap(S.mpl.colormaps['inferno'](np.linspace(0.12, 1.0, 256)), name='inferno_lift')
MESH = dict(smooth_shading=True, ambient=0.22, diffuse=0.85, specular=0.35, specular_power=25)
ASSETS = OV.ASSETS


def plotter(win=(2200, 1700)):
    p = pv.Plotter(off_screen=True, window_size=win)
    p.set_background(BG)
    p.enable_anti_aliasing('ssaa')
    p.enable_depth_peeling(12)
    return p


def cam(p, centre, dvec, dist, up=(0, 0, 1), angle=22):
    dvec = np.asarray(dvec, float) / np.linalg.norm(dvec)
    p.camera_position = [tuple(centre + dist * dvec), tuple(centre), up]
    p.camera.view_angle = angle


def shot(p, name):
    f = ASSETS / f'hero_{name}.png'
    p.screenshot(str(f))
    p.close()
    return np.asarray(Image.open(f).convert('RGB'))


def crop_dark(im, pad=20):
    m = im.max(2) > 12
    rr, cc = np.where(m)
    h, w = m.shape
    return im[max(rr.min() - pad, 0):min(rr.max() + pad, h), max(cc.min() - pad, 0):min(cc.max() + pad, w)]


# ── P04: whole-brain + slab meshes in scanner space ───────────────────────────
def kj_meshes():
    d = OV.load()
    wb = OV.colour_by_label(WB.surf_world(d['lab'] > 0, d['aff'], sigma=0.9), d['lab'], d['aff'])
    out = {'wb': wb}
    for k, m in d['regions'].items():
        mi, ai = OV.iso(m.astype(np.uint8), d['saff'])
        li, _ = OV.iso(np.where(m, d['sl'], 0), d['saff'])
        mesh = OV.colour_by_label(WB.surf_world(mi > 0, ai, sigma=0.9), li, ai)
        mesh.point_data['glu'] = WB.cest_of_points(mesh.points, d['cests'][k], d['saff'], d['sshape'])
        out[k] = mesh
    return out


def panel_a(M, win=(2200, 1700), name='a', dist=520):
    p = plotter(win)
    p.add_mesh(M['wb'], scalars='rgb', rgb=True, opacity=0.16, **MESH)
    p.add_mesh(M['slab'], scalars='glu', cmap=CMAP, clim=(R.VMIN, R.VMAX), show_scalar_bar=False, **MESH)
    c = np.asarray(M['wb'].center)
    cam(p, c, (0.80, 0.55, 0.42), dist)
    return shot(p, name)


def panel_b(M):
    """Exploded stack, bottom -> top: slab WM ROIs, CC + cingulum, corona radiata."""
    p = plotter((1700, 2100))
    lift = 34.0
    for i, k in enumerate(['slab', 'cc', 'cr']):
        m = M[k].translate((0, 0, i * lift), inplace=False)
        p.add_mesh(m, scalars='rgb', rgb=True, **MESH)
        # faint plate under each layer
        b = m.bounds
        plate = pv.Plane(center=((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, b[4] - 1.5),
                         i_size=(b[1] - b[0]) * 1.12, j_size=(b[3] - b[2]) * 1.08)
        p.add_mesh(plate, color='#9FB3C8', opacity=0.07, lighting=False)
        p.add_mesh(plate.extract_feature_edges(boundary_edges=True), color='#9FB3C8', opacity=0.5, line_width=1.5)
    c = np.asarray(M['slab'].center) + (0, 0, lift)
    cam(p, c, (0.55, -0.62, 0.58), 330)
    return shot(p, 'b')


def cached_region_mesh(s, key):
    d = R.subject(s)
    reg = d['regions'][key]
    surf = R.surface(reg)
    f = R.fill_nearest(d['cest'], d['ok'] & reg)
    f = R.prep(ndi.gaussian_filter(f, (0.7, 0.7, 0.35)), 1)
    surf.point_data['glu'] = R.sample(surf, f, 'glu')
    return surf, d['centre']


def panel_c(s):
    p = plotter((1600, 1500))
    centre = None
    for key, col in [('cc', PINK), ('cr', TEAL)]:
        m, centre = cached_region_mesh(s, key)
        p.add_mesh(m, scalars='glu', cmap=CMAP, clim=(R.VMIN, R.VMAX), show_scalar_bar=False, **MESH)
    d, up = R.VIEWS['superior-anterior']
    cam(p, centre, d, 250, up=tuple(float(u) for u in up))
    return shot(p, f'c_{s}')


def ages_table():
    rows = [(s, a, R.paper_mean(s, 'cc'), R.paper_mean(s, 'cr')) for s, a in D.subjects_by_age()]
    t = pd.DataFrame(rows, columns=['s', 'age', 'cc', 'cr'])
    t['diff'] = t.cr - t.cc
    fits = {v: smf.ols(f'{v} ~ age', t).fit() for v in ('cc', 'cr', 'diff')}
    return t, fits


def dark_axes(ax):
    ax.set_facecolor(BG)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(FG)
    ax.tick_params(colors=FG, which='both')
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)


def make():
    S.apply()
    M = kj_meshes()
    print('P04 meshes ready', flush=True)
    A = crop_dark(panel_a(M))
    B = crop_dark(panel_b(M))
    Cy, Co = crop_dark(panel_c('P01')), crop_dark(panel_c('P16'))
    # cover: portrait, no text
    cov = crop_dark(panel_a(M, win=(3000, 3000), name='cover_raw', dist=430), pad=40)
    canvas = Image.new('RGB', (2400, 3600), 'black')
    im = Image.fromarray(cov)
    im = im.resize((2300, int(2300 * im.height / im.width)), Image.LANCZOS)
    canvas.paste(im, ((2400 - im.width) // 2, int(3600 * 0.58 - im.height / 2)))   # lower-middle: room for masthead
    canvas.save(S.OUT / 'Cover_KJ.png')
    print('cover saved', flush=True)

    t, fits = ages_table()
    sl = {k: fits[k].params.age * 10 for k in fits}
    pd_ = fits['diff'].pvalues.age

    # ── compose (183 mm, black) ─────────────────────────────────────────────
    W = S.DOUBLE
    aw = 112
    ah = aw * A.shape[0] / A.shape[1]
    bw = W - aw - 6
    bh = ah
    row2 = 52
    FH = 4 + ah + 6 + row2 + 8
    F = S.MMFigure(W, FH)
    F.fig.patch.set_facecolor(BG)
    lab = dict(color=FG, fontweight='bold', fontsize=8, va='top', ha='left')

    def put(img, x, y, w, h):
        ax = F.ax(x, y, w, h)
        S.img_ax(ax)
        ax.imshow(img)
        return ax

    # a
    put(A, 2, 4, aw, ah)
    F.text(3, 2.2, 'a', **lab)
    F.text(8, 2.4, 'Glutamate-weighted CEST inside the white-matter atlas', color=FG, fontsize=6.5,
           ha='left', fontweight='bold')
    F.text(8, 5.4, 'whole-brain JHU atlas (glass) · 3D GluCEST slab (solid) · 26 y', color='#A0A0A0', fontsize=5.2,
           ha='left')
    # b
    bx = aw + 5
    bimg_w = min(bw, bh * B.shape[1] / B.shape[0])
    put(B, bx + (bw - bimg_w) / 2, 4, bimg_w, bh)
    F.text(bx, 2.2, 'b', **lab)
    F.text(bx + 5, 2.4, 'Segmentation hierarchy', color=FG, fontsize=6.5, ha='left', fontweight='bold')
    for txt, frac, col in [('corona radiata', 0.13, TEAL), ('CC + cingulum', 0.45, PINK), ('slab WM ROIs', 0.80, FG)]:
        F.text(bx + bw - 1, 4 + bh * frac, txt, color=col, fontsize=5.5, ha='right', fontweight='bold')
    # colour bar under a
    cax = F.ax(10, 4 + ah - 4, 32, 1.6)
    sm = S.mpl.cm.ScalarMappable(cmap=CMAP, norm=S.mpl.colors.Normalize(R.VMIN, R.VMAX))
    cb = F.fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=[0, 4, 8, 12])
    cb.outline.set_edgecolor(FG); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(colors=FG, width=0.4, length=1.5, labelsize=5)
    F.text(26, 4 + ah - 6.2, 'GluCEST (%)', color=FG, fontsize=5.5)

    # c: young vs old
    y2 = 4 + ah + 6
    F.text(3, y2 - 1.5, 'c', **lab)
    F.text(8, y2 - 1.3, '23 vs 63 years', color=FG, fontsize=6.5, ha='left', fontweight='bold')
    cw = 42
    for j, (img, s) in enumerate([(Cy, 'P01'), (Co, 'P16')]):
        x = 4 + j * (cw + 3)
        h = min(row2 - 9, cw * img.shape[0] / img.shape[1])
        put(img, x, y2 + 2, cw, h)
        age = dict(D.subjects_by_age())[s]
        F.text(x + cw / 2, y2 + 2 + h + 1.8, f'{age} y', color=FG, fontsize=6, fontweight='bold')
        F.text(x + cw * 0.27, y2 + 2 + h + 4.6, f'CC+cing {R.paper_mean(s, "cc"):.2f}%', color=PINK,
               fontsize=4.8, fontweight='bold')
        F.text(x + cw * 0.75, y2 + 2 + h + 4.6, f'CR {R.paper_mean(s, "cr"):.2f}%', color=TEAL,
               fontsize=4.8, fontweight='bold')
    F.text(4 + cw + 1.5, y2 + 1.5, 'CR lateral · CC + cingulum medial', color='#8A8A8A', fontsize=4.6)

    # d: divergence plot
    dx = 4 + 2 * cw + 3 + 9
    F.text(dx - 7, y2 - 1.5, 'd', **lab)
    F.text(dx - 2, y2 - 1.3, 'Regionally divergent age associations', color=FG, fontsize=6.5, ha='left',
           fontweight='bold')
    ax = F.ax(dx, y2 + 3, W - dx - 4, row2 - 10)
    dark_axes(ax)
    xs = np.linspace(20, 76, 50)
    for v, col, name in [('cc', PINK, 'CC + cingulum'), ('cr', TEAL, 'Corona radiata')]:
        f = fits[v]
        pr = f.get_prediction(pd.DataFrame({'age': xs})).summary_frame(alpha=0.05)
        ax.fill_between(xs, pr.mean_ci_lower, pr.mean_ci_upper, color=col, alpha=0.18, lw=0)
        ax.plot(xs, pr['mean'], color=col, lw=1.1)
        ax.scatter(t.age, t[v], s=9, color=col, edgecolor=BG, lw=0.3, zorder=3)
    ax.set_xlim(20, 76)
    ax.set_ylim(5.0, 8.6)
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('GluCEST (%)')
    ax.text(0.02, 0.97, f'CC + cingulum  {S.minus(f"{sl["cc"]:+.2f}")} pp/decade', transform=ax.transAxes,
            color=PINK, fontsize=5.4, va='top', fontweight='bold')
    ax.text(0.02, 0.89, f'Corona radiata  {S.minus(f"{sl["cr"]:+.2f}")} pp/decade', transform=ax.transAxes,
            color=TEAL, fontsize=5.4, va='top', fontweight='bold')
    ax.text(0.98, 0.04, f'slope difference CR $-$ (CC + cing.)\n+{sl["diff"]:.2f} pp/decade, P = {pd_:.3f}',
            transform=ax.transAxes, color=FG, fontsize=5.4, va='bottom', ha='right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    F.text(W / 2, FH - 1.8, 'n = 17 participants (23-73 y), 7 T. Fits: OLS with 95% CI; slope difference from '
           'the within-participant CR - (CC + cingulum) regression on age.', color='#8A8A8A', fontsize=4.6)
    pdf, png = S.OUT / 'Fig_hero_cover.pdf', S.OUT / 'Fig_hero_cover.png'
    for f in (pdf, png):
        f.unlink(missing_ok=True)
    F.fig.savefig(pdf, facecolor=BG)
    F.fig.savefig(png, dpi=600, facecolor=BG)
    S.plt.close(F.fig)
    print('saved', pdf.name, png.name, '| slopes', {k: round(v, 3) for k, v in sl.items()}, 'P diff', round(pd_, 4))


if __name__ == '__main__':
    make()
