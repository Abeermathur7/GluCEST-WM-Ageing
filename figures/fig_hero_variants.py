"""Hero cover figure, colour variants v2, v3, ... (dark). Layout from Fig_hero_cover minus panel d:

a  whole-brain JHU WM atlas as glass + GluCEST slab (solid)        b  exploded segmentation (more spacing)
c  23 vs 63 y GluCEST on CC + cingulum and corona radiata (full width)

Each theme sets: GluCEST colormap (a, c), glass tint for the atlas (tract-family colours or one hue),
and the segmentation palette for b. All GluCEST maps are perceptually ordered; dark ends are lifted so
low values stay visible on black. Meshes are built once and reused for every theme.

Outputs: Fig_hero_cover_v<N>.pdf/.png and Cover_P04_v<N>.png (2:3, no text) for each theme.
Run:  python fig_hero_variants.py [v2 v4 ...] [--recompose]   (default: all themes; --recompose reuses saved renders; --b re-renders panel b only)
"""
from __future__ import annotations

import sys

import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from PIL import Image

import figdata as D
import nature_style as S
import render3d as R
import fig_3d_wholebrain as WB
import fig_3d_overview as OV
import fig_hero_cover as H

BG, FG = 'black', '#EDEDED'


def lifted(cmap, lo=0.12, hi=1.0):
    c = cmap if not isinstance(cmap, str) else S.mpl.colormaps[cmap]
    return ListedColormap(c(np.linspace(lo, hi, 256)))


# segmentation keys: genu, body, splen, cing, acr, scr, pcr, ic, assoc, stem
THEMES = {
    'v2': dict(name='Ember / neon', glu=lifted('magma', 0.15), glass='family',
               seg=dict(genu='#F7A6D8', body='#E0479E', splen='#9D1F6B', cing='#FFC857',
                        acr='#7AE7FF', scr='#22B8CF', pcr='#0B7285', ic='#5C7CFA', assoc='#8D99AE', stem='#6C5B7B')),
    'v3': dict(name='Ice', glu=lifted(sns.color_palette('mako', as_cmap=True), 0.15), glass='#6FA8DC',
               seg=dict(genu='#FFD1BA', body='#FF8C61', splen='#CE4A2E', cing='#FFE066',
                        acr='#B8F2E6', scr='#5EDFC4', pcr='#2A9D8F', ic='#8DA9C4', assoc='#5C6B7A', stem='#3E4C59')),
    'v4': dict(name='Viridis', glu=lifted('viridis', 0.08), glass='#B8B8B8',
               seg=dict(genu='#DCCBFF', body='#9B6BFF', splen='#6A3FD1', cing='#F472B6',
                        acr='#A7F3D0', scr='#34D399', pcr='#047857', ic='#7DD3FC', assoc='#9CA3AF', stem='#6B7280')),
    'v5': dict(name='Plasma', glu=lifted('plasma', 0.05), glass='#8C8CFF',
               seg=dict(genu='#FDE68A', body='#FBBF24', splen='#B45309', cing='#F87171',
                        acr='#BFDBFE', scr='#60A5FA', pcr='#1D4ED8', ic='#C4B5FD', assoc='#94A3B8', stem='#64748B')),
    'v6': dict(name='Cividis (most colour-blind safe)', glu=lifted('cividis', 0.10), glass='#9FB3C8',
               seg=dict(genu='#E9D5FF', body='#C084FC', splen='#7E22CE', cing='#FDBA74',
                        acr='#CCFBF1', scr='#2DD4BF', pcr='#0F766E', ic='#93C5FD', assoc='#A1A1AA', stem='#71717A')),
    'v7': dict(name='Rocket / teal', glu=lifted(sns.color_palette('rocket', as_cmap=True), 0.15), glass='family',
               seg=dict(genu='#FBCFE8', body='#EC4899', splen='#9D174D', cing='#FACC15',
                        acr='#99F6E4', scr='#14B8A6', pcr='#115E59', ic='#60A5FA', assoc='#A8A29E', stem='#78716C')),
}


def label_key(l):
    l = int(l)
    return {3: 'genu', 4: 'body', 5: 'splen', 35: 'cing', 36: 'cing', 37: 'cing', 38: 'cing',
            23: 'acr', 24: 'acr', 25: 'scr', 26: 'scr', 27: 'pcr', 28: 'pcr'}.get(
        l, 'ic' if 17 <= l <= 22 else 'stem' if (1 <= l <= 2 or 6 <= l <= 16) else 'assoc')


def build():
    """P04 meshes with per-vertex atlas labels stored, so any palette can be applied later."""
    d = OV.load()
    wb = WB.surf_world(d['lab'] > 0, d['aff'], sigma=0.9)
    wb.point_data['lab'] = WB.label_of_points(wb.points, d['lab'], d['aff'])
    M = {'wb': wb}
    regions = dict(d['regions'])
    regions['cconly'] = np.isin(d['sl'], [3, 4, 5])        # genu, body, splenium
    regions['cing'] = np.isin(d['sl'], [35, 36])           # cingulum (cingulate gyrus) R/L
    for k, m in regions.items():
        mi, ai = OV.iso(m.astype(np.uint8), d['saff'])
        li, _ = OV.iso(np.where(m, d['sl'], 0), d['saff'])
        mesh = WB.surf_world(mi > 0, ai, sigma=0.9)
        mesh.point_data['lab'] = WB.label_of_points(mesh.points, li, ai)
        if k in d['cests']:
            mesh.point_data['glu'] = WB.cest_of_points(mesh.points, d['cests'][k], d['saff'], d['sshape'])
        M[k] = mesh
    young = {k: H.cached_region_mesh('P01', k) for k in ('cc', 'cr')}
    old = {k: H.cached_region_mesh('P16', k) for k in ('cc', 'cr')}
    return M, young, old


def paint(mesh, seg):
    rgb = np.array([R.hex2rgb(seg[label_key(l)]) for l in mesh.point_data['lab']])
    m = mesh.copy()
    m.point_data['rgb'] = (rgb * 255).astype(np.uint8)
    return m


def render_a(M, th, win=(2200, 1700), dist=520, name='a'):
    p = H.plotter(win)
    if th['glass'] == 'family':
        p.add_mesh(paint(M['wb'], dict(th['seg'])), scalars='rgb', rgb=True, opacity=0.16, **H.MESH)
    else:
        p.add_mesh(M['wb'], color=th['glass'], opacity=0.13, **H.MESH)
    p.add_mesh(M['slab'], scalars='glu', cmap=th['glu'], clim=(R.VMIN, R.VMAX), show_scalar_bar=False, **H.MESH)
    H.cam(p, np.asarray(M['wb'].center), (0.80, 0.55, 0.42), dist)
    return H.shot(p, name)


B_LIFT, B_SEP = 52.0, 22.0       # layer spacing; extra lift of the cingulum above the corpus callosum
B_LAYERS = [('slab', 0.0, True), ('cconly', B_LIFT, True), ('cing', B_LIFT + B_SEP, False),
            ('cr', 2 * B_LIFT + B_SEP, True)]


def render_b(M, th, tag):
    p = H.plotter((1700, 2600))
    for k, z, plate_on in B_LAYERS:
        m = paint(M[k], th['seg']).translate((0, 0, z), inplace=False)
        p.add_mesh(m, scalars='rgb', rgb=True, **H.MESH)
        if not plate_on:
            continue
        b = M['slab' if k == 'cconly' else k].translate((0, 0, z), inplace=False).bounds
        plate = H.pv.Plane(center=((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, b[4] - 1.5),
                           i_size=(b[1] - b[0]) * 1.12, j_size=(b[3] - b[2]) * 1.08)
        p.add_mesh(plate, color='#9FB3C8', opacity=0.07, lighting=False)
        p.add_mesh(plate.extract_feature_edges(boundary_edges=True), color='#9FB3C8', opacity=0.5, line_width=1.5)
    c = np.asarray(M['slab'].center) + (0, 0, B_LAYERS[-1][1] / 2)
    H.cam(p, c, (0.55, -0.62, 0.58), 440)
    return H.shot(p, f'b_{tag}')


def render_c(meshes, th, tag):
    p = H.plotter((1600, 1500))
    for k in ('cc', 'cr'):
        m, centre = meshes[k]
        p.add_mesh(m, scalars='glu', cmap=th['glu'], clim=(R.VMIN, R.VMAX), show_scalar_bar=False, **H.MESH)
    d, up = R.VIEWS['superior-anterior']
    H.cam(p, centre, d, 250, up=tuple(float(u) for u in up))
    return H.shot(p, f'c_{tag}')


def B_LABELS(seg):
    """(text, vertical position as a fraction of the b image, colour) for each layer, top to bottom."""
    return [('corona radiata', 0.17, seg['scr']), ('cingulum', 0.40, seg['cing']),
            ('corpus callosum', 0.56, seg['body']), ('slab WM ROIs', 0.84, FG)]


def compose(tag, th, A, B, Cy, Co):
    seg = th['seg']
    cc_col, cr_col = seg['body'], seg['scr']
    W = S.DOUBLE
    # Nature-style type, sized for reading at 183 mm: letters 11 pt bold, titles 9 pt bold, labels 8 pt
    LET, TTL, LBL, TCK = 11, 9, 8, 7.5
    aw = 98
    ah = aw * A.shape[0] / A.shape[1]
    bw = W - aw - 6
    row2 = 56
    top = 8                                          # room for the 9 pt panel titles
    FH = top + ah + 9 + row2 + 3
    F = S.MMFigure(W, FH)
    F.fig.patch.set_facecolor(BG)
    lab = dict(color=FG, fontweight='bold', fontsize=LET, va='top', ha='left')
    ttl = dict(color=FG, fontweight='bold', fontsize=TTL, va='top', ha='left')

    def put(img, x, y, w, h):
        ax = F.ax(x, y, w, h)
        S.img_ax(ax)
        ax.imshow(img)

    put(A, 2, top, aw, ah)
    F.text(1, 1.2, 'a', **lab)
    F.text(7, 1.6, 'GluCEST in the white-matter atlas', **ttl)
    bx = aw + 5
    label_col = 25                                   # right-hand column for the 8 pt layer labels
    bimg_h = ah + 2
    bimg_w = bimg_h * B.shape[1] / B.shape[0]
    if bimg_w > bw - label_col:
        bimg_w = bw - label_col
        bimg_h = bimg_w * B.shape[0] / B.shape[1]
    by = top + (ah - bimg_h) / 2
    put(B, bx, by, bimg_w, bimg_h)
    F.text(bx - 1, 1.2, 'b', **lab)
    F.text(bx + 5, 1.6, 'Segmentation hierarchy', **ttl)
    for txt, frac, col in B_LABELS(seg):
        # layer centres as fractions of the b image height
        F.text(W - 1, by + bimg_h * frac, txt, color=col, fontsize=LBL, ha='right', fontweight='bold')
    cax = F.ax(8, top + ah - 3, 36, 2.2)
    sm = S.mpl.cm.ScalarMappable(cmap=th['glu'], norm=S.mpl.colors.Normalize(R.VMIN, R.VMAX))
    cb = F.fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=[0, 4, 8, 12])
    cb.outline.set_edgecolor(FG)
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(colors=FG, width=0.6, length=2, labelsize=TCK)
    F.text(26, top + ah - 6.2, 'GluCEST (%)', color=FG, fontsize=LBL, fontweight='bold')

    # c: full width, two large renders
    y2 = top + ah + 9
    F.text(1, y2 - 2.8, 'c', **lab)
    F.text(7, y2 - 2.4, '23 vs 63 years', **ttl)
    ch = row2 - 9
    ages = dict(D.subjects_by_age())
    widths = [ch * im.shape[1] / im.shape[0] for im in (Cy, Co)]
    gap = 22
    x = (W - sum(widths) - gap) / 2
    for img, s, w in zip((Cy, Co), ('P01', 'P16'), widths):
        put(img, x, y2 + 1, w, ch)
        F.text(x + w / 2, y2 + 1 + ch + 3, f'{ages[s]} y', color=FG, fontsize=TTL, fontweight='bold')
        x += w + gap
    pdf, png = S.OUT / f'Fig_hero_cover_{tag}.pdf', S.OUT / f'Fig_hero_cover_{tag}.png'
    for f in (pdf, png):
        f.unlink(missing_ok=True)
    F.fig.savefig(pdf, facecolor=BG)
    F.fig.savefig(png, dpi=600, facecolor=BG)
    S.plt.close(F.fig)


def cover(M, th, tag):
    cov = H.crop_dark(render_a(M, th, win=(3000, 3000), dist=430, name=f'cover_{tag}'), pad=40)
    canvas = Image.new('RGB', (2400, 3600), 'black')
    im = Image.fromarray(cov)
    im = im.resize((2300, int(2300 * im.height / im.width)), Image.LANCZOS)
    canvas.paste(im, ((2400 - im.width) // 2, int(3600 * 0.58 - im.height / 2)))
    canvas.save(S.OUT / f'Cover_P04_{tag}.png')


def make(tags):
    S.apply()
    S.dark()
    M, young, old = build()
    print('meshes ready', flush=True)
    for tag in tags:
        th = THEMES[tag]
        A = H.crop_dark(render_a(M, th, name=f'a_{tag}'))
        B = H.crop_dark(render_b(M, th, tag))
        Cy, Co = H.crop_dark(render_c(young, th, f'young_{tag}')), H.crop_dark(render_c(old, th, f'old_{tag}'))
        compose(tag, th, A, B, Cy, Co)
        cover(M, th, tag)
        print('saved', tag, th['name'], flush=True)


def recompose(tags):
    """Rebuild the composed figures from the renders already saved in assets_3d (no re-rendering)."""
    S.apply()
    S.dark()
    load = lambda n: H.crop_dark(np.asarray(Image.open(H.ASSETS / f'hero_{n}.png').convert('RGB')))
    for tag in tags:
        compose(tag, THEMES[tag], load(f'a_{tag}'), load(f'b_{tag}'), load(f'c_young_{tag}'), load(f'c_old_{tag}'))
        print('recomposed', tag, flush=True)


def rerender_b(tags):
    """Re-render only panel b (segmentation stack) for each theme, then recompose with saved a and c."""
    S.apply()
    S.dark()
    M, _, _ = build()
    for tag in tags:
        render_b(M, THEMES[tag], tag)
    recompose(tags)


if __name__ == '__main__':
    tags = [a for a in sys.argv[1:] if a in THEMES] or list(THEMES)
    if '--recompose' in sys.argv:
        recompose(tags)
    elif '--b' in sys.argv:
        rerender_b(tags)
    else:
        make(tags)
