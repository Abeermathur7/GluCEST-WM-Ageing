"""Hero figure, two panels, colour themes v2-v7, each in a dark and a white version.

a  Whole-brain JHU WM atlas as glass + the 3D GluCEST slab white matter (solid).
b  Exploded segmentation of the same participant (P04, 26 y), two matching stacks at one scale:
   left  ROIs (slab WM, corpus callosum, cingulum, corona radiata) in segmentation colours,
   right GluCEST painted on the same ROIs (each ROI coloured from its own voxels only).

Text: Arial, letters 11 pt bold, titles 9 pt bold, labels 8 pt; no grey text.
Outputs (overwrite the earlier three-panel versions):
   Fig_hero_cover_v<N>.pdf/.png         black background
   Fig_hero_cover_v<N>_white.pdf/.png   white background (comparison)
Run:  python fig_hero_ab.py [v2 v5 ...] [--recompose | --b]   (--b: re-render panel b only)
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pyvista as pv
from PIL import Image

import figdata as D
import nature_style as S
import render3d as R
import fig_3d_wholebrain as WB
import fig_3d_overview as OV
import fig_hero_cover as H
import fig_hero_variants as V

MODES = {'dark': dict(bg='black', fg='#EDEDED', plate='#9FB3C8', suffix=''),
         'white': dict(bg='white', fg='#111111', plate='#6B7C8F', suffix='_white')}
LIFT = 52.0                    # equal spacing between the four layers
LAYERS = [('slab', 0.0, True), ('cconly', LIFT, True), ('cing', 2 * LIFT, True), ('cr', 3 * LIFT, True)]
LAYER_NAME = {'slab': '24 mm WM 3D slab', 'cconly': 'corpus callosum', 'cing': 'cingulum', 'cr': 'corona radiata'}
STACK_WIN = (1500, 2800)
STACK_DIR = (0.55, -0.62, 0.42)
PLATES = False                 # grey base under each stack layer (removed at the user's request)
TILT = {}                      # degrees per layer (e.g. {'cr': 18}); empty = all layers match the slab


def tilt(mesh, k, pivot):
    """Rotate a layer about the screen-horizontal axis through `pivot` so its top faces the camera more."""
    ang = TILT.get(k, 0.0)
    if not ang:
        return mesh
    d = np.array(STACK_DIR, float)
    d /= np.linalg.norm(d)
    axis = np.cross(d, (0, 0, 1))
    axis /= np.linalg.norm(axis)
    # sign that turns the layer normal (0, 0, 1) towards the camera direction d
    sign = 1 if np.dot(_rot((0, 0, 1), axis, ang), d) > d[2] else -1
    return mesh.rotate_vector(axis, sign * ang, point=tuple(pivot), inplace=False)


def _rot(v, axis, deg):
    """Rodrigues rotation of vector v about unit axis by deg."""
    v, k, t = np.asarray(v, float), np.asarray(axis, float), np.radians(deg)
    return v * np.cos(t) + np.cross(k, v) * np.sin(t) + k * np.dot(k, v) * (1 - np.cos(t))
# c: whole-brain views (RAS: x = right, y = anterior, z = superior)
VIEWS = [('Right', (1.0, 0.15, 0.25), (0, 0, 1)), ('Left', (-1.0, 0.15, 0.25), (0, 0, 1)),
         ('Anterior', (0.15, 1.0, 0.30), (0, 0, 1)), ('Superior', (0.0, -0.20, 1.0), (0, 1, 0))]
LET, TTL, LBL, TCK = 15, 9, 8, 7.5        # panel letters 15 pt bold


def build():
    """P04 meshes with labels; GluCEST for every b layer (slab, CC only, cingulum, CR)."""
    d = OV.load()
    v = D.vol(OV.SUBJ)
    to_nii = lambda a: np.flip(np.transpose(a, (1, 0, 2)), axis=1)
    cest = to_nii(v['CESTmap'].astype(float))
    ok = (d['sl'] > 0) & np.isfinite(cest) & (cest != 0) & (cest > -10) & (cest < 30)
    regions = {'slab': d['regions']['slab'], 'cconly': np.isin(d['sl'], [3, 4, 5]),
               'cing': np.isin(d['sl'], [35, 36]), 'cr': d['regions']['cr']}
    wb = WB.surf_world(d['lab'] > 0, d['aff'], sigma=0.9)
    wb.point_data['lab'] = WB.label_of_points(wb.points, d['lab'], d['aff'])
    M = {'wb': wb}
    for k, m in regions.items():
        mi, ai = OV.iso(m.astype(np.uint8), d['saff'])
        li, _ = OV.iso(np.where(m, d['sl'], 0), d['saff'])
        mesh = WB.surf_world(mi > 0, ai, sigma=0.9)
        mesh.point_data['lab'] = WB.label_of_points(mesh.points, li, ai)
        mesh.point_data['glu'] = WB.cest_of_points(mesh.points, OV.region_cest(cest, ok, m), d['saff'],
                                                   d['sshape'])
        M[k] = mesh
    return M


def plotter(win, mode):
    p = pv.Plotter(off_screen=True, window_size=win)
    p.set_background(MODES[mode]['bg'])
    p.enable_anti_aliasing('ssaa')
    p.enable_depth_peeling(12)
    return p


def crop(im, mode, pad=20):
    m = im.max(2) > 12 if mode == 'dark' else im.min(2) < 243
    rr, cc = np.where(m)
    h, w = m.shape
    return (max(rr.min() - pad, 0), min(rr.max() + pad, h), max(cc.min() - pad, 0), min(cc.max() + pad, w))


def cut(im, b):
    return im[b[0]:b[1], b[2]:b[3]]


def glass_scene(M, th, mode, win):
    p = plotter(win, mode)
    if th['glass'] == 'family':
        p.add_mesh(V.paint(M['wb'], th['seg']), scalars='rgb', rgb=True,
                   opacity=0.16 if mode == 'dark' else 0.22, **H.MESH)
    else:
        p.add_mesh(M['wb'], color=th['glass'], opacity=0.13 if mode == 'dark' else 0.20, **H.MESH)
    p.add_mesh(M['slab'], scalars='glu', cmap=th['glu'], clim=(R.VMIN, R.VMAX), show_scalar_bar=False, **H.MESH)
    return p


def render_a(M, th, mode, tag):
    p = glass_scene(M, th, mode, (2200, 1700))
    H.cam(p, np.asarray(M['wb'].center), (0.80, 0.55, 0.42), 520)
    return H.shot(p, f'ab_a_{tag}_{mode}')


def render_views(M, th, mode, tag):
    imgs = []
    for name, d, up in VIEWS:
        p = glass_scene(M, th, mode, (1600, 1600))
        H.cam(p, np.asarray(M['wb'].center), d, 500, up=up)
        imgs.append(H.shot(p, f'ab_c{name}_{tag}_{mode}'))
    return imgs


def render_stack(M, th, mode, tag, kind):
    """kind 'seg': segmentation colours; kind 'glu': GluCEST on each ROI."""
    p = plotter(STACK_WIN, mode)
    pc = MODES[mode]['plate']
    centres = {}
    sb = M['slab'].bounds
    for k, z, plate_on in LAYERS:
        centres[k] = np.asarray(M[k].center) + (0, 0, z)
        zb = M[k].bounds[4] + z
        pivot = ((sb[0] + sb[1]) / 2, (sb[2] + sb[3]) / 2, zb)      # same pivot for the layer and its plate
        m = tilt(M[k].translate((0, 0, z), inplace=False), k, pivot)
        if kind == 'seg':
            p.add_mesh(V.paint(m, th['seg']), scalars='rgb', rgb=True, **H.MESH)
        else:
            p.add_mesh(m, scalars='glu', cmap=th['glu'], clim=(R.VMIN, R.VMAX), show_scalar_bar=False, **H.MESH)
        if plate_on and PLATES:
            # identical plate under every layer: slab footprint, just below that layer's lowest point
            plate = pv.Plane(center=(pivot[0], pivot[1], zb - 1.5),
                             i_size=(sb[1] - sb[0]) * 1.05, j_size=(sb[3] - sb[2]) * 1.04)
            plate = tilt(plate, k, pivot)
            p.add_mesh(plate, color=pc, opacity=0.07 if mode == 'dark' else 0.10, lighting=False)
            p.add_mesh(plate.extract_feature_edges(boundary_edges=True), color=pc, opacity=0.5, line_width=1.5)
    c = np.asarray(M['slab'].center) + (0, 0, LAYERS[-1][1] / 2)
    H.cam(p, c, STACK_DIR, 560)
    p.enable_parallel_projection()      # orthographic: every layer seen at exactly the same angle
    p.reset_camera()
    p.render()
    rows = {}                       # image row of each layer centre, for the side labels
    ren = p.renderer
    for k, w in centres.items():
        ren.SetWorldPoint(*w, 1.0)
        ren.WorldToDisplay()
        rows[k] = STACK_WIN[1] - ren.GetDisplayPoint()[1]
    name = f'ab_b{kind}_{tag}_{mode}'
    (H.ASSETS / f'hero_{name}_rows.json').write_text(json.dumps(rows))
    return H.shot(p, name), rows


def compose(tag, th, mode, A, Bs, Bg, rows, Cs, names=None, colours=None, stem='Fig_hero_cover_'):
    """rows: {layer: fraction of the (cropped) stack image height}; Cs: the four whole-brain views.
    names / colours: layer label text and colour (default: the WM layers); stem: output file prefix."""
    md = MODES[mode]
    fg, bg = md['fg'], md['bg']
    seg = th['seg']
    names = names or LAYER_NAME
    W = S.DOUBLE
    gap_ab, gap_b, label_col = 5, 1.0, 29
    # split the width so panel a and the two b stacks come out the same height
    ra, rb = A.shape[1] / A.shape[0], Bs.shape[0] / Bs.shape[1]
    K = W - 2 - gap_ab - gap_b - label_col
    aw = (rb * K / 2) / (1 / ra + rb / 2)
    ah = aw / ra
    sw = (K - aw) / 2                                            # width of each stack
    sh = sw * Bs.shape[0] / Bs.shape[1]
    top = 7.5                                                    # panel letters only
    body = max(ah, sh + 3)
    cbar_h = 11                                                  # colour-bar strip under a (no overlap)
    vgap = 2
    vw = (W - 4 - 3 * vgap) / 4
    vh = vw * Cs[0].shape[0] / Cs[0].shape[1]
    yc = top + body + cbar_h
    FH = yc + 6 + vh + 2
    F = S.MMFigure(W, FH)
    F.fig.patch.set_facecolor(bg)
    lab = dict(color=fg, fontweight='bold', fontsize=LET, va='top', ha='left')

    def put(img, x, y, w, h):
        ax = F.ax(x, y, w, h)
        ax.set_axis_off()
        ax.imshow(img)

    ya = top + (body - ah) / 2
    put(A, 2, ya, aw, ah)
    F.text(1, 1.2, 'a', **lab)
    ycb = top + body + 2.5
    xcb = max(2 + aw / 2 - 4, 37)                                # keep the 8 pt label inside the page
    cax = F.ax(xcb, ycb, 36, 2.2)
    sm = S.mpl.cm.ScalarMappable(cmap=th['glu'], norm=S.mpl.colors.Normalize(R.VMIN, R.VMAX))
    cb = F.fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=[0, 4, 8, 12])
    cb.outline.set_edgecolor(fg)
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(colors=fg, labelcolor=fg, width=0.6, length=2, labelsize=TCK)
    F.text(xcb - 2, ycb + 1.1, 'GluCEST Contrast (%)', color=fg, fontsize=LBL, fontweight='bold', ha='right')

    bx = 2 + aw + gap_ab
    yb = top + (body - sh - 3) / 2
    F.text(bx - 1, 1.2, 'b', **lab)
    for i, (img, sub) in enumerate([(Bs, 'ROIs'), (Bg, 'GluCEST')]):
        x = bx + i * (sw + gap_b)
        F.text(x + sw / 2, yb + 1.5, sub, color=fg, fontsize=LBL + 0.5, fontweight='bold')
        put(img, x, yb + 3, sw, sh)
    colour = colours or {'slab': None, 'cconly': seg['body'], 'cing': seg['cing'], 'cr': seg['scr']}
    colour = {k: (fg if c is None else c) for k, c in colour.items()}
    if mode == 'white':     # darken light label colours so text stays legible on white
        colour = {k: (c if c == fg else tuple(0.62 * np.array(R.hex2rgb(c)))) for k, c in colour.items()}
    for k, frac in rows.items():
        F.text(W - 1, yb + 3 + sh * frac, names[k], color=colour[k], fontsize=LBL, ha='right',
               fontweight='bold')

    # c: four whole-brain views
    F.text(1, yc - 1.4, 'c', **lab)
    for i, ((name, *_), img) in enumerate(zip(VIEWS, Cs)):
        x = 2 + i * (vw + vgap)
        F.text(x + vw / 2, yc + 2.2, name, color=fg, fontsize=LBL + 0.5, fontweight='bold')
        put(img, x, yc + 5, vw, vh)

    pdf = S.OUT / f'{stem}{tag}{md["suffix"]}.pdf'
    png = S.OUT / f'{stem}{tag}{md["suffix"]}.png'
    for f in (pdf, png):
        f.unlink(missing_ok=True)
    F.fig.savefig(pdf, facecolor=bg)
    F.fig.savefig(png, dpi=600, facecolor=bg)
    S.plt.close(F.fig)


def finish(tag, th, mode, A, Bs, Bg, rows, Cs, **kw):
    A = cut(A, crop(A, mode))
    b = crop(np.maximum(Bs, Bg) if mode == 'dark' else np.minimum(Bs, Bg), mode)   # one frame for both stacks
    frac = {k: (r - b[0]) / (b[1] - b[0]) for k, r in rows.items()}
    stackc = np.maximum.reduce(Cs) if mode == 'dark' else np.minimum.reduce(Cs)
    c = crop(stackc, mode)                                                          # one frame for all views
    compose(tag, th, mode, A, cut(Bs, b), cut(Bg, b), frac, [cut(im, c) for im in Cs], **kw)


def make(tags, recompose=False, b_only=False):
    S.apply()
    if not recompose:
        M = build()
        print('meshes ready', flush=True)
    for tag in tags:
        th = V.THEMES[tag]
        for mode in MODES:
            ld = lambda n: np.asarray(Image.open(H.ASSETS / f'hero_{n}.png').convert('RGB'))
            if b_only:                  # re-render the b stacks, reuse saved a and c
                A, Cs = ld(f'ab_a_{tag}_{mode}'), [ld(f'ab_c{name}_{tag}_{mode}') for name, *_ in VIEWS]
                Bs, rows = render_stack(M, th, mode, tag, 'seg')
                Bg, _ = render_stack(M, th, mode, tag, 'glu')
            elif recompose:
                A, Bs, Bg = ld(f'ab_a_{tag}_{mode}'), ld(f'ab_bseg_{tag}_{mode}'), ld(f'ab_bglu_{tag}_{mode}')
                rows = json.loads((H.ASSETS / f'hero_ab_bseg_{tag}_{mode}_rows.json').read_text())
                Cs = [ld(f'ab_c{name}_{tag}_{mode}') for name, *_ in VIEWS]
            else:
                A = render_a(M, th, mode, tag)
                Bs, rows = render_stack(M, th, mode, tag, 'seg')
                Bg, _ = render_stack(M, th, mode, tag, 'glu')
                Cs = render_views(M, th, mode, tag)
            finish(tag, th, mode, A, Bs, Bg, rows, Cs)
        print('saved', tag, flush=True)


if __name__ == '__main__':
    make([a for a in sys.argv[1:] if a in V.THEMES] or list(V.THEMES), recompose='--recompose' in sys.argv,
         b_only='--b' in sys.argv)
