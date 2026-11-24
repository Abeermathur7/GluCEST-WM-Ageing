"""Grey-matter hero figure and combined tissue-hierarchy figure (participant P04, 26 y), dark + white.

Fig_hero_GM_v2[_white]        a  whole-brain Harvard-Oxford cortical atlas (glass, lobe colours) + slab GM GluCEST
                              b  exploded slab GM: 24 mm GM 3D slab, frontal, cingulate, parietal, occipital (ROIs | GluCEST)
                              c  four whole-brain views
Fig_hero_combined_v2[_white]  a  whole brain segmented (GM glass over WM)   b  WM segmented (JHU tracts)
                              c  24 mm 3D slab, GM above WM (exploded); top row segmentation, bottom row GluCEST

IMPORTANT, label names: in the Harvard-Oxford ROI images the value is (atlas index + 1), but the .tsv/.json
names were read from the 0-based index, so every GM name is shifted by one (e.g. value 1 is the frontal pole,
not the insula). Verified in P04's native atlas by location. Names here are corrected: name(v) = HO[v - 1].
GM ROIs are restricted to segmented GM (slab seg: 2 = GM, 3 = WM), as in the paper's GM analysis.

Run:  python fig_hero_gm.py [gm] [combined] [wmenc] [wmov] [gmov] [--recompose]   (default: all)
"""
from __future__ import annotations

import json
import sys

import nibabel as nib
import numpy as np
import pyvista as pv
from PIL import Image
from scipy import ndimage as ndi

import figdata as D
import nature_style as S
import render3d as R
import fig_3d_wholebrain as WB
import fig_3d_overview as OV
import fig_hero_cover as H
import fig_hero_variants as V
import fig_hero_ab as AB

P = 'D:/7T_GLUCEST_Repeatability/S148_rep1/nifti/'
TAG = 'v2'
TH = V.THEMES[TAG]
DS = 2

# ── corrected Harvard-Oxford names and lobe groups (0-based atlas index = value - 1) ─────────────────
_HO = json.loads((S.CACHE / 'labels_KJ.json').read_text())['GM']          # keyed by value, names shifted


def ho_name(v):
    return 'Frontal Pole' if v == 1 else _HO.get(str(v - 1), f'label {v}')


GROUP_IDX = {
    'frontal': {0, 2, 3, 4, 5, 6, 24, 25, 32, 40},
    'cingulate': {26, 27, 28, 29},
    'parietal': {16, 17, 18, 19, 20, 30, 41, 42},
    'occipital': {21, 22, 23, 31, 35, 39, 46, 47},
    'temporal': {7, 8, 9, 10, 11, 12, 13, 14, 15, 33, 34, 36, 37, 38, 43, 44, 45},
    'insula': {1},
}
GM_COL = {'frontal': '#F59E0B', 'cingulate': '#EC4899', 'parietal': '#22D3EE', 'occipital': '#A78BFA',
          'temporal': '#84CC16', 'insula': '#F87171', 'none': '#9CA3AF'}
TISSUE = {'GM': '#E07A5F', 'WM': '#3D85C6'}


def group_of(v):
    i = int(v) - 1
    for g, s in GROUP_IDX.items():
        if i in s:
            return g
    return 'none'


def paint_groups(mesh):
    rgb = np.array([R.hex2rgb(GM_COL[group_of(v)]) if v > 0 else R.hex2rgb(GM_COL['none'])
                    for v in mesh.point_data['lab']])
    m = mesh.copy()
    m.point_data['rgb'] = (rgb * 255).astype(np.uint8)
    return m


# ── data ───────────────────────────────────────────────────────────────────────────────────────────
def load():
    d = OV.load()                                   # JHU whole brain + slab WM (verified orientation)
    ld = lambda f: np.asarray(nib.load(P + f).dataobj)
    g = nib.load(P + 'MP2RAGE_UNI_hdbet_GMrois.nii.gz')
    gm_wb = np.asarray(g.dataobj).astype(np.int16)[::DS, ::DS, ::DS]
    gm_aff = g.affine @ np.diag([DS, DS, DS, 1])
    f = nib.load(P + 'MP2RAGE_UNI_hdbet_fast3_seg.nii.gz')
    seg_wb = np.asarray(f.dataobj).astype(np.int16)[::DS, ::DS, ::DS]
    seg_aff = f.affine @ np.diag([DS, DS, DS, 1])
    sl_gm = ld('CESTslab_GMrois.nii').astype(int)
    sl_seg = ld('CESTslab_seg.nii').astype(int)
    v = D.vol(OV.SUBJ)
    to_nii = lambda a: np.flip(np.transpose(a, (1, 0, 2)), axis=1)
    cest = to_nii(v['CESTmap'].astype(float))
    roi = to_nii(v['roi'] > 0)
    valid = np.isfinite(cest) & (cest != 0) & (cest > -10) & (cest < 30) & roi
    gm_slab = (sl_gm > 0) & (sl_seg == 2) & roi
    wm_slab = d['regions']['slab']
    assert ((d['sl'] > 0) & (sl_seg == 3)).sum() > 10 * ((d['sl'] > 0) & (sl_seg == 2)).sum(), 'seg coding'
    return dict(d=d, gm_wb=gm_wb, gm_aff=gm_aff, seg_wb=seg_wb, seg_aff=seg_aff, sl_gm=np.where(gm_slab, sl_gm, 0),
                gm_slab=gm_slab, wm_slab=wm_slab, cest=cest, valid=valid, saff=d['saff'], sshape=d['sshape'])


def slab_mesh(X, mask, labels, glu_region=None):
    mi, ai = OV.iso(mask.astype(np.uint8), X['saff'])
    li, _ = OV.iso(np.where(mask, labels, 0), X['saff'])
    m = WB.surf_world(mi > 0, ai, sigma=0.9)
    m.point_data['lab'] = WB.label_of_points(m.points, li, ai)
    region = mask if glu_region is None else glu_region
    m.point_data['glu'] = WB.cest_of_points(m.points, OV.region_cest(X['cest'], X['valid'] & region, region),
                                            X['saff'], X['sshape'])
    return m


def build():
    X = load()
    M = {}
    # whole-brain GM atlas (HO), coloured by lobe
    M['wb_gm'] = WB.surf_world(X['gm_wb'] > 0, X['gm_aff'], sigma=0.9)
    M['wb_gm'].point_data['lab'] = WB.label_of_points(M['wb_gm'].points, X['gm_wb'], X['gm_aff'])
    # whole-brain tissue segmentation (FAST: 2 = GM, 3 = WM)
    M['wb_seg_gm'] = WB.surf_world(X['seg_wb'] == 2, X['seg_aff'], sigma=0.9)
    M['wb_seg_wm'] = WB.surf_world(X['seg_wb'] == 3, X['seg_aff'], sigma=0.9)
    # clean whole-brain hull: GM + WM, closed and hole-filled, heavily smoothed (no skull-strip ragged edges)
    hull = ndi.binary_fill_holes(ndi.binary_closing(X['seg_wb'] >= 2, iterations=1))
    M['wb_hull'] = WB.surf_world(hull, X['seg_aff'], sigma=1.3)
    # whole-brain JHU WM with labels, and GluCEST inside the slab
    d = X['d']
    M['wb_wm'] = WB.surf_world(d['lab'] > 0, d['aff'], sigma=0.9)
    M['wb_wm'].point_data['lab'] = WB.label_of_points(M['wb_wm'].points, d['lab'], d['aff'])
    M['wb_wm'].point_data['glu'] = WB.cest_of_points(M['wb_wm'].points, d['cest_all'], d['saff'], d['sshape'])
    # slab GM layers
    sg = X['sl_gm']
    grp = np.vectorize(group_of)(np.maximum(sg, 1))
    M['gm_slab'] = slab_mesh(X, X['gm_slab'], sg)
    for g in ('frontal', 'cingulate', 'parietal', 'occipital'):        # occipital has its own layer (was merged)
        M[f'gm_{g}'] = slab_mesh(X, X['gm_slab'] & (grp == g), sg)
    # slab WM (JHU) and slab GM + WM tissue
    M['wm_slab'] = slab_mesh(X, X['wm_slab'], d['sl'])
    for k, vals in (('wm_cc', [3, 4, 5]), ('wm_cing', [35, 36]), ('wm_cr', [23, 24, 25, 26, 27, 28])):
        M[k] = slab_mesh(X, X['wm_slab'] & np.isin(d['sl'], vals), d['sl'])
    tissue = X['gm_slab'] | X['wm_slab']
    M['tissue_slab'] = slab_mesh(X, tissue, np.where(X['gm_slab'], 1, 0))
    return M


# ── generic renderers ──────────────────────────────────────────────────────────────────────────────
def stack(Mm, layers, mode, name, kind, painter, lift=95.0, plates=True):   # GM slab is wide: more spacing
    """Exploded orthographic stack. layers: [(mesh key, label key)] bottom -> top."""
    p = AB.plotter(AB.STACK_WIN, mode)
    pc = AB.MODES[mode]['plate']
    sb = Mm[layers[0][0]].bounds
    centres = {}
    for i, (k, lk) in enumerate(layers):
        z = i * lift
        m = Mm[k].translate((0, 0, z), inplace=False)
        centres[lk] = np.asarray(m.center)
        if kind == 'seg':
            p.add_mesh(painter(m), scalars='rgb', rgb=True, **H.MESH)
        else:
            p.add_mesh(m, scalars='glu', cmap=TH['glu'], clim=(R.VMIN, R.VMAX), show_scalar_bar=False, **H.MESH)
        if not plates:
            continue
        zb = Mm[k].bounds[4] + z
        plate = pv.Plane(center=((sb[0] + sb[1]) / 2, (sb[2] + sb[3]) / 2, zb - 1.5),
                         i_size=(sb[1] - sb[0]) * 1.05, j_size=(sb[3] - sb[2]) * 1.04)
        p.add_mesh(plate, color=pc, opacity=0.07 if mode == 'dark' else 0.10, lighting=False)
        p.add_mesh(plate.extract_feature_edges(boundary_edges=True), color=pc, opacity=0.5, line_width=1.5)
    c = np.asarray(Mm[layers[0][0]].center) + (0, 0, (len(layers) - 1) * lift / 2)
    H.cam(p, c, AB.STACK_DIR, 560)
    p.enable_parallel_projection()
    p.reset_camera()
    p.render()
    rows = {}
    for lk, w in centres.items():
        p.renderer.SetWorldPoint(*w, 1.0)
        p.renderer.WorldToDisplay()
        rows[lk] = AB.STACK_WIN[1] - p.renderer.GetDisplayPoint()[1]
    (H.ASSETS / f'hero_{name}_rows.json').write_text(json.dumps(rows))
    return H.shot(p, name), rows


def glass(p, mesh, mode, painter=None, color=None, op=(0.16, 0.22)):
    o = op[0] if mode == 'dark' else op[1]
    if painter is not None:
        p.add_mesh(painter(mesh), scalars='rgb', rgb=True, opacity=o, **H.MESH)
    else:
        p.add_mesh(mesh, color=color, opacity=o, **H.MESH)


GREY_GLASS = {'dark': ('#9A9A9A', 0.14), 'white': ('#7A7A7A', 0.18)}


def grey_glass(p, mesh, mode):
    """Single-colour, low-opacity whole-brain reference (no lobe colours)."""
    col, op = GREY_GLASS[mode]
    p.add_mesh(mesh, color=col, opacity=op, **H.MESH)


def scene(mode, win, centre, d, dist, adds, up=(0, 0, 1)):
    p = AB.plotter(win, mode)
    for f in adds:
        f(p)
    H.cam(p, centre, d, dist, up=up)
    return p


def glu_kw(nan='#5E5E5E'):
    return dict(scalars='glu', cmap=TH['glu'], clim=(R.VMIN, R.VMAX), show_scalar_bar=False, nan_color=nan, **H.MESH)


# ── GM hero figure ─────────────────────────────────────────────────────────────────────────────────
GM_LAYERS = [('gm_slab', 'slab'), ('gm_frontal', 'frontal'), ('gm_cingulate', 'cingulate'),
             ('gm_parietal', 'parietal'), ('gm_occipital', 'occipital')]
GM_NAMES = {'slab': '24 mm GM 3D slab', 'frontal': 'frontal', 'cingulate': 'cingulate', 'parietal': 'parietal',
            'occipital': 'occipital'}
GM_LABEL_COL = {'slab': None, 'frontal': GM_COL['frontal'], 'cingulate': GM_COL['cingulate'],
                'parietal': GM_COL['parietal'], 'occipital': GM_COL['occipital']}


def gm_figure(M, mode, recompose=False):
    ld = lambda n: np.asarray(Image.open(H.ASSETS / f'hero_{n}.png').convert('RGB'))
    nm = lambda s: f'gm_{s}_{mode}'
    if recompose:
        A, Bs, Bg = ld(nm('a')), ld(nm('bseg')), ld(nm('bglu'))
        rows = json.loads((H.ASSETS / f'hero_{nm("bseg")}_rows.json').read_text())
        Cs = [ld(nm(f'c{v[0]}')) for v in AB.VIEWS]
    else:
        c0 = np.asarray(M['wb_gm'].center)
        A = H.shot(scene(mode, (2200, 1700), c0, (0.80, 0.55, 0.42), 520,
                         [lambda p: grey_glass(p, M['wb_gm'], mode),
                          lambda p: p.add_mesh(M['gm_slab'], **glu_kw())]), nm('a'))
        Bs, rows = stack(M, GM_LAYERS, mode, nm('bseg'), 'seg', paint_groups)
        Bg, _ = stack(M, GM_LAYERS, mode, nm('bglu'), 'glu', paint_groups)
        Cs = []
        for name, d, up in AB.VIEWS:
            Cs.append(H.shot(scene(mode, (1600, 1600), c0, d, 500,
                                   [lambda p: grey_glass(p, M['wb_gm'], mode),
                                    lambda p: p.add_mesh(M['gm_slab'], **glu_kw())], up=up), nm(f'c{name}')))
    AB.finish(TAG, TH, mode, A, Bs, Bg, rows, Cs, names=GM_NAMES, colours=GM_LABEL_COL, stem='Fig_hero_GM_')


# ── WM figure enclosed in the grey whole-brain reference ────────────────────────────────────────────
WM_LAYERS = [('wm_slab', 'slab'), ('wm_cc', 'cconly'), ('wm_cing', 'cing'), ('wm_cr', 'cr')]


def wm_enclosed_figure(M, mode, recompose=False):
    ld = lambda n: np.asarray(Image.open(H.ASSETS / f'hero_{n}.png').convert('RGB'))
    nm = lambda s: f'wmenc_{s}_{mode}'
    wmpaint = lambda m: V.paint(m, TH['seg'])

    def adds():
        return [lambda p: grey_glass(p, M['wb_gm'], mode),
                lambda p: p.add_mesh(wmpaint(M['wb_wm']), scalars='rgb', rgb=True,
                                     opacity=0.30 if mode == 'dark' else 0.38, **H.MESH),
                lambda p: p.add_mesh(M['wm_slab'], **glu_kw())]
    if recompose:
        A, Bs, Bg = ld(nm('a')), ld(nm('bseg')), ld(nm('bglu'))
        rows = json.loads((H.ASSETS / f'hero_{nm("bseg")}_rows.json').read_text())
        Cs = [ld(nm(f'c{v[0]}')) for v in AB.VIEWS]
    else:
        c0 = np.asarray(M['wb_gm'].center)
        A = H.shot(scene(mode, (2200, 1700), c0, (0.80, 0.55, 0.42), 520, adds()), nm('a'))
        Bs, rows = stack(M, WM_LAYERS, mode, nm('bseg'), 'seg', wmpaint, lift=52.0)
        Bg, _ = stack(M, WM_LAYERS, mode, nm('bglu'), 'glu', wmpaint, lift=52.0)
        Cs = [H.shot(scene(mode, (1600, 1600), c0, d, 500, adds(), up=up), nm(f'c{name}'))
              for name, d, up in AB.VIEWS]
    AB.finish(TAG, TH, mode, A, Bs, Bg, rows, Cs, stem='Fig_hero_WM_enclosed_')


# ── WM overview: whole brain -> WM ROIs -> 24 mm slab (top), then the original a and b ─────────────
def wm_overview_figure(M, mode, recompose=False):
    """Top row (a): whole brain -> WM ROIs -> 24 mm WM 3D slab, one camera and one scale.
    Bottom row: (b) JHU atlas glass + slab GluCEST, (c) the exploded WM subregion stack."""
    ld = lambda n: np.asarray(Image.open(H.ASSETS / f'hero_{n}.png').convert('RGB'))
    nm = lambda s: f'wmov_{s}_{mode}'
    wmpaint = lambda m: V.paint(m, TH['seg'])
    if recompose:
        P_ = [ld(nm(f'p{i}')) for i in range(3)]
        A = ld(nm('a'))
        Bs, Bg = ld(nm('bseg')), ld(nm('bglu'))
        rows = json.loads((H.ASSETS / f'hero_{nm("bseg")}_rows.json').read_text())
    else:
        c0 = np.asarray(M['wb_seg_gm'].center)
        view = (0.80, 0.55, 0.42)
        brain_col = '#C9CED6' if mode == 'dark' else '#8C939C'      # cool grey glass
        P_ = [H.shot(scene(mode, (2000, 1700), c0, view, 520,
                           [lambda p: grey_glass(p, M['wb_gm'], mode),   # as in the enclosed-WM figure, panel a
                            lambda p: p.add_mesh(wmpaint(M['wb_wm']), scalars='rgb', rgb=True,
                                                 opacity=0.30 if mode == 'dark' else 0.38, **H.MESH),
                            lambda p: p.add_mesh(M['wm_slab'], **glu_kw())]), nm('p0')),
              H.shot(scene(mode, (2000, 1700), c0, view, 520,
                           [lambda p: p.add_mesh(wmpaint(M['wb_wm']), scalars='rgb', rgb=True, **H.MESH)]), nm('p1')),
              H.shot(scene(mode, (2000, 1700), c0, view, 520,
                           [lambda p: p.add_mesh(wmpaint(M['wm_slab']), scalars='rgb', rgb=True, **H.MESH)]), nm('p2'))]
        A = H.shot(scene(mode, (2200, 1700), np.asarray(M['wb_wm'].center), view, 520,
                         [lambda p: p.add_mesh(wmpaint(M['wb_wm']), scalars='rgb', rgb=True,
                                               opacity=0.16 if mode == 'dark' else 0.22, **H.MESH),
                          lambda p: p.add_mesh(M['wm_slab'], **glu_kw())]), nm('a'))
        # stacks without the grey base plates
        Bs, rows = stack(M, WM_LAYERS, mode, nm('bseg'), 'seg', wmpaint, lift=52.0, plates=False)
        Bg, _ = stack(M, WM_LAYERS, mode, nm('bglu'), 'glu', wmpaint, lift=52.0, plates=False)
    compose_overview(mode, P_, A, Bs, Bg, rows)


def gm_overview_figure(M, mode, recompose=False):
    """GM counterpart of the WM overview: (a) whole brain -> GM ROIs (Harvard-Oxford) -> 24 mm GM 3D slab,
    (b) grey cortical-atlas glass + slab GM GluCEST, (c) GM lobe stack without base plates."""
    ld = lambda n: np.asarray(Image.open(H.ASSETS / f'hero_{n}.png').convert('RGB'))
    nm = lambda s: f'gmov_{s}_{mode}'
    if recompose:
        P_ = [ld(nm(f'p{i}')) for i in range(3)]
        A, Bs, Bg = ld(nm('a')), ld(nm('bseg')), ld(nm('bglu'))
        rows = json.loads((H.ASSETS / f'hero_{nm("bseg")}_rows.json').read_text())
    else:
        c0 = np.asarray(M['wb_seg_gm'].center)
        view = (0.80, 0.55, 0.42)
        brain_col = '#C9CED6' if mode == 'dark' else '#8C939C'
        P_ = [H.shot(scene(mode, (2000, 1700), c0, view, 520,
                           [lambda p: grey_glass(p, M['wb_gm'], mode),   # grey brain as in b, slab GM as ROIs
                            lambda p: p.add_mesh(paint_groups(M['gm_slab']), scalars='rgb', rgb=True, **H.MESH)]),
                     nm('p0')),
              H.shot(scene(mode, (2000, 1700), c0, view, 520,
                           [lambda p: p.add_mesh(paint_groups(M['wb_gm']), scalars='rgb', rgb=True, **H.MESH)]),
                     nm('p1')),
              H.shot(scene(mode, (2000, 1700), c0, view, 520,
                           [lambda p: p.add_mesh(paint_groups(M['gm_slab']), scalars='rgb', rgb=True, **H.MESH)]),
                     nm('p2'))]
        A = H.shot(scene(mode, (2200, 1700), np.asarray(M['wb_gm'].center), view, 520,
                         [lambda p: grey_glass(p, M['wb_gm'], mode),
                          lambda p: p.add_mesh(M['gm_slab'], **glu_kw())]), nm('a'))
        Bs, rows = stack(M, GM_LAYERS, mode, nm('bseg'), 'seg', paint_groups, plates=False)
        Bg, _ = stack(M, GM_LAYERS, mode, nm('bglu'), 'glu', paint_groups, plates=False)
    compose_overview(mode, P_, A, Bs, Bg, rows,
                     titles=('Whole brain', 'Harvard-Oxford GM ROIs', '24 mm GM 3D slab'),
                     names=GM_NAMES, colours=GM_LABEL_COL, stem='Fig_hero_GM_overview_')


def compose_overview(mode, P_, A, Bs, Bg, rows, titles=('Whole brain', 'JHU atlas WM ROIs', '24 mm WM 3D slab'),
                     names=None, colours=None, stem='Fig_hero_WM_overview_', letters=True):
    # letters=False: no panel letters, their positions go to <stem>..._letters.json (for embedding in a bigger figure)
    md = AB.MODES[mode]
    fg, bg = md['fg'], md['bg']
    seg = TH['seg']
    # crops: one window for the three progression images (same scale), one for both stacks
    both = np.maximum.reduce(P_) if mode == 'dark' else np.minimum.reduce(P_)
    w = AB.crop(both, mode)
    P_ = [AB.cut(im, w) for im in P_]
    A = AB.cut(A, AB.crop(A, mode))
    bb = AB.crop(np.maximum(Bs, Bg) if mode == 'dark' else np.minimum(Bs, Bg), mode)
    frac = {k: (r - bb[0]) / (bb[1] - bb[0]) for k, r in rows.items()}
    Bs, Bg = AB.cut(Bs, bb), AB.cut(Bg, bb)

    W = S.DOUBLE
    arrow = 10
    pw = (W - 4 - 2 * arrow) / 3
    ph = pw * P_[0].shape[0] / P_[0].shape[1]
    top = 7.5
    row1 = top + 6 + ph                                      # titles above the images
    gap_ab, gap_b, label_col = 5, 1.0, 29
    ra, rb = A.shape[1] / A.shape[0], Bs.shape[0] / Bs.shape[1]
    K = W - 2 - gap_ab - gap_b - label_col
    aw = (rb * K / 2) / (1 / ra + rb / 2)
    ah = aw / ra
    sw = (K - aw) / 2
    sh = sw * Bs.shape[0] / Bs.shape[1]
    y2 = row1 + 8
    body = max(ah, sh + 3)
    FH = y2 + body + 13
    F = S.MMFigure(W, FH)
    F.fig.patch.set_facecolor(bg)
    lab = dict(color=fg, fontweight='bold', fontsize=AB.LET, va='top', ha='left')

    def put(img, x, y, w_, h_):
        ax = F.ax(x, y, w_, h_)
        ax.set_axis_off()
        ax.imshow(img)

    # a: progression
    pos = {'a': (1, 1.2)}
    F.text(1, 1.2, 'a', **lab) if letters else None
    for i, (img, title) in enumerate(zip(P_, titles)):
        x = 2 + i * (pw + arrow)
        F.text(x + pw / 2, top + 2.5, title, color=fg, fontsize=AB.LBL + 0.5, fontweight='bold')
        put(img, x, top + 6, pw, ph)
        if i < 2:
            F.text(x + pw + arrow / 2, top + 6 + ph / 2, '→', color=fg, fontsize=18, fontweight='bold')
    # b: atlas glass + slab GluCEST (unchanged), colour bar underneath
    ya = y2 + (body - ah) / 2
    pos['b'] = (1, y2 - 6.5)
    F.text(1, y2 - 6.5, 'b', **lab) if letters else None
    put(A, 2, ya, aw, ah)
    ycb = y2 + body + 2.5
    xcb = max(2 + aw / 2 - 4, 37)
    cax = F.ax(xcb, ycb, 36, 2.2)
    sm = S.mpl.cm.ScalarMappable(cmap=TH['glu'], norm=S.mpl.colors.Normalize(R.VMIN, R.VMAX))
    cb = F.fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=[0, 4, 8, 12])
    cb.outline.set_edgecolor(fg)
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(colors=fg, labelcolor=fg, width=0.6, length=2, labelsize=AB.TCK)
    F.text(xcb - 2, ycb + 1.1, 'GluCEST Contrast (%)', color=fg, fontsize=AB.LBL, fontweight='bold', ha='right')
    # c: WM subregion stack (unchanged)
    bx = 2 + aw + gap_ab
    yb = y2 + (body - sh - 3) / 2
    pos['c'] = (bx - 1, y2 - 6.5)
    F.text(bx - 1, y2 - 6.5, 'c', **lab) if letters else None
    for i, (img, sub) in enumerate([(Bs, 'ROIs'), (Bg, 'GluCEST')]):
        x = bx + i * (sw + gap_b)
        F.text(x + sw / 2, yb + 1.5, sub, color=fg, fontsize=AB.LBL + 0.5, fontweight='bold')
        put(img, x, yb + 3, sw, sh)
    names = names or AB.LAYER_NAME
    colour = colours or {'slab': None, 'cconly': seg['body'], 'cing': seg['cing'], 'cr': seg['scr']}
    colour = {k: (fg if c is None else c) for k, c in colour.items()}
    if mode == 'white':
        colour = {k: (c if c == fg else tuple(0.62 * np.array(R.hex2rgb(c)))) for k, c in colour.items()}
    for k, fr in frac.items():
        F.text(W - 1, yb + 3 + sh * fr, names[k], color=colour[k], fontsize=AB.LBL, ha='right',
               fontweight='bold')
    tail = '' if letters else '_noletters'
    if not letters:
        (S.OUT / f'{stem}{TAG}{md["suffix"]}{tail}_letters.json').write_text(json.dumps(dict(W=W, H=FH, pos=pos)))
    for suf in ('.pdf', '.png'):
        f = S.OUT / f'{stem}{TAG}{md["suffix"]}{tail}{suf}'
        f.unlink(missing_ok=True)
        F.fig.savefig(f, facecolor=bg, dpi=600 if suf == '.png' else None)
    S.plt.close(F.fig)


def wm_overview_noletters(mode='white'):
    # Figure 1 (WM overview) without panel letters, from the saved renders, for stacking on top of other panels
    ld = lambda n: np.asarray(Image.open(H.ASSETS / f'hero_{n}.png').convert('RGB'))
    nm = lambda s: f'wmov_{s}_{mode}'
    rows = json.loads((H.ASSETS / f'hero_{nm("bseg")}_rows.json').read_text())
    compose_overview(mode, [ld(nm(f'p{i}')) for i in range(3)], ld(nm('a')), ld(nm('bseg')), ld(nm('bglu')), rows,
                     letters=False)


# ── combined tissue-hierarchy figure ────────────────────────────────────────────────────────────────
def combined_figure(M, mode, recompose=False):
    ld = lambda n: np.asarray(Image.open(H.ASSETS / f'hero_{n}.png').convert('RGB'))
    nm = lambda s: f'comb_{s}_{mode}'
    keys = ['wb_seg', 'wm_seg', 'slab_seg', 'wb_glu', 'wm_glu', 'slab_glu']
    if recompose:
        I = {k: ld(nm(k)) for k in keys}
    else:
        c0 = np.asarray(M['wb_seg_wm'].center)
        view = (0.80, 0.55, 0.42)
        wmpaint = lambda m: V.paint(m, TH['seg'])
        I = {
            'wb_seg': H.shot(scene(mode, (2000, 1700), c0, view, 520,
                                   [lambda p: glass(p, M['wb_seg_gm'], mode, color=TISSUE['GM'], op=(0.12, 0.18)),
                                    lambda p: p.add_mesh(M['wb_seg_wm'], color=TISSUE['WM'], **H.MESH)]),
                             nm('wb_seg')),
            'wm_seg': H.shot(scene(mode, (2000, 1700), c0, view, 520,
                                   [lambda p: p.add_mesh(wmpaint(M['wb_wm']), scalars='rgb', rgb=True, **H.MESH)]),
                             nm('wm_seg')),
            'wb_glu': H.shot(scene(mode, (2000, 1700), c0, view, 520,
                                   [lambda p: glass(p, M['wb_seg_gm'], mode, color=TISSUE['GM'], op=(0.18, 0.26)),
                                    lambda p: p.add_mesh(M['tissue_slab'], **glu_kw())]), nm('wb_glu')),
            'wm_glu': H.shot(scene(mode, (2000, 1700), c0, view, 520,
                                   [lambda p: p.add_mesh(M['wb_wm'], **glu_kw())]), nm('wm_glu')),
        }
        two = [('wm_slab', 'WM'), ('gm_slab', 'GM')]
        tissue_paint = {'wm_slab': lambda m: _solid(m, TISSUE['WM']), 'gm_slab': lambda m: _solid(m, TISSUE['GM'])}
        I['slab_seg'], _ = _two_stack(M, two, mode, nm('slab_seg'), 'seg', tissue_paint)
        I['slab_glu'], _ = _two_stack(M, two, mode, nm('slab_glu'), 'glu', tissue_paint)
    compose_combined(mode, I)


def _solid(m, col):
    m = m.copy()
    m.point_data['rgb'] = np.tile((np.array(R.hex2rgb(col)) * 255).astype(np.uint8), (m.n_points, 1))
    return m


def _two_stack(M, layers, mode, name, kind, paints, lift=48.0):
    p = AB.plotter((1600, 2000), mode)
    pc = AB.MODES[mode]['plate']
    sb = M['wm_slab'].bounds
    for i, (k, _) in enumerate(layers):
        z = i * lift
        m = M[k].translate((0, 0, z), inplace=False)
        if kind == 'seg':
            p.add_mesh(paints[k](m), scalars='rgb', rgb=True, **H.MESH)
        else:
            p.add_mesh(m, **glu_kw())
        zb = M[k].bounds[4] + z
        plate = pv.Plane(center=((sb[0] + sb[1]) / 2, (sb[2] + sb[3]) / 2, zb - 1.5),
                         i_size=(sb[1] - sb[0]) * 1.35, j_size=(sb[3] - sb[2]) * 1.15)
        p.add_mesh(plate, color=pc, opacity=0.07 if mode == 'dark' else 0.10, lighting=False)
        p.add_mesh(plate.extract_feature_edges(boundary_edges=True), color=pc, opacity=0.5, line_width=1.5)
    H.cam(p, np.asarray(M['wm_slab'].center) + (0, 0, lift / 2), AB.STACK_DIR, 560)
    p.enable_parallel_projection()
    p.reset_camera()
    return H.shot(p, name), None


def compose_combined(mode, I):
    md = AB.MODES[mode]
    fg, bg = md['fg'], md['bg']
    for k in I:
        I[k] = AB.cut(I[k], AB.crop(I[k], mode))
    W = S.DOUBLE
    left, arrow_w, gap_r = 14, 9, 4
    cw = (W - left - 2 - 2 * arrow_w) / 3
    rh = [max(cw * I[k].shape[0] / I[k].shape[1] for k in ks) for ks in (('wb_seg', 'wm_seg', 'slab_seg'),
                                                                           ('wb_glu', 'wm_glu', 'slab_glu'))]
    top = 15
    FH = top + rh[0] + gap_r + rh[1] + 14
    F = S.MMFigure(W, FH)
    F.fig.patch.set_facecolor(bg)
    lab = dict(color=fg, fontweight='bold', fontsize=AB.LET, va='top', ha='left')
    cols = [('a', 'Whole brain', 'wb'), ('b', 'White matter (JHU atlas)', 'wm'), ('c', '24 mm 3D slab: GM + WM', 'slab')]
    for j, (letter, title, key) in enumerate(cols):
        x = left + j * (cw + arrow_w)
        F.text(x - 1, 1.2, letter, **lab)
        F.text(x + cw / 2, 9.5, title, color=fg, fontsize=AB.LBL + 1, fontweight='bold')
        y = top
        for r, suf in enumerate(('seg', 'glu')):
            im = I[f'{key}_{suf}']
            h = cw * im.shape[0] / im.shape[1]
            ax = F.ax(x, y + (rh[r] - h) / 2, cw, h)
            ax.set_axis_off()
            ax.imshow(im)
            if j < 2:
                F.text(x + cw + arrow_w / 2, y + rh[r] / 2, '\u2192', color=fg, fontsize=18, fontweight='bold')
            y += rh[r] + gap_r
    for r, txt in enumerate(('Segmentation', 'GluCEST')):
        yy = top + (rh[0] + gap_r if r else 0) + rh[r] / 2
        F.text(5, yy, txt, color=fg, fontsize=AB.LBL + 0.5, fontweight='bold', rotation=90)
    # keys: tissue + slab layer names, and the colour bar
    yk = top + rh[0] + gap_r + rh[1] + 3
    for i, (nm_, col) in enumerate([('GM', TISSUE['GM']), ('WM', TISSUE['WM'])]):
        xx = left + i * 14
        F.fig.add_artist(S.mpl.patches.Rectangle((xx / F.W, 1 - (yk + 1.6) / F.H), 2.6 / F.W, 2.6 / F.H,
                                                 transform=F.fig.transFigure, color=col, lw=0))
        F.text(xx + 3.6, yk + 0.3, nm_, color=fg, fontsize=AB.LBL, fontweight='bold', ha='left')
    cax = F.ax(W / 2 + 4, yk, 40, 2.2)
    sm = S.mpl.cm.ScalarMappable(cmap=TH['glu'], norm=S.mpl.colors.Normalize(R.VMIN, R.VMAX))
    cb = F.fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=[0, 4, 8, 12])
    cb.outline.set_edgecolor(fg)
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(colors=fg, labelcolor=fg, width=0.6, length=2, labelsize=AB.TCK)
    F.text(W / 2 + 2, yk + 1.1, 'GluCEST Contrast (%)', color=fg, fontsize=AB.LBL, fontweight='bold', ha='right')
    for suf in ('.pdf', '.png'):
        f = S.OUT / f'Fig_hero_combined_{TAG}{md["suffix"]}{suf}'
        f.unlink(missing_ok=True)
        F.fig.savefig(f, facecolor=bg, dpi=600 if suf == '.png' else None)
    S.plt.close(F.fig)


def make(which=('wmov', 'gmov'), modes=('white',), recompose=False):
    # wmov = Figure 1, gmov = GM overview (supplementary); recompose=True uses the saved renders (no drive)
    S.apply()
    M = None if recompose else build()
    fn = {'gm': gm_figure, 'gmov': gm_overview_figure, 'wmov': wm_overview_figure, 'wmenc': wm_enclosed_figure}
    for mode in modes:
        for w in which:
            fn[w](M, mode, recompose)
            print(w, mode, 'done', flush=True)


if __name__ == '__main__':
    S.apply()
    which = [a for a in sys.argv[1:] if a in ('gm', 'combined', 'wmenc', 'wmov', 'gmov')] or ['gm', 'combined', 'wmenc', 'wmov', 'gmov']
    rec = '--recompose' in sys.argv
    M = None if rec else build()
    print('meshes ready', flush=True)
    for mode in AB.MODES:
        if 'gm' in which:
            gm_figure(M, mode, rec)
            print('GM figure', mode, flush=True)
        if 'gmov' in which:
            gm_overview_figure(M, mode, rec)
            print('GM overview figure', mode, flush=True)
        if 'wmov' in which:
            wm_overview_figure(M, mode, rec)
            print('WM overview figure', mode, flush=True)
        if 'wmenc' in which:
            wm_enclosed_figure(M, mode, rec)
            print('WM enclosed figure', mode, flush=True)
        if 'combined' in which:
            combined_figure(M, mode, rec)
            print('combined figure', mode, flush=True)
