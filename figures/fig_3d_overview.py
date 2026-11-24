"""Nature Communications 3D overview figure + transparent assets: whole-brain WM atlas -> GluCEST slab ->
slab WM ROIs -> CC + cingulum / corona radiata, each with GluCEST.

Row 1 (atlas):   a whole-brain JHU WM, lateral, with the GluCEST slab outlined
                 b whole-brain JHU WM, superior, slab outlined
                 c slab WM ROIs   d CC + cingulum   e corona radiata          (superior, slab scale)
Row 2 (GluCEST): f/g whole-brain WM painted with GluCEST inside the slab (grey outside)
                 h slab WM ROIs   i CC + cingulum   j corona radiata          (0-12 %, inferno)

One participant (P04, 26 y: the only whole-brain native atlas on the data drive), one scanner (RAS) space,
so whole-brain and slab renders are geometrically consistent. Cache -> slab NIfTI orientation is
asserted (exact label match). No translucent brain overlay.

Outputs: Fig_3D_overview_KJ.pdf/.png (183 mm) and assets_3d/*.png (transparent, ~2000 px, for Illustrator).
Run:  python fig_3d_overview.py [cmap]
"""
from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv
from scipy import ndimage as ndi

import figdata as D
import nature_style as S
import render3d as R
import fig_3d_wholebrain as WB

CMAP = sys.argv[1] if len(sys.argv) > 1 else 'inferno_lift'
SUBJ, FOLDER = WB.SUBJ, WB.FOLDER
ASSETS = S.OUT / 'assets_3d'
ASSETS.mkdir(exist_ok=True)
C = S.SUB_COL
CC_CING, CR = R.CC_CING, R.CR
SLAB_EDGE = '#C9D3DD'   # light edges on black
WIN = (2000, 1600)

VIEWS = {'lateral': (np.array([0.90, 0.30, 0.38]), (0, 0, 1)),     # from the right, anterior to the right
         'superior': (np.array([0.0, -0.25, 1.0]), (0, 1, 0))}      # top-down, anterior at the top


def iso(lab, aff):
    """Repeat voxels along the 2 mm slice axis so slab surfaces are not stair-stepped."""
    z = np.sqrt((aff[:3, :3] ** 2).sum(0))
    f = np.maximum(np.round(z / z.min()).astype(int), 1)
    for ax, k in enumerate(f):
        if k > 1:
            lab = np.repeat(lab, k, axis=ax)
    a2 = aff @ np.diag([1 / f[0], 1 / f[1], 1 / f[2], 1])
    a2[:3, 3] -= a2[:3, :3] @ ((f - 1) / 2)
    return lab, a2


def region_cest(cest, ok, region):
    """GluCEST from this region's voxels only, extended to neighbours, light smoothing (slab grid)."""
    f = R.fill_nearest(cest, ok & region)
    return ndi.gaussian_filter(f, (0.7, 0.7, 0.35))


def load():
    lab, aff, bm, baff, cest_all, saff, sshape = WB.load()
    sl = np.asarray(nib.load(FOLDER + 'CESTslab_WMrois.nii').dataobj).astype(int)
    smask = np.asarray(nib.load(FOLDER + 'CESTslab_mask.nii').dataobj) > 0
    v = D.vol(SUBJ)
    to_nii = lambda a: np.flip(np.transpose(a, (1, 0, 2)), axis=1)
    cest = to_nii(v['CESTmap'].astype(float))
    roi = to_nii(v['roi'] > 0)
    sl = np.where(roi, sl, 0)
    ok = (sl > 0) & np.isfinite(cest) & (cest != 0) & (cest > -10) & (cest < 30)
    regions = {'slab': sl > 0, 'cc': np.isin(sl, CC_CING), 'cr': np.isin(sl, CR)}
    cests = {k: region_cest(cest, ok, m) for k, m in regions.items()}
    return dict(lab=lab, aff=aff, cest_all=cest_all, saff=saff, sshape=sshape, sl=sl,
                smask=smask | roi, regions=regions, cests=cests)


def colour_by_label(mesh, lab, aff):
    l = WB.label_of_points(mesh.points, lab, aff)
    rgb = np.array([R.hex2rgb(WB.FAMILY.get(int(x), ('', WB.OUT_SLAB))[1]) for x in l])
    mesh.point_data['rgb'] = (rgb * 255).astype(np.uint8)
    return mesh


def slab_box(d):
    """Slab extent (all 12 slices), cropped in-plane to the slab white matter + 6 mm, as a world-space
    box (faces) and its 12 edges (lines)."""
    ij = np.argwhere(d['regions']['slab'].any(2))
    (i0, j0), (i1, j1) = ij.min(0) - 6, ij.max(0) + 6
    k1 = d['sshape'][2]
    box = pv.Box(bounds=(i0 - 0.5, i1 + 0.5, j0 - 0.5, j1 + 0.5, -0.5, k1 - 0.5), quads=True)
    box = box.transform(d['saff'], inplace=False)
    return box, box.extract_feature_edges(feature_angle=30, boundary_edges=False, manifold_edges=False)


def render(meshes, centre, view, dist, name):
    dvec, up = VIEWS[view]
    dvec = dvec / np.linalg.norm(dvec)
    p = pv.Plotter(off_screen=True, window_size=WIN)
    p.set_background('black')
    p.enable_anti_aliasing('ssaa')
    for mesh, kw in meshes:
        p.add_mesh(mesh, **kw)
    p.camera_position = [tuple(centre + dist * dvec), tuple(centre), up]
    p.camera.view_angle = 22
    f = ASSETS / f'{name}.png'
    p.screenshot(str(f), transparent_background=True)
    p.close()
    from PIL import Image
    im = np.asarray(Image.open(f).convert('RGBA'))
    return im


def crop_rgba(imgs, pad=12):
    a = np.any([im[..., 3] > 8 for im in imgs], axis=0)
    rr, cc = np.where(a)
    h, w = a.shape
    b = max(rr.min() - pad, 0), min(rr.max() + pad, h), max(cc.min() - pad, 0), min(cc.max() + pad, w)
    return [im[b[0]:b[1], b[2]:b[3]] for im in imgs]


def make():
    d = load()
    lab_kw = dict(scalars='rgb', rgb=True, **R.MESH_KW)
    glu_kw = dict(scalars='glu', cmap=CMAP, clim=(R.VMIN, R.VMAX), nan_color=WB.OUT_SLAB,
                  show_scalar_bar=False, **R.MESH_KW)
    box_kw = dict(color='#9AA0A6', opacity=0.10, lighting=False)
    edge_kw = dict(color=SLAB_EDGE, line_width=2.5)

    # whole-brain WM (0.69 mm atlas, downsampled x2)
    wb = colour_by_label(WB.surf_world(d['lab'] > 0, d['aff']), d['lab'], d['aff'])
    wb_glu = wb.copy()
    wb_glu.point_data['glu'] = WB.cest_of_points(wb.points, d['cest_all'], d['saff'], d['sshape'])
    box, edges = slab_box(d)
    wb_c = np.asarray(wb.center)

    # slab ROIs (1 x 1 x 2 mm -> isotropic); colour each region only from its own labels
    slab_mesh = {}
    for k, m in d['regions'].items():
        mi, ai = iso(m.astype(np.uint8), d['saff'])
        li, _ = iso(np.where(m, d['sl'], 0), d['saff'])
        mesh = colour_by_label(WB.surf_world(mi > 0, ai, sigma=0.9), li, ai)
        mesh.point_data['glu'] = WB.cest_of_points(mesh.points, d['cests'][k], d['saff'], d['sshape'])
        slab_mesh[k] = mesh
    sl_c = np.asarray(slab_mesh['slab'].center)

    print('meshes built; rendering', flush=True)
    imgs = {
        'a': render([(wb, lab_kw), (box, box_kw), (edges, edge_kw)], wb_c, 'lateral', 560,
                    f'{SUBJ}_wholeWM_atlas_lateral'),
        'b': render([(wb, lab_kw), (box, box_kw), (edges, edge_kw)], wb_c, 'superior', 560,
                    f'{SUBJ}_wholeWM_atlas_superior'),
        'f': render([(wb_glu, glu_kw)], wb_c, 'lateral', 560, f'{SUBJ}_wholeWM_glucest_lateral_{CMAP}'),
        'g': render([(wb_glu, glu_kw)], wb_c, 'superior', 560, f'{SUBJ}_wholeWM_glucest_superior_{CMAP}'),
    }
    for (lt, gt), k, nm in [(('c', 'h'), 'slab', 'slabWM'), (('d', 'i'), 'cc', 'CCcing'), (('e', 'j'), 'cr', 'CR')]:
        imgs[lt] = render([(slab_mesh[k], lab_kw)], sl_c, 'superior', 330, f'{SUBJ}_{nm}_atlas_superior')
        imgs[gt] = render([(slab_mesh[k], glu_kw)], sl_c, 'superior', 330, f'{SUBJ}_{nm}_glucest_superior_{CMAP}')
    # common crops: whole-brain panels share one, slab panels share another
    wbk, slk = ['a', 'b', 'f', 'g'], ['c', 'd', 'e', 'h', 'i', 'j']
    imgs.update(zip(wbk, crop_rgba([imgs[k] for k in wbk])))
    imgs.update(zip(slk, crop_rgba([imgs[k] for k in slk])))

    # ── compose (183 mm, Nature Comms) ─────────────────────────────────────
    x0, g, t = 2.5, 1.6, 4.2
    big = 44.0
    small = (S.DOUBLE - x0 - 2 - 2 * big - 4 * g) / 3
    Hw, Ww = imgs['a'].shape[:2]
    Hs, Ws = imgs['c'].shape[:2]
    ph = max(big * Hw / Ww, small * Hs / Ws, big * imgs['b'].shape[0] / imgs['b'].shape[1])
    FH = 2 * (ph + t) + 3 + 13
    F = S.MMFigure(S.DOUBLE, FH)
    widths = [big, big, small, small, small]
    titles = [('a', 'Whole-brain WM atlas', S.FG), ('b', 'Whole-brain WM, superior', S.FG),
              ('c', 'Slab WM ROIs', S.FG), ('d', 'CC + cingulum', C['CC body']), ('e', 'Corona radiata', C['SCR']),
              ('f', 'GluCEST in slab', S.FG), ('g', 'GluCEST in slab, superior', S.FG),
              ('h', 'GluCEST: slab WM', S.FG), ('i', 'GluCEST: CC + cing.', C['CC body']),
              ('j', 'GluCEST: corona radiata', C['SCR'])]
    for r in range(2):
        y = 1 + r * (ph + t + 3)
        x = x0
        for c in range(5):
            letter, title, col = titles[r * 5 + c]
            w = widths[c]
            F.letter(letter, x - 2.2, y + 0.3)
            F.text(x + w / 2, y + 1.7, title, color=col, fontweight='bold', fontsize=6)
            ax = F.ax(x, y + t, w, ph)
            S.img_ax(ax)
            ax.imshow(imgs[letter])
            x += w + g
    # divider between whole-brain and slab-scale panels
    xd = x0 + 2 * big + 1.5 * g
    F.fig.add_artist(S.mpl.lines.Line2D([xd / F.W] * 2, [1 - 2 / F.H, 1 - (2 * (ph + t) + 4) / F.H],
                                        transform=F.fig.transFigure, color='#404040', lw=0.5))
    # legend row
    yb = 2 * (ph + t) + 3 + 3
    key = [('CC genu', C['CC genu']), ('CC body', C['CC body']), ('CC splenium', C['CC splenium']),
           ('Cingulum', C['Cingulum']), ('Ant. CR', C['ACR']), ('Sup. CR', C['SCR']), ('Post. CR', C['PCR']),
           ('Int. capsule', '#3A8FC6'), ('Assoc./other', '#D9C38A'), ('Brainstem/cereb.', '#B9A7C9')]
    for i, (nm, col) in enumerate(key):
        xx = x0 + (i % 5) * 19.5
        yy = yb + (i // 5) * 3.2
        F.fig.add_artist(S.mpl.patches.Rectangle((xx / F.W, 1 - (yy + 0.9) / F.H), 1.8 / F.W, 1.8 / F.H,
                                                 transform=F.fig.transFigure, color=col, lw=0))
        F.text(xx + 2.5, yy, nm, ha='left', fontsize=5.2)
    xs = x0 + 5 * 19.5 + 2
    F.fig.add_artist(S.mpl.patches.Rectangle((xs / F.W, 1 - (yb + 0.9) / F.H), 1.8 / F.W, 1.8 / F.H,
                                             transform=F.fig.transFigure, facecolor=S.LIGHT, edgecolor=SLAB_EDGE,
                                             lw=0.5))
    F.text(xs + 2.5, yb, 'GluCEST slab (12 × 2 mm)', ha='left', fontsize=5.2)
    F.fig.add_artist(S.mpl.patches.Rectangle((xs / F.W, 1 - (yb + 4.1) / F.H), 1.8 / F.W, 1.8 / F.H,
                                             transform=F.fig.transFigure, color=WB.OUT_SLAB, lw=0))
    F.text(xs + 2.5, yb + 3.2, 'outside slab (no GluCEST)', ha='left', fontsize=5.2)
    cax = F.ax(S.DOUBLE - 40, yb - 0.6, 30, 1.8)
    sm = S.mpl.cm.ScalarMappable(cmap=CMAP, norm=S.mpl.colors.Normalize(R.VMIN, R.VMAX))
    cb = F.fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=[0, 4, 8, 12])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=1.5, labelsize=5)
    F.text(S.DOUBLE - 42, yb + 0.3, 'GluCEST (%)', ha='right', fontsize=5.5)
    age = dict(D.subjects_by_age())[SUBJ]
    F.text(S.DOUBLE / 2, FH - 1.8, f'Participant {SUBJ} ({age} y). a, f: lateral view from the right, anterior to '
           'the right; b-e, g-j: superior view, anterior at the top. c-e and h-j share one (enlarged) scale.',
           fontsize=4.8, color=S.GREY)
    F.save(f'Fig_3D_overview_{SUBJ}_{CMAP}')
    print('assets in', ASSETS)


if __name__ == '__main__':
    make()
