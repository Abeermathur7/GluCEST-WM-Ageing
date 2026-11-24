"""Whole-brain JHU white-matter atlas in 3D (ITK-SNAP style) with GluCEST painted where the slab covers it.

a/b: all 48 JHU WM labels in native MP2RAGE space (right-lateral, superior views), coloured by tract family.
c/d: same surfaces; inside the GluCEST slab the surface shows GluCEST (0-12 %), outside it is grey.

Only participant P04 (26 y) has the whole-brain native atlas on the data drive (S148_rep1).
Geometry: everything is placed in scanner (RAS, mm) coordinates through the NIfTI affines, so renders
show true anatomy. Cached GluCEST (.mat orientation) -> slab NIfTI grid: transpose (1,0,2) + flip axis 1
(verified: exact label match with CESTslab_WMrois.nii).

Run:  python fig_3d_wholebrain.py [cmap]      (default inferno)
"""
from __future__ import annotations

import sys

import nibabel as nib
import numpy as np
import pandas as pd
import pyvista as pv
from scipy import ndimage as ndi

import figdata as D
import nature_style as S
import render3d as R

SUBJ, FOLDER = 'P04', 'D:/7T_GLUCEST_Repeatability/S148_rep1/nifti/'
CMAP = sys.argv[1] if len(sys.argv) > 1 else 'inferno_lift'
DS = 2                         # downsample the 0.69 mm atlas to ~1.4 mm for rendering
C = S.SUB_COL

FAMILY = {  # label -> (family, colour)
    **{l: ('Corpus callosum', c) for l, c in [(3, C['CC genu']), (4, C['CC body']), (5, C['CC splenium'])]},
    35: ('Cingulum', C['Cingulum']), 36: ('Cingulum', C['Cingulum']),
    37: ('Cingulum', '#F0A060'), 38: ('Cingulum', '#F0A060'),
    23: ('Corona radiata', C['ACR']), 24: ('Corona radiata', C['ACR']),
    25: ('Corona radiata', C['SCR']), 26: ('Corona radiata', C['SCR']),
    27: ('Corona radiata', C['PCR']), 28: ('Corona radiata', C['PCR']),
}
for l in range(17, 23):
    FAMILY[l] = ('Internal capsule', ['#56B4E9', '#56B4E9', '#3A8FC6', '#3A8FC6', '#0072B2', '#0072B2'][l - 17])
for l in (29, 30, 31, 32, 33, 34, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48):
    FAMILY[l] = ('Association / other', '#E69F00' if l in (41, 42) else '#F0E442' if l in (43, 44) else '#D9C38A')
for l in list(range(1, 3)) + list(range(6, 17)):
    FAMILY[l] = ('Brainstem / cerebellar / fornix', '#B9A7C9')
FAM_ORDER = ['Corpus callosum', 'Cingulum', 'Corona radiata', 'Internal capsule',
             'Association / other', 'Brainstem / cerebellar / fornix']
FAM_SWATCH = {'Corpus callosum': C['CC body'], 'Cingulum': C['Cingulum'], 'Corona radiata': C['SCR'],
              'Internal capsule': '#3A8FC6', 'Association / other': '#D9C38A',
              'Brainstem / cerebellar / fornix': '#B9A7C9'}
OUT_SLAB = '#5E5E5E'   # dim grey on black (was #CFCFCF on white)


def load():
    wb = nib.load(FOLDER + 'MP2RAGE_UNI_hdbet_WMrois.nii.gz')
    lab = np.asarray(wb.dataobj).astype(np.int16)[::DS, ::DS, ::DS]
    aff = wb.affine @ np.diag([DS, DS, DS, 1])
    brain = nib.load(FOLDER + 'MP2RAGE_INV2_SPM8bias_hdbet_mask.nii.gz')
    bm = np.asarray(brain.dataobj)[::DS, ::DS, ::DS] > 0
    baff = brain.affine @ np.diag([DS, DS, DS, 1])
    slab = nib.load(FOLDER + 'CESTslab_WMrois.nii')
    v = D.vol(SUBJ)
    to_nii = lambda a: np.flip(np.transpose(a, (1, 0, 2)), axis=1)
    sl_lab = np.asarray(slab.dataobj).astype(int)
    assert (to_nii(D.sub(SUBJ)['wm_atlas'].astype(int)) == sl_lab).all(), 'slab orientation mismatch'
    cest = to_nii(v['CESTmap'].astype(float))
    roi = to_nii(v['roi'] > 0)
    ok = (sl_lab > 0) & roi & np.isfinite(cest) & (cest != 0) & (cest > -10) & (cest < 30)
    cest_f = ndi.gaussian_filter(R.fill_nearest(cest, ok), (0.7, 0.7, 0.35))
    return lab, aff, bm, baff, cest_f, slab.affine, slab.shape


def surf_world(mask, aff, sigma=0.7):
    m = ndi.gaussian_filter(np.pad(mask.astype(np.float32), 2), sigma)
    g = pv.ImageData(dimensions=m.shape, spacing=(1, 1, 1), origin=(-2, -2, -2))
    g.point_data['m'] = m.ravel(order='F')
    s = g.contour([0.5], scalars='m').smooth_taubin(n_iter=30, pass_band=0.05)
    s = s.transform(aff, inplace=False)
    return s.compute_normals(auto_orient_normals=True)


def label_of_points(pts, lab, aff):
    filled = R.fill_nearest(lab, lab > 0)
    ijk = np.round(nib.affines.apply_affine(np.linalg.inv(aff), pts)).astype(int)
    ijk = np.clip(ijk, 0, np.array(lab.shape) - 1)
    return filled[ijk[:, 0], ijk[:, 1], ijk[:, 2]]


def cest_of_points(pts, cest_f, saff, sshape):
    ijk = nib.affines.apply_affine(np.linalg.inv(saff), pts)
    inside = np.all((ijk >= -0.5) & (ijk <= np.array(sshape) - 0.5), axis=1)
    vals = ndi.map_coordinates(cest_f, ijk.T, order=1, mode='nearest')
    return np.where(inside, vals, np.nan)


VIEWS = {'lateral': (np.array([0.90, 0.30, 0.38]), (0, 0, 1)),      # RAS (x=R, y=A): from right, slightly anterior/up
         'superior': (np.array([0.0, -0.25, 1.0]), (0, 1, 0))}       # top-down, anterior at the top


def plot(mesh_kw, centre, view, dist=560, hull=None):
    d, up = VIEWS[view]
    d = d / np.linalg.norm(d)
    p = pv.Plotter(off_screen=True, window_size=(1600, 1300))
    p.set_background('black')
    p.enable_anti_aliasing('ssaa')
    if hull is not None:
        p.add_mesh(hull, color='#EDEDED', opacity=0.12, smooth_shading=True)
    for mesh, kw in mesh_kw:
        p.add_mesh(mesh, **kw)
    p.camera_position = [tuple(centre + dist * d), tuple(centre), up]
    p.camera.view_angle = 22
    return p


def make():
    lab, aff, bm, baff, cest_f, saff, sshape = load()
    names = pd.read_csv(FOLDER + 'MP2RAGE_UNI_hdbet_WMrois.tsv', sep='\t')
    wm = surf_world(lab > 0, aff)
    hull = None        # translucent brain outline removed at the user's request (11-18)
    l = label_of_points(wm.points, lab, aff)
    rgb = np.array([R.hex2rgb(FAMILY.get(int(x), ('', OUT_SLAB))[1]) for x in l])
    wm.point_data['rgb'] = (rgb * 255).astype(np.uint8)
    glu = cest_of_points(wm.points, cest_f, saff, sshape)
    wm_glu = wm.copy()
    wm_glu.point_data['glu'] = glu
    centre = np.asarray(wm.center)
    # slab outline (12 x 2 mm box) in world space
    box = pv.Box(bounds=(-0.5, sshape[0] - 0.5, -0.5, sshape[1] - 0.5, -0.5, sshape[2] - 0.5)).transform(saff, inplace=False)
    # restrict the drawn box to the brain's extent in-plane: just its edges
    edges = box.extract_feature_edges(feature_angle=30)

    lab_kw = dict(scalars='rgb', rgb=True, **R.MESH_KW)
    glu_kw = dict(scalars='glu', cmap=CMAP, clim=(R.VMIN, R.VMAX), nan_color=OUT_SLAB,
                  show_scalar_bar=False, **R.MESH_KW)
    edge_kw = dict(color='#404040', line_width=2.5)
    imgs = {}
    for view in ('lateral', 'superior'):
        imgs['lab', view] = R.shot(plot([(wm, lab_kw)], centre, view, hull=hull), f'{SUBJ}_wb_lab_{view}')
        imgs['glu', view] = R.shot(plot([(wm_glu, glu_kw)], centre, view, hull=hull),
                                   f'{SUBJ}_wb_glu_{view}_{CMAP}')
    b = R.bbox(list(imgs.values()))
    imgs = {k: R.crop(v, b) for k, v in imgs.items()}
    H, W = next(iter(imgs.values())).shape[:2]

    # ── compose ──────────────────────────────────────────────────────────────
    x0, g, t, side = 3, 2.0, 4.0, 40
    pw = (S.DOUBLE - x0 - side - g - 2) / 2
    ph = pw * H / W
    FH = 2 * (ph + t) + 3 + 6
    F = S.MMFigure(S.DOUBLE, FH)
    cells = [[('a', ('lab', 'lateral'), 'JHU white-matter atlas: lateral'),
              ('b', ('lab', 'superior'), 'JHU white-matter atlas: superior')],
             [('c', ('glu', 'lateral'), 'GluCEST within the slab: lateral'),
              ('d', ('glu', 'superior'), 'GluCEST within the slab: superior')]]
    for r, row in enumerate(cells):
        y = 1 + r * (ph + t + 3)
        for c, (letter, key, title) in enumerate(row):
            x = x0 + c * (pw + g)
            F.letter(letter, x - 2, y + 0.3)
            F.text(x + pw / 2, y + 1.6, title, fontweight='bold', fontsize=6.5)
            ax = F.ax(x, y + t, pw, ph)
            S.img_ax(ax)
            ax.imshow(imgs[key])
    # side column: family key (top) and colour bar (bottom)
    sx = x0 + 2 * pw + g + 4
    F.text(sx, 1 + t + 1, 'Tract family', ha='left', fontweight='bold', fontsize=6)
    for i, fam in enumerate(FAM_ORDER):
        yy = 1 + t + 5 + i * 3.4
        F.fig.add_artist(S.mpl.patches.Rectangle((sx / F.W, 1 - (yy + 0.9) / F.H), 1.8 / F.W, 1.8 / F.H,
                                                 transform=F.fig.transFigure, color=FAM_SWATCH[fam], lw=0))
        F.text(sx + 2.6, yy, fam, ha='left', fontsize=5.5)
    yb = 1 + ph + t + 3 + t + 3
    cax = F.ax(sx, yb, side - 8, 2.0)
    sm = S.mpl.cm.ScalarMappable(cmap=CMAP, norm=S.mpl.colors.Normalize(R.VMIN, R.VMAX))
    cb = F.fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=[0, 4, 8, 12])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=1.5, labelsize=5)
    cb.set_label('GluCEST (%)', labelpad=1.5, fontsize=5.5)
    F.fig.add_artist(S.mpl.patches.Rectangle((sx / F.W, 1 - (yb + 10.9) / F.H), 1.8 / F.W, 1.8 / F.H,
                                             transform=F.fig.transFigure, color=OUT_SLAB, lw=0))
    F.text(sx + 2.6, yb + 10, 'outside GluCEST slab', ha='left', fontsize=5.5)
    age = dict(D.subjects_by_age())[SUBJ]
    F.text(S.DOUBLE / 2, FH - 2.2, f'Participant {SUBJ} ({age} y), native MP2RAGE space; JHU ICBM-DTI-81 labels '
           f'({len(names)} present). Lateral view from the right, anterior to the right.', fontsize=5, color=S.GREY)
    F.save(f'Fig_3D_wholebrain_{SUBJ}_{CMAP}')


if __name__ == '__main__':
    make()
