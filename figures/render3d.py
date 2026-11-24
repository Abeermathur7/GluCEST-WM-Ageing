"""Shared 3D rendering helpers (PyVista/VTK, off-screen) for the white-matter ROI / GluCEST renders.

Data: corrected JHU atlas + MP2RAGE UNI from sub_<s>.npz, GluCEST from vol_<s>.npz (GluCEST slab
space, 1 x 1 x 2 mm). Array axes: 0 = anterior->posterior, 1 = right->left, 2 = inferior->superior.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pyvista as pv
from PIL import Image
from scipy import ndimage as ndi

import figdata as D
import nature_style as S

S.dark()                      # every 3D figure: black background, light text
BG = 'black'
CMAP_DEFAULT = 'inferno_lift'
VMIN, VMAX = 0, 12
ZUP = 2                       # upsample 2 mm slices to 1 mm for smooth surfaces
PAD = 3
CC_CING = [3, 4, 5, 35, 36]   # genu, body, splenium, cingulum (cingulate gyrus) R/L
CR = [23, 24, 25, 26, 27, 28]  # anterior / superior / posterior corona radiata R/L
RDIR = S.CACHE / 'render3d'
RDIR.mkdir(exist_ok=True)

C = S.SUB_COL
LABEL_COL = {3: C['CC genu'], 4: C['CC body'], 5: C['CC splenium'], 35: C['Cingulum'], 36: C['Cingulum'],
             23: C['ACR'], 24: C['ACR'], 25: C['SCR'], 26: C['SCR'], 27: C['PCR'], 28: C['PCR']}
OTHER_COL = '#C8C8C8'

# Camera presets: (direction from the ROI centre to the camera, view-up), in render axes
# (0 = posterior, 1 = RIGHT, 2 = superior; see _rh). Renders show true anatomy (not mirrored).
VIEWS = {
    'superior': ([-0.15, 0.0, 1.0], (-1, 0, 0)),          # top-down, anterior at the top
    'superior-anterior': ([-0.6, 0.0, 0.85], (0, 0, 1)),
    'anterior-right-superior': ([-0.5, 0.35, 0.8], (0, 0, 1)),
    'left-superior': ([-0.15, -0.75, 0.65], (0, 0, 1)),
    'posterior-left-superior': ([0.5, -0.35, 0.8], (0, 0, 1)),
    'anterior-low': ([-0.85, -0.3, 0.45], (0, 0, 1)),
    'anterior-left-superior': ([-0.50, -0.35, 0.80], (0, 0, 1)),
    # matches Example_orientation.png (ITK-SNAP): right-lateral, anterior on the right of the image,
    # camera slightly above the horizontal and turned a little toward the front
    'right-lateral': ([-0.30, 0.90, 0.38], (0, 0, 1)),
}
VIEW = 'superior-anterior'     # slab-only figures; 'right-lateral' matches Example_orientation.png


def set_view(name):
    global VIEW
    assert name in VIEWS, f'unknown view {name}; choose from {list(VIEWS)}'
    VIEW = name
WIN = (1600, 1300)
MESH_KW = dict(smooth_shading=True, ambient=0.25, diffuse=0.8, specular=0.15, specular_power=15)


def hex2rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255


def prep(a, order, cval=0.0):
    """Pad, then upsample the slice axis to isotropic 1 mm."""
    a = np.pad(a, PAD, constant_values=cval)
    return ndi.zoom(a, (1, 1, ZUP), order=order, mode='nearest')


def grid_of(a, name):
    g = pv.ImageData(dimensions=a.shape, spacing=(1, 1, 1))
    g.point_data[name] = np.asarray(a, np.float32).ravel(order='F')
    return g


def fill_nearest(vals, valid):
    """Replace invalid voxels with the value of the nearest valid voxel, so a surface sampled at the
    ROI boundary sees in-ROI values rather than background."""
    idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return vals[tuple(idx)]


def surface(mask):
    m = ndi.gaussian_filter(prep(mask.astype(np.float32), 1), 0.8)
    s = grid_of(m, 'm').contour([0.5], scalars='m')
    return s.smooth_taubin(n_iter=40, pass_band=0.05).compute_normals(auto_orient_normals=True)


def sample(surf, vol, name):
    return np.asarray(surf.sample(grid_of(vol, name))[name])


def _rh(a):
    """Cached arrays are (posterior, LEFT, superior): a left-handed frame, which VTK would render as a
    mirror image. Flipping axis 1 gives (posterior, RIGHT, superior), right-handed = true anatomy."""
    return np.ascontiguousarray(np.flip(a, axis=1))


@lru_cache(maxsize=None)
def subject(s):
    sub, v = D.sub(s), D.vol(s)
    atlas = _rh(sub['wm_atlas']).astype(int)
    roi = _rh(v['roi']) > 0
    cest = _rh(v['CESTmap']).astype(float)
    wm = (atlas > 0) & roi
    ok = wm & np.isfinite(cest) & (cest != 0) & (cest > -10) & (cest < 30)
    regions = {'all': wm, 'cc': np.isin(atlas, CC_CING) & roi, 'cr': np.isin(atlas, CR) & roi}
    centre = (np.argwhere(wm).mean(0) + PAD) * np.array([1, 1, ZUP])
    return dict(atlas=atlas, roi=roi, cest=cest, ok=ok, uni=_rh(sub['uni']).astype(float),
                regions=regions, centre=centre)


def plotter(centre, dist):
    p = pv.Plotter(off_screen=True, window_size=WIN)
    p.set_background(BG)
    p.enable_anti_aliasing('ssaa')
    d, up = VIEWS[VIEW]
    d = np.array(d, float) / np.linalg.norm(d)
    p.camera_position = [tuple(centre + dist * d), tuple(centre), tuple(float(u) for u in up)]
    p.camera.view_angle = 22
    return p


def render_labels(s, key, dist=300):
    d = subject(s)
    reg = d['regions'][key]
    surf = surface(reg)
    lab = prep(fill_nearest(np.where(reg, d['atlas'], 0), reg), 0)
    # nearest-voxel lookup: interpolating label numbers would invent labels at seams (4 & 35 -> 20)
    ijk = np.clip(np.round(surf.points).astype(int), 0, np.array(lab.shape) - 1)
    lab = lab[ijk[:, 0], ijk[:, 1], ijk[:, 2]].astype(int)
    rgb = np.array([hex2rgb(LABEL_COL.get(l, OTHER_COL)) for l in lab])
    surf.point_data['rgb'] = (rgb * 255).astype(np.uint8)
    p = plotter(d['centre'], dist)
    p.add_mesh(surf, scalars='rgb', rgb=True, **MESH_KW)
    return p


def render_cest(s, key, dist=300, cmap=CMAP_DEFAULT):
    """GluCEST from this region's voxels only, extended outward to the surface, light smoothing."""
    d = subject(s)
    reg = d['regions'][key]
    surf = surface(reg)
    f = fill_nearest(d['cest'], d['ok'] & reg)
    f = prep(ndi.gaussian_filter(f, (0.7, 0.7, 0.35)), 1)
    surf.point_data['glu'] = sample(surf, f, 'glu')
    p = plotter(d['centre'], dist)
    p.add_mesh(surf, scalars='glu', cmap=cmap, clim=(VMIN, VMAX), show_scalar_bar=False, **MESH_KW)
    return p


def render_anat(s, dist=560):
    """MP2RAGE UNI on orthogonal planes through the slab, inside a faint brain hull."""
    d = subject(s)
    a = np.where(d['roi'], d['uni'], np.nan)
    lo, hi = np.nanpercentile(a, [1, 99])
    a = np.clip((a - lo) / (hi - lo), 0, 1)
    a = prep(np.nan_to_num(a, nan=-1.0), 0, cval=-1.0)
    c = d['centre']
    planes = grid_of(a, 'uni').slice_orthogonal(x=c[0] + 6, y=c[1], z=a.shape[2] - PAD * ZUP - 3)
    p = plotter(c, dist)
    for block in planes:
        b = block.threshold(-0.5, scalars='uni')          # keep brain voxels only
        if b.n_points:
            p.add_mesh(b, scalars='uni', cmap='gray', clim=(0, 1), show_scalar_bar=False,
                       ambient=0.6, diffuse=0.5, show_edges=False)
    return p


def paper_mean(s, key):
    """Voxel-weighted regional GluCEST from subregion_means.csv (orientation-corrected atlas, 'raw'
    variant): the source of the paper's numbers. (roi_means.csv is an older extraction: do not use.)"""
    d = D.subregions('raw')
    d = d[d.Subject == s]
    if key == 'cc':
        d = d[d.Family == 'CC + cingulum']
    elif key == 'cr':
        d = d[d.Family == 'Corona radiata']
    return float(np.average(d.GluCEST, weights=d.n_vox))


def shot(p, name):
    f = RDIR / f'{name}.png'
    p.screenshot(str(f))
    p.close()
    return np.asarray(Image.open(f).convert('RGB'))


def bbox(imgs, pad=8):
    """Common crop box (clamped to the image) over a set of same-size renders."""
    nw = np.any([im[..., :3].max(axis=2) > 12 for im in imgs], axis=0)      # non-black
    rr, cc = np.where(nw)
    h, w = nw.shape
    return max(rr.min() - pad, 0), min(rr.max() + pad, h), max(cc.min() - pad, 0), min(cc.max() + pad, w)


def crop(im, b):
    return im[b[0]:b[1], b[2]:b[3]]
