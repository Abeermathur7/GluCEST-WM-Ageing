"""Step 2: WM-subregion extraction with verified atlas orientation, a high-resolution
anatomical underlay and a WM partial-volume (PV) fraction.  P18 excluded.

ORIENTATION (important)
  The slab-space NIfTIs written by the atlas pipeline (CESTslab.nii, CESTslab_seg.nii,
  CESTslab_WMrois.nii, ...) relate to the .mat arrays by   transpose(1,0,2) + flip of axis 0
  (anterior/posterior).  Evidence, per scan: Pearson r between the .mat S0image and
  CESTslab.nii is ~1.000 with the flip and ~0.45-0.67 without it (logged to
  _cache/orientation_check.csv).  utils_main.py applies only the transpose, so the notebook's
  atlas ROI values are A/P-mirrored.  Here every NIfTI-derived array is mapped with
  nifti_to_mat() below.

Per scan this writes _cache/sub_<scan>.npz with (in .mat orientation):
  wm_atlas  - JHU WM label image (correct orientation)
  uni       - MP2RAGE UNI (hd-bet) resampled into the slab grid by 3x3x4 sub-voxel averaging
  pv_wm/gm  - fraction of sub-samples labelled WM/GM by FSL FAST (MP2RAGE_UNI_hdbet_fast3_seg)
and tables:
  _cache/subregion_means.csv  (scan-1, every subject x subregion x voxel-selection variant)
  _cache/subregion_retest.csv (REGISTRY pairs with atlases for both scans)
"""
from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import utils_main as U  # noqa: E402

import nature_style as S  # noqa: E402
CACHE = S.CACHE
EXCLUDE = {'P18'}

# ── subregion definitions (JHU ICBM-DTI-81 labels as named in CESTslab_WMrois.txt) ──
SUBREGIONS = {
    'CC genu':     ['Genu of corpus callosum'],
    'CC body':     ['Body of corpus callosum'],
    'CC splenium': ['Splenium of corpus callosum'],
    'Cingulum':    ['Cingulum (cingulate gyrus) L', 'Cingulum (cingulate gyrus) R'],
    'ACR':         ['Anterior corona radiata L', 'Anterior corona radiata R'],
    'SCR':         ['Superior corona radiata L', 'Superior corona radiata R'],
    'PCR':         ['Posterior corona radiata L', 'Posterior corona radiata R'],
    'SLF':         ['Superior longitudinal fasciculus L', 'Superior longitudinal fasciculus R'],
    'sFOF':        ['Superior fronto-occipital fasciculus L', 'Superior fronto-occipital fasciculus R'],
    'ALIC':        ['Anterior limb of internal capsule L', 'Anterior limb of internal capsule R'],
    'EC':          ['External capsule L', 'External capsule R'],
}
FAMILY = {'CC genu': 'CC + cingulum', 'CC body': 'CC + cingulum', 'CC splenium': 'CC + cingulum',
          'Cingulum': 'CC + cingulum', 'ACR': 'Corona radiata', 'SCR': 'Corona radiata',
          'PCR': 'Corona radiata'}


# Atlas folders for repeat scans missing from utils_main.SCAN_FOLDERS (identified from the .mat
# refpath session names and confirmed by S0 vs CESTslab.nii correlation, see README):
#   P14 1 -> S076 (r = 1.000), P14 2 -> S076_rep2 (r = 1.000)
#   P02 3 (session techdev_20260512_S179) -> S179_rep3 (r = 1.000)
#   P02 4 (session S179_rep4) has no atlas folder on F:; nearest is S179_rep3 (r = 0.911, a different
#   session) -> rejected by MIN_R_S0, so the P02 pair is excluded from the subregion retest.
LOCAL_SCAN_FOLDERS = {('P14', 1): 'S076', ('P14', 2): 'S076_rep2', ('P02', 3): 'S179_rep3', ('P02', 4): 'S179_rep3'}
MIN_R_S0 = 0.95   # same-session matches give r >= 0.96; cross-session matches give r <= 0.91


def nifti_to_mat(a):
    return np.transpose(a, (1, 0, 2))[::-1]


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m].ravel(), b[m].ravel())[0, 1])


def pick_nifti_dir(folder, s0):
    """All nifti dirs with a WM atlas under folder; choose the one whose CESTslab.nii
    matches this scan's S0 (r ~ 1); fall back to a populated atlas."""
    cands = []
    for h in sorted((U.BASE / folder).rglob('CESTslab_WMrois.nii')):
        d = h.parent
        atlas = np.asarray(nib.load(str(h)).dataobj)
        r = np.nan
        refp = next((d / n for n in ('CESTslab.nii', 'CESTslabX.nii') if (d / n).exists()), None)
        if refp is not None:
            ref = np.squeeze(np.asarray(nib.load(str(refp)).dataobj, float))
            if ref.ndim == 4:      # multi-volume reference: use the first volume
                ref = ref[..., 0]
            if ref.ndim == 3 and (ref.shape[1], ref.shape[0], ref.shape[2]) == s0.shape:
                r = corr(nifti_to_mat(ref), s0)
        cands.append((d, r, (atlas > 0).sum()))
    if not cands:
        return None, np.nan
    with_r = [c for c in cands if np.isfinite(c[1])]
    best = max(with_r, key=lambda c: c[1]) if with_r else max(cands, key=lambda c: c[2])
    if best[2] == 0:  # empty placeholder atlas
        best = max(cands, key=lambda c: c[2])
    return best[0], best[1]


def seg_for(out, folder, d=None):
    """Tissue mask (0 none, 1 CSF, 2 GM, 3 WM) in the .mat frame. Default: the .mat segmask. GLUCEST_SEG=nifti: the
    scan's CESTslab_seg.nii (S0-matched nifti dir), cut to the same roimask as the .mat one. Never falls back."""
    if S.SEG_MODE != 'nifti':
        return np.asarray(out.segmask, np.uint8)
    if d is None:
        d, _ = pick_nifti_dir(folder, np.asarray(out.S0image, float))
    if d is None:
        raise FileNotFoundError(f'no nifti dir for {folder}')
    p = d / 'CESTslab_seg.nii'
    if p.exists():
        seg = nifti_to_mat(np.asarray(nib.load(str(p)).dataobj)).astype(np.uint8)
    else:
        # only P07 (S170) lacks the slab file: majority-vote resample of its whole-brain FAST seg into the slab grid
        # (reproduces the existing CESTslab_seg.nii files at 94-95% of labelled voxels, checked 2026-11-19)
        seg = fast_to_slab(d)
        print(f'  WARNING {folder}: no CESTslab_seg.nii, using FAST seg resampled into the slab')
    if not seg.any():
        raise ValueError(f'{p} is empty')
    assert seg.shape == np.asarray(out.segmask).shape, (folder, seg.shape)
    return np.where(np.asarray(out.roimask) > 0, seg, 0).astype(np.uint8)


def fast_to_slab(d):
    """MP2RAGE FAST seg (1 CSF, 2 GM, 3 WM) -> slab grid, majority label over 3 x 3 x 4 sub-samples per voxel (.mat frame)."""
    seg_p = next((b / 'MP2RAGE_UNI_hdbet_fast3_seg.nii.gz' for b in (d, d.parent / 'extra_files')
                  if (b / 'MP2RAGE_UNI_hdbet_fast3_seg.nii.gz').exists()), None)
    if seg_p is None:
        raise FileNotFoundError(f'no CESTslab_seg.nii or FAST seg for {d}')
    slab, seg = nib.load(str(d / 'CESTslab_WMrois.nii')), nib.load(str(seg_p))
    Sv = np.asarray(seg.dataobj, np.int16)
    sh = slab.shape[:3]
    M = np.linalg.inv(seg.affine) @ slab.affine
    ii, jj, kk = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in sh], indexing='ij')
    votes = np.zeros(sh + (4,), np.int16)
    for a in (-1 / 3, 0, 1 / 3):
        for b in (-1 / 3, 0, 1 / 3):
            for c in (-0.375, -0.125, 0.125, 0.375):
                P = np.stack([ii + a, jj + b, kk + c, np.ones(sh, np.float32)], 0).reshape(4, -1)
                lab = map_coordinates(Sv, (M @ P)[:3], order=0, mode='constant').reshape(sh)
                for k in range(4):
                    votes[..., k] += lab == k
    return nifti_to_mat(votes.argmax(-1)).astype(np.uint8)


def labels_for(d):
    lbl = U._roi_labels(d, 'WM')
    if not lbl:  # label txt may sit in a sibling export folder (P17)
        for t in d.parent.parent.rglob('CESTslab_WMrois.txt'):
            lbl = U._roi_labels(t.parent, 'WM')
            if lbl:
                break
    return lbl


def resample_uni(d, slab_img):
    for base in (d, d.parent / 'extra_files'):   # P07 keeps the hd-bet/FAST outputs in extra_files/
        uni_p = base / 'MP2RAGE_UNI_hdbet.nii.gz'
        seg_p = base / 'MP2RAGE_UNI_hdbet_fast3_seg.nii.gz'
        if uni_p.exists() and seg_p.exists():
            break
    else:
        return None
    uni, seg = nib.load(str(uni_p)), nib.load(str(seg_p))
    Uv = np.asarray(uni.dataobj, np.float32); Sv = np.asarray(seg.dataobj, np.int16)
    sh = slab_img.shape
    M = np.linalg.inv(uni.affine) @ slab_img.affine
    ii, jj, kk = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in sh], indexing='ij')
    o = (np.arange(3) + 0.5) / 3 - 0.5
    oz = (np.arange(4) + 0.5) / 4 - 0.5
    acc = np.zeros(sh); wm = np.zeros(sh); gm = np.zeros(sh); n = 0
    for a in o:
        for b in o:
            for c in oz:
                P = np.stack([ii + a, jj + b, kk + c, np.ones(sh, np.float32)], 0).reshape(4, -1)
                q = (M @ P)[:3]
                acc += map_coordinates(Uv, q, order=1, mode='constant').reshape(sh)
                lab = map_coordinates(Sv, q, order=0, mode='constant').reshape(sh)
                wm += lab == 3; gm += lab == 2; n += 1
    return {k: nifti_to_mat(v / n).astype(np.float32) for k, v in (('uni', acc), ('pv_wm', wm), ('pv_gm', gm))}


def process_scan(tag, scan_key, folder):
    out = U.load_scan(U.SCANS[scan_key])
    s0 = np.asarray(out.S0image, float)
    d, r = pick_nifti_dir(folder, s0)
    if d is None:
        return None, dict(scan=tag, folder=folder, nifti_dir='', r_S0=np.nan, note='no atlas')
    atlas_img = nib.load(str(d / 'CESTslab_WMrois.nii'))
    atlas = nifti_to_mat(np.asarray(atlas_img.dataobj)).astype(np.int16)
    rs = resample_uni(d, atlas_img)
    seg = seg_for(out, folder, d)
    chk = dict(scan=tag, folder=folder, nifti_dir=str(d), r_S0=r)
    if rs is not None:
        a, b = rs['pv_wm'] > 0.5, seg == 3
        chk['dice_WM_FAST_vs_mat'] = 2 * (a & b).sum() / (a.sum() + b.sum())
        a = a[::-1]  # same mapping without the A/P flip (= utils_main convention)
        chk['dice_WM_FAST_vs_mat_noflip'] = 2 * (a & b).sum() / (a.sum() + b.sum())
    data = dict(cest=np.asarray(out.CESTmap, np.float32), seg=seg, roi=np.asarray(out.roimask, np.uint8),
                t1=np.asarray(out.T1img, np.float32), atlas=atlas, lbl=labels_for(d))
    if rs is not None:
        data.update(rs)
    return data, chk


def roi_rows(data, zsel, base):
    """Per-subregion means for a z-selection (array of slice indices), several variants."""
    c, sg, rm, t1, at = (data[k][:, :, zsel] for k in ('cest', 'seg', 'roi', 't1', 'atlas'))
    pv = data['pv_wm'][:, :, zsel] if 'pv_wm' in data else None
    valid = np.isfinite(c) & (rm > 0) & (c > -10) & (c < 30) & (c != 0)
    name2val = {v: k for k, v in data['lbl'].items()}
    rows = []
    for sub, labs in SUBREGIONS.items():
        vals = [name2val[l] for l in labs if l in name2val]
        reg = np.isin(at, vals) & valid
        variants = {'raw': reg, 'segWM': reg & (sg == 3)}
        if pv is not None:
            variants['pvWM90'] = reg & (pv > 0.9)
        for var, m in variants.items():
            if m.sum() < 5:
                continue
            mt = m & (t1 > 0)
            rows.append(dict(base, Subregion=sub, Family=FAMILY.get(sub, 'Other'), Variant=var,
                             n_vox=int(m.sum()), GluCEST=float(c[m].mean()),
                             T1=float(t1[mt].mean()) if mt.any() else np.nan,
                             WMfrac_seg=float((sg[m] == 3).mean()),
                             WMfrac_pv=float(pv[m].mean()) if pv is not None else np.nan))
    return rows


def main():
    checks, rows = [], []
    subjects = sorted([(k, a) for k, a in U.AGE_MAP.items() if k.split(' – ')[0] not in EXCLUDE], key=lambda x: x[1])
    for key, age in subjects:
        s = key.split(' – ')[0]
        data, chk = process_scan(f'{s}_1', key, U.SUBJECT_MAP[key][0])
        checks.append(chk)
        if data is None:
            print('  no atlas', key); continue
        np.savez_compressed(CACHE / f'sub_{s}.npz', wm_atlas=data['atlas'],
                            **{k: data[k] for k in ('uni', 'pv_wm', 'pv_gm') if k in data})
        rows += roi_rows(data, np.arange(1, data['cest'].shape[2] - 1), dict(Subject=s, Age=age))
        print(f'  {key}: r_S0={chk["r_S0"]:.3f} dice={chk.get("dice_WM_FAST_vs_mat", np.nan):.3f}')
    pd.DataFrame(rows).to_csv(CACHE / 'subregion_means.csv', index=False)

    rrows = []
    folders = {**U.SCAN_FOLDERS, **LOCAL_SCAN_FOLDERS}
    for ini, sid, zt1, zt2, s1, s2, n1, n2 in U.REPRO_REGISTRY:
        if ini in EXCLUDE or (ini, n1) not in folders or (ini, n2) not in folders:
            print(f'  retest {ini}: skipped (no atlas folder for scan {n1} and/or {n2})')
            continue
        per, ok = [], True
        for n, zt, sw in ((n1, zt1, s1), (n2, zt2, s2)):
            data, chk = process_scan(f'{ini}_{n}', U.scan_key(ini, n), folders[(ini, n)])
            checks.append(chk)
            if data is None or not (chk['r_S0'] >= MIN_R_S0):
                print(f'  retest {ini}: scan {n} atlas rejected (r_S0 = {chk["r_S0"]:.3f} < {MIN_R_S0})')
                ok = False
                break
            z_all = np.arange(data['cest'].shape[2])[zt]
            per.append((data, z_all[np.arange(*sw.indices(len(z_all)))]))
        if not ok:
            continue
        nsl = min(len(per[0][1]), len(per[1][1]))
        for j, (data, z) in enumerate(per):
            for r in roi_rows(data, z[:nsl], dict(Subject=ini, scan='P13'[j])):
                rrows.append(r)
        print(f'  retest {ini}: done ({nsl} slices)')
    pd.DataFrame(rrows).to_csv(CACHE / 'subregion_retest.csv', index=False)
    pd.DataFrame(checks).to_csv(CACHE / 'orientation_check.csv', index=False)


if __name__ == '__main__':
    main()
