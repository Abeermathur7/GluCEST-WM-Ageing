"""Re-extract scan-rescan repeatability for Table 2 (WM subregions, CC split into genu/body/splenium) and a new
GM lobe table, with the current scan-folder mapping (P01 fixed 2026-10-21) and orientation-corrected atlases.

Needs the data drive.  Run:  python recompute_retest_tables.py
Writes:
  _cache/subregion_retest.csv          (old copy kept as _cache/subregion_retest_pre2026-11-23.csv)
  _cache/gm_lobe_retest.csv            per subject / scan / lobe means
  Table2_WM_subregion_CoV_split.csv    per-participant CoV (%) + mean
  Table3_GM_lobe_CoV.csv               per-participant CoV (%) + mean
"""
from __future__ import annotations

import shutil

import nibabel as nib
import numpy as np
import pandas as pd

import extract_subregions as X
import figdata as D
import nature_style as S

U = X.U
CACHE = S.CACHE

# Harvard-Oxford cortical index (0-based, = atlas value - 1, as in gm_subregion_age.py) -> lobe
LOBE = {**{i: 'Frontal' for i in (0, 2, 3, 4, 5, 6, 24, 25, 32, 40)},
        **{i: 'Cingulate' for i in (26, 27, 28, 29)},
        **{i: 'Parietal' for i in (16, 17, 18, 19, 20, 30, 41, 42)},
        **{i: 'Occipital' for i in (21, 22, 23, 31, 35, 39, 46, 47)},
        **{i: 'Temporal' for i in (7, 8, 9, 10, 11, 12, 13, 14, 15, 33, 34, 36, 37, 38, 43, 44, 45)},
        1: 'Insula'}
LOBES = ['Frontal', 'Cingulate', 'Parietal', 'Occipital', 'Temporal', 'Insula']
WM_COLS = {'CC genu': ['CC genu'], 'CC body': ['CC body'], 'CC splenium': ['CC splenium'],
           'CC (all)': ['CC genu', 'CC body', 'CC splenium'], 'Cingulum': ['Cingulum'],
           'CR': ['ACR', 'SCR', 'PCR'], 'LF': ['SLF'], 'FOF': ['sFOF'],
           'CC + Cing.': ['CC genu', 'CC body', 'CC splenium', 'Cingulum']}


def cov(a, b):
    """paper formula: |a - b| / sqrt(2) over the mean of |a|, |b|, in %"""
    return np.abs(a - b) / np.sqrt(2) / ((np.abs(a) + np.abs(b)) / 2) * 100


def gm_rows(data, nifti_dir, z, base):
    p = nifti_dir / 'CESTslab_GMrois.nii'
    if not p.exists():
        return []
    gm = X.nifti_to_mat(np.asarray(nib.load(str(p)).dataobj)).astype(np.int16)[:, :, z]   # A/P flip included
    c, sg, rm = data['cest'][:, :, z], data['seg'][:, :, z], data['roi'][:, :, z]
    ok = np.isfinite(c) & (rm > 0) & (c > -10) & (c < 30) & (c != 0) & (sg == 2)
    lobe = np.vectorize(lambda v: LOBE.get(int(v) - 1, '') if v > 0 else '')(gm)
    rows = []
    for L in LOBES + ['All GM regions']:
        m = ok & ((lobe != '') if L == 'All GM regions' else (lobe == L))
        if m.sum() >= 5:
            rows.append(dict(base, Lobe=L, n_vox=int(m.sum()), GluCEST=float(c[m].mean())))
    return rows


def main():
    folders = {**U.SCAN_FOLDERS, **X.LOCAL_SCAN_FOLDERS}
    wrows, grows = [], []
    for ini, sid, zt1, zt2, s1, s2, n1, n2 in U.REPRO_REGISTRY:
        if ini in X.EXCLUDE or (ini, n1) not in folders or (ini, n2) not in folders:
            print(f'  {ini}: skipped (no folder for scan {n1} and/or {n2})')
            continue
        per = []
        for n, zt, sw in ((n1, zt1, s1), (n2, zt2, s2)):
            data, chk = X.process_scan(f'{ini}_{n}', U.scan_key(ini, n), folders[(ini, n)])
            if data is None or not (chk['r_S0'] >= X.MIN_R_S0):
                print(f'  {ini}: scan {n} rejected (r_S0 = {chk.get("r_S0", np.nan):.3f})')
                per = None
                break
            z_all = np.arange(data['cest'].shape[2])[zt]
            per.append((data, chk, z_all[np.arange(*sw.indices(len(z_all)))]))
        if not per:
            continue
        nsl = min(len(per[0][2]), len(per[1][2]))
        for j, (data, chk, z) in enumerate(per):
            base = dict(Subject=ini, scan='P13'[j])
            wrows += X.roi_rows(data, z[:nsl], base)
            grows += gm_rows(data, X.Path(chk['nifti_dir']), z[:nsl], base)
        print(f'  {ini}: done ({nsl} slices, scans {n1}/{n2} from {folders[(ini, n1)]} / {folders[(ini, n2)]})')

    old = CACHE / 'subregion_retest.csv'
    if old.exists():
        shutil.copy(old, CACHE / 'subregion_retest_pre2026-11-23.csv')
    wr, gr = pd.DataFrame(wrows), pd.DataFrame(grows)
    wr.to_csv(old, index=False)
    gr.to_csv(CACHE / 'gm_lobe_retest.csv', index=False)

    age = dict(D.subjects_by_age())

    def pooled(d, key, groups):
        out = {}
        for col, members in groups.items():
            x = d[d[key].isin(members)].assign(w=lambda t: t.GluCEST * t.n_vox)
            g = x.groupby(['Subject', 'scan'])[['w', 'n_vox']].sum()
            m = (g.w / g.n_vox).unstack('scan')
            out[col] = cov(m['A'], m['B'])
        t = pd.DataFrame(out)
        t.insert(0, 'Age', [age.get(s, np.nan) for s in t.index])
        t = t.sort_values('Age')
        t.loc['Mean'] = t.mean()
        t.index = [str(i + 1) if s != 'Mean' else 'Mean' for i, s in enumerate(t.index)]   # no identifiers in outputs
        t.index.name = 'Participant'
        return t.round(2)

    t2 = pooled(wr[wr.Variant == 'raw'], 'Subregion', WM_COLS)
    t3 = pooled(gr, 'Lobe', {L: [L] for L in LOBES + ['All GM regions']})
    t2.to_csv(S.OUT / 'Table2_WM_subregion_CoV_split.csv')
    t3.to_csv(S.OUT / 'Table3_GM_lobe_CoV.csv')
    print(t2.to_string(), '\n\n', t3.to_string())


if __name__ == '__main__':
    main()
