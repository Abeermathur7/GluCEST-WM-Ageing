"""Load the cached data written by extract_data.py (P18 already excluded)."""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from nature_style import CACHE

EXCLUDE = {'P18'}


def _drop(df):
    return df[~df['Subject'].isin(EXCLUDE)].reset_index(drop=True)


@lru_cache
def tissue_means():
    return _drop(pd.read_csv(CACHE / 'tissue_means.csv'))


def gm_wm_wide():
    t = tissue_means()
    w = t.pivot(index='Subject', columns='Tissue', values='GluCEST')
    w['Age'] = t.drop_duplicates('Subject').set_index('Subject')['Age']
    return w.sort_values('Age')


@lru_cache
def roi_means():
    return _drop(pd.read_csv(CACHE / 'roi_means.csv'))


@lru_cache
def voxels():
    return _drop(pd.read_csv(CACHE / 'voxel_sample.csv.gz'))


@lru_cache
def retest():
    return _drop(pd.read_csv(CACHE / 'retest_means.csv'))


def subjects_by_age():
    t = tissue_means().drop_duplicates('Subject').sort_values(['Age', 'Subject'])
    return list(zip(t['Subject'], t['Age']))


@lru_cache(maxsize=32)
def vol(s):
    assert s not in EXCLUDE
    d = np.load(CACHE / f'vol_{s}.npz')
    return {k: d[k] for k in d.files}


def labels(s):
    return json.loads((CACHE / f'labels_{s}.json').read_text())


@lru_cache
def subregions(variant='raw'):
    """Orientation-corrected per-subject WM subregion means (extract_subregions.py).
    variant: 'raw' (atlas ∩ valid), 'segWM' (∩ .mat segmask WM), 'pvWM90' (∩ FAST WM PV > 0.9)."""
    d = _drop(pd.read_csv(CACHE / 'subregion_means.csv'))
    return d[d.Variant == variant].reset_index(drop=True) if variant else d


@lru_cache
def subregion_retest(variant='raw'):
    d = _drop(pd.read_csv(CACHE / 'subregion_retest.csv'))
    return d[d.Variant == variant].reset_index(drop=True) if variant else d


SUB_ORDER = ['CC genu', 'CC body', 'CC splenium', 'Cingulum', 'ACR', 'SCR', 'PCR', 'SLF', 'sFOF',
             'ALIC', 'EC']


def family_means(variant='raw'):
    """Per-subject voxel-weighted mean over subregions of each family."""
    d = subregions(variant)
    d = d[d.Family != 'Other'].copy()
    d['w'] = d.n_vox
    g = d.groupby(['Subject', 'Age', 'Family'])
    out = g.apply(lambda x: pd.Series({
        'GluCEST': np.average(x.GluCEST, weights=x.w),
        'T1': np.average(x.T1.fillna(x.T1.mean()), weights=x.w),
        'WMfrac_pv': np.average(x.WMfrac_pv, weights=x.w),
        'WMfrac_seg': np.average(x.WMfrac_seg, weights=x.w),
        'n_vox': x.w.sum()}), include_groups=False).reset_index()
    return out


def subregion_label_map(s):
    """{atlas integer value -> subregion name} for subject s (JHU numbering)."""
    from extract_subregions import SUBREGIONS
    lbl = {int(k): v for k, v in labels(s)['WM'].items()}
    name2sub = {n: sub for sub, names in SUBREGIONS.items() for n in names}
    return {k: name2sub[v] for k, v in lbl.items() if v in name2sub}


def paint(s, z, values, mask_valid=True):
    """Paint {subregion: value} onto subject s's (orientation-corrected) atlas at slice z."""
    at = sub(s)['wm_atlas'][:, :, z]
    out = np.full(at.shape, np.nan)
    for k, name in subregion_label_map(s).items():
        if name in values:
            out[at == k] = values[name]
    if mask_valid:
        out[vol(s)['roi'][:, :, z] == 0] = np.nan
    return np.ma.masked_invalid(out)


def subregion_index(s, z):
    """Integer image: index into SUB_ORDER (+1), 0 = none."""
    at = sub(s)['wm_atlas'][:, :, z]
    out = np.zeros(at.shape, int)
    for k, name in subregion_label_map(s).items():
        out[at == k] = SUB_ORDER.index(name) + 1
    return out


def tract_group(label: str):
    l = label.lower()
    if 'corpus callosum' in l:
        return 'Corpus callosum'
    if 'corona radiata' in l:
        return 'Corona radiata'
    return None


def tract_groups():
    """Per-subject mean GluCEST / T1 for the two tract families (mean of ROI means,
    as in the notebook's glu_grp)."""
    r = roi_means()
    r = r[r['Type'] == 'WM'].copy()
    r['Group'] = r['ROIlabel'].map(tract_group)
    return (r.dropna(subset=['Group']).groupby(['Subject', 'Age', 'Group'])[['GluCEST', 'T1']]
            .mean().reset_index())


def brain_box(s, pad=3, slices=range(1, 11)):
    roi = vol(s)['roi'][:, :, list(slices)].any(2)
    rr, cc = np.where(roi)
    return rr.min() - pad, rr.max() + pad + 1, cc.min() - pad, cc.max() + pad + 1


def common_crop(subjects, pad=3):
    """Same-size crop window for several subjects, each centred on its own brain."""
    boxes = {s: brain_box(s, pad) for s in subjects}
    H = max(b[1] - b[0] for b in boxes.values())
    W = max(b[3] - b[2] for b in boxes.values())

    def crop(a, s):
        r0, r1, c0, c1 = boxes[s]
        rc, cc = (r0 + r1) // 2, (c0 + c1) // 2
        r0, c0 = max(rc - H // 2, 0), max(cc - W // 2, 0)
        return a[r0:r0 + H, c0:c0 + W]
    return crop, H, W


@lru_cache(maxsize=32)
def sub(s):
    """Outputs of extract_subregions.py: correctly oriented WM atlas, resampled MP2RAGE UNI,
    FAST WM/GM partial-volume fractions (all in .mat orientation)."""
    assert s not in EXCLUDE
    d = np.load(CACHE / f'sub_{s}.npz')
    return {k: d[k] for k in d.files}


def underlay(s, z, kind='S0', pct=(1, 99)):
    v = vol(s)
    if kind == 'UNI':  # MP2RAGE UNI resampled to the slab: best GM/WM contrast
        a = sub(s)['uni'][:, :, z].astype(float)
        brain = (a > 0) & (v['roi'][:, :, z] > 0)
        lo, hi = np.percentile(a[brain], pct)
        return np.ma.masked_where(~brain, np.clip((a - lo) / (hi - lo), 0, 1))
    if kind == 'S0':
        a = v['S0image'][:, :, z].astype(float)
    else:  # inverted T1 map -> T1-weighted-like contrast
        t1 = v['segimage'][:, :, z].astype(float)
        a = np.where(t1 > 0, 1.0 / np.clip(t1, 300, None), 0)
    brain = v['roi'][:, :, z] > 0
    lo, hi = np.percentile(a[brain], [1, 99.5])
    return np.ma.masked_where(~brain, np.clip((a - lo) / (hi - lo), 0, 1))


def cest_masked(s, z, tissues=(2, 3)):
    v = vol(s)
    c = v['CESTmap'][:, :, z].astype(float)
    ok = np.isin(v['seg'][:, :, z], tissues) & (v['roi'][:, :, z] > 0) & np.isfinite(c) & (c != 0)
    return np.ma.masked_where(~ok, c)


def gm_atlas(s):
    """Harvard-Oxford slab atlas in the GluCEST (.mat) frame. Caches written before 2026-11-19 stored it A/P reversed
    (atlas_for() lacked the flip); those are flipped here, newer caches carry atlas_fixed and are used as is."""
    v = vol(s)
    g = v['gm_atlas'].astype(int)
    return g if 'atlas_fixed' in v else np.flip(g, axis=0)
