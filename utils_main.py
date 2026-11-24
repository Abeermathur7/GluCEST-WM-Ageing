"""Data loading, math, and plotting helpers for the GluCEST reproducibility notebook
(Glucest_ANalysis_figures.ipynb).

Layout of this module:
  - Config: file paths, subject registries, ROI label abbreviations
  - Scan loading
  - Tissue / ROI dataframe builders
  - Core stats (COV, regression + CI, partial correlation)
  - Plot style
  - Figures: distribution comparison, slice comparison, age scatter,
    per-ROI age grid, partial-correlation panels, COV summary tables
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import copy
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns
from scipy.stats import linregress, pearsonr, t as t_dist
import statsmodels.formula.api as smf
# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def _on_drive(rel):
    """the data drive mounts under different letters; use the first that has rel (F: kept as the default)"""
    return next((Path(f'{L}:/') / rel for L in 'FEDGH' if (Path(f'{L}:/') / rel).exists()), Path('F:/') / rel)


DATA_DIR = _on_drive('GluCEST_CESTCORRECTEDMTR_data')                       # flat dir of .mat scans
BASE     = _on_drive('GlucestData_withNifit/7T_GLUCEST_Repeatability')      # GM/WM NIfTI atlases

TISSUE    = {1: 'CSF', 2: 'GM', 3: 'WM'}
CEST_FILL = -7

SCANS = {
    'P11 – Scan 1': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P11_01_SEGMENTED.mat',
    'P11 – Scan 2': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P11_02_SEGMENTED.mat',
    'P11 – Scan 3': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P11_03_SEGMENTED.mat',
    'P12 – Scan 1': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P12_01_SEGMENTED.mat',
    'P12 – Scan 2': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P12_02_SEGMENTED.mat',
    'P06 – Scan 1': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P06_01_SEGMENTED.mat',
    'P06 – Scan 2': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P06_02_SEGMENTED.mat',
    'P03 – Scan 1': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P03_01_SEGMENTED.mat',
    'P03 – Scan 2': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P03_02_SEGMENTED.mat',
    'P01 – Scan 1': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P01_01_SEGMENTED.mat',
    'P01 – Scan 2': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P01_02_SEGMENTED.mat',
    'P01 – Scan 3': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P01_03_SEGMENTED.mat',
    'P02 – Scan 1': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P02_01_SEGMENTED.mat',
    'P02 – Scan 2': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P02_02_SEGMENTED.mat',
    'P02 – Scan 3': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P02_03_SEGMENTED.mat',
    'P02 – Scan 4': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P02_04_SEGMENTED.mat',
    'P15 – Scan 1': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P15_01_SEGMENTED.mat',
    'P15 – Scan 2': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P15_02_SEGMENTED.mat',
    'P05 – Scan 1': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P05_01_SEGMENTED.mat',
    'P05 – Scan 2': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P05_02_SEGMENTED.mat',
    'P18 – Scan 1': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P18_01_SEGMENTED.mat',
    'P08 – Scan 1': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P08_01_SEGMENTED.mat',
    'P08 – Scan 2': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P08_02_SEGMENTED.mat',
    'P16 – Scan 1': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P16_01_SEGMENTED.mat',
    'P16 – Scan 2': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P16_02_SEGMENTED.mat',
    'P10 – Scan 1': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P10_01_SEGMENTED.mat',
    'P13 – Scan 1': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P13_01_SEGMENTED.mat',
    'P14 – Scan 1': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P14_01_SEGMENTED.mat',
    'P14 – Scan 2': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P14_02_SEGMENTED.mat',
    'P09 – Scan 1': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P09_01_SEGMENTED.mat',
    'P17 – Scan 1': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P17_01_SEGMENTED.mat',
    'P07 – Scan 1': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P07_01_SEGMENTED.mat',
    'P07 – Scan 1': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P07_01_SEGMENTED.mat',
    'P04 – Scan 1': 'CEST3DNEW_prep_tfl_3D_glucest_GLUCEST_P04_01_SEGMENTED.mat',
}

# Age (years) at Scan 1, keyed the same way as SCANS.
AGE_MAP = {
    'P11 – Scan 1': 47, 'P12 – Scan 1': 47, 'P06 – Scan 1': 30, 'P01 – Scan 1': 23,
    'P02 – Scan 1': 24, 'P15 – Scan 1': 63, 'P18 – Scan 1': 78, 'P03 – Scan 1': 25,
    'P05 – Scan 1': 27, 'P08 – Scan 1': 34, 'P16 – Scan 1': 63, 'P10 – Scan 1': 42,
    'P13 – Scan 1': 54, 'P14 – Scan 1': 57, 'P09 – Scan 1': 38,
    'P17 – Scan 1': 73, 'P07 – Scan 1': 31,'P04 – Scan 1' : 26,
}

# scan label -> (top-level subject folder under BASE, age). The GM/WM atlas is
# found under this folder via rglob, so the exact session-subfolder depth
# doesn't matter.
SUBJECT_MAP = {
    'P11 – Scan 1': ('S06_rep1',                    47),
    'P12 – Scan 1': ('S007_S007',                   47),
    'P06 – Scan 1': ('S095',                        30),
    'P01 – Scan 1': ('S167_rep1',                   23),
    'P15 – Scan 1': ('S002_rep1',                   63),
    'P03 – Scan 1': ('P03_S00_rep1',                 25),
    'P05 – Scan 1': ('S092',                        27),
    'P08 – Scan 1': ('S138_rep1',                   34),
    'P16 – Scan 1': ('S004',                        63),
    'P10 – Scan 1': ('S008',                        42),
    'P18 – Scan 1': ('S145_GluCEST_DFS_5_18_2026',  78),
    'P14 – Scan 1': ('S076',                        57),
    'P09 – Scan 1': ('S088',                        38),
    'P02 – Scan 1': ('S179_rep1/SC7T_20260422_S197', 24),
    'P13 – Scan 1': ('S009',                        54),
    'P07 – Scan 1': ('S170',                        31),
    'P17 – Scan 1': ('S096',                        73),
    'P04 – Scan 1': ('S148_rep1',                   26),
}

# (initials, scan#) -> top-level folder under BASE, for subjects with a
# second/third repeat scan that also needs its own atlas located.
SCAN_FOLDERS = {
    ('P11', 1): 'S06_rep1',          ('P11', 2): 'S06_rep2',
    ('P12', 1): 'S007_S007',         ('P12', 2): 'S007_rep2_May4th',
    ('P06', 1): 'S095',              ('P06', 2): 'S095_rep2',
    ('P01', 2): 'S167_rep2',         ('P01', 3): 'S167_rep3',   # S0-matched 2026-10-21 (was shifted: rep1/rep2)
    ('P15', 1): 'S002_rep1',         ('P15', 2): 'S002_rep2',
    ('P16', 1): 'S004',              ('P16', 2): 'S004_06092026',
    ('P03', 1): 'P03_S00_rep1',       ('P03', 2): 'P03_S00_rep2',
    ('P05', 1): 'S092',              ('P05', 2): 'S092_S002',
    ('P08', 1): 'S138_rep1',         ('P08', 2): 'S138_rep2',
    ('P14', 1): 'S076',              ('P14', 2): 'S076_rep2',
}

# SOURCE OF TRUTH for scan-rescan repeatability (same as the notebook's REPRO_SLICES cell):
#   initials: (scan # A, slice A, scan # B, slice B) -- 0-based slices into the 12-slice volume.
# The slice pair sets the A->B offset; REPRO_REGISTRY applies that offset to every slice in the
# slab (edge slices 0 and 11 dropped), so CoV/ICC use all overlapping slices.
REPRO_SLICES = {
    'P01': (2, 6, 3, 6),
    'P03': (1, 6, 2, 6),
    'P05': (1, 6, 2, 7),
    'P11': (1, 6, 2, 6),   # was z7; S0/WM-matched 2026-10-07
    'P12': (1, 6, 2, 7),
    'P06': (1, 5, 2, 6),   # was z7
    'P08': (1, 5, 2, 5),   # was z7
    'P14': (1, 6, 2, 5),   # was z7; low-confidence (in-plane head position differs)
    'P15': (1, 6, 2, 6),
    'P16': (1, 6, 2, 6),
}


def repro_registry(repro=None, lo=1, hi=11):
    """REPRO_SLICES -> REGISTRY-format tuples (initials, ID, z-range A, z-range B, all, all, scan A, scan B)."""
    rows = []
    for ini, (nA, zA, nB, zB) in (repro or REPRO_SLICES).items():
        off = zB - zA
        a0, a1 = max(lo, lo - off), min(hi, hi - off)
        rows.append((ini, ini, slice(a0, a1), slice(a0 + off, a1 + off), slice(None), slice(None), nA, nB))
    return rows


REPRO_REGISTRY = repro_registry()

# Per-subject slice-alignment for the primary two-scan reproducibility comparison:
# (initials, subject ID, z-trim scan A, z-trim scan B, slice window A, slice
#  window B, scan# A, scan# B). Drives plot_slice_comparison, compute_registry_cov,
# and build_subreg_cov so every figure/table uses identical slice indexing.
REGISTRY = [
    ('P11', 'S006', slice(1, -1), slice(None, -1), slice(1, 10), slice(1, 10), 1, 2),
    ('P12', 'S007', slice(1, -1), slice(1, -1),    slice(1, 10), slice(1, 10), 1, 2),
    ('P06', 'S095', slice(1, -1), slice(1, -1),    slice(0, 10), slice(1, 11), 1, 2),
    ('P01', 'S167', slice(1, -1), slice(1, -1),    slice(1, 11), slice(1, 11), 2, 3),
    ('P02', 'S179', slice(1, -1), slice(1, -1),    slice(1, 10), slice(1, 10), 3, 4),
    ('P15', 'S002', slice(1, -1), slice(1, -1),    slice(1, 10), slice(0, 10), 1, 2),
    ('P16', 'S004', slice(1, -1), slice(1, -1),    slice(1, 10), slice(0, 10), 1, 2),
    ('P03', 'S000', slice(1, -1), slice(1, -1),    slice(1, 10), slice(0, 10), 1, 2),
    ('P05', 'S092', slice(1, -1), slice(1, -1),    slice(1, 10), slice(0, 10), 1, 2),
    ('P14', 'S076', slice(1, -1), slice(1, -1),    slice(1, 10), slice(0, 10), 1, 2),
]

# ROI-label shortening tables (Nature-Comms-style figure/table space savers).
ABBREV_WM = [
    ('corpus callosum',             'CC'),
    ('corona radiata',              'CR'),
    ('longitudinal fasciculus',     'LF'),
    ('fronto-occipital fasciculus', 'FOF'),
    ('Cingulum (cingulate gyrus)',  'Cingulum'),
    ('internal capsule',            'IC'),
]
ABBREV_GM = [
    ('Superior',        'Sup.'),
    ('Middle',          'Mid.'),
    ('Inferior',        'Inf.'),
    ('Frontal',         'Front.'),
    ('Temporal',        'Temp.'),
    ('Occipital',       'Occ.'),
    ('Parietal',        'Par.'),
    ('Cingulate',       'Cing.'),
    ('Parahippocampal', 'Parahipp.'),
    ('Precentral',      'Precent.'),
    ('Postcentral',     'Postcent.'),
    ('Precuneous',      'Precun.'),
    ('Lingual',         'Ling.'),
    ('Gyrus',           'Gy.'),
    ('Cortex',          'Ctx.'),
    (' (anterior)',     ' (ant)'),
    (' (posterior)',    ' (pos)'),
]

TCOL = {'CSF': '#4DBBD5', 'GM': '#E64B35', 'WM': '#00A087', 'Whole Brain': '#7B4FA6'}
# GROUP_COL = {'Corpus Callosum and Cincgulum WM': '#3C5488', 'Corona Radiata WM': '#E07A1A'}
GROUP_COL = {'Corpus Callosum & Cingulum': '#3C5488','Corona Radiata': '#E07A1A'}

def scan_key(initials: str, n: int) -> str:
    """'P11', 1 -> 'P11 – Scan 1' (en-dash, matches SCANS keys)."""
    return f'{initials} – Scan {n}'


def shorten_wm(name: str) -> str:
    for old, new in ABBREV_WM:
        name = name.replace(old, new)
    return name


def shorten_gm(name: str) -> str:
    for old, new in ABBREV_GM:
        name = name.replace(old, new)
    return name


def wm_group(roi_label: str) -> str | None:
    """Classify a WM ROI label as 'Corpus Callosum & Cingulum', 'Corona
    Radiata', or None (excluded from either bucket). Return values must match
    GROUP_COL's keys exactly."""
    l = roi_label.lower()
    ## Change regions here
    if 'corpus callosum'in l :
        #or 'cingulum' in l:
        return 'Corpus Callosum & Cingulum'
    if 'corona radiata' in l:
        return 'Corona Radiata'
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Scan loading
# ─────────────────────────────────────────────────────────────────────────────

def load_scan(fname, data_dir: Path = DATA_DIR):
    """Load a segmented GluCEST .mat 'output' struct; falls back to h5py for
    MATLAB v7.3 files."""
    try:
        raw = sio.loadmat(str(data_dir / fname), squeeze_me=True, struct_as_record=False)
        return raw['output']
    except Exception:
        import h5py
        import types
        with h5py.File(data_dir / fname, 'r') as f:
            grp = f['output']
            out = types.SimpleNamespace()
            for k in grp.keys():
                out.__dict__[k] = np.array(grp[k]).T
            return out


# ─────────────────────────────────────────────────────────────────────────────
# Core stats
# ─────────────────────────────────────────────────────────────────────────────

def pct_cov(mu1: float, mu2: float) -> float:
    """% coefficient of variation between two scan means: |mu1-mu2|/sqrt(2)
    over the mean of |mu1|,|mu2|, as a percent. The notebook's most-repeated
    formula (reproducibility COV between exactly two scans)."""
    return (abs(mu1 - mu2) / np.sqrt(2)) / ((abs(mu1) + abs(mu2)) / 2) * 100


def pct_cv(vals) -> float:
    """Coefficient of variation across >=2 scan means (equals pct_cov for n=2)."""
    vals = np.asarray(vals, dtype=float)
    return np.std(vals, ddof=1) / np.mean(np.abs(vals)) * 100


def add_regression_ci(ax, x, y, color, extend=2.0, lw=1.2, alpha=0.15):
    """Fit + plot a linear regression line with a 95% CI ribbon on `ax`.
    Returns (r, p) from a Pearson correlation of the raw x, y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m, b = np.polyfit(x, y, 1)
    x_fit = np.linspace(x.min() - extend, x.max() + extend, 300)
    y_fit = m * x_fit + b
    n, x_m = len(x), x.mean()
    se = np.sqrt(np.sum((y - (m * x + b)) ** 2) / (n - 2))
    ci = (t_dist.ppf(0.975, n - 2) * se
          * np.sqrt(1 / n + (x_fit - x_m) ** 2 / np.sum((x - x_m) ** 2)))
    ax.plot(x_fit, y_fit, color=color, lw=lw, zorder=3)
    ax.fill_between(x_fit, y_fit - ci, y_fit + ci, color=color, alpha=alpha, zorder=2, linewidth=0)
    return pearsonr(x, y)


def partial_corr(y, x, covariate):
    """Regress `covariate` out of both y and x (linear fit), then correlate the
    residuals. Returns (resid_y, resid_x, r, p) with p from a Student-t test on
    the partial correlation (df = n-3)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    covariate = np.asarray(covariate, dtype=float)
    m, b = np.polyfit(covariate, y, 1)
    resid_y = y - (m * covariate + b)
    m, b = np.polyfit(covariate, x, 1)
    resid_x = x - (m * covariate + b)
    r, _ = pearsonr(resid_y, resid_x)
    n = len(resid_y)
    t_stat = r * np.sqrt((n - 3) / (1 - r ** 2))
    p = 2 * t_dist.sf(abs(t_stat), df=n - 3)
    return resid_y, resid_x, r, p


def p_str(p: float) -> str:
    return f'p = {p:.3f}' if p >= 0.001 else 'p < 0.001'


# ─────────────────────────────────────────────────────────────────────────────
# Group-difference modeling (e.g. does Age predict value_col differently across
# groups, such as Expected vs Atypical WM in t1_grp?)
# ─────────────────────────────────────────────────────────────────────────────

def _group_term(group_col, reference):
    """patsy term for group_col, optionally forcing which level is the
    Treatment-coding reference/baseline (default: patsy's alphabetical pick)."""
    if reference is None:
        return group_col
    return f"C({group_col}, Treatment(reference='{reference}'))"


def fit_age_group_interaction(df, value_col='GluCEST', group_col='Group', covariates=('T1',),
                               reference=None):
    """OLS interaction model testing whether the Age slope on `value_col`
    differs between groups (e.g. Expected WM vs Atypical WM). Fits
    `value_col ~ Age * group_col + covariates`. Returns the fitted
    statsmodels result.

    `reference` picks which group level is the Treatment-coding baseline: the
    bare 'Age' coefficient is that group's own slope, and the interaction term
    is the *other* group's slope relative to it. Default (None) is patsy's
    alphabetical pick. See simple_slopes_by_group to get every group's own
    slope + p-value without picking one by hand.

    Caveat: treats every row as independent. If `df` has one row per subject
    per group (e.g. t1_grp, where each subject contributes both an 'Expected
    WM' and an 'Atypical WM' row), that's a repeated-measures structure this
    model does NOT account for -- see fit_age_group_interaction_mixed."""
    import statsmodels.formula.api as smf
    cov_terms = f" + {' + '.join(covariates)}" if covariates else ''
    formula = f'{value_col} ~ Age * {_group_term(group_col, reference)}{cov_terms}'
    return smf.ols(formula, data=df).fit()


def fit_age_group_interaction_mixed(df, value_col='GluCEST', group_col='Group',
                                     covariates=('T1',), subject_col='Subject', reference=None):
    """Mixed-effects alternative to fit_age_group_interaction: identical fixed
    effects (including the same `reference` re-leveling option), plus a random
    intercept per `subject_col` -- the statistically correct choice when each
    subject contributes one row per group."""
    import statsmodels.formula.api as smf
    cov_terms = f" + {' + '.join(covariates)}" if covariates else ''
    formula = f'{value_col} ~ Age * {_group_term(group_col, reference)}{cov_terms}'
    return smf.mixedlm(formula, data=df, groups=df[subject_col]).fit()


def fit_age_group_interaction_gee(df, value_col='GluCEST', group_col='Group',
                                   covariates=('T1',), subject_col='Subject',
                                   reference=None, cov_struct=None):
    """GEE alternative to fit_age_group_interaction_mixed: identical fixed-effects
    formula (`value_col ~ Age * group_col + covariates`), but instead of a random
    intercept per subject, treats each subject's rows as one correlated cluster
    (default working correlation: exchangeable) and estimates population-averaged
    effects with cluster-robust (sandwich) standard errors.

    Prefer GEE over the mixed model when you want population-averaged effects
    (what happens to the 'typical' subject) rather than subject-specific ones,
    or when you'd rather not assume a distribution for the random effects --
    GEE only assumes a working correlation structure, and its standard errors
    stay valid even if that structure is mis-specified."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    cov_struct = cov_struct if cov_struct is not None else sm.cov_struct.Exchangeable()
    cov_terms = f" + {' + '.join(covariates)}" if covariates else ''
    formula = f'{value_col} ~ Age * {_group_term(group_col, reference)}{cov_terms}'
    return smf.gee(formula, groups=df[subject_col], data=df, cov_struct=cov_struct).fit()


def get_interaction_terms(model, group_col='Group'):
    """Pull the Age:group_col interaction row(s) out of a fitted model (OLS or
    mixed-effects). statsmodels names the term after whichever group level is
    NOT picked as reference, e.g. 'Age:Group[T.Expected WM]' -- returns
    {term_name: (coef, p_value)} for every matching term."""
    needle = f'Age:{group_col}'
    hits = [name for name in model.pvalues.index if needle in name]
    if not hits:
        raise KeyError(f"No interaction term containing '{needle}' found. "
                        f"Available terms: {list(model.pvalues.index)}")
    return {name: (model.params[name], model.pvalues[name]) for name in hits}


def simple_slopes_by_group(df, value_col='GluCEST', group_col='Group', covariates=('T1',),
                            groups=None, mixed=False, gee=False, subject_col='Subject'):
    """Each group's own Age-slope with its own p-value, obtained by refitting
    the interaction model once per group as the Treatment-coding reference (the
    'relevel and rerun' trick -- see fit_age_group_interaction's `reference`
    arg) rather than reading only the reference group's slope off one table.
    Set mixed=True to use fit_age_group_interaction_mixed, or gee=True to use
    fit_age_group_interaction_gee, instead of OLS (mixed and gee are mutually
    exclusive). Returns one row per group: slope, std_err, p_value."""
    if mixed and gee:
        raise ValueError('simple_slopes_by_group: pass at most one of mixed=True, gee=True')
    groups = list(groups) if groups is not None else sorted(df[group_col].dropna().unique())
    fit_fn = (fit_age_group_interaction_mixed if mixed else
              fit_age_group_interaction_gee if gee else fit_age_group_interaction)
    rows = []
    for grp in groups:
        kwargs = dict(value_col=value_col, group_col=group_col, covariates=covariates, reference=grp)
        if mixed or gee:
            kwargs['subject_col'] = subject_col
        model = fit_fn(df, **kwargs)
        rows.append({'Group': grp, 'slope': model.params['Age'],
                     'std_err': model.bse['Age'], 'p_value': model.pvalues['Age']})
    return pd.DataFrame(rows)


def report_simple_slopes(slopes_df):
    """Print each group's slope with a plain significance verdict, from
    simple_slopes_by_group's output. Returns slopes_df unchanged."""
    for _, row in slopes_df.iterrows():
        verdict = 'significant' if row['p_value'] < 0.05 else 'not significant'
        print(f"{row['Group']}: slope = {row['slope']:.4f} +/- {row['std_err']:.4f}, "
              f"{p_str(row['p_value'])}  ->  {verdict}")
    return slopes_df


def plot_model_implied_lines(model, df, x_col='Age', y_col='GluCEST', group_col='Group',
                              groups=None, colors=None, covariates=('T1',),
                              show_points=True, ylabel=None, title='', savepath=None):
    """Plot a fitted interaction model's predicted y_col vs x_col, one line per
    group, holding each covariate fixed at its sample mean. This is the most
    direct visualization of what an Age * Group interaction coefficient means:
    if the lines diverge, the groups age differently; if they're near-parallel,
    they don't. Works with both OLS and mixed-effects results from
    fit_age_group_interaction(_mixed) -- prediction uses fixed effects only."""
    groups = list(groups) if groups is not None else sorted(df[group_col].dropna().unique())
    colors = colors or GROUP_COL
    cov_means = {c: df[c].mean() for c in covariates}
    ylabel = ylabel or y_col

    with paper_style():
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        x_range = np.linspace(df[x_col].min(), df[x_col].max(), 100)
        for grp in groups:
            col = colors.get(grp)
            if show_points:
                sub = df[df[group_col] == grp]
                ax.scatter(sub[x_col], sub[y_col], color=col, s=22, alpha=0.55,
                           linewidths=0.4, edgecolors='white', zorder=3)
            pred_df = pd.DataFrame({x_col: x_range, group_col: grp,
                                     **{c: cov_means[c] for c in covariates}})
            pred = model.predict(pred_df)
            ax.plot(x_range, pred, color=col, lw=2.0, label=grp, zorder=4)
        ax.set_xlabel(x_col, fontsize=10, labelpad=3)
        ax.set_ylabel(ylabel, fontsize=10, labelpad=3)
        ax.set_title(title,fontsize=10, fontweight='bold', pad=4)
        ax.legend(frameon=False, fontsize=10, loc='best')
        clean_axes(ax)
        plt.tight_layout()
        if savepath:
            plt.savefig(savepath, bbox_inches='tight')
        plt.show()


def report_interaction(model, group_col='Group'):
    """Print each Age x group_col interaction coefficient with a plain-language
    verdict on statistical significance (p < 0.05). Returns the same dict as
    get_interaction_terms."""
    terms = get_interaction_terms(model, group_col=group_col)
    for name, (coef, p) in terms.items():
        verdict = ('significant -- supports differential aging between groups' if p < 0.05
                   else 'not significant -- no confirmed evidence of differential aging')
        print(f'{name}: beta = {coef:.4f}, {p_str(p)}  ->  {verdict}')
    return terms


# ─────────────────────────────────────────────────────────────────────────────
# Effect size & power analysis
# ─────────────────────────────────────────────────────────────────────────────

def cohens_d(x1, x2, paired=False):
    """Cohen's d effect size for the difference between two samples.

    paired=False (independent samples -- e.g. two different groups of
    subjects): d = (mean(x1) - mean(x2)) / pooled_sd, the classic two-sample d.

    paired=True (same subjects measured twice -- e.g. one subject's Corpus
    Callosum & Cingulum GluCEST vs their own Corona Radiata GluCEST; x1[i] and
    x2[i] must be the same subject, same order, same length):
        d = mean(x1 - x2) / std(x1 - x2)
    This is the effect size a paired t-test / repeated-measures design
    actually tests, and is usually larger than the unpaired d on the same data
    because subtracting each subject's own baseline removes between-subject
    variance that has nothing to do with the effect you're testing."""
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    if paired:
        if len(x1) != len(x2):
            raise ValueError("paired Cohen's d needs equal-length, same-order arrays")
        diff = x1 - x2
        return diff.mean() / diff.std(ddof=1)
    n1, n2 = len(x1), len(x2)
    pooled_var = ((n1 - 1) * x1.var(ddof=1) + (n2 - 1) * x2.var(ddof=1)) / (n1 + n2 - 2)
    return (x1.mean() - x2.mean()) / np.sqrt(pooled_var)


def interpret_effect_size(d):
    """Cohen's (1988) rule-of-thumb label for |d|: <0.2 negligible, <0.5
    small, <0.8 medium, else large."""
    ad = abs(d)
    if ad < 0.2:
        return 'negligible'
    if ad < 0.5:
        return 'small'
    if ad < 0.8:
        return 'medium'
    return 'large'


def power_for_d(d, n, alpha=0.05, paired=True):
    """Achieved (post-hoc) power to detect effect size `d` at sample size `n`
    and significance `alpha` (two-sided t-test). `n` is the number of pairs
    when paired=True (matches cohens_d(..., paired=True)), or the per-group n
    (equal group sizes assumed) when paired=False."""
    from statsmodels.stats.power import TTestPower, TTestIndPower
    if paired:
        return TTestPower().solve_power(effect_size=abs(d), nobs=n, alpha=alpha, power=None)
    return TTestIndPower().solve_power(effect_size=abs(d), nobs1=n, alpha=alpha, power=None, ratio=1.0)


def n_for_d_power(d, power=0.80, alpha=0.05, paired=True):
    """Sample size needed to detect effect size `d` at `power` (default 80%)
    and `alpha` (default 0.05, two-sided). Returns n as pairs (paired=True) or
    per-group n (paired=False), rounded up to the next whole subject."""
    from statsmodels.stats.power import TTestPower, TTestIndPower
    if paired:
        n = TTestPower().solve_power(effect_size=abs(d), nobs=None, alpha=alpha, power=power)
    else:
        n = TTestIndPower().solve_power(effect_size=abs(d), nobs1=None, alpha=alpha, power=power, ratio=1.0)
    return int(np.ceil(n))


def report_power(d, n, alpha=0.05, paired=True):
    """Print Cohen's d (with its plain-language size label), the achieved
    power at the observed n, and the n that would be needed for 80% power at
    this effect size. Returns {'d', 'label', 'achieved_power', 'n_for_80pct'}."""
    label = interpret_effect_size(d)
    achieved = power_for_d(d, n, alpha=alpha, paired=paired)
    needed = n_for_d_power(d, power=0.80, alpha=alpha, paired=paired)
    kind = 'pairs' if paired else 'per group'
    print(f"Cohen's d = {d:.3f} ({label})")
    print(f'Observed n = {n} {kind}  ->  achieved power = {achieved:.1%}  (alpha = {alpha})')
    print(f'n needed for 80% power at this effect size: {needed} {kind}')
    return {'d': d, 'label': label, 'achieved_power': achieved, 'n_for_80pct': needed}


# ─────────────────────────────────────────────────────────────────────────────
# Plot style
# ─────────────────────────────────────────────────────────────────────────────

_PAPER_RC = {
    'font.family': 'Arial', 'font.size': 7,
    'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
}


@contextmanager
def paper_style(**overrides):
    """Context manager for the notebook's figure style. Replaces the repeated
    `mpl.rcParams.update({...}) ... mpl.rcParams.update(mpl.rcParamsDefault)`
    pattern with `with paper_style(): ...` (auto-restores on exit)."""
    with mpl.rc_context({**_PAPER_RC, **overrides}):
        yield


def clean_axes(ax):
    ax.tick_params(axis='both', which='both', length=3, width=0.6)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('bottom', 'left'):
        ax.spines[sp].set_linewidth(0.6)


# ─────────────────────────────────────────────────────────────────────────────
# Tissue / ROI dataframe builders
# ─────────────────────────────────────────────────────────────────────────────

def build_tissue_roi_df(scan_specs, tissue=TISSUE):
    """scan_specs: iterable of (label, out, z_trim_slice, roi_window_slice).
    Returns (tissue_df, whole_brain_roi_df): long-form per-voxel CEST(%) tables,
    replicating the notebook's repeated per-subject loading loop."""
    rows, roi_rows = [], []
    for label, out, zt, s in scan_specs:
        c = np.asarray(out.CESTmap, dtype=float)[:, :, zt]
        seg = np.asarray(out.segmask, dtype=int)[:, :, zt]
        roi = np.asarray(out.roimask, dtype=int)[:, :, zt]
        idx = np.arange(*s.indices(c.shape[2]))
        cest, seg, roi = c[:, :, idx], seg[:, :, idx], roi[:, :, idx]
        valid = np.isfinite(cest) & (roi > 0) & (cest > -10) & (cest < 30) & (cest != 0)
        for tid, tname in tissue.items():
            mask = (seg == tid) & valid
            rows += [{'Scan': label, 'Tissue': tname, 'CEST (%)': v} for v in cest[mask]]
        roi_rows += [{'Scan': label, 'CEST (%)': v} for v in cest[(roi > 0) & valid]]
    return pd.DataFrame(rows), pd.DataFrame(roi_rows)


def build_all_scans_df(scans=SCANS, tissue=TISSUE):
    """Whole-dataset per-voxel CEST(%) table (used for the top-level whole-brain
    histogram and the inter-subject comparison section)."""
    records = []
    for label, fname in scans.items():
        out = load_scan(fname)
        cest = np.asarray(out.CESTmap, dtype=float)[:, :, 1:-1]
        seg = np.asarray(out.segmask, dtype=int)[:, :, 1:-1]
        for tid, tname in tissue.items():
            mask = (seg == tid) & np.isfinite(cest) & (cest >= -10) & (cest <= 30) & (cest != 0)
            for v in cest[mask]:
                records.append({'Scan': label, 'Tissue': tname, 'CEST (%)': v})
    return pd.DataFrame(records)


def print_tissue_cov(df, scan_a, scan_b, tissues=('CSF', 'GM', 'WM'), label=None):
    print(f'Tissue COV {label or f"{scan_a} vs {scan_b}"}:')
    for tissue in tissues:
        mu1 = df[(df['Scan'] == scan_a) & (df['Tissue'] == tissue)]['CEST (%)'].mean()
        mu2 = df[(df['Scan'] == scan_b) & (df['Tissue'] == tissue)]['CEST (%)'].mean()
        print(f'  {tissue}: {pct_cov(mu1, mu2):.2f}%')


def print_whole_brain_cov(roi_df, scan_a, scan_b):
    mu1 = roi_df[roi_df['Scan'] == scan_a]['CEST (%)'].mean()
    mu2 = roi_df[roi_df['Scan'] == scan_b]['CEST (%)'].mean()
    print(f'Whole Brain COV: {pct_cov(mu1, mu2):.2f}%')


# ─────────────────────────────────────────────────────────────────────────────
# ROI atlas helpers
# ─────────────────────────────────────────────────────────────────────────────

def _roi_labels(ndir: Path, roi_type: str) -> dict:
    lbl = {}
    txt = ndir / f'CESTslab_{roi_type}rois.txt'
    if txt.exists():
        for line in txt.read_text(encoding='utf-8', errors='replace').splitlines():
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                try:
                    lbl[int(parts[0])] = parts[1].strip()
                except ValueError:
                    pass
    return lbl


def atlas_to_mat_frame(vol):
    """Map a CESTslab NIfTI volume onto the transposed .mat frame used here
    (np.transpose(mat, (1, 0, 2))). The NIfTI relates to the raw .mat arrays by
    transpose(1,0,2) PLUS an anterior/posterior flip, so in this frame A/P is
    axis 1. Without it, JHU ROIs are A/P-mirrored (genu<->splenium, ACR<->PCR).
    Verified per scan: WM Dice 0.75-0.85 with the flip vs 0.51-0.65 without
    (nature_figures_agent3/_cache/orientation_check.csv)."""
    return vol[:, ::-1, :]


def load_roi_atlas(ndir: Path, roi_type: str):
    """Load a WM/GM ROI atlas (full volume, untrimmed) + its int->name label map.
    Returned in NIfTI space (utils_wmh pairs it with other NIfTIs); wrap with
    atlas_to_mat_frame() before combining it with .mat data."""
    nii = ndir / f'CESTslab_{roi_type}rois.nii'
    if not nii.exists():
        return None, {}
    vol = nib.load(str(nii)).get_fdata().astype(int)
    return vol, _roi_labels(ndir, roi_type)


def locate_nifti_dirs(scan_folders=SCAN_FOLDERS, base=BASE, atlas='CESTslab_WMrois.nii'):
    """Resolve each (initials, scan#) -> folder to the directory containing its
    WM/GM ROI atlas (found via rglob), keyed by scan label."""
    nifti_dirs = {}
    for (initials, n), folder in scan_folders.items():
        root = base / folder
        if not root.is_dir():
            print(f'✗ folder missing   {initials} {n}  ->  {root}')
            continue
        hits = sorted(root.rglob(atlas))
        if not hits:
            print(f'✗ no {atlas}        {initials} {n}  ->  {root}')
            continue
        if len(hits) > 1:
            print(f'! {len(hits)} matches for {initials} {n}; using first')
        nifti_dirs[scan_key(initials, n)] = hits[0].parent
        print(f'✓ {initials} {n}  ->  {hits[0].parent}')
    print(f'\nLoaded {len(nifti_dirs)} / {len(scan_folders)} scans')
    return nifti_dirs


def _subject_map_atlas_paths(subject_map, base, roi_type):
    """Yield (key, age, atlas_path, label_dir) for each SUBJECT_MAP entry whose
    WM/GM atlas can be found via rglob under its subject folder.

    When multiple atlases are found (e.g. duplicate nifti/XNIFTI export
    folders), some may be empty placeholders and/or missing their label .txt.
    We pick the atlas that actually has nonzero ROI data, and separately look
    across all candidate folders for a label .txt to pair with it (label
    numbering is a fixed atlas scheme, so it's valid regardless of which
    export folder it came from). Getting this wrong silently produces bare
    numeric ROI labels that fail to match group classifiers like wm_group."""
    for key, (folder, age) in subject_map.items():
        hits = sorted((base / folder).rglob(f'CESTslab_{roi_type}rois.nii'))
        if not hits:
            print(f'  SKIP {key}: {roi_type} ROI file not found under {base / folder}')
            continue
        chosen = hits[0]
        label_dir = chosen.parent
        if len(hits) > 1:
            populated = [h for h in hits if np.any(nib.load(str(h)).get_fdata() > 0)]
            if populated:
                chosen = populated[0]
            label_dir = chosen.parent
            if not (label_dir / f'CESTslab_{roi_type}rois.txt').exists():
                labeled_dirs = [h.parent for h in hits
                                if (h.parent / f'CESTslab_{roi_type}rois.txt').exists()]
                if labeled_dirs:
                    label_dir = labeled_dirs[0]
            note = '' if (label_dir / f'CESTslab_{roi_type}rois.txt').exists() \
                else '  (WARNING: no label .txt found for this subject)'
            print(f'  ! {key}: {len(hits)} {roi_type} atlases under {folder}; '
                  f'using {chosen} (labels from {label_dir}){note}')
        yield key, age, chosen, label_dir


def build_roi_glucest_df(roi_type, subject_map=SUBJECT_MAP, base=BASE, min_voxels=5):
    """Per-ROI mean GluCEST(%) vs Age, for WM or GM ROIs."""
    rows = []
    for key, age, atlas_path, ndir in _subject_map_atlas_paths(subject_map, base, roi_type):
        out = load_scan(SCANS[key])
        cest = np.transpose(np.asarray(out.CESTmap, float)[:, :, 1:-1], (1, 0, 2))
        roi = np.transpose(np.asarray(out.roimask, int)[:, :, 1:-1], (1, 0, 2))
        valid = np.isfinite(cest) & (roi > 0) & (cest > -10) & (cest < 30) & (cest != 0)

        vol = atlas_to_mat_frame(nib.load(str(atlas_path)).get_fdata().astype(int))[:, :, 1:-1]
        lbl = _roi_labels(ndir, roi_type)
        subj = key.split(' ')[0]
        for rv in np.unique(vol[vol > 0]):
            mask = (vol == rv) & valid
            if mask.sum() < min_voxels:
                continue
            rows.append({'Subject': subj, 'Age': age, 'ROIvalue': int(rv),
                         'ROIlabel': lbl.get(int(rv), str(rv)),
                         'GluCEST': float(cest[mask].mean()), 'n_vox': int(mask.sum())})
    return pd.DataFrame(rows)


def build_roi_mt_df(roi_type, mt_field, subject_map=SUBJECT_MAP, base=BASE, min_voxels=5):
    """Per-ROI mean MT vs Age, for WM or GM ROIs. Unlike GluCEST, MT has no
    physiological range clamp applied -- just finite & nonzero."""
    rows = []
    for key, age, atlas_path, ndir in _subject_map_atlas_paths(subject_map, base, roi_type):
        out = load_scan(SCANS[key])
        mt = np.transpose(np.asarray(getattr(out, mt_field), float)[:, :, 1:-1], (1, 0, 2))
        roi = np.transpose(np.asarray(out.roimask, int)[:, :, 1:-1], (1, 0, 2))
        valid = np.isfinite(mt) & (roi > 0) & (mt != 0)

        vol = atlas_to_mat_frame(nib.load(str(atlas_path)).get_fdata().astype(int))[:, :, 1:-1]
        lbl = _roi_labels(ndir, roi_type)
        subj = key.split(' ')[0]
        for rv in np.unique(vol[vol > 0]):
            mask = (vol == rv) & valid
            if mask.sum() < min_voxels:
                continue
            rows.append({'Subject': subj, 'Age': age, 'ROIvalue': int(rv),
                         'ROIlabel': lbl.get(int(rv), str(rv)),
                         'MT': float(mt[mask].mean())})
    return pd.DataFrame(rows)


def build_roi_glucest_t1_df(roi_type, subject_map=SUBJECT_MAP, base=BASE, min_voxels=5):
    """Per-ROI mean GluCEST(%) AND T1 vs Age (feeds the T1-dependence / partial
    correlation sections), for WM or GM ROIs."""
    rows = []
    for key, age, atlas_path, ndir in _subject_map_atlas_paths(subject_map, base, roi_type):
        out = load_scan(SCANS[key])
        cest = np.transpose(np.asarray(out.CESTmap, float)[:, :, 1:-1], (1, 0, 2))
        t1 = np.transpose(np.asarray(out.T1img, float)[:, :, 1:-1], (1, 0, 2))
        roi = np.transpose(np.asarray(out.roimask, int)[:, :, 1:-1], (1, 0, 2))
        valid = (np.isfinite(cest) & np.isfinite(t1) & (roi > 0)
                 & (cest > -10) & (cest < 30) & (cest != 0) & (t1 > 0))

        vol = atlas_to_mat_frame(nib.load(str(atlas_path)).get_fdata().astype(int))[:, :, 1:-1]
        lbl = _roi_labels(ndir, roi_type)
        subj = key.split(' ')[0]
        for rv in np.unique(vol[vol > 0]):
            mask = (vol == rv) & valid
            if mask.sum() < min_voxels:
                continue
            rows.append({'Subject': subj, 'Age': age, 'ROIvalue': int(rv),
                         'ROIlabel': lbl.get(int(rv), str(rv)),
                         'GluCEST': float(cest[mask].mean()),
                         'T1': float(t1[mask].mean())})
    return pd.DataFrame(rows)


def build_region_glucest_t1_df(subject_map=SUBJECT_MAP, base=BASE, group_fn=wm_group, min_voxels=5):
    """Per-subject mean GluCEST, T1 and tissue fraction (Frac) for GM, WM and
    each WM tract group from `group_fn` (feeds the 'does GluCEST add anything
    beyond T1?' section). Tract groups pool all their atlas voxels (no CSF
    exclusion) so partial volume shows up in Frac instead of being hidden.

    Frac: for GM/WM, that tissue's share of all segmented slab voxels (atrophy
    proxy); for a tract group, the share of its voxels segmask calls WM
    (partial-volume proxy -- low Frac means CSF/GM contamination)."""
    rows = []
    for key, age, atlas_path, ndir in _subject_map_atlas_paths(subject_map, base, 'WM'):
        out = load_scan(SCANS[key])
        cest = np.transpose(np.asarray(out.CESTmap, float)[:, :, 1:-1], (1, 0, 2))
        t1 = np.transpose(np.asarray(out.T1img, float)[:, :, 1:-1], (1, 0, 2))
        seg = np.transpose(np.asarray(out.segmask, int)[:, :, 1:-1], (1, 0, 2))
        roi = np.transpose(np.asarray(out.roimask, int)[:, :, 1:-1], (1, 0, 2))
        valid = (np.isfinite(cest) & np.isfinite(t1) & (roi > 0)
                 & (cest > -10) & (cest < 30) & (cest != 0) & (t1 > 0))
        n_brain = (valid & (seg > 0)).sum()

        regions = {TISSUE[tid]: (seg == tid) & valid for tid in (2, 3)}
        vol = atlas_to_mat_frame(nib.load(str(atlas_path)).get_fdata().astype(int))[:, :, 1:-1]
        lbl = _roi_labels(ndir, 'WM')
        for rv in np.unique(vol[vol > 0]):
            group = group_fn(lbl.get(int(rv), str(rv)))
            if group:
                regions[group] = regions.get(group, np.zeros_like(valid)) | ((vol == rv) & valid)

        for name, mask in regions.items():
            if mask.sum() < min_voxels:
                continue
            frac = mask.sum() / n_brain if name in TISSUE.values() else (seg[mask] == 3).mean()
            rows.append({'Subject': key.split(' ')[0], 'Age': age, 'Region': name,
                         'GluCEST': float(cest[mask].mean()), 'T1': float(t1[mask].mean()),
                         'Frac': float(frac)})
    return pd.DataFrame(rows)


def build_tissue_age_df(field, value_col, age_map=AGE_MAP, tissue=TISSUE,
                         range_clamp=True, whole_brain=True):
    """Per-subject, per-tissue mean of `field` (a scan-struct attribute name,
    e.g. 'CESTmap') vs Age, with an appended 'Whole Brain' tissue row averaging
    the three tissue means. range_clamp=True applies the GluCEST physiological
    range filter (-10 < v < 30, v != 0); False just requires finite & nonzero
    (for MT, which has no such clamp in the notebook)."""
    rows = []
    for label, age in age_map.items():
        out = load_scan(SCANS[label])
        vals = np.asarray(getattr(out, field), float)[:, :, 1:-1]
        seg = np.asarray(out.segmask, int)[:, :, 1:-1]
        roi = np.asarray(out.roimask, int)[:, :, 1:-1]
        if range_clamp:
            valid = np.isfinite(vals) & (roi > 0) & (vals > -10) & (vals < 30) & (vals != 0)
        else:
            valid = np.isfinite(vals) & (roi > 0) & (vals != 0)
        initials = label.split(' – ')[0]
        for tid, tname in tissue.items():
            rows.append({'Subject': initials, 'Tissue': tname,
                         value_col: vals[(seg == tid) & valid].mean(), 'Age': age})

    means = pd.DataFrame(rows)
    if whole_brain:
        wb = means.groupby(['Subject', 'Age'])[value_col].mean().reset_index()
        wb['Tissue'] = 'Whole Brain'
        means = pd.concat([means, wb], ignore_index=True)
    return means


def detect_mt_field(sample_scan_label, age_map=AGE_MAP):
    """Auto-detect the MT field name on the scan struct (any attribute whose
    name contains 'mt'), falling back to 'MTRmap'."""
    probe = load_scan(SCANS[sample_scan_label if sample_scan_label else next(iter(age_map))])
    candidates = [k for k in vars(probe).keys() if 'mt' in k.lower()]
    print('Candidate MT fields found on struct:', candidates)
    field = candidates[0] if candidates else 'MTRmap'
    print('Using field:', field)
    return field


def diagnose_group_coverage(df_glucest, df_glucest_t1, group_fn=wm_group, group_col='Group'):
    """Find where subjects get silently dropped between a GluCEST-only ROI
    dataframe (e.g. df_wm, from build_roi_glucest_df) and the paired
    GluCEST+T1 version (e.g. df_wm_t1, from build_roi_glucest_t1_df) that
    t1_grp / the interaction models are built from. Two distinct things can
    cause a subject to vanish at the group stage:
      1. T1 validity is stricter than GluCEST-only -- df_glucest_t1 requires
         a valid T1img in the same voxels, so a subject present in
         df_glucest for a given ROI can be missing from df_glucest_t1 there.
      2. ROI label text doesn't match group_fn's substring rules (e.g.
         wm_group's 'corpus callosum' / 'cingulum' / 'corona radiata' checks),
         so the ROI silently gets group_col = None and is dropped.
    Prints a per-group before/after subject count and returns a dict with the
    exact subjects lost to cause (1) for each group."""
    df_glucest = df_glucest.copy()
    df_glucest_t1 = df_glucest_t1.copy()
    df_glucest[group_col] = df_glucest['ROIlabel'].apply(group_fn)
    df_glucest_t1[group_col] = df_glucest_t1['ROIlabel'].apply(group_fn)

    all_subjects = set(df_glucest['Subject'].unique())
    groups = sorted(set(df_glucest[group_col].dropna().unique())
                     | set(df_glucest_t1[group_col].dropna().unique()))
    report = {}
    for grp in groups:
        glu_subj = set(df_glucest[df_glucest[group_col] == grp]['Subject'].unique())
        t1_subj = set(df_glucest_t1[df_glucest_t1[group_col] == grp]['Subject'].unique())
        dropped_by_t1 = sorted(glu_subj - t1_subj)
        missing_entirely = sorted(all_subjects - glu_subj)
        report[grp] = {'dropped_by_t1': dropped_by_t1, 'missing_entirely': missing_entirely}
        print(f'{grp}: {len(glu_subj)} subjects (GluCEST-only) -> {len(t1_subj)} subjects (GluCEST+T1)')
        if dropped_by_t1:
            print(f'  dropped by the T1 requirement: {dropped_by_t1}')
        if missing_entirely:
            print(f'  MISSING from this group entirely (no matching ROI label found): {missing_entirely}')
            for subj in missing_entirely:
                labels = sorted(df_glucest[df_glucest['Subject'] == subj]['ROIlabel'].unique())
                print(f"    {subj}'s actual ROI labels ({len(labels)}): {labels}")

    labeled_rois = set(df_glucest['ROIlabel'].unique())
    unmatched = sorted({r for r in labeled_rois if group_fn(r) is None})
    if unmatched:
        print(f"\nROI labels not matching any group (excluded by group_fn, "
              f"first 10 of {len(unmatched)}): {unmatched[:10]}")
    return report


def build_roi_partial_corr_table(df, abbrev_fn, min_n=6):
    """Per-ROI partial correlation of GluCEST vs Age with T1 regressed out."""
    results = []
    for roi_val, sub in df.groupby('ROIvalue'):
        sub = sub.dropna(subset=['Age', 'T1', 'GluCEST'])
        n = len(sub)
        if n < min_n:
            continue
        _, _, r, p = partial_corr(sub['GluCEST'], sub['Age'], sub['T1'])
        results.append({'ROIvalue': roi_val, 'ROIlabel': abbrev_fn(sub['ROIlabel'].iloc[0]),
                         'n': n, 'partial_r': r, 'partial_p': p})
    return pd.DataFrame(results).sort_values('partial_p').reset_index(drop=True)


def build_subreg_cov(roi_type, nifti_dirs, registry=REGISTRY, tissue=None):
    """Per-subject, per-sub-region COV between the two REGISTRY-aligned scans,
    using each subject's z-trim/slice-window exactly as the tissue-level COV
    and slice-comparison figures do. tissue: segmask class to keep inside each
    ROI (e.g. 3 = WM for WM tracts, excluding CSF/GM partial volume); None keeps
    every atlas voxel (previous behaviour)."""
    abbrev = shorten_wm if roi_type == 'WM' else shorten_gm
    rows = []
    for initials, sid, zt1, zt2, s1, s2, n1, n2 in registry:
        k1, k2 = scan_key(initials, n1), scan_key(initials, n2)
        nd1, nd2 = nifti_dirs.get(k1), nifti_dirs.get(k2)
        if nd1 is None or nd2 is None:
            print(f'SKIP {initials} ({roi_type}): NIFTI_DIRS entry missing')
            continue
        vol1, lbl = load_roi_atlas(nd1, roi_type)
        vol2, _ = load_roi_atlas(nd2, roi_type)
        if vol1 is None or vol2 is None:
            print(f'SKIP {initials} ({roi_type}): atlas NIfTI not found')
            continue

        out1, out2 = load_scan(SCANS[k1]), load_scan(SCANS[k2])
        c1 = np.transpose(np.asarray(out1.CESTmap, float), (1, 0, 2))[:, :, zt1]
        c2 = np.transpose(np.asarray(out2.CESTmap, float), (1, 0, 2))[:, :, zt2]
        r1 = np.transpose(np.asarray(out1.roimask, int), (1, 0, 2))[:, :, zt1]
        r2 = np.transpose(np.asarray(out2.roimask, int), (1, 0, 2))[:, :, zt2]
        a1 = atlas_to_mat_frame(vol1)[:, :, zt1]
        a2 = atlas_to_mat_frame(vol2)[:, :, zt2]

        idx1 = np.arange(*s1.indices(c1.shape[2]))
        idx2 = np.arange(*s2.indices(c2.shape[2]))
        n_sl = min(len(idx1), len(idx2))
        idx1, idx2 = idx1[:n_sl], idx2[:n_sl]

        c1s, r1s, a1s = c1[:, :, idx1], r1[:, :, idx1], a1[:, :, idx1]
        c2s, r2s, a2s = c2[:, :, idx2], r2[:, :, idx2], a2[:, :, idx2]
        v1 = np.isfinite(c1s) & (r1s > 0) & (c1s > -10) & (c1s < 30) & (c1s != 0)
        v2 = np.isfinite(c2s) & (r2s > 0) & (c2s > -10) & (c2s < 30) & (c2s != 0)
        if tissue is not None:
            g1 = np.transpose(np.asarray(out1.segmask, int), (1, 0, 2))[:, :, zt1][:, :, idx1]
            g2 = np.transpose(np.asarray(out2.segmask, int), (1, 0, 2))[:, :, zt2][:, :, idx2]
            v1 &= g1 == tissue
            v2 &= g2 == tissue

        row = {'Subject': initials, 'ID': sid, 'n Slices': n_sl}
        all_rv = np.union1d(np.unique(a1s[a1s > 0]), np.unique(a2s[a2s > 0]))
        for rv in all_rv:
            m1 = (a1s == rv) & v1
            m2 = (a2s == rv) & v2
            if m1.sum() < 5 or m2.sum() < 5:
                continue
            row[abbrev(lbl.get(int(rv), str(int(rv))))] = pct_cov(
                float(c1s[m1].mean()), float(c2s[m2].mean()))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_registry_cov(registry=REGISTRY):
    """Two-scan whole-brain + per-tissue COV for every subject in REGISTRY,
    using the same slice-alignment windows as plot_slice_comparison."""
    rows = []
    for initials, sid, zt1, zt2, s1, s2, n1, n2 in registry:
        k1, k2 = scan_key(initials, n1), scan_key(initials, n2)
        out1, out2 = load_scan(SCANS[k1]), load_scan(SCANS[k2])

        cest1 = np.asarray(out1.CESTmap, float)[:, :, zt1]
        seg1 = np.asarray(out1.segmask, int)[:, :, zt1]
        roi1 = np.asarray(out1.roimask, int)[:, :, zt1]
        cest2 = np.asarray(out2.CESTmap, float)[:, :, zt2]
        seg2 = np.asarray(out2.segmask, int)[:, :, zt2]
        roi2 = np.asarray(out2.roimask, int)[:, :, zt2]

        idx1 = np.arange(*s1.indices(cest1.shape[2]))
        idx2 = np.arange(*s2.indices(cest2.shape[2]))
        n_sl = min(len(idx1), len(idx2))
        idx1, idx2 = idx1[:n_sl], idx2[:n_sl]

        c1, s1v, r1 = cest1[:, :, idx1], seg1[:, :, idx1], roi1[:, :, idx1]
        c2, s2v, r2 = cest2[:, :, idx2], seg2[:, :, idx2], roi2[:, :, idx2]
        v1 = np.isfinite(c1) & (s1v > 0) & (c1 > -10) & (c1 < 30) & (c1 != 0)
        v2 = np.isfinite(c2) & (s2v > 0) & (c2 > -10) & (c2 < 30) & (c2 != 0)

        row = {'Subject': initials, 'ID': sid, 'n Scans': 2, 'n Slices': n_sl}
        for tid, tname in TISSUE.items():
            mu1 = c1[(s1v == tid) & v1].mean()
            mu2 = c2[(s2v == tid) & v2].mean()
            row[tname] = pct_cov(mu1, mu2)
        m1 = c1[(r1 > 0) & v1].mean()
        m2 = c2[(r2 > 0) & v2].mean()
        row['Whole Brain'] = pct_cov(m1, m2)
        rows.append(row)
    return pd.DataFrame(rows)


def format_cov_df(cov_df, meta_cols):
    """Round/format a COV dataframe for inline `display()` (as opposed to the
    matplotlib-rendered figure table -- see render_cov_table)."""
    d = cov_df.copy()
    for c in [col for col in d.columns if col not in meta_cols]:
        d[c] = d[c].map(lambda x: f'{x:.3f}' if pd.notna(x) else '–')
    return d.set_index('Subject')


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_distribution_comparison(df, scan_colors, tissues=('CSF', 'GM', 'WM'),
                                  kind='kde', xlim=(-5, 25), clip=None,
                                  title='', savepath=None, figsize=(5.5, 2.0)):
    """3-panel (one per tissue) distribution comparison across scans.
    scan_colors: {scan_label: color}, in legend order. kind: 'kde' or 'hist'."""
    clip = clip or xlim
    _df = df[df['Scan'].isin(scan_colors)]
    with paper_style():
        fig, axes = plt.subplots(1, len(tissues), figsize=figsize)
        for ax, tissue in zip(axes, tissues):
            for scan_label, col in scan_colors.items():
                vals = _df[(_df['Scan'] == scan_label) & (_df['Tissue'] == tissue)]['CEST (%)']
                label = scan_label.split(' – ')[1] if ' – ' in scan_label else scan_label.split()[-1]
                if kind == 'kde':
                    sns.kdeplot(vals, ax=ax, color=col, fill=True, alpha=0.25, linewidth=1.2,
                                clip=clip, label=label)
                else:
                    sns.histplot(vals, ax=ax, color=col, fill=True, alpha=0.25, linewidth=1.2,
                                 binrange=xlim, stat='count', element='step', label=label)
            ax.set_xlim(*xlim)
            ax.set_ylim(bottom=0)
            ax.set_xlabel('GluCEST (%)')
            ax.set_ylabel(('Density' if kind == 'kde' else 'Count') if ax is axes[0] else '')
            ax.set_title(tissue, fontweight='bold', pad=3)
            clean_axes(ax)
        axes[1].legend(fontsize=6, frameon=False, loc='upper right')
        if title:
            fig.suptitle(title, fontsize=8, fontweight='bold', y=1.02)
        plt.tight_layout(w_pad=1.5)
        if savepath:
            plt.savefig(savepath, bbox_inches='tight')
        plt.show()


def plot_slice_comparison(out1, out2, zt1, zt2, s1, s2, title,
                           row_labels=('Scan 1', 'Scan 2'), tissue=TISSUE,
                           vmin=-7, vmax=12, interpolation=None, savepath=None):
    """2-row slice grid (segmentation-masked GluCEST maps, jet colormap) with a
    per-tissue + overall COV text box and a shared colorbar. Returns the COV
    text lines. Replaces the notebook's repeated 'Slices_XX' figures."""
    cest1 = np.asarray(out1.CESTmap, float)[:, :, zt1]
    seg1 = np.asarray(out1.segmask, int)[:, :, zt1]
    cest2 = np.asarray(out2.CESTmap, float)[:, :, zt2]
    seg2 = np.asarray(out2.segmask, int)[:, :, zt2]

    idx1 = np.arange(*s1.indices(cest1.shape[2]))
    idx2 = np.arange(*s2.indices(cest2.shape[2]))
    n_sl = min(len(idx1), len(idx2))
    idx1, idx2 = idx1[:n_sl], idx2[:n_sl]

    c1, sg1 = cest1[:, :, idx1], seg1[:, :, idx1]
    c2, sg2 = cest2[:, :, idx2], seg2[:, :, idx2]
    v1 = np.isfinite(c1) & (sg1 > 0) & (c1 > -10) & (c1 < 30) & (c1 != 0)
    v2 = np.isfinite(c2) & (sg2 > 0) & (c2 > -10) & (c2 < 30) & (c2 != 0)

    cov_lines = []
    for tid, tname in tissue.items():
        mu1 = c1[(sg1 == tid) & v1].mean()
        mu2 = c2[(sg2 == tid) & v2].mean()
        cov_lines.append(f'{tname}: {pct_cov(mu1, mu2):.1f}%')
    mu1_all = c1[(sg1 > 0) & v1].mean()
    mu2_all = c2[(sg2 > 0) & v2].mean()
    cov_lines.append(f'Overall: {pct_cov(mu1_all, mu2_all):.1f}%')
    cov_text = '\n'.join(cov_lines)

    with paper_style(**{'figure.dpi': 200}):
        cmap_jet = copy(plt.cm.jet)
        cmap_jet.set_bad('black')
        fig, axes = plt.subplots(2, n_sl, figsize=(n_sl * 0.9, 1.8),
                                  gridspec_kw=dict(hspace=0.02, wspace=0.02))
        imshow_kw = dict(cmap=cmap_jet, vmin=vmin, vmax=vmax, aspect='equal')
        if interpolation:
            imshow_kw['interpolation'] = interpolation
        for pi, (i1, i2) in enumerate(zip(idx1, idx2)):
            for row, (cest, seg, idx) in enumerate([(cest1, seg1, i1), (cest2, seg2, i2)]):
                im = axes[row, pi].imshow(
                    np.ma.masked_where(seg[:, :, idx] == 0, cest[:, :, idx]), **imshow_kw)
                axes[row, pi].set(xticks=[], yticks=[])
            axes[0, pi].set_title(str(i1 + 1) if i1 == i2 else f'{i1+1}|{i2+1}', fontsize=6, pad=1.5)

        for row, label in enumerate(row_labels):
            axes[row, 0].set_ylabel(label, fontsize=7, fontweight='bold', labelpad=3)

        fig.text(0.01, -0.04, cov_text, fontsize=5.5, va='top', ha='left', fontfamily='Arial',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='#cccccc', linewidth=0.4, alpha=0.9))
        fig.subplots_adjust(right=0.87)
        cax = fig.add_axes([0.89, 0.15, 0.012, 0.70])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label('GluCEST (%)', fontsize=7, labelpad=4)
        cb.ax.tick_params(labelsize=6, length=2)
        fig.suptitle(title, fontsize=8, fontweight='bold', y=1.03)
        if savepath:
            plt.savefig(savepath, bbox_inches='tight', dpi=300)
        plt.show()
    return cov_lines


def plot_age_scatter(df, value_col, groups, colors, ylabel, title='', group_col='Tissue',
                      show_labels=True, show_n=False, savepath=None, figsize=None, ylim=None,
                      slope_diff=False):
    """One scatter+regression+95%CI panel per group (tissue, or Expected/Atypical
    WM group), plotted against Age. Covers GluCEST-vs-Age, MT-vs-Age, T1-vs-Age,
    and the Expected/Atypical WM variants of each.
    slope_diff=True (two groups): adds the age-slope difference (groups[1] - groups[0],
    per decade) and its P, from regressing the within-subject difference on Age."""
    if slope_diff:
        w = (df[df[group_col].isin(groups[:2])]
             .pivot_table(index=['Subject', 'Age'], columns=group_col, values=value_col)
             .dropna().reset_index())
        res = linregress(w['Age'].astype(float), w[groups[1]] - w[groups[0]])
        diff_str = (f'Slope difference = {10 * res.slope:.2f} per decade,  '
                    f'{p_str(res.pvalue)}  (n={len(w)})')
    figsize = figsize or (2.4 * len(groups) + 0.5, 2.4)
    with paper_style():
        fig, axes = plt.subplots(1, len(groups), figsize=figsize)
        axes = np.atleast_1d(axes)
        for ax, grp in zip(axes, groups):
            sub = df[df[group_col] == grp].dropna(subset=['Age', value_col]).reset_index(drop=True)
            x, y = sub['Age'].values.astype(float), sub[value_col].values.astype(float)
            col = colors[grp]
            ax.scatter(x, y, color=col, s=22, zorder=4, linewidths=0.4,
                       edgecolors='white', clip_on=False)
            if show_labels:
                for _, row in sub.iterrows():
                    ax.text(row['Age'] + 0.8, row[value_col], row['Subject'],
                            fontsize=12, va='center', color='#555555', clip_on=False)
                    print()
            r, p = add_regression_ci(ax, x, y, col)
            n_str = f'  (n={len(x)})' if show_n else ''
            ax.set_title(f'{grp}{n_str}\n$r$ = {r:.2f},  {p_str(p)}', fontsize=12, fontweight='bold', pad=3)
            ax.set_xlabel('Age (years)', fontsize=12, labelpad=3)
            ax.set_ylabel(ylabel if ax is axes[0] else '', fontsize=12, labelpad=3)
            clean_axes(ax)
            if ylim:
                ax.set_ylim(*ylim)
        if title:
            fig.suptitle(title, fontsize=12, fontweight='bold', y=1.04)
        plt.tight_layout(w_pad=2)
        if slope_diff:
            fig.text(0.5, -0.06, diff_str, ha='center', va='top', fontsize=11)
        if savepath:
            plt.savefig(savepath, bbox_inches='tight')
        plt.show()


def plot_roi_age_grid(df, value_col, abbrev_fn, color, ylabel, title,
                       min_subjects=3, n_per_row=3, show_labels=False, savepath=None):
    """Grid of per-ROI scatter+regression-vs-age panels, one panel per ROI with
    >= min_subjects subjects present. Covers the 4 WM/GM x GluCEST/MT ROI grids."""
    roi_counts = df.groupby('ROIvalue')['Subject'].nunique()
    plot_rois = roi_counts[roi_counts >= min_subjects].sort_values(ascending=False).index.tolist()
    n_rows = int(np.ceil(len(plot_rois) / n_per_row))

    with paper_style():
        fig, all_axes = plt.subplots(n_rows, n_per_row, figsize=(7.5, 2.4 * n_rows),
                                      gridspec_kw=dict(hspace=0.60, wspace=0.38))
        all_axes = np.array(all_axes).reshape(n_rows, n_per_row)

        for idx, roi_val in enumerate(plot_rois):
            row_i, col_i = divmod(idx, n_per_row)
            ax = all_axes[row_i, col_i]
            sub = df[df.ROIvalue == roi_val].dropna(subset=['Age', value_col]).reset_index(drop=True)
            x, y = sub['Age'].values.astype(float), sub[value_col].values.astype(float)

            ax.scatter(x, y, color=color, s=22, zorder=4, linewidths=0.4,
                       edgecolors='white', clip_on=False)
            if show_labels:
                for _, row in sub.iterrows():
                    ax.text(row['Age'] + 0.8, row[value_col], row['Subject'],
                            fontsize=5, va='center', color='#555555', clip_on=False)
            r, p = add_regression_ci(ax, x, y, color)
            name = abbrev_fn(sub['ROIlabel'].iloc[0])
            ax.set_title(f'{name}\n$r$ = {r:.2f},  {p_str(p)}', fontsize=7, fontweight='bold', pad=3)
            ax.set_xlabel('Age (years)', fontsize=8, labelpad=3)
            ax.set_ylabel(ylabel if col_i == 0 else '', fontsize=8, labelpad=3)
            clean_axes(ax)

        for idx in range(len(plot_rois), n_rows * n_per_row):
            row_i, col_i = divmod(idx, n_per_row)
            all_axes[row_i, col_i].set_visible(False)

        fig.suptitle(title, fontsize=8, fontweight='bold', y=1.01)
        if savepath:
            plt.savefig(savepath, bbox_inches='tight')
        plt.show()
    return plot_rois


def plot_roi_partial_corr_grid(df, partial_df, color, title,
                                xlabel='Age residual (T1 removed)',
                                ylabel='GluCEST residual (T1 removed)',
                                n_per_row=3, savepath=None):
    """Grid of per-ROI partial-correlation panels (GluCEST vs Age, T1 regressed
    out), one panel per row of `partial_df` (see build_roi_partial_corr_table)."""
    plot_rois = partial_df['ROIvalue'].tolist()
    n_rows = int(np.ceil(len(plot_rois) / n_per_row))

    with paper_style():
        fig, all_axes = plt.subplots(n_rows, n_per_row, figsize=(7.5, 2.4 * n_rows),
                                      gridspec_kw=dict(hspace=0.65, wspace=0.40))
        all_axes = np.array(all_axes).reshape(n_rows, n_per_row)

        for idx, roi_val in enumerate(plot_rois):
            row_i, col_i = divmod(idx, n_per_row)
            ax = all_axes[row_i, col_i]
            sub = df[df.ROIvalue == roi_val].dropna(subset=['Age', 'T1', 'GluCEST']).reset_index(drop=True)
            resid_glu, resid_age, _, _ = partial_corr(sub['GluCEST'], sub['Age'], sub['T1'])
            n = len(sub)

            ax.scatter(resid_age, resid_glu, color=color, s=22, zorder=4,
                       linewidths=0.4, edgecolors='white', clip_on=False)
            add_regression_ci(ax, resid_age, resid_glu, color, extend=0.05 * np.ptp(resid_age))
            ax.axhline(0, color='#cccccc', lw=0.6, ls='--', zorder=1)
            ax.axvline(0, color='#cccccc', lw=0.6, ls='--', zorder=1)

            row = partial_df[partial_df.ROIvalue == roi_val].iloc[0]
            ax.set_title(f"{row['ROIlabel']}  (n={n})\nPartial $r$ = {row['partial_r']:.2f},  {p_str(row['partial_p'])}",
                         fontsize=7, fontweight='bold', pad=3)
            ax.set_xlabel(xlabel, fontsize=8, labelpad=3)
            ax.set_ylabel(ylabel if col_i == 0 else '', fontsize=8, labelpad=3)
            clean_axes(ax)

        for idx in range(len(plot_rois), n_rows * n_per_row):
            row_i, col_i = divmod(idx, n_per_row)
            all_axes[row_i, col_i].set_visible(False)

        fig.suptitle(title, fontsize=8, fontweight='bold', y=1.01)
        if savepath:
            plt.savefig(savepath, bbox_inches='tight')
        plt.show()


def plot_tissue_partial_corr_row(df, groups, colors, x_var, cov_var, xlabel, ylabel,
                                  group_col='Tissue', show_n=False,
                                  ax_row=None, savepath=None, suptitle=None):
    """One row of partial-correlation panels (one per tissue/group), regressing
    `cov_var` out of both GluCEST and `x_var` (each in {'Age', 'T1'}). Pass
    ax_row from an existing subplot grid to compose a multi-row figure (e.g. the
    combined T1<->Age partial-correlation panel); otherwise draws its own figure."""
    created_fig = ax_row is None
    if created_fig:
        with paper_style():
            fig, ax_row = plt.subplots(1, len(groups), figsize=(7.5, 2.4))
            ax_row = np.atleast_1d(ax_row)
            _draw_partial_corr_row(df, groups, colors, x_var, cov_var, xlabel, ylabel,
                                    group_col, show_n, ax_row)
            if suptitle:
                fig.suptitle(suptitle, fontsize=8, fontweight='bold', y=1.04)
            plt.tight_layout(w_pad=2)
            if savepath:
                plt.savefig(savepath, bbox_inches='tight')
            plt.show()
    else:
        _draw_partial_corr_row(df, groups, colors, x_var, cov_var, xlabel, ylabel,
                                group_col, show_n, ax_row)


def _draw_partial_corr_row(df, groups, colors, x_var, cov_var, xlabel, ylabel,
                            group_col, show_n, ax_row):
    for ax, grp in zip(ax_row, groups):
        sub = df[df[group_col] == grp].reset_index(drop=True)
        age = sub['Age'].values.astype(float)
        glu = sub['GluCEST'].values.astype(float)
        t1 = sub['T1'].values.astype(float)
        col = colors[grp]
        x_raw = t1 if x_var == 'T1' else age
        cov = age if cov_var == 'Age' else t1
        resid_glu, resid_x, r, p = partial_corr(glu, x_raw, cov)
        n = len(resid_glu)
        n_str = f'  (n={n})' if show_n else ''

        ax.scatter(resid_x, resid_glu, color=col, s=22, zorder=4,
                   linewidths=0.4, edgecolors='white', clip_on=False)
        add_regression_ci(ax, resid_x, resid_glu, col, extend=0.05 * np.ptp(resid_x))
        ax.axhline(0, color='#cccccc', lw=0.6, ls='--', zorder=1)
        ax.axvline(0, color='#cccccc', lw=0.6, ls='--', zorder=1)
        ax.set_title(f'{grp}{n_str}\nPartial $r$ = {r:.2f},  {p_str(p)}', fontsize=7, fontweight='bold', pad=3)
        ax.set_xlabel(xlabel, fontsize=8, labelpad=3)
        ax.set_ylabel(ylabel if ax is ax_row[0] else '', fontsize=8, labelpad=3)
        clean_axes(ax)


def render_cov_table(cov_df, meta_cols, title, int_cols=(), cols_per_table=6,
                      fontsize=9, savepath=None):
    """Nature-Comms-style COV summary table, matplotlib-rendered. Paginates into
    multiple figures if there are more data columns than `cols_per_table` (the
    WM/GM sub-region tables can have dozens of ROI columns)."""
    data_cols = [c for c in cov_df.columns if c not in meta_cols]
    if cov_df.empty or not data_cols:
        print(f'No data for: {title}')
        return
    n_pages = max(1, -(-len(data_cols) // cols_per_table))

    for page, start in enumerate(range(0, len(data_cols), cols_per_table)):
        cur_cols = data_cols[start:start + cols_per_table]
        col_labels = list(meta_cols) + [f'{c}\nCOV (%)' for c in cur_cols]
        table_data = [
            [int(r[c]) if c in int_cols else r[c] for c in meta_cols]
            + [f'{r[c]:.1f}' if pd.notna(r.get(c)) else '–' for c in cur_cols]
            for _, r in cov_df.iterrows()
        ]

        n_rows, n_cols = len(table_data), len(col_labels)
        col_w = [0.08] * len(meta_cols) + [0.12] * len(cur_cols)
        fig_w = max(10.0, sum(col_w) * 10)
        fig_h = (n_rows + 1.8) * 0.50

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_axis_off()
        tbl = ax.table(cellText=table_data, colLabels=col_labels, cellLoc='center',
                        loc='center', colWidths=col_w)
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(fontsize)
        for (ri, ci), cell in tbl.get_celld().items():
            cell.set_edgecolor('none')
            cell.set_linewidth(0)
            cell.PAD = 0.12
            if ri == 0:
                cell.set_text_props(fontweight='bold', color='#1a1a1a', fontsize=fontsize)
                cell.set_facecolor('#e8e8e8')
            else:
                cell.set_facecolor('white' if ri % 2 == 1 else '#f5f5f5')
                cell.set_text_props(fontsize=fontsize)
        for ci in range(n_cols):
            tbl[0, ci].visible_edges = 'T'
            tbl[0, ci].set_linewidth(0)
            tbl[0, ci].set_edgecolor('#2a2a2a')
            tbl[1, ci].visible_edges = 'T'
            tbl[1, ci].set_linewidth(0.7)
            tbl[1, ci].set_edgecolor('#999999')
            tbl[n_rows, ci].visible_edges = 'B'
            tbl[n_rows, ci].set_linewidth(1.2)
            tbl[n_rows, ci].set_edgecolor('#2a2a2a')

        page_title = f'{title} (Part {page + 1} of {n_pages})' if n_pages > 1 else title
        fig.suptitle(page_title, fontsize=11, fontweight='bold', y=0.98)
        plt.tight_layout()
        if savepath:
            stem = Path(savepath).stem + (f'_part{page + 1}' if n_pages > 1 else '')
            plt.savefig(Path(savepath).with_stem(stem), bbox_inches='tight', dpi=300)
        plt.show()
