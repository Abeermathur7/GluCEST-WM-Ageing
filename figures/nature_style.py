"""Shared Nature-style figure settings and helpers.

Conventions (Nature / Nature Communications artwork guide + the reference paper
s41467-026-70707-6):
  * Arial, 5-7 pt text; panel letters 8 pt bold lowercase, top-left
  * single column 89 mm, double column 183 mm, max height ~170 mm (fits a page with legend)
  * 0.5 pt axes, ticks out, no top/right spines, no frames on legends
  * colour-blind-safe Okabe-Ito categorical palette; perceptually uniform colormaps
  * vector PDF (TrueType fonts embedded, editable text) + 600 dpi PNG
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

MM = 1 / 25.4
SINGLE = 89          # mm
ONEHALF = 120        # mm
DOUBLE = 183         # mm

OUT = Path(__file__).resolve().parent
CACHE = OUT / '_cache'
# GLUCEST_SEG=nifti: tissue masks from each scan's CESTslab_seg.nii instead of the .mat segmask; separate cache and
# output folder so the two sets never mix (make_niftiseg_set.py drives it).
SEG_MODE = __import__('os').environ.get('GLUCEST_SEG', 'mat')
if SEG_MODE == 'nifti':
    CACHE = OUT / '_cache_niftiseg'
    OUT = OUT / 'NIfTI_seg_figures'
CACHE.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)

# ── colour ───────────────────────────────────────────────────────────────────
OKABE = dict(black='#000000', orange='#E69F00', sky='#56B4E9', green='#009E73',
             yellow='#F0E442', blue='#0072B2', vermillion='#D55E00', purple='#CC79A7')
TISSUE_COL = {'GM': '#D55E00', 'WM': '#0072B2', 'CSF': '#56B4E9'}
TRACT_COL = {'Corpus callosum': '#CC79A7', 'Corona radiata': '#009E73'}
FAMILY_COL = {'CC + cingulum': '#CC79A7', 'Corona radiata': '#009E73', 'Other': '#7F7F7F'}
# WM subregions: callosal/cingulate = reddish-purples, corona radiata = greens,
# association / capsular tracts = Okabe-Ito orange / yellow / blues.
SUB_COL = {'CC genu': '#E7A3CF', 'CC body': '#CC79A7', 'CC splenium': '#8E3F75', 'Cingulum': '#D55E00',
           'ACR': '#8FD3B7', 'SCR': '#009E73', 'PCR': '#00573F',
           'SLF': '#E69F00', 'sFOF': '#F0E442', 'ALIC': '#56B4E9', 'EC': '#0072B2'}
GREY = '#7F7F7F'
LIGHT = '#D9D9D9'


def _cmap_glu():
    """Custom perceptually ordered map (dark indigo -> teal -> gold -> cream):
    monotonic lightness, distinct from viridis in hue, CVD-friendly."""
    anchors = ['#140b34', '#2b2f6f', '#1f6a8a', '#2a9d8f', '#8cc06b', '#e9c46a', '#fbf4d0']
    return LinearSegmentedColormap.from_list('glucest', anchors, N=256)


GLU_CMAP = _cmap_glu()
mpl.colormaps.register(GLU_CMAP, force=True)

RC = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 6, 'axes.titlesize': 6.5, 'axes.labelsize': 6,
    'xtick.labelsize': 5.5, 'ytick.labelsize': 5.5, 'legend.fontsize': 5.5,
    'axes.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 2, 'axes.titlepad': 3, 'axes.titleweight': 'bold',
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'xtick.major.size': 2, 'ytick.major.size': 2, 'xtick.major.pad': 1.5, 'ytick.major.pad': 1.5,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'lines.linewidth': 0.8, 'lines.markersize': 3, 'patch.linewidth': 0.5,
    'legend.frameon': False, 'legend.handlelength': 1.2, 'legend.borderaxespad': 0.2,
    'legend.handletextpad': 0.4, 'legend.labelspacing': 0.3,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold', 'mathtext.default': 'regular',
    'savefig.dpi': 600, 'figure.dpi': 150, 'savefig.transparent': False,
    'image.interpolation': 'nearest', 'axes.unicode_minus': True,
}


def apply():
    mpl.rcParams.update(RC)


# ── dark theme (3D render figures; user request 2026-11-18: black background for all) ───────────────
BG_DARK, FG = '#000000', '#EDEDED'
DARK_RC = {
    'figure.facecolor': BG_DARK, 'savefig.facecolor': BG_DARK, 'axes.facecolor': BG_DARK,
    'text.color': FG, 'axes.labelcolor': FG, 'axes.edgecolor': FG, 'xtick.color': FG, 'ytick.color': FG,
    'xtick.labelcolor': FG, 'ytick.labelcolor': FG,
}
# inferno without its black end (12-100 %): low values stay visible on a black background
INFERNO_LIFT = mpl.colors.ListedColormap(mpl.colormaps['inferno'](np.linspace(0.12, 1.0, 256)),
                                         name='inferno_lift')
mpl.colormaps.register(INFERNO_LIFT, force=True)


def dark():
    mpl.rcParams.update(DARK_RC)


apply()


# ── figure / layout helpers ──────────────────────────────────────────────────
class MMFigure:
    """Figure laid out in millimetres, origin top-left (like Illustrator)."""

    def __init__(self, w_mm, h_mm):
        self.W, self.H = w_mm, h_mm
        self.fig = plt.figure(figsize=(w_mm * MM, h_mm * MM))

    def ax(self, x, y, w, h, **kw):
        return self.fig.add_axes([x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H], **kw)

    def letter(self, s, x, y, size=8):
        self.fig.text(x / self.W, 1 - y / self.H, s, fontsize=size, fontweight='bold',
                      va='top', ha='left')

    def text(self, x, y, s, **kw):
        kw.setdefault('va', 'center'); kw.setdefault('ha', 'center')
        return self.fig.text(x / self.W, 1 - y / self.H, s, **kw)

    def hline(self, x0, x1, y, **kw):
        kw.setdefault('color', 'k'); kw.setdefault('lw', 0.5)
        self.fig.add_artist(mpl.lines.Line2D([x0 / self.W, x1 / self.W], [1 - y / self.H] * 2, **kw))

    def save(self, name):
        save(self.fig, name)


def save(fig, name):
    pdf, png = OUT / f'{name}.pdf', OUT / f'{name}.png'
    pdf.parent.mkdir(parents=True, exist_ok=True)
    for p in (pdf, png):
        p.unlink(missing_ok=True)   # OneDrive cloud-only placeholders cannot be overwritten in place
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    plt.close(fig)
    print('saved', pdf.name, png.name)


def despine(ax, left=True, bottom=True):
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)


def img_ax(ax):
    ax.set_axis_off()


def colorbar_inset(fig, im, cax, label, ticks=None, orientation='horizontal', label_side='left'):
    """Slim colourbar in the reference-paper style: thin bar, label beside it."""
    cb = fig.colorbar(im, cax=cax, orientation=orientation, ticks=ticks)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=1.5, width=0.4, pad=1, labelsize=5.5)
    if orientation == 'horizontal':
        if label_side == 'left':
            cb.ax.text(-0.04, 0.5, label, transform=cb.ax.transAxes, ha='right', va='center', fontsize=6)
        else:
            cb.set_label(label, fontsize=6, labelpad=1.5)
    else:
        cb.set_label(label, fontsize=6, labelpad=2)
    return cb


# ── stats formatting ─────────────────────────────────────────────────────────
def fmt_p(p, prefix=True):
    if p < 1e-3:
        e = int(np.floor(np.log10(p)))
        m = p / 10 ** e
        s = rf'{m:.1f} $\times$ 10$^{{{e}}}$'
        return (r'$\it{P}$ = ' + s) if prefix else s
    s = f'{p:.3f}' if p < 0.01 else f'{p:.2f}'
    return (r'$\it{P}$ = ' + s) if prefix else s


def fmt_r(r, p, sym='r'):
    r = 0.0 if abs(r) < 0.005 else r
    return rf'$\it{{{sym}}}$ = {r:.2f}'.replace('-', '−') + ', ' + fmt_p(p)


def minus(s):
    return str(s).replace('-', '−')


def ols_ci(x, y, xs=None, alpha=0.05):
    """OLS fit with 95% CI of the mean; returns xs, yhat, lo, hi, (r, p)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    b1, b0 = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100) if xs is None else xs
    yhat = b0 + b1 * xs
    resid = y - (b0 + b1 * x)
    s = np.sqrt((resid ** 2).sum() / (n - 2))
    se = s * np.sqrt(1 / n + (xs - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum())
    t = stats.t.ppf(1 - alpha / 2, n - 2)
    return xs, yhat, yhat - t * se, yhat + t * se, stats.pearsonr(x, y)


def reg_plot(ax, x, y, color, s=10, marker='o', band=True, lw=0.8, zorder=3, label=None):
    xs, yh, lo, hi, (r, p) = ols_ci(x, y)
    if band:
        ax.fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0, zorder=zorder - 1)
    ax.plot(xs, yh, color=color, lw=lw, zorder=zorder)
    ax.scatter(x, y, s=s, marker=marker, facecolor=color, edgecolor='white', linewidth=0.35,
               zorder=zorder + 1, label=label, clip_on=False)
    return r, p


def mean_ci(v, alpha=0.05):
    v = np.asarray(v, float)
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    h = se * stats.t.ppf(1 - alpha / 2, len(v) - 1)
    return m, m - h, m + h


def boot_ci(v, n=10000, alpha=0.05, seed=1):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    bs = rng.choice(v, size=(n, len(v)), replace=True).mean(1)
    return np.percentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)]), bs


def paired_stats(a, b):
    """b - a: mean diff, 95% CI, Cohen's dz, paired t P, Wilcoxon P."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    m, lo, hi = mean_ci(d)
    t, p = stats.ttest_rel(b, a)
    w = stats.wilcoxon(b, a).pvalue
    return dict(diff=m, lo=lo, hi=hi, dz=d.mean() / d.std(ddof=1), t=t, p=p, p_w=w, n=len(d))


def half_violin(ax, data, pos, color, side='left', width=0.35, alpha=0.35, bw=None, lw=0.5):
    """One-sided KDE ('raincloud' cloud)."""
    data = np.asarray(data, float)
    kde = stats.gaussian_kde(data, bw_method=bw)
    ys = np.linspace(data.min() - 0.15 * np.ptp(data), data.max() + 0.15 * np.ptp(data), 200)
    dens = kde(ys)
    dens = dens / dens.max() * width
    sgn = -1 if side == 'left' else 1
    ax.fill_betweenx(ys, pos, pos + sgn * dens, color=color, alpha=alpha, lw=0)
    ax.plot(pos + sgn * dens, ys, color=color, lw=lw)
