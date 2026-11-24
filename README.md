# GluCEST-WM-Ageing

Analysis and figure code for *3D glutamate-weighted CEST (GluCEST) MRI at 7T: repeatability and age-related
changes across white-matter subregions*.

`figures/GluCEST_WM_Ageing_Paper_Figures.ipynb` recreates every figure in the manuscript (Figures 1–4 and the
supplementary B0/B1 and gray-matter figures), Tables 1–3, and the statistics behind Figure 4: the OLS slope
difference, the mixed model `GluCEST ~ Age * Region + (1 | Participant)`, a corpus-callosum-only sensitivity
check, and GluCEST beyond T1. The notebook is saved with its outputs, so you can read the figures without
running it.

## Layout

```
utils_main.py   data paths, participant table, ROI / tissue dataframe builders
figures/        notebook, figure modules, and the CSV tables (Tables 1–3, Fig. 3 statistics)
```

## Running

```
pip install -r figures/requirements.txt
cd figures && jupyter notebook GluCEST_WM_Ageing_Paper_Figures.ipynb
```

The raw 7T data and participant-level caches are not included (participant privacy). Participants are coded
P01–P18 in order of age; P18 is excluded from all analyses (n = 17; 10 repeat-scan pairs). With access to the
data, set the drive paths at the top of `utils_main.py`. 3D rendering needs a working OpenGL context
(pyvista / vtk).
