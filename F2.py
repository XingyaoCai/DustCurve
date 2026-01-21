import os
import pathlib

import warnings
warnings.filterwarnings("ignore")

import FunctionLib as FL

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
config = {
    "font.size": 18,
    "font.family": "serif",
    "figure.figsize": (25, 14),

    "axes.titlesize": 28,
    "axes.labelsize": 24,
    "legend.fontsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,

    "axes.linewidth": 2,
    "axes.edgecolor": "black",

    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,

    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "xtick.minor.width": 2,
    "ytick.minor.width": 2,

    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
}
mpl.rcParams.update(config)

import astropy.units as u
import astropy

DJAv4Catalog = FL.Spectrum_Catalog()
DJAv4Catalog.load_from_pkl(("./DJAV4.2Catalog.pkl"))

print("="*50)
print(f"Total {(DJAv4Catalog.sample_num())} sample objectss loaded.")
print("="*50)

dn_index_conditions = np.array([[0, 0.51], [0.51, 0.72], [0.72, 1.5]])
def dn_condition_func(dn_index):
    for i, (low, high) in enumerate(dn_index_conditions):
        if low < dn_index <= high:
            return i
    return -1

analyst_list = []
for dn_idx in range(len(dn_index_conditions)):
    condition_name = f"DN_Index_{dn_index_conditions[dn_idx][0]:.2f}_to_{dn_index_conditions[dn_idx][1]:.2f}"
    analyst = FL.DustAttenuationAnalyst(DJAv4Catalog)

    surveyid_subid_list = []

    for surveyid_subid, entry in DJAv4Catalog.catalog_iterator():
        if not entry['sample_flag']:
            continue

        # 获取 Flux Ratio
        try:
            measurements = entry['dust_curve_parameters']['Corrected_Flux_Density_Measurements']
            dn_index = measurements['red_flux_density'] / measurements['blue_flux_density']
        except KeyError:
            continue

        # 检查是否属于当前的 Flux Ratio Bin
        if dn_condition_func(dn_index) != dn_idx:
            continue
        surveyid_subid_list.append(surveyid_subid)

    print(f"Processing Condition: {condition_name}, Total Samples: {len(surveyid_subid_list)}")

    # 3. 载入目标 ID 列表
    analyst.load_and_group_objects(surveyid_subid_list)
    analyst.compute_average_spectra()

    analyst.compute_attenuation_curves(mode='smoothed')

    # 5. 拟合与分析
    analyst.fit_polynomial(order=3)
    # 假设我们想让无穷远处截距匹配 Calzetti 的推断值 -0.898
    analyst.derive_k_curve_parameters(intercept_val=-0.89877)
    analyst_list.append(analyst)

fig, ax = plt.subplots()
if_ref=True
colors_list = plt.cm.viridis(np.linspace(0,1,len(analyst_list)))
for analyst in analyst_list:

    dn_index_range = dn_index_conditions[analyst_list.index(analyst)]
    condition_name = f"Dn {dn_index_range[0]:.2f}-{dn_index_range[1]:.2f}"

    analyst.plot_reddening_curve_comparison(
    if_show=False,
    if_input=True, fig=fig, ax=ax,
    curve_color=colors_list[analyst_list.index(analyst)],
    curve_label=condition_name,
    draw_background=if_ref  # 第一次绘制，需要参考线
)
    if_ref=False  # 之后不需要参考线
plt.show()
