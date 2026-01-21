# %%
import astropy.units as u
import astropy
from scipy.signal import savgol_filter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import FunctionLib as FL
from collections import defaultdict
import os
import pathlib
import warnings

warnings.filterwarnings("ignore")


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


DJAv4Catalog = FL.Spectrum_Catalog()
DJAv4Catalog.load_from_pkl(("./DJAV4.2Catalog.pkl"))

print("="*50)
print(f"Total {(DJAv4Catalog.sample_num())} sample objectss loaded.")
print("="*50)

# %%
dn_index_conditions = np.array([[0, 0.5], [0.5, 0.65], [0.65, 0.8], [0.8, 2]])


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
            dn_index = measurements['red_flux_density'] / \
                measurements['blue_flux_density']
        except KeyError:
            continue

        # 检查是否属于当前的 Flux Ratio Bin
        if dn_condition_func(dn_index) != dn_idx:
            continue
        surveyid_subid_list.append(surveyid_subid)

    print(
        f"Processing Condition: {condition_name}, Total Samples: {len(surveyid_subid_list)}")

    # 3. 载入目标 ID 列表
    analyst.load_and_group_objects(surveyid_subid_list)
    analyst.compute_average_spectra()

    analyst.compute_attenuation_curves(mode='smoothed')

    # 5. 拟合与分析
    analyst.fit_polynomial(order=3)
    # 假设我们想让无穷远处截距匹配 Calzetti 的推断值 -0.898
    analyst.derive_k_curve_parameters(intercept_val=-0.89877)
    analyst_list.append(analyst)
    analyst.plot_fit_vs_calzetti()           # 拟合效果


fig, ax = plt.subplots()
if_ref = True
colors_list = plt.cm.viridis(np.linspace(0, 1, len(analyst_list)))
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
    if_ref = False  # 之后不需要参考线

# %%


# %%
redshift_conditions = np.array([[3, 4], [4, 5.0], [5., 7.2]])
# redshift_conditions=np.array([[3,3.5],[3.5,4.3],[4.3,5.2],[5.2,5.9],[5.9,7.2]])


def redshift_condition_func(redshift):
    for i, (low, high) in enumerate(redshift_conditions):
        if low < redshift <= high:
            return i
    return -1


analyst_list = []

for redshift_idx in range(len(redshift_conditions)):
    condition_name = f"Redshift_{redshift_conditions[redshift_idx][0]:.2f}_to_{redshift_conditions[redshift_idx][1]:.2f}"
    analyst = FL.DustAttenuationAnalyst(DJAv4Catalog)

    surveyid_subid_list = []

    for surveyid_subid, entry in DJAv4Catalog.catalog_iterator():
        if not entry['sample_flag']:
            continue

        # 获取 Flux Ratio
        try:
            redshift = entry['determined_redshift']
        except KeyError:
            continue

        # 检查是否属于当前的 Flux Ratio Bin
        if redshift_condition_func(redshift) != redshift_idx:
            continue
        surveyid_subid_list.append(surveyid_subid)

    print(
        f"Processing Condition: {condition_name}, Total Samples: {len(surveyid_subid_list)}")
    # 3. 载入目标 ID 列表
    analyst.load_and_group_objects(surveyid_subid_list)
    analyst.compute_average_spectra()
    analyst.compute_attenuation_curves(mode='smoothed')
    # 5. 拟合与分析
    analyst.fit_polynomial(order=3)
    # 假设我们想让无穷远处截距匹配 Calzet
    analyst.derive_k_curve_parameters(intercept_val=-0.89877)
    analyst_list.append(analyst)

# %%


# %%

fig, ax = plt.subplots(figsize=(16, 14))
if_ref = True
colors_list = plt.cm.viridis(np.linspace(0, 1, len(analyst_list)))
for analyst in analyst_list:

    redshift_range = redshift_conditions[analyst_list.index(analyst)]
    condition_name = f"Redshift {redshift_range[0]:.2f}"

    analyst.plot_reddening_curve_comparison(
        if_show=False,
        if_input=True, fig=fig, ax=ax,
        curve_color=colors_list[analyst_list.index(analyst)],
        curve_label=condition_name,
        draw_background=if_ref  # 第一次绘制，需要参考线
    )
    if_ref = False  # 之后不需要参考线

# %%
has = []
for surveyid_subid, entry in DJAv4Catalog.catalog_iterator():
    if not entry['sample_flag']:
        continue
    params = entry.get('dust_curve_parameters', {})
    line_params = params.get('Best_Balmer_Decrement_Line_Parameters', {})

    ha = line_params.get('halpha_flux', 0)
    has.append(ha)

# %%
plt.hist(has, range=(0, 2e-17))

# %%
halpha_flux_threshold = [(0, 2e-18), (2e-18, 5e-18), (5e-18, 5e-17)]
# redshift_conditions=np.array([[3,3.5],[3.5,4.3],[4.3,5.2],[5.2,5.9],[5.9,7.2]])


def halpha_condition_func(halpha_flux):
    for i, (low, high) in enumerate(halpha_flux_threshold):
        if low < halpha_flux <= high:
            return i
    return -1


analyst_list = []

for halpha_idx in range(len(halpha_flux_threshold)):
    condition_name = f"Halpha_{halpha_flux_threshold[halpha_idx][0]:.2e}_to_{halpha_flux_threshold[halpha_idx][1]:.2e}"
    analyst = FL.DustAttenuationAnalyst(DJAv4Catalog)

    surveyid_subid_list = []

    for surveyid_subid, entry in DJAv4Catalog.catalog_iterator():
        if not entry['sample_flag']:
            continue

        # 获取 Flux Ratio
        try:
            params = entry.get('dust_curve_parameters', {})
            line_params = params.get(
                'Best_Balmer_Decrement_Line_Parameters', {})
            halpha_flux = line_params.get('halpha_flux', 0)
        except KeyError:
            continue

        # 检查是否属于当前的 Flux Ratio Bin
        if halpha_condition_func(halpha_flux) != halpha_idx:
            continue
        surveyid_subid_list.append(surveyid_subid)

    print(
        f"Processing Condition: {condition_name}, Total Samples: {len(surveyid_subid_list)}")
    # 3. 载入目标 ID 列表
    analyst.load_and_group_objects(surveyid_subid_list)
    analyst.compute_average_spectra()
    analyst.compute_attenuation_curves(mode='smoothed')
    # 5. 拟合与分析
    analyst.fit_polynomial(order=3)
    # 假设我们想让无穷远处截距匹配 Calzet
    analyst.derive_k_curve_parameters(intercept_val=-0.89877)
    analyst_list.append(analyst)

# %%

fig, ax = plt.subplots(figsize=(16, 14))
if_ref = True
colors_list = plt.cm.viridis(np.linspace(0, 1, len(analyst_list)))
for analyst in analyst_list:

    halpha_range = halpha_flux_threshold[analyst_list.index(analyst)]
    condition_name = f"Halpha {halpha_range[0]:.2e} to {halpha_range[1]:.2e}"

    analyst.plot_reddening_curve_comparison(
        if_show=False,
        if_input=True, fig=fig, ax=ax,
        curve_color=colors_list[analyst_list.index(analyst)],
        curve_label=condition_name,
        draw_background=if_ref  # 第一次绘制，需要参考线
    )
    if_ref = False  # 之后不需要参考线

# %%
dn_index_conditions = np.array([[0, 0.72], [0.72, 1.5]])


def dn_condition_func(dn_index):
    for i, (low, high) in enumerate(dn_index_conditions):
        if low < dn_index <= high:
            return i
    return -1


redshift_conditions = np.array([[3, 4], [4, 5.0], [5., 7.2]])
# redshift_conditions=np.array([[3,3.5],[3.5,4.3],[4.3,5.2],[5.2,5.9],[5.9,7.2]])


def redshift_condition_func(redshift):
    for i, (low, high) in enumerate(redshift_conditions):
        if low < redshift <= high:
            return i
    return -1


for dn_idx in range(len(dn_index_conditions)):

    dn_range = dn_index_conditions[dn_idx]
    print(f"Processing Dn Index Range: {dn_range[0]:.2f} to {dn_range[1]:.2f}")

    analyst_list = []

    for redshift_idx in range(len(redshift_conditions)):
        condition_name = f"Redshift_{redshift_conditions[redshift_idx][0]:.2f}_to_{redshift_conditions[redshift_idx][1]:.2f}"
        analyst = FL.DustAttenuationAnalyst(DJAv4Catalog)

        surveyid_subid_list = []

        for surveyid_subid, entry in DJAv4Catalog.catalog_iterator():
            if not entry['sample_flag']:
                continue

            try:
                measurements = entry['dust_curve_parameters']['Corrected_Flux_Density_Measurements']
                dn_index = measurements['red_flux_density'] / \
                    measurements['blue_flux_density']
                if dn_condition_func(dn_index) != dn_idx:
                    continue

            except KeyError:
                continue

            # 获取 Flux Ratio
            try:
                redshift = entry['determined_redshift']
            except KeyError:
                continue

            # 检查是否属于当前的 Flux Ratio Bin
            if redshift_condition_func(redshift) != redshift_idx:
                continue
            surveyid_subid_list.append(surveyid_subid)

        print(
            f"Processing Condition: {condition_name}, Total Samples: {len(surveyid_subid_list)}")
        # 3. 载入目标 ID 列表
        analyst.load_and_group_objects(surveyid_subid_list)
        analyst.compute_average_spectra()
        analyst.compute_attenuation_curves(mode='smoothed')
        # 5. 拟合与分析
        analyst.fit_polynomial(order=3)
        # 假设我们想让无穷远处截距匹配 Calzet
        analyst.derive_k_curve_parameters(intercept_val=-0.89877)
        analyst_list.append(analyst)

    fig, ax = plt.subplots(figsize=(16, 14))
    if_ref = True
    colors_list = plt.cm.viridis(np.linspace(0, 1, len(analyst_list)))
    for analyst in analyst_list:

        condition_name = f"Redshift {redshift_conditions[analyst_list.index(analyst)][0]:.2f}"

        analyst.plot_reddening_curve_comparison(
            if_show=False,
            if_input=True, fig=fig, ax=ax,
            curve_color=colors_list[analyst_list.index(analyst)],
            curve_label=condition_name,
            draw_background=if_ref  # 第一次绘制，需要参考线
        )
        if_ref = False  # 之后不需要参考线

    ax.title.set_text(
        f"Dn Index Range: {dn_range[0]:.2f} to {dn_range[1]:.2f}")

    plt.show()

# %%
