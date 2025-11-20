import os
os.environ["OPENBLAS_NUM_THREADS"] = "64"

from copy import deepcopy as copy
import matplotlib.pyplot as plt

import warnings

import astropy.units as u
import FunctionLib as FL
import inspect
from tqdm import tqdm
import astropy
import wave
import numpy as np
import pandas as pd
import os
import pathlib
import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
from collections import defaultdict
import re
import scipy
from astropy.io import fits as asfits
from s3fit import FitFrame

mpl.rcParams['font.family'] = 'serif'


import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

DJAv4Catalog = FL.Spectrum_Catalog()
DJAv4Catalog.load_from_pkl(os.path.expanduser(
    './DJAV4.2Catalog.pkl'))
print(DJAv4Catalog.sample_num())

DJAv4Catalog.to_dataframe()

s3fit_results_path = pathlib.Path('./s3fit_results/')
s3fit_results_path.mkdir(parents=True, exist_ok=True)

figs_path=pathlib.Path('./figs_spectrum_fits/')
figs_path.mkdir(parents=True, exist_ok=True)

ssp_config = {'main': {'pars': [[-1000, 1000, 'free'], [100, 1200, 'free'], [0, 5.0, 'free'],
                        [0, 0.94, 'free'], [-1, 1, 'free']],
                'info': {'age_min': -2.25, 'age_max': 'universe', 'met_sel': 'solar', 'sfh_name': 'exponential'} },
        'young': {'pars': [[None, None, 'ssp:main:0'], [None, None, 'ssp:main:1'], [None, None, 'ssp:main:2'],
                            [-2, -1, 'free'], [-1, -1, 'fix']],
                'info': {'age_min': -2.25, 'age_max': 0, 'met_sel': 'solar', 'sfh_name': 'constant'} } }

ssp_file = './models/popstar_for_s3fit.fits'

el_config = {'NLR': {'pars':       [[ -500,   500, 'free'], [1,  750, 'free'], [0, 5, 'free'], [1.3, 4.3, 'free'], [4, None, 'fix']],
                    'info': {'line_used': ['all']}},
             'outflow_1': {'pars': [[-2000,   100, 'free'], [750, 2500, 'free'], [0, 5, 'free'], [1.3, 4.3, 'free'], [4, None, 'fix']],
                           'info': {'line_used': ['all']}},
             }


model_config = {'ssp': {'enable': True, 'config': ssp_config, 'file': ssp_file},
                'el': {'enable': True, 'config': el_config, 'use_pyneb': True}}


def plot_spectrum_and_save(survey_id_subid, disperser_filter, converted_flux, converted_flux_error, converted_wave, redshift, save_path=figs_path):
    fig, ax=plt.subplots(figsize=(25,14))
    ax.plot(converted_wave.value, converted_flux.value, color='blue', lw=1, label='Observed Spectrum')
    ax.fill_between(converted_wave.value, converted_flux.value - 1.0*converted_flux_error.value,
                    converted_flux.value + 1.0*converted_flux_error.value, color='gray', alpha=0.5, label='1$\sigma$ Error Region')
    ax.fill_between(converted_wave.value, converted_flux.value - 3.0*converted_flux_error.value,
                    converted_flux.value + 3.0*converted_flux_error.value, color='gray', alpha=0.25, label='3$\sigma$ Error Region')
    oiii_mask = (converted_wave.value > 4960*(1+redshift)) & (converted_wave.value < 5030*(1+redshift))
    halpha_mask = (converted_wave.value > 6550*(1+redshift)) & (converted_wave.value < 6570*(1+redshift))
    y_range_threshold = np.nanmax(converted_flux.value[halpha_mask])*1.2
    ax.set_ylim(-0.5*y_range_threshold, 1.0*y_range_threshold)
    ax.set_xlim(4501*(1+redshift), 6750*(1+redshift))



    ax.minorticks_on()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')


    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_tick_params(width=2, direction='in', which='both', labelsize=20)
    ax.yaxis.set_tick_params(width=2, direction='in', which='both', labelsize=20)
    ax.xaxis.set_tick_params(length=6, which='major')
    ax.xaxis.set_tick_params(length=3, which='minor')
    ax.yaxis.set_tick_params(length=6, which='major')
    ax.yaxis.set_tick_params(length=3, which='minor')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.grid(visible=True, which='major', color='gray', linestyle='--', linewidth=1)
    ax.grid(visible=True, which='minor', color='lightgray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Observed Wavelength (Å)', fontsize=24)
    ax.set_ylabel(r'$F_\lambda$ ($\rm{erg\ s^{-1}\ cm^{-2}\ \AA^{-1}}$)', fontsize=24)
    #add another x axis on the top to show rest-frame wavelength
    ax_top = ax.secondary_xaxis('top', functions=(lambda x: x / (1 + redshift), lambda x: x * (1 + redshift)))
    ax_top.minorticks_on()
    ax_top.set_xlabel('Rest-frame Wavelength (Å)', fontsize=24)
    ax_top.xaxis.set_tick_params(width=2, direction='in', which='both', labelsize=20)
    ax_top.xaxis.set_tick_params(length=6, which='major')
    ax_top.xaxis.set_tick_params(length=3, which='minor')
    fig.suptitle(f'Spectrum for {survey_id_subid} - {disperser_filter}', fontsize=28)
    ax.legend(fontsize=20, loc='upper center')
    plt.savefig(save_path/(f'spectrum_{survey_id_subid}_{disperser_filter}.pdf'), bbox_inches='tight')
    plt.close(fig)

    return


for survey_id_subid, catalog in DJAv4Catalog.catalog_iterator():
    try:
        if not catalog['sample_flag']:
            continue
        for key in catalog['line_ratios'].keys():
            if catalog['line_ratios'][key]>2.86:
                continue


        for disperser_filter, _ in catalog['lines_fit'].items():
            grating_filepath=catalog['grating_filepaths'][disperser_filter]

            redshift=catalog['determined_redshift']

            disperser_name, filter_name = disperser_filter.split('-')

            if pathlib.Path.joinpath(s3fit_results_path, f's3fit_{survey_id_subid}_{disperser_filter}.pkl.gz').exists():
                print(f's3fit result for {survey_id_subid} - {disperser_filter} exists, skip fitting.')
                continue

            with asfits.open(grating_filepath) as grating_spectrum_fits:

                observed_wave=grating_spectrum_fits[1].data['wave']*u.micron
                observed_flux=grating_spectrum_fits[1].data['flux']*u.mJy
                observed_flux_error=grating_spectrum_fits[1].data['err']*u.mJy

                non_nan_mask = np.isfinite(observed_flux.value) & np.isfinite(observed_flux_error.value)


                target_flux_lambda_unit = u.erg / u.s / u.cm**2 / u.AA

                converted_flux = observed_flux.to(target_flux_lambda_unit, equivalencies=u.spectral_density(observed_wave))
                converted_flux_error = observed_flux_error.to(target_flux_lambda_unit, equivalencies=u.spectral_density(observed_wave))

                converted_wave = observed_wave.to(u.AA)


                error_threshold = 10*np.nanmedian(converted_flux_error.value) * u.erg / u.s / u.cm**2 / u.AA
                non_nan_mask &= (converted_flux_error < error_threshold)




                spec_valid_range = []
                if non_nan_mask.any():
                    mask_int = non_nan_mask.astype(int)
                    padded_mask = np.concatenate(([0], mask_int, [0]))
                    diff = np.diff(padded_mask)

                    starts_indices = np.where(diff == 1)[0]
                    ends_indices = np.where(diff == -1)[0] - 1

                    for start_idx, end_idx in zip(starts_indices, ends_indices):
                        start_wave = converted_wave[start_idx].value
                        end_wave = converted_wave[end_idx].value
                        spec_valid_range.append([start_wave, end_wave])

                resolution_fits=asfits.open(pathlib.Path('~/jwst_disperser_resolution').expanduser()/('jwst_nirspec_'+disperser_name+'_disp.fits'))
                R_array=resolution_fits[1].data['R']
                R_wave=(resolution_fits[1].data['WAVELENGTH']*u.micron).to(u.AA).value

                k,b=np.polyfit(R_wave, R_array, 1)

                spec_R_array=k*converted_wave.value+b

                plot_spectrum_and_save(survey_id_subid, disperser_filter, converted_flux, converted_flux_error, converted_wave, redshift, save_path=figs_path)


                FF_biSFH=FitFrame(spec_wave_w=converted_wave.value[non_nan_mask], spec_flux_w=converted_flux.value[non_nan_mask], spec_ferr_w=converted_flux_error.value[non_nan_mask], spec_R_inst_w=spec_R_array[non_nan_mask], spec_valid_range=None, v0_redshift=redshift, model_config=model_config, num_mocks=0, print_step=False,plot_step=False,examine_result=False)
                FF_biSFH.main_fit()
                FF_biSFH.save_to_file(pathlib.Path.joinpath(s3fit_results_path, f's3fit_{survey_id_subid}_{disperser_filter}.pkl.gz'))


    except Exception as e:
        print(e)
        continue



for survey_id_subid, catalog in DJAv4Catalog.catalog_iterator():

    try:
        if not catalog['sample_flag']:
            continue
        for key in catalog['line_ratios'].keys():
            if catalog['line_ratios'][key]<=2.86:
                continue

        for disperser_filter, _ in catalog['lines_fit'].items():
            grating_filepath=catalog['grating_filepaths'][disperser_filter]

            redshift=catalog['determined_redshift']

            disperser_name, filter_name = disperser_filter.split('-')

            with asfits.open(grating_filepath) as grating_spectrum_fits:

                observed_wave=grating_spectrum_fits[1].data['wave']*u.micron
                observed_flux=grating_spectrum_fits[1].data['flux']*u.mJy
                observed_flux_error=grating_spectrum_fits[1].data['err']*u.mJy

                non_nan_mask = np.isfinite(observed_flux.value) & np.isfinite(observed_flux_error.value)


                target_flux_lambda_unit = u.erg / u.s / u.cm**2 / u.AA

                converted_flux = observed_flux.to(target_flux_lambda_unit, equivalencies=u.spectral_density(observed_wave))
                converted_flux_error = observed_flux_error.to(target_flux_lambda_unit, equivalencies=u.spectral_density(observed_wave))

                converted_wave = observed_wave.to(u.AA)


                error_threshold = 10*np.nanmedian(converted_flux_error.value) * u.erg / u.s / u.cm**2 / u.AA
                non_nan_mask &= (converted_flux_error < error_threshold)


                spec_valid_range = []
                if non_nan_mask.any():
                    mask_int = non_nan_mask.astype(int)
                    padded_mask = np.concatenate(([0], mask_int, [0]))
                    diff = np.diff(padded_mask)

                    starts_indices = np.where(diff == 1)[0]
                    ends_indices = np.where(diff == -1)[0] - 1

                    for start_idx, end_idx in zip(starts_indices, ends_indices):
                        start_wave = converted_wave[start_idx].value
                        end_wave = converted_wave[end_idx].value
                        spec_valid_range.append([start_wave, end_wave])

                resolution_fits=asfits.open(pathlib.Path('~/jwst_disperser_resolution').expanduser()/('jwst_nirspec_'+disperser_name+'_disp.fits'))
                R_array=resolution_fits[1].data['R']
                R_wave=(resolution_fits[1].data['WAVELENGTH']*u.micron).to(u.AA).value

                k,b=np.polyfit(R_wave, R_array, 1)

                spec_R_array=k*converted_wave.value+b

                plot_spectrum_and_save(survey_id_subid, disperser_filter, converted_flux, converted_flux_error, converted_wave, redshift, save_path=figs_path)


                FF_biSFH=FitFrame(spec_wave_w=converted_wave.value[non_nan_mask], spec_flux_w=converted_flux.value[non_nan_mask], spec_ferr_w=converted_flux_error.value[non_nan_mask], spec_R_inst_w=spec_R_array[non_nan_mask], spec_valid_range=None, v0_redshift=redshift, model_config=model_config, num_mocks=0, print_step=False,plot_step=False,examine_result=False)
                FF_biSFH.main_fit()
                FF_biSFH.save_to_file(pathlib.Path.joinpath(s3fit_results_path, f's3fit_{survey_id_subid}_{disperser_filter}.pkl.gz'))


    except Exception as e:
        print(e)
        continue


