import stpsf
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
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from collections import defaultdict
import re
import scipy
from astropy.io import fits as asfits
import logging
from datetime import datetime

mpl.rcParams['font.family'] = 'serif'

log_dir = os.path.expanduser("~/logs")
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join(log_dir, f"fitting_log_{timestamp}.log")
logging.basicConfig(
level=logging.INFO,
format="%(asctime)s [%(levelname)s] %(message)s",
handlers=[
logging.FileHandler(log_file) # Only log to a file, removed StreamHandler
]
)

warnings.filterwarnings("ignore")

DJAv4Catalog = FL.Spectrum_Catalog()
DJAv4Catalog.load_from_pkl(os.path.expanduser(
'./DJAV4.2Catalog.pkl'))
print(DJAv4Catalog.sample_num())

DJAv4Catalog.to_dataframe()

def calculate_halpha_hbeta_psf_for_given_catalog(id, catalog, DJAv4Catalog=DJAv4Catalog):
    if catalog['sample_flag'] == False:
        return 0


    if 'grating_within_coverage' not in catalog:
        return 0

    if catalog['grating_slitloss_correction'].keys():
        return 0



    for filter, grating_filepath in catalog['grating_within_coverage'].items():


        filter_name=filter.split('-')[1]
        halpha_wavelength = 6564.61 * (1 + catalog['determined_redshift']) * u.AA
        hbeta_wavelength = 4862.68 * (1 + catalog['determined_redshift']) * u.AA


        with asfits.open(grating_filepath) as grating_spectrum_fits:
            x_offset_grating = grating_spectrum_fits[1].header.get('SRCXPOS',0.0)
            y_offset_grating = grating_spectrum_fits[1].header.get('SRCYPOS',0.0)
            slitlit_status=grating_spectrum_fits[1].header.get('SHUTSTA','')

            nirspec= stpsf.NIRSpec()
            nirspec.options['source_offset_x']= x_offset_grating*0.20
            nirspec.options['source_offset_y']= y_offset_grating*0.46

            nirspec.filter=filter_name
            nirspec.image_mask='Three adjacent MSA open shutters'

            halpha_wavelength=halpha_wavelength.to(u.m).value
            hbeta_wavelength=hbeta_wavelength.to(u.m).value

            selection_mask=np.zeros((48,48),dtype=bool)
            if slitlit_status=='1x1':
                    x1,x2=23,25
                    y1,y2=22,26
            elif slitlit_status=='x11':
                    x1,x2=23,25
                    y1,y2=22,26
            elif slitlit_status=='11x':
                    x1,x2=23,25
                    y1,y2=22,26
            else:
                    x1,x2=23,25
                    y1,y2=22,26

            selection_mask[y1:y2,x1:x2]=True

            OVERSAMPLE=4

            halpha_psf=nirspec.calc_psf(monochromatic=halpha_wavelength,oversample=OVERSAMPLE)
            hbeta_psf=nirspec.calc_psf(monochromatic=hbeta_wavelength,oversample=OVERSAMPLE)

            halpha_selection_flux=halpha_psf[3].data[selection_mask]
            hbeta_selection_flux=hbeta_psf[3].data[selection_mask]

            halpha_total_flux=np.nansum(halpha_psf[3].data)
            hbeta_total_flux=np.nansum(hbeta_psf[3].data)

            halpha_psf_fraction=np.nansum(halpha_selection_flux)/halpha_total_flux
            hbeta_psf_fraction=np.nansum(hbeta_selection_flux)/hbeta_total_flux

            catalog['grating_slitloss_correction'][f'halpha_psf_fraction_{filter_name}']=halpha_psf_fraction
            catalog['grating_slitloss_correction'][f'hbeta_psf_fraction_{filter_name}']=hbeta_psf_fraction

            DJAv4Catalog.update_catalog_item(id, catalog)
            DJAv4Catalog.save_catalog_to_pkl(pathlib.Path('/home/xingyaocai/DustCurve/DJAV4.2Catalog.pkl'))

    return 1
for id, catalog in (DJAv4Catalog.catalog_iterator()):
    if catalog['sample_flag'] == False:
        continue

    result=calculate_halpha_hbeta_psf_for_given_catalog(id, catalog, DJAv4Catalog=DJAv4Catalog)
    if result==1:
        print(f'Processed catalog id: {id}')

DJAv4Catalog.save_catalog_to_pkl(pathlib.Path('/home/xingyaocai/DustCurve/DJAV4.2Catalog.pkl'))
