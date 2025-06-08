import wave
from matplotlib.image import resample
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import scipy
from scipy import integrate
import scipy.optimize
import astropy
import astropy.io as io
import astropy.nddata
import astropy.constants as const
import astropy.units as units
import specutils
from tqdm import tqdm

import inspect

import FunctionLib as FL
import warnings
warnings.filterwarnings("ignore")


DJA_File_Path_str=os.path.expanduser('~/DJAv4/')

DJA_Catalog_DataFrame=pd.read_csv(os.path.expanduser('~/DustCurve/DJAv4Catalog.csv'))
DJA_File_List_All=[]
number_file=0

if os.path.exists(DJA_File_Path_str):
    for root_dir in os.listdir(DJA_File_Path_str):
        if root_dir.startswith('.'):
            continue
        Root_File_Path_str=os.path.join(DJA_File_Path_str, root_dir)
        DJA_File_List=np.array(os.listdir(Root_File_Path_str))
        number_file+=len(DJA_File_List)
        for file_name in DJA_File_List:
            if file_name.endswith('.fits'):
                DJA_File_List_All.append(os.path.join(Root_File_Path_str, file_name))

print('Number of files in DJAv4:', number_file)

redshift_is=np.where(DJA_Catalog_DataFrame['z']!= np.nan)[0]

for file_name_str in tqdm((DJA_File_List_All[0:2000])):
    if file_name_str.endswith('.fits'):
        if file_name_str.split('/')[-1].split('_')[1]=='prism-clear':
            continue
        redshift_quantity=FL.Load_Redshift(DJA_Catalog_DataFrame, file_name_str.split('/')[-1])
        if isinstance(redshift_quantity, IndexError):
            continue
        if np.isnan(redshift_quantity):
            continue
        if redshift_quantity < 2* astropy.units.dimensionless_unscaled:
            continue


        line_to_fit_restframe_wavelength_quantity=4863.0 * units.Angstrom

        hdul=io.fits.open(os.path.join(DJA_File_Path_str, file_name_str))

        wave= hdul[1].data['wave']* units.micron
        flux= hdul[1].data['flux']*units.mJy

        if wave[0]/(1+redshift_quantity.value)>0.45* units.micron:
            continue

        spectrum= FL.Spectrum_1d(
            observed_wavelengths=wave,
            redshift=redshift_quantity,
            observed_flux_nu=flux
        )
        spectrum.set_boundarys(lower_boundary=0.45 * units.micron, upper_boundary=0.55 * units.micron)


        spectrum.set_boundarys(0.45 * units.micron,0.55 * units.micron)


        fitter=FL.SpectralLineFitter(spectrum, 4863.0 * units.Angstrom, max_components=8, max_iterations=100000,figname=f'{file_name_str.split("/")[-1].split(".")[0]}_fit.png')

        fitter.iterative_gaussian_fitting(
    line_restframe_wavelength=4863.0 * units.Angstrom,
    tolerance=10 * units.Angstrom,
    plot_results=False
)
        fitter.plot_final_decomposition(4863.0 * units.Angstrom)

