import warnings
import FunctionLib as FL
import inspect
from tqdm import tqdm
import astropy
import wave
from matplotlib.image import resample
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict
import re
import scipy

mpl.rcParams['font.family'] = 'serif'


warnings.filterwarnings("ignore")

DJAv4Catalog=FL.Spectrum_Catalog()
DJAv4Catalog.load_from_pkl(os.path.expanduser('~/DustCurve/spectrum_catalog.pkl'))


i=0
for id_subid, catalog in DJAv4Catalog.catalog_iterator():
    i+=1
    if not catalog.get('prism_filepath'):
        continue

    spectrum=FL.Load_Spectrum_From_Fits(catalog['prism_filepath'], redshift=catalog['prism_redshift'])
    spectrum.set_boundarys(1268*astropy.units.Angstrom, 2580*astropy.units.Angstrom)

    Fitter=FL.SpectralLineFitter(spectrum, 4863*astropy.units.Angstrom)

    exp_fit_result=Fitter.fit_exponential()
    if exp_fit_result.get('error'):
        print(f'Exponential fit error for {id_subid}: {exp_fit_result["error"]}')
        continue
    print(f'Exponential fit result for {id_subid}:'+str(exp_fit_result['parameters']))


    if i>10:
        break
