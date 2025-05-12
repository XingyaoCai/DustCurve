import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import astropy
import astropy.io as io
import astropy.units as units
import specutils


def Load_Redshift(Pandas_Dataframe, File_Name):
    """
    Load the redshift from the DJA catalog dataframe.
    Parameters:
    Pandas_Dataframe : pandas.DataFrame
        The dataframe containing the DJA catalog data.
    File_Name : str
        The name of the file for which to load the redshift.
    Returns:
    np.float32
        The redshift value for the specified file.

    Errors:
    Returns the error message if the file is not found in the dataframe.
    """
    try:
        index = int(np.where(Pandas_Dataframe['file'] == File_Name)[0][0])
        redshift = Pandas_Dataframe.iloc[index]['z']
        return np.float32(redshift)
    except Exception as e:
        return e

def Load_N_Rescale_Spectra(Fits_FilePath):
    """
    Load the spectra from the FITS file and pack the spectra into a SpecUtils Spectrum1D object with the F_lambda.

    Parameters
    ----------
    Fits_FilePath : str
        The path to the FITS file containing the spectra.

    Returns
    -------
    specutils.Spectrum1D
        The Spectrum1D object containing the spectra.

    Errors
    -------
    Returns the error message if the FITS file cannot be opened or if there is an issue with the data.
    """
    with io.fits.open(Fits_FilePath) as hdul:

        try:
            Spectra_Data = hdul[1].data
            Spectra_Header = hdul[1].header

            Wavelength = Spectra_Data['wave']*units.micron
            Flux = Spectra_Data['flux']*units.uJy

            Flux_Lambda = Flux.to(units.erg / (units.cm**2 * units.s * units.AA), equivalencies=units.spectral_density(Wavelength))

            return specutils.Spectrum1D(flux=Flux_Lambda, spectral_axis=Wavelength, meta=Spectra_Header)

        except Exception as e:
            return e
