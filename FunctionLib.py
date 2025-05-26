import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import astropy
import astropy.io as io
import astropy.units as units
import specutils

import inspect


def Load_Redshift(Pandas_Dataframe, File_Name):
    """
    This function retrieves the redshift value for a given spectra file from the DJA dataframe.
    Parameters
    ----------
    Pandas_Dataframe : pandas.DataFrame
        The dataframe containing the DJA catalog data.
    File_Name : str
        The name of the file for which to load the redshift.
    Returns
    -------
    np.float32
        The redshift value for the specified file.

    Errors
    -------
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
        The Spectrum1D object containing the spectra, with the flux in F_lambda units.

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
        Error= Spectra_Data['err']*units.uJy

        Flux_Lambda = Flux.to(units.erg / (units.cm**2 * units.s * units.AA), equivalencies=units.spectral_density(Wavelength))
        Error_Lambda = Error.to(units.erg / (units.cm**2 * units.s * units.AA), equivalencies=units.spectral_density(Wavelength))
        Error_Lambda = astropy.nddata.StdDevUncertainty(Error_Lambda)


        return specutils.Spectrum1D(flux=Flux_Lambda, spectral_axis=Wavelength, meta=Spectra_Header, uncertainty=Error_Lambda)

      except Exception as e:
        return e

def Calibrate_Spectra_To_RestFrame(Spectrum, Redshift):

    """
    Calibrate the spectra to the rest frame using the redshift value.

    Parameters
    ----------
    Spectrum : specutils.Spectrum1D
        The Spectrum1D object containing the spectra.
    Redshift : float
        The redshift value to use for calibration.

    Returns
    -------
    specutils.Spectrum1D
        The calibrated Spectrum1D object.

    Errors
    -------
    Returns the error message if there is an issue with the calibration.
    """
    try:
        Restframe_Spectrum_Wavelength = Spectrum.spectral_axis / (1 + Redshift)
        Spectrum = specutils.Spectrum1D(
            flux=Spectrum.flux,
            spectral_axis=Restframe_Spectrum_Wavelength,
            uncertainty=Spectrum.uncertainty,
            meta=Spectrum.meta
        )
        return Spectrum
    except Exception as e:
        return e


def Free(*args):
    """
    Free up memory by deleting specified variables from the local or global namespace.

    Parameters
    ----------
    args : str
        Variable names to be deleted.
        If a variable name is not a string, an error message will be printed.
        If a variable is not found in the local or global namespace, a warning will be printed.

    Returns
    -------
    None
        This function does not return any value. It only deletes the specified variables and triggers garbage collection.

    """
    caller_globals = inspect.currentframe().f_back.f_globals
    caller_locals = inspect.currentframe().f_back.f_locals

    if not args:
        return

    for var_name in args:
        if not isinstance(var_name, str):
            print(f"Error: Variable name '{var_name}' must be a string")
            continue

        if var_name in caller_locals:
            namespace = caller_locals
        elif var_name in caller_globals:
            namespace = caller_globals
        else:
            print(f"Warning: Variable '{var_name}' not found in local or global namespace")
            continue

        try:
            del namespace[var_name]
        except Exception as e:
            print(f"Error: Could not delete variable '{var_name}': {e}")
            continue
    import gc
    gc.collect()
    return None
