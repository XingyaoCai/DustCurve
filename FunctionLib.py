import collections
import inspect
import os
import re

import astropy
import astropy.io
import astropy.nddata
import astropy.units
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import tqdm

import warnings
import copy

from pathlib import Path # Standard and correct way to import



def Load_Spectrum_From_Fits(file_path, redshift=None, index_in_hdulist=1, wavelength_key='wave', wavelength_unit='micron', flux_key='flux', flux_unit='uJy', error_key='error', error_unit='uJy'):
    """
    Load a spectrum from DJA catalog from the given file path, packed it into the Spectrum_1d class, and return the class instance.

    Parameters
    ----------
    file_path : str
        The path to the spectrum file.
    redshift : astropy.units.Quantity, optional
        The redshift value to apply to the spectrum. If not provided, the redshift from the file will be used.

    Returns
    -------
    Spectrum_1d
        An instance of the Spectrum_1d class containing the loaded spectrum data.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    if not file_path.endswith('.fits'):
        raise ValueError(f"The file {file_path} is not a valid FITS file.")

    with astropy.io.fits.open(file_path) as hdulist:
        if index_in_hdulist >= len(hdulist):
            raise IndexError(
                f"Index {index_in_hdulist} is out of bounds for the HDU list.")

        data = hdulist[index_in_hdulist].data

        wavelength_data = data[wavelength_key]
        flux_data = data[flux_key]

        wavelength_unit = astropy.units.Unit(wavelength_unit)
        flux_unit = astropy.units.Unit(flux_unit)

        if redshift is None:
            redshift = Load_Spectrum_Redshift(
                file_path, pd.read_csv(os.path.expanduser('~/DustCurve/DJAv4Catalog.csv')))
        if isinstance(redshift, str):
            redshift = float(redshift)
        elif isinstance(redshift, (int, float)):
            redshift = astropy.units.Quantity(
                redshift, unit=astropy.units.dimensionless_unscaled)
        elif isinstance(redshift, astropy.units.Quantity):
            if redshift.unit.is_equivalent(astropy.units.dimensionless_unscaled):
                redshift = redshift.to_value(
                    astropy.units.dimensionless_unscaled)
            else:
                raise ValueError("Redshift must be a dimensionless quantity.")
        else:
            raise TypeError(
                "Redshift must be a string, int, float, or astropy.units.Quantity.")

        spectrum = Spectrum_1d(
            observed_wavelengths=wavelength_data * wavelength_unit,
            observed_flux_nu=flux_data * flux_unit,
            redshift=redshift
        )

        return spectrum


def Load_Spectrum_Redshift(filepath, catalog):
    """
    Load the redshift from a spectrum file from the catalog.

    Parameters
    ----------
    filepath : str
        The file path of the spectrum file.
    catalog : pandas.DataFrame
        The catalog DataFrame containing spectrum information.

    Returns
    -------
    redshift : astropy.units.Quantity or None
        The redshift of the spectrum, or None if not found.
    """
    if not isinstance(catalog, pd.DataFrame):
        try:
            catalog = pd.read_csv(catalog)
        except Exception as e:
            raise ValueError(
                "Catalog must be a pandas DataFrame or a valid CSV file path.") from e

    filename = filepath.split('/')[-1]

    index_in_catalog = catalog[catalog['file'] == filename].index

    if len(index_in_catalog) == 0:
        return None

    redshift_value = catalog.loc[index_in_catalog[0], 'z']

    if np.isnan(redshift_value):
        redshift_value = catalog.loc[index_in_catalog[0], 'zfit']
        return astropy.units.Quantity(redshift_value, unit=astropy.units.dimensionless_unscaled)
    return astropy.units.Quantity(redshift_value, unit=astropy.units.dimensionless_unscaled)


def Load_Spectrum_Grade(filepath, catalog):
    """
    Load the grade of a spectrum file from the catalog.

    Parameters
    ----------
    filepath : str
        The file path of the spectrum file.
    catalog : pandas.DataFrame
        The catalog DataFrame containing spectrum information.

    Returns
    -------
    grade : int or None
        The grade of the spectrum if found, otherwise None.
    """
    if not isinstance(catalog, pd.DataFrame):
        try:
            catalog = pd.read_csv(catalog)
        except Exception as e:
            raise ValueError(
                "Catalog must be a pandas DataFrame or a valid CSV file path.") from e

    filename = filepath.split('/')[-1]

    index_in_catalog = catalog[catalog['file'] == filename].index

    if len(index_in_catalog) == 0:
        return None

    grade_value = catalog.loc[index_in_catalog[0], 'grade']

    if np.isnan(grade_value):
        grade_value = catalog.loc[index_in_catalog[0], 'grade_fit']
        return int(grade_value)
    return int(grade_value)


def Load_N_Rescale_Spectra(Fits_FilePath):
    """
    Load the spectra from the FITS file and pack the spectra into astropy.nddata.NDDataArray objects.

    Parameters
    ----------
    Fits_FilePath : str
        The path to the FITS file containing the spectra.

    Returns
    -------
    Observed Wavelength : astropy.nddata.NDDataArray
        The observed wavelengths of the spectrum.
    Observed Flux (F_lambda) : astropy.nddata.NDDataArray
        The observed flux values of the spectrum in F_lambda units, containing the error as uncertainty.
    Observed Flux (F_mu) : astropy.nddata.NDDataArray
        The observed flux values of the spectrum in F_mu units, containing the error as uncertainty.

    Errors
    -------
    Returns the error message if the FITS file cannot be opened or if there is an issue with the data.
    """
    with astropy.io.fits.open(Fits_FilePath) as hdul:

        try:
            Spectra_Data = hdul[1].data

            Wavelength = Spectra_Data['wave']*astropy.units.micron
            Flux = Spectra_Data['flux']*astropy.units.uJy
            Error = Spectra_Data['err']*astropy.units.uJy

            Flux_Lambda = Flux.to(astropy.units.erg / (astropy.units.cm**2 * astropy.units.s *
                                  astropy.units.AA), equivalencies=astropy.units.spectral_density(Wavelength))
            Error_Lambda = Error.to(astropy.units.erg / (astropy.units.cm**2 * astropy.units.s *
                                    astropy.units.AA), equivalencies=astropy.units.spectral_density(Wavelength))
            Error_Lambda = astropy.nddata.StdDevUncertainty(Error_Lambda)
            Error = astropy.nddata.StdDevUncertainty(Error)

            return astropy.nddata.NDDataArray(Wavelength), astropy.nddata.NDDataArray(Flux_Lambda, uncertainty=Error_Lambda), astropy.nddata.NDDataArray(Flux, uncertainty=Error)

        except Exception as e:
            return e

# def Calibrate_Spectra_To_RestFrame(Spectrum, Redshift):

#     """
#     Calibrate the spectra to the rest frame using the redshift value.

#     Parameters
#     ----------
#     Spectrum : specutils.Spectrum1D
#         The Spectrum1D object containing the spectra.
#     Redshift : float
#         The redshift value to use for calibration.

#     Returns
#     -------
#     specutils.Spectrum1D
#         The calibrated Spectrum1D object.

#     Errors
#     -------
#     Returns the error message if there is an issue with the calibration.
#     """
#     try:
#         Restframe_Spectrum_Wavelength = Spectrum.spectral_axis / (1 + Redshift)
#         Spectrum = specutils.Spectrum1D(
#             flux=Spectrum.flux,
#             spectral_axis=Restframe_Spectrum_Wavelength,
#             uncertainty=Spectrum.uncertainty,
#             meta=Spectrum.meta
#         )
#         return Spectrum
#     except Exception as e:
#         return e


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
            print(
                f"Warning: Variable '{var_name}' not found in local or global namespace")
            continue

        try:
            del namespace[var_name]
        except Exception as e:
            print(f"Error: Could not delete variable '{var_name}': {e}")
            continue
    import gc
    gc.collect()
    return None


class Spectrum_1d:
    """

    A class to store and manipulate a 1D spectrum. The initialization requires the observed
    wavelengths, fluxes in F_nu or F_lambda units, and the redshift value. The initialization
    will automatically calculate the rest frame wavelengths and rest frame fluxes and store them
    as astropy.nddata.NDDataArray objects.

    Attributes
    ----------
    observed_wavelengths : astropy.nddata.NDDataArray
        The observed wavelengths of the spectrum.
    observed_flux_nu : astropy.nddata.NDDataArray
        The observed flux values in F_nu units.
    observed_flux_lambda : astropy.nddata.NDDataArray
        The observed flux values in F_lambda units.
    redshift : astropy.units.Quantity
        The redshift value of the spectrum.
    restframe_wavelengths : astropy.nddata.NDDataArray
        The rest frame wavelengths of the spectrum.
    restframe_flux_nu : astropy.nddata.NDDataArray
        The rest frame flux values in F_nu units, corrected for cosmological dimming.
    restframe_flux_lambda : astropy.nddata.NDDataArray
        The rest frame flux values in F_lambda units, corrected for cosmological dimming.
    processing_wavelengths : astropy.nddata.NDDataArray
        The wavelengths currently being processed or analyzed, typically a subset of the rest-frame wavelengths.
    processing_flux_lambda: astropy.nddata.NDDataArray
        The F_lambda flux currently being processed, corresponding to the processing_wavelengths.
    processing_flux_nu: astropy.nddata.NDDataArray
        The F_nu flux currently being processed, corresponding to the processing_wavelengths.
    """

    def __init__(self, observed_wavelengths, redshift, observed_flux_nu=None, observed_flux_lambda=None):
        """
        Initializes the Spectrum_1d object with the given parameters.

        Parameters
        ----------
        observed_wavelengths : astropy.nddata.NDDataArray or astropy.units.Quantity
            The observed wavelengths of the spectrum, should contain the unit.
        redshift : float, int, or astropy.units.Quantity
            The redshift value of the spectrum.
        observed_flux_nu : astropy.nddata.NDDataArray or astropy.units.Quantity, optional
            The observed flux values of the spectrum in F_nu units, should contain the unit and uncertainty.
        observed_flux_lambda : astropy.nddata.NDDataArray or astropy.units.Quantity, optional
            The observed flux values of the spectrum in F_lambda units, should contain the unit and uncertainty.
        """
        # Check that at least one flux is provided
        if observed_flux_lambda is None and observed_flux_nu is None:
            raise ValueError(
                "At least one of observed_flux_nu or observed_flux_lambda must be provided.")

        # Handle wavelengths - allow both NDDataArray and Quantity
        if isinstance(observed_wavelengths, astropy.nddata.NDDataArray):
            self.observed_wavelengths = observed_wavelengths
        elif isinstance(observed_wavelengths, astropy.units.Quantity):
            self.observed_wavelengths = astropy.nddata.NDDataArray(
                observed_wavelengths)
        else:
            raise TypeError(
                "observed_wavelengths must be an astropy.nddata.NDDataArray or astropy.units.Quantity object.")

        # Handle redshift
        if isinstance(redshift, (float, int)):
            self.redshift = redshift * astropy.units.dimensionless_unscaled
        elif isinstance(redshift, astropy.units.Quantity):
            if redshift.unit.is_equivalent(astropy.units.dimensionless_unscaled):
                self.redshift = redshift
            else:
                raise ValueError("Redshift must be dimensionless.")
        else:
            raise TypeError(
                "Redshift must be a float, int, or dimensionless astropy.units.Quantity.")

        # Calculate rest-frame wavelengths
        obs_wave_data_attr = self.observed_wavelengths.data
        if isinstance(obs_wave_data_attr, astropy.units.Quantity):
            obs_wave_values = obs_wave_data_attr.value
        else:
            obs_wave_values = obs_wave_data_attr

        rest_wave_data = obs_wave_values / (1 + self.redshift.value)

        rest_wave_uncertainty_values = None
        if self.observed_wavelengths.uncertainty is not None:
            rest_wave_uncertainty_values = self.observed_wavelengths.uncertainty.array / \
                (1 + self.redshift.value)
            rest_wave_uncertainty = type(self.observed_wavelengths.uncertainty)(
                rest_wave_uncertainty_values)
        else:
            rest_wave_uncertainty = None

        self.restframe_wavelengths = astropy.nddata.NDDataArray(
            data=rest_wave_data,
            uncertainty=rest_wave_uncertainty,
            unit=self.observed_wavelengths.unit
        )

        # Handle F_nu flux
        if observed_flux_nu is not None:
            if isinstance(observed_flux_nu, astropy.nddata.NDDataArray):
                self.observed_flux_nu = observed_flux_nu
            elif isinstance(observed_flux_nu, astropy.units.Quantity):
                self.observed_flux_nu = astropy.nddata.NDDataArray(
                    observed_flux_nu)
            else:
                raise TypeError(
                    "observed_flux_nu must be an astropy.nddata.NDDataArray or astropy.units.Quantity object.")
        else:
            self.observed_flux_nu = None

        # Handle F_lambda flux
        if observed_flux_lambda is not None:
            if isinstance(observed_flux_lambda, astropy.nddata.NDDataArray):
                self.observed_flux_lambda = observed_flux_lambda
            elif isinstance(observed_flux_lambda, astropy.units.Quantity):
                self.observed_flux_lambda = astropy.nddata.NDDataArray(
                    observed_flux_lambda)
            else:
                raise TypeError(
                    "observed_flux_lambda must be an astropy.nddata.NDDataArray or astropy.units.Quantity object.")
        else:
            self.observed_flux_lambda = None

        # Auto-convert between flux types if only one is provided
        if self.observed_flux_nu is None and self.observed_flux_lambda is not None:
            self._convert_lambda_to_nu()
        elif self.observed_flux_lambda is None and self.observed_flux_nu is not None:
            self._convert_nu_to_lambda()

        # Calculate rest-frame fluxes
        z_factor = 1 + self.redshift.value
        self.restframe_flux_nu = None
        self.restframe_flux_lambda = None

        if self.observed_flux_nu is not None:
            rest_flux_nu_data = self.observed_flux_nu.data * z_factor
            rest_uncert_nu = None
            if self.observed_flux_nu.uncertainty is not None:
                rest_uncert_nu_data = self.observed_flux_nu.uncertainty.array * z_factor
                rest_uncert_nu = type(self.observed_flux_nu.uncertainty)(rest_uncert_nu_data)
            self.restframe_flux_nu = astropy.nddata.NDDataArray(
                data=rest_flux_nu_data,
                uncertainty=rest_uncert_nu,
                unit=self.observed_flux_nu.unit
            )

        if self.observed_flux_lambda is not None:
            rest_flux_lambda_data = self.observed_flux_lambda.data * (z_factor)
            rest_uncert_lambda = None
            if self.observed_flux_lambda.uncertainty is not None:
                rest_uncert_lambda_data = self.observed_flux_lambda.uncertainty.array * (z_factor)
                rest_uncert_lambda = type(self.observed_flux_lambda.uncertainty)(rest_uncert_lambda_data)
            self.restframe_flux_lambda = astropy.nddata.NDDataArray(
                data=rest_flux_lambda_data,
                uncertainty=rest_uncert_lambda,
                unit=self.observed_flux_lambda.unit
            )

        # Set default processing frame to be the rest frame
        self.processing_wavelengths = self.restframe_wavelengths
        self.processing_flux_lambda = self.restframe_flux_lambda
        self.processing_flux_nu = self.restframe_flux_nu

        self.processing_wavelengths = self.processing_wavelengths.convert_unit_to(
            astropy.units.AA)

        # Handle NaN values by converting them to 0
        self._handle_nan_values()

    def _handle_nan_values(self):
        """Convert NaN values to 0 in wavelengths and fluxes."""
        attributes_to_clean = [
            'observed_wavelengths', 'restframe_wavelengths',
            'observed_flux_nu', 'observed_flux_lambda',
            'restframe_flux_nu', 'restframe_flux_lambda'
        ]

        for attr_name in attributes_to_clean:
            attr = getattr(self, attr_name)
            if attr is not None:
                data_array = attr.data
                if isinstance(data_array, astropy.units.Quantity):
                    nan_mask = np.isnan(data_array.value)
                    if np.any(nan_mask):
                        new_values = data_array.value.copy()
                        new_values[nan_mask] = 0
                        attr.data = new_values * data_array.unit
                else:
                    nan_mask = np.isnan(data_array)
                    if np.any(nan_mask):
                        attr.data[nan_mask] = 0

                if attr.uncertainty is not None:
                    uncertainty_array = attr.uncertainty.array
                    if uncertainty_array is not None:
                        nan_mask = np.isnan(uncertainty_array)
                        if np.any(nan_mask):
                            uncertainty_array[nan_mask] = 0

    def _get_quantity_from_nddata(self, nddata_array):
        if nddata_array is None:
            return None
        data_attr = nddata_array.data
        if isinstance(data_attr, astropy.units.Quantity):
            return data_attr
        elif nddata_array.unit is not None:
            return data_attr * nddata_array.unit
        else:
            raise ValueError(
                "NDDataArray is missing unit information for quantity conversion.")

    def _convert_lambda_to_nu(self):
        if self.observed_flux_lambda is None:
            raise ValueError(
                "Cannot convert from F_lambda: observed_flux_lambda is None.")

        F_lambda_quantity_to_convert = self._get_quantity_from_nddata(
            self.observed_flux_lambda)
        obs_wave_quantity = self._get_quantity_from_nddata(self.observed_wavelengths)
        target_F_nu_unit = astropy.units.erg / \
            (astropy.units.cm**2 * astropy.units.s * astropy.units.Hz)
        F_nu_converted = F_lambda_quantity_to_convert.to(
            target_F_nu_unit,
            equivalencies=astropy.units.spectral_density(obs_wave_quantity)
        )
        flux_nu_data = F_nu_converted.value
        uncertainty_nu_obj = None
        if self.observed_flux_lambda.uncertainty is not None:
            uncertainty_F_lambda_values = self.observed_flux_lambda.uncertainty.array
            uncertainty_F_lambda_quantity = uncertainty_F_lambda_values * \
                self.observed_flux_lambda.unit
            uncertainty_F_nu_converted = uncertainty_F_lambda_quantity.to(
                target_F_nu_unit,
                equivalencies=astropy.units.spectral_density(
                    obs_wave_quantity)
            )
            uncertainty_nu_data = uncertainty_F_nu_converted.value
            uncertainty_nu_obj = type(
                self.observed_flux_lambda.uncertainty)(uncertainty_nu_data)

        self.observed_flux_nu = astropy.nddata.NDDataArray(
            data=flux_nu_data,
            uncertainty=uncertainty_nu_obj,
            unit=target_F_nu_unit
        )

    def _convert_nu_to_lambda(self):
        if self.observed_flux_nu is None:
            raise ValueError(
                "Cannot convert from F_nu: observed_flux_nu is None.")

        F_nu_quantity_to_convert = self._get_quantity_from_nddata(
            self.observed_flux_nu)
        obs_wave_quantity = self._get_quantity_from_nddata(self.observed_wavelengths)
        target_F_lambda_unit = astropy.units.erg / \
            (astropy.units.cm**2 * astropy.units.s * astropy.units.AA)
        F_lambda_converted = F_nu_quantity_to_convert.to(
            target_F_lambda_unit,
            equivalencies=astropy.units.spectral_density(obs_wave_quantity)
        )
        flux_lambda_data = F_lambda_converted.value
        uncertainty_lambda_obj = None
        if self.observed_flux_nu.uncertainty is not None:
            uncertainty_F_nu_values = self.observed_flux_nu.uncertainty.array
            uncertainty_F_nu_quantity = uncertainty_F_nu_values * self.observed_flux_nu.unit
            uncertainty_F_lambda_converted = uncertainty_F_nu_quantity.to(
                target_F_lambda_unit,
                equivalencies=astropy.units.spectral_density(
                    obs_wave_quantity)
            )
            uncertainty_lambda_data = uncertainty_F_lambda_converted.value
            uncertainty_lambda_obj = type(
                self.observed_flux_nu.uncertainty)(uncertainty_lambda_data)

        self.observed_flux_lambda = astropy.nddata.NDDataArray(
            data=flux_lambda_data,
            uncertainty=uncertainty_lambda_obj,
            unit=target_F_lambda_unit
        )

    def set_boundarys(self, lower_boundary=None, upper_boundary=None):
        """
        Set the lower and upper boundaries for the spectrum on the rest-frame data.

        Parameters
        ----------
        lower_boundary : astropy.units.Quantity, optional
            The lower boundary of the spectrum. If None, no lower boundary is set.
        upper_boundary : astropy.units.Quantity, optional
            The upper boundary of the spectrum. If None, no upper boundary is set.
        """
        indices = None
        if lower_boundary is not None and upper_boundary is None:
            lower_boundary = lower_boundary.to(self.restframe_wavelengths.unit)
            indices = self.restframe_wavelengths.data >= lower_boundary.value
        elif upper_boundary is not None and lower_boundary is None:
            upper_boundary = upper_boundary.to(self.restframe_wavelengths.unit)
            indices = self.restframe_wavelengths.data <= upper_boundary.value
        elif lower_boundary is not None and upper_boundary is not None:
            lower_boundary = lower_boundary.to(self.restframe_wavelengths.unit)
            upper_boundary = upper_boundary.to(self.restframe_wavelengths.unit)
            indices = (self.restframe_wavelengths.data >= lower_boundary.value) & \
                      (self.restframe_wavelengths.data <= upper_boundary.value)

        if indices is not None:
            self.processing_wavelengths = self.restframe_wavelengths[indices]
            self.processing_flux_lambda = self.restframe_flux_lambda[indices] if self.restframe_flux_lambda is not None else None
            self.processing_flux_nu = self.restframe_flux_nu[indices] if self.restframe_flux_nu is not None else None

        self.processing_wavelengths = self.processing_wavelengths.convert_unit_to(
            astropy.units.AA)

    def get_flux(self, unit='lambda', lower_boundary=None, upper_boundary=None):
        """
        Applies boundaries and returns the processed flux array in the specified unit.

        Parameters
        ----------
        unit : str, optional
            The desired flux unit. Can be 'lambda' or 'nu'. Defaults to 'lambda'.
        lower_boundary : astropy.units.Quantity, optional
            The lower boundary to apply to the spectrum.
        upper_boundary : astropy.units.Quantity, optional
            The upper boundary to apply to the spectrum.

        Returns
        -------
        astropy.nddata.NDDataArray
            The processed flux array.
        """
        # The original logic of get_flux was slightly confusing.
        # This implementation first sets the boundaries and then returns the requested flux.
        if lower_boundary is not None or upper_boundary is not None:
            self.set_boundarys(lower_boundary, upper_boundary)

        if unit.lower() in ['lambda', 'f_lambda', 'flambda']:
            return self.processing_flux_lambda
        elif unit.lower() in ['nu', 'f_nu', 'fnu']:
            return self.processing_flux_nu
        else:
            raise ValueError("Unit must be 'lambda' or 'nu'.")

    def show(self, if_show=True, flux_type='flambda', plot_range='processing',if_RestFrame=True):
        """
        Display the processed spectrum (defaults to F_lambda vs wavelength).

        Parameters
        ----------
        if_show : bool, optional
            If True, the plot will be displayed. Default is True.
        flux_type : str, optional
            The type of flux to plot. Can be 'flambda' or 'fnu'.
            Default is 'flambda'.
        plot_range : str, optional
            The range of the spectrum to plot. Can be 'processing' or 'full'.
            Default is 'processing'.

        if_RestFrame : bool, optional
            If True, the rest-frame spectrum will be plotted. Default is True.

        Returns
        -------
        fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        """
        fig, ax = plt.subplots(figsize=(20, 10))

        if self.processing_wavelengths is not None and self.processing_flux_lambda is not None:
            if flux_type.lower() not in ['flambda', 'f_lambda', 'lambda', 'fnu', 'f_nu', 'nu']:
                raise ValueError(
                    "flux_type must be 'flambda' or 'f_lambda' for F_lambda flux.")
            if plot_range.lower() not in ['processing', 'full']:
                raise ValueError(
                    "plot_range must be 'processing' or 'full'.")
            if flux_type.lower() == 'flambda' or flux_type.lower() == 'f_lambda' or flux_type.lower() == 'lambda':

                if plot_range.lower() == 'full':
                    if if_RestFrame:
                        wave_data = self.restframe_wavelengths.data
                        wave_unit = self.restframe_wavelengths.unit

                        flux_data = self.restframe_flux_lambda.data
                        flux_unit = self.restframe_flux_lambda.unit
                    else:
                        wave_data = self.observed_wavelengths.data
                        wave_unit = self.observed_wavelengths.unit

                        flux_data = self.observed_flux_lambda.data
                        flux_unit = self.observed_flux_lambda.unit
                elif plot_range.lower() == 'processing':
                    if if_RestFrame:
                        wave_data = self.processing_wavelengths.data
                        wave_unit = self.processing_wavelengths.unit

                        flux_data = self.processing_flux_lambda.data
                        flux_unit = self.processing_flux_lambda.unit
                    else:
                        wave_data = self.observed_wavelengths.data
                        wave_unit = self.observed_wavelengths.unit

                        flux_data = self.observed_flux_lambda.data
                        flux_unit = self.observed_flux_lambda.unit

            elif flux_type.lower() == 'fnu' or flux_type.lower() == 'f_nu' or flux_type.lower() == 'nu':
                if plot_range.lower() == 'full':
                    if if_RestFrame:
                        wave_data = self.restframe_wavelengths.data
                        wave_unit = self.restframe_wavelengths.unit

                        flux_data = self.restframe_flux_nu.data
                        flux_unit = self.restframe_flux_nu.unit
                    else:
                        wave_data = self.observed_wavelengths.data
                        wave_unit = self.observed_wavelengths.unit

                        flux_data = self.observed_flux_nu.data
                        flux_unit = self.observed_flux_nu.unit
                elif plot_range.lower() == 'processing':
                    if if_RestFrame:
                        wave_data = self.processing_wavelengths.data
                        wave_unit = self.processing_wavelengths.unit

                        flux_data = self.processing_flux_nu.data
                        flux_unit = self.processing_flux_nu.unit
                    else:
                        wave_data = self.observed_wavelengths.data
                        wave_unit = self.observed_wavelengths.unit

                        flux_data = self.observed_flux_nu.data
                        flux_unit = self.observed_flux_nu.unit
            else:
                raise ValueError(
                    "flux_type must be 'flambda' or 'fnu'.")

            ax.plot(wave_data, flux_data,label=f'Processed Spectrum ({flux_type})', color='blue')

            if flux_type.lower() in ['flambda', 'f_lambda', 'lambda']:
                ax.set_ylabel(f'Rest Flux ({flux_unit})' if if_RestFrame else f'Observed Flux ({flux_unit})')
                ax.set_xlabel(f'Rest Wavelength ({wave_unit})' if if_RestFrame else f'Observed Wavelength ({wave_unit})')
            elif flux_type.lower() in ['fnu', 'f_nu', 'nu']:
                ax.set_xlabel(f'Rest Wavelength ({wave_unit})' if if_RestFrame else f'Observed Wavelength ({wave_unit})')
                ax.set_ylabel(f'Rest Flux ({flux_unit})' if if_RestFrame else f'Observed Flux ({flux_unit})')
            ax.set_title(f'Spectrum at Redshift {self.redshift.value:.3f}')
            ax.legend()
            if if_show:
                plt.show()
        return fig, ax

    def dual_boundarys(self, if_process=False, unit=None):
        """
        Return the lower and upper boundaries of the spectrum.

        Parameters
        ----------
        if_process : bool, optional
            If True, returns the boundaries of the currently processed data.
            Otherwise, returns the boundaries of the full rest-frame spectrum. Default is False.
        unit : astropy.units.Unit, optional
            The unit to which the boundaries should be converted.

        Returns
        -------
        tuple
            A tuple containing the lower and upper boundaries of the spectrum.
        """
        wavelengths_to_use = self.processing_wavelengths if if_process else self.restframe_wavelengths
        mask= self.restframe_flux_lambda.data!=0
        wavelengths_to_use = wavelengths_to_use[mask]

        if wavelengths_to_use is not None:
            lower_boundary = wavelengths_to_use.data.min() * wavelengths_to_use.unit
            upper_boundary = wavelengths_to_use.data.max() * wavelengths_to_use.unit
            if unit is not None:
                lower_boundary = lower_boundary.to(unit)
                upper_boundary = upper_boundary.to(unit)
            return lower_boundary, upper_boundary

        return None, None

    def reset(self):
        """Reset the processing wavelengths and fluxes to the original rest-frame values."""
        self.processing_wavelengths = self.restframe_wavelengths
        self.processing_flux_lambda = self.restframe_flux_lambda
        self.processing_flux_nu = self.restframe_flux_nu
        self.processing_wavelengths = self.processing_wavelengths.convert_unit_to(
            astropy.units.AA)

    def __repr__(self):
        """String representation of the spectrum object."""
        wave_data_repr = "N/A"
        num_points_str = "N/A"

        if self.restframe_wavelengths is not None:
            rest_wave_values = self.restframe_wavelengths.data
            if isinstance(rest_wave_values, astropy.units.Quantity):
                rest_wave_values = rest_wave_values.value
            if rest_wave_values is not None and len(rest_wave_values) > 0:
                wave_data_repr = f"{rest_wave_values.min():.2f} - {rest_wave_values.max():.2f} {self.restframe_wavelengths.unit}"
                num_points_str = str(len(rest_wave_values))
            else:
                wave_data_repr = f"Empty {self.restframe_wavelengths.unit}"
                num_points_str = "0"

        return f"Spectrum_1d(z={self.redshift.value:.3f}, λ_res={wave_data_repr}, {num_points_str} points)"


class SpectralLineFitter:
    """
    A class to fit Gaussian components to spectral lines in a spectrum.

    Attributes
    ----------
    spectrum: Spectrum_1d
        An instance of the Spectrum_1d class containing the observed wavelengths and fluxes.
    line_restframe_wavelengths: astropy.units.Quantity or list of astropy.units.Quantity
        The rest-frame wavelengths of the spectral lines to fit.
    max_components: int
        The maximum number of Gaussian components to fit to each spectral line.
    max_iterations: int
        The maximum number of iterations for the fitting process.
    fit_results: list of dict
        A list to store the fitting results for each spectral line.

    Methods
    -------
    Details of the methods will be attached after the method definitions.
    gaussian(x, amplitude, mean, stddev)

    """

    def __init__(self, spectrum, line_restframe_wavelengths, max_components=8, max_iterations=100000):
        """
        Initializes the SpectralLineFitter with a spectrum and spectral lines to fit.

        Parameters
        ----------
        spectrum: Spectrum_1d
            An instance of the Spectrum_1d class containing the observed wavelengths and fluxes.
        line_restframe_wavelengths: astropy.units.Quantity or list of astropy.units.Quantity
            The rest-frame wavelengths of the spectral lines to fit.
        max_components: int, optional
            The maximum number of Gaussian components to fit to each spectral line (default is 8).
        max_iterations: int, optional
            The maximum number of iterations for the fitting process (default is 100000).
        """

        if not isinstance(spectrum, Spectrum_1d):
            raise TypeError("spectrum must be an instance of Spectrum_1d.")

        self.spectrum = spectrum

        if isinstance(line_restframe_wavelengths, astropy.units.Quantity):
            self.line_restframe_wavelengths = [line_restframe_wavelengths]
        elif isinstance(line_restframe_wavelengths, list) and all(isinstance(w, astropy.units.Quantity) for w in line_restframe_wavelengths):
            self.line_restframe_wavelengths = line_restframe_wavelengths
        else:
            raise TypeError(
                "line_restframe_wavelengths must be an astropy.units.Quantity or a list of astropy.units.Quantity objects.")

        self.max_components = max_components
        self.max_iterations = max_iterations
        self.fit_results = []

    def gaussian(self, x, amplitude, mean, stddev):
        """
        Gaussian function for fitting.

        Parameters
        ----------
        x : array-like
            The independent variable (wavelengths).
        amplitude : float
            The height of the Gaussian peak.
        mean : float
            The position of the center of the Gaussian.
        stddev : float
            The standard deviation (width) of the Gaussian.

        Returns
        -------
        array-like
            The values of the Gaussian function at x.
        """
        if isinstance(x, astropy.nddata.NDDataArray):
            x = x.data
            return astropy.nddata.NDDataArray(
                data=amplitude * np.exp(-0.5 * ((x - mean) / stddev) ** 2),
                unit=x.unit
            )
        elif isinstance(x, astropy.units.Quantity):
            return amplitude * np.exp(-0.5 * ((x.value - mean) / stddev) ** 2) * x.unit
        else:
            return amplitude * np.exp(-0.5 * ((x - mean) / stddev) ** 2)

    def gaussian_with_offset(self, x, amplitude, mean, stddev, offset=0):
        """
        Gaussian function with an offset for fitting.

        Parameters
        ----------
        x : array-like
            The independent variable (wavelengths).
        amplitude : float
            The height of the Gaussian peak.
        mean : float
            The position of the center of the Gaussian.
        stddev : float
            The standard deviation (width) of the Gaussian.
        offset : float, optional
            An offset to be added to the Gaussian (default is 0).

        Returns
        -------
        array-like
            The values of the Gaussian function at x with an offset.

        """

        if isinstance(x, astropy.nddata.NDDataArray):
            x = x.data
            return astropy.nddata.NDDataArray(
                data=amplitude *
                np.exp(-0.5 * ((x - mean) / stddev) ** 2) + offset,
                unit=x.unit
            )
        elif isinstance(x, astropy.units.Quantity):
            return (amplitude * np.exp(-0.5 * ((x.value - mean) / stddev) ** 2) + offset) * x.unit

        else:
            return amplitude * np.exp(-0.5 * ((x - mean) / stddev) ** 2) + offset

    def power_law(self, x, amplitude, exponent):
        """
        Power law function for fitting.
        Parameters
        ----------
            x : array-like
                The independent variable (wavelengths).
            amplitude : float
                The amplitude of the power law.
            exponent : float
            The exponent of the power law.
        Returns
        -------
            array-like
                The values of the power law function at x.
        """
        if isinstance(x, astropy.nddata.NDDataArray):
            x = x.data
            return astropy.nddata.NDDataArray(
                data=amplitude * (x ** exponent),
                unit=x.unit
            )
        elif isinstance(x, astropy.units.Quantity):
            return amplitude * (x.value ** exponent) * x.unit
        else:
            return amplitude * (x ** exponent)

    def fit_power_law(self, initial_guess=None):
        """
        Fits a power law function to the spectrum data, the initial guess will be generated based on the observed fluxes.
        Parameters
        ----------
        initial_guess : list, optional
            A list containing the initial guesses for the power law parameters [amplitude, exponent]. If None, the initial guess will be generated based on the observed fluxes (default is None).
        Returns
        -------
        dict
            A dictionary containing the fit results, including the fitted parameters and the covariance matrix.
        """
        try:
            # Extract observed wavelengths and fluxes
            obs_wavelengths = self.spectrum.processing_wavelengths.convert_unit_to(
                astropy.units.AA).data
            obs_flux_lambda = self.spectrum.processing_flux_lambda.data

            # Initial guess for the power law parameters
            if initial_guess is None:
                # Use log-log linear regression for better initial guess
                valid_mask = (obs_wavelengths > 0) & (obs_flux_lambda > 0)

                amplitude_guess = obs_flux_lambda[np.argmin(obs_wavelengths)]
                exponent_guess = -1.0
                initial_guess = [amplitude_guess, exponent_guess]

            # Fit the power law using scipy.optimize.curve_fit
            popt, pcov = scipy.optimize.curve_fit(self.power_law,
                                                  obs_wavelengths,
                                                  obs_flux_lambda,
                                                  p0=initial_guess,
                                                  maxfev=self.max_iterations*100)

            y_fit = self.power_law(obs_wavelengths, *popt)

            return {
                'success': True,
                'parameters': {
                    'amplitude': popt[0] * self.spectrum.processing_flux.unit,
                    'decay': popt[1]
                },
                'fitted_curve': astropy.nddata.NDDataArray(
                    data=y_fit,
                    unit=self.spectrum.processing_flux.unit
                ),
                'covariance': pcov,
                'integrated_flux': None,
                'integration_error': None
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def fit_single_gaussian(self, initial_guess=None):
        """
        Fits a single Gaussian to the spectrum data, the initial guess will be generated based on the observed fluxes.

        Returns
        -------
        dict
            A dictionary containing the fit results, including the fitted parameters and the covariance matrix.
        """

        try:
            # Extract observed wavelengths and fluxes
            obs_wavelengths = self.spectrum.processing_wavelengths.convert_unit_to(
                astropy.units.AA).data
            obs_flux_lambda = self.spectrum.processing_flux.data
            # NumPy array

            # Initial guess for the Gaussian parameters
            amplitude_guess = obs_flux_lambda.max()  # Initial guess for the amplitude
            # Initial guess for the mean (wavelength of max flux)
            mean_guess = obs_wavelengths[np.argmax(obs_flux_lambda)]
            # Initial guess for the width, can be adjusted
            stddev_guess = 10 * astropy.units.AA
            # int or float here

            if initial_guess is not None:
                amplitude_guess = initial_guess[0]
                mean_guess = initial_guess[1]
                stddev_guess = initial_guess[2] * astropy.units.AA
            elif not isinstance(stddev_guess, astropy.units.Quantity):
                raise TypeError(
                    "stddev_guess must be an astropy.units.Quantity object.")
            else:
                stddev_guess = stddev_guess.to(astropy.units.AA)
                initial_guess = [amplitude_guess,
                                 mean_guess, stddev_guess.value]

            # print(f"Initial guess for Gaussian parameters: {initial_guess}")

            # Fit the Gaussian using scipy.optimize.curve_fit

            popt, pcov = scipy.optimize.curve_fit(self.gaussian,
                                                  obs_wavelengths,
                                                  obs_flux_lambda,
                                                  p0=initial_guess,
                                                  maxfev=self.max_iterations)

            y_fit = self.gaussian(obs_wavelengths, *popt)

            integrated_flux, integration_error = scipy.integrate.quad(self.gaussian, obs_wavelengths.min(), obs_wavelengths.max(
            ), args=tuple(popt), epsabs=0) * self.spectrum.processing_flux.unit * self.spectrum.processing_wavelengths.unit

            return {
                'success': True,
                'parameters': {
                    'amplitude': popt[0] * self.spectrum.processing_flux.unit,
                    'mean': popt[1] * astropy.units.AA,
                    'stddev': popt[2] * astropy.units.AA,
                },
                'fitted_curve': astropy.nddata.NDDataArray(
                    data=y_fit,
                    unit=self.spectrum.processing_flux.unit
                ),
                'covariance': pcov,
                'integrated_flux': integrated_flux,
                'integration_error': integration_error
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def fit_single_gaussian_with_offset(self, initial_guess=None):
        """
        Fits a single Gaussian with an offset (using the gaussian_with_offset method defined above) to the spectrum data, the initial guess is an optional parameter, if not provided, it will be generated based on the observed fluxes.
        Parameters
        ----------
        initial_guess : list, optional
            A list containing the initial guesses for the Gaussian parameters [amplitude, mean, stddev, offset]. If None, the initial guess will be generated based on the observed fluxes (default is None).
        Returns
        -------
        dict
            A dictionary containing the fit results, including the fitted parameters and the covariance matrix.
        """

        try:
            # Extract observed wavelengths and fluxes from self.spectrum
            obs_wavelengths = self.spectrum.processing_wavelengths.convert_unit_to(
                astropy.units.AA).data
            obs_flux_lambda = self.spectrum.processing_flux_lambda.convert_unit_to(
                astropy.units.erg / (astropy.units.cm**2 * astropy.units.s * astropy.units.AA)).data

            # NumPy Array now

            # Initial guess for the Gaussian parameters

            if initial_guess is None:
                amplitude_guess = obs_flux_lambda.max()
                mean_guess = obs_wavelengths[np.argmax(obs_flux_lambda)]
                stddev_guess = 2
                # Use median as initial guess for offset
                offset_guess = np.median(obs_flux_lambda)

                initial_guess = [amplitude_guess,
                                 mean_guess, stddev_guess, offset_guess]
            else:
                if len(initial_guess) != 4:
                    raise ValueError(
                        "initial_guess must contain exactly 4 parameters: [amplitude, mean, stddev, offset].")

            popt, pcov = scipy.optimize.curve_fit(self.gaussian_with_offset,
                                                  obs_wavelengths,
                                                  obs_flux_lambda,
                                                  p0=initial_guess,
                                                  maxfev=self.max_iterations)

            y_fit = self.gaussian_with_offset(obs_wavelengths, *popt)

            integrated_flux, integration_error = (scipy.integrate.quad(self.gaussian, obs_wavelengths.min(), obs_wavelengths.max(
            ), args=tuple(popt[0:3]), epsabs=0)) * self.spectrum.processing_flux_lambda.unit * self.spectrum.processing_wavelengths.unit

            return {
                'success': True,
                'parameters': {
                    'amplitude': popt[0] * self.spectrum.processing_flux_lambda.unit,
                    'mean': popt[1] * astropy.units.AA,
                    'stddev': popt[2] * astropy.units.AA,
                    'offset': popt[3] * self.spectrum.processing_flux_lambda.unit
                },
                'fitted_curve': astropy.nddata.NDDataArray(
                    data=y_fit,
                    unit=self.spectrum.processing_flux_lambda.unit
                ),
                'fitted_wavelengths': astropy.nddata.NDDataArray(
                    data=obs_wavelengths,
                    unit=self.spectrum.processing_wavelengths.unit
                ),
                'residual_flux': astropy.nddata.NDDataArray(
                    data=obs_flux_lambda - y_fit,
                    unit=self.spectrum.processing_flux_lambda.unit
                ),
                'covariance': pcov,
                'integrated_flux': integrated_flux,
                'integration_error': integration_error
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def fit_exponential(self, initial_guess=None):
        """
        Fits an exponential function to the spectrum data, the initial guess will be generated based on the observed fluxes.

        Parameters
        ----------
        initial_guess : list, optional
            A list containing the initial guesses for the exponential parameters [amplitude, decay]. If None, the initial guess will be generated based on the observed fluxes (default is None).

        Returns
        -------
        dict
            A dictionary containing the fit results, including the fitted parameters and the covariance matrix.
        """

        try:
            # Extract observed wavelengths and fluxes
            obs_wavelengths = self.spectrum.processing_wavelengths.convert_unit_to(
                astropy.units.AA).data
            obs_flux_lambda = self.spectrum.processing_flux.data

            # Initial guess for the exponential parameters
            if initial_guess is None:
                amplitude_guess = obs_flux_lambda[np.argmin(obs_wavelengths)]
                decay_guess = -0.5

            initial_guess = [amplitude_guess, decay_guess]

            # Fit the exponential using scipy.optimize.curve_fit
            popt, pcov = scipy.optimize.curve_fit(self.exponential,
                                                  obs_wavelengths,
                                                  obs_flux_lambda,
                                                  p0=initial_guess,
                                                  maxfev=self.max_iterations)

            y_fit = self.exponential(obs_wavelengths, *popt)

            # dont need integrated flux for exponential fit

            return {
                'success': True,
                'parameters': {
                    'amplitude': popt[0] * self.spectrum.processing_flux.unit,
                    'decay': popt[1]
                },
                'fitted_curve': astropy.nddata.NDDataArray(
                    data=y_fit,
                    unit=self.spectrum.processing_flux.unit
                ),
                'covariance': pcov,
                'integrated_flux': None,
                'integration_error': None
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def check_line(self, line_restframe_wavelength, mean_fit, tolerance=10 * astropy.units.AA):
        """
        Checks if the line_restframe_wavelength is within the tolerance of the mean_fit.

        Parameters
        ----------
        line_restframe_wavelength : astropy.units.Quantity
            The rest-frame wavelength of the spectral line.
        mean_fit : astropy.units.Quantity
            The mean wavelength from the Gaussian fit.
        tolerance : astropy.units.Quantity, optional
            The tolerance range for checking (default is 10 * astropy.units.AA).

        Returns
        -------
        bool
            True if the line is within the tolerance, False otherwise.
        """
        if not isinstance(line_restframe_wavelength, astropy.units.Quantity):
            raise TypeError(
                "line_restframe_wavelength must be an astropy.units.Quantity object.")
        if not isinstance(mean_fit, astropy.units.Quantity):
            raise TypeError(
                "mean_fit must be an astropy.units.Quantity object.")
        if not isinstance(tolerance, astropy.units.Quantity):
            raise TypeError(
                "tolerance must be an astropy.units.Quantity object.")

        return abs(line_restframe_wavelength - mean_fit) <= tolerance

    def plot_fit_result(self, fit_result, component_index=0, is_residual=False):
        """
        Plots the fit result with Gaussian components fit overlay.

        Parameters
        ----------
        fit_result : dict
            The fit result dictionary containing the fitted parameters and the fitted curve, should be generated by fit_single_gaussian.
        component_index : int, optional
            The index of the component to plot (default is 0).
        is_residual : bool, optional
            If True, plot the residuals instead of the fitted curve (default is False).
        """
        if not isinstance(fit_result, dict):
            raise TypeError(
                "fit_result must be a dictionary containing the fit results.")

        if 'fitted_curve' not in fit_result or 'parameters' not in fit_result:
            raise ValueError(
                "fit_result must contain 'fitted_curve' and 'parameters' keys.")

        obs_wavelengths = self.spectrum.processing_wavelengths.convert_unit_to(
            astropy.units.AA).data
        obs_flux_lambda = self.spectrum.processing_flux.data

        fig, ax = plt.subplots(figsize=(20, 10))

        if is_residual:
            ax.plot(obs_wavelengths, obs_flux_lambda,
                    label=f"Residual Spectrum {component_index}", color='blue', alpha=0.5)
            title = f'Component {component_index} - Gaussian Fit Residuals'

        else:
            ax.plot(obs_wavelengths, obs_flux_lambda,
                    label=f"Observed Spectrum {component_index}", color='blue', alpha=0.5)
            title = f'Component {component_index} - Gaussian Fit Result'

        if fit_result['success']:
            ax.plot(obs_wavelengths, fit_result['fitted_curve'].data,
                    label=f"Fitted Curve {component_index}", color='red', alpha=0.7)

        ax.set_xlabel("Wavelength (Angstrom)", fontsize=14)
        ax.set_ylabel("Flux (erg/cm^2/s/Angstrom)", fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.legend()
        ax.set_xlim(obs_wavelengths.min(), obs_wavelengths.max())

        if not is_residual:
            flux_margin = 0.1 * np.nanmax(obs_flux_lambda)
            ax.set_ylim(np.nanmin(obs_flux_lambda) - flux_margin,
                        np.nanmax(obs_flux_lambda) + flux_margin)

        plt.grid()
        plt.show()
        plt.close(fig)

    def print_fit_summary(self, fit_result, component_index=0):
        """
        Prints a summary of the fit results for a specific component.

        Parameters
        ----------
        fit_result : dict
            The fit result dictionary containing the fitted parameters and the covariance matrix, should be generated by fit_single_gaussian.
        component_index : int, optional
            The index of the component to print the summary for (default is 0).
        """
        if not isinstance(fit_result, dict):
            raise TypeError(
                "fit_result must be a dictionary containing the fit results.")

        if 'parameters' not in fit_result or 'covariance' not in fit_result:
            raise ValueError(
                "fit_result must contain 'parameters' and 'covariance' keys.")

        if fit_result['success']:
            params = fit_result['parameters']
            print(f"\n{'='*60}")
            print(f"Component {component_index} Fit Parameters:")
            print(f"Amplitude: {params['amplitude']:.3e}")
            print(f"Mean (Rest-frame Wavelength): {params['mean']:.3f} ")
            print(f"Standard Deviation (Width): {params['stddev']:.3f}")
            print(f"Integrated Flux: {fit_result['integrated_flux']:.3e}")
            print(f"Integration Error: {fit_result['integration_error']:.3e}")
            print(f"Covariance Matrix:\n{fit_result['covariance']}")
            print(f"{'='*60}\n")

        else:
            print(f"\n{'='*60}")
            print(
                f"Component {component_index} Fit Failed: {fit_result['error']}")
            print(f"{'='*60}\n")

    def iterative_gaussian_fitting(self, line_restframe_wavelength, tolerance=10 * astropy.units.AA, plot_results=True):
        """
        Iteratively fits Gaussian components to a spectral line until the fit is successful or the maximum number of components is reached.

        Parameters
        ----------
        line_restframe_wavelength : astropy.units.Quantity
            The rest-frame wavelength of the spectral line to fit.
        tolerance : astropy.units.Quantity, optional
            The tolerance range for checking the fit (default is 10 * astropy.units.AA).
        plot_results : bool, optional
            If True, plots the fit results (default is True).

        Returns
        -------
        tuple
            A tuple containing:
            - fit_result: list of dict
                The fit results for each component, including the fitted parameters and the covariance matrix.
            - line_integrated_flux: astropy.units.Quantity
                The integrated flux of the spectral line.
        """

        self.fit_results = []  # Reset fit results for each line
        line_integrated_flux = 0 * self.spectrum.processing_flux.unit * \
            self.spectrum.processing_wavelengths.unit

        for component_index in range(self.max_components):

            if not isinstance(line_restframe_wavelength, astropy.units.Quantity):
                raise TypeError(
                    "line_restframe_wavelength must be an astropy.units.Quantity object.")

            # print(f"\n{'='*60}")
            # print(f"Fitting Component {component_index + 1} for Line at {line_restframe_wavelength:.3f}")
            # print(f"Tolerance: {tolerance:.3f}")
            # print(f"{'='*60}")

            if component_index == self.max_components - 1:
                indice = np.argmin(
                    (self.spectrum.processing_wavelengths.data - line_restframe_wavelength.value))
                initial_guess = [self.spectrum.processing_flux.data[indice],
                                 line_restframe_wavelength.value,
                                 5]
            else:
                initial_guess = None

            fit_result = self.fit_single_gaussian(initial_guess=initial_guess)
            if not fit_result['success']:
                print(
                    f"Component {component_index + 1} fit failed: {fit_result['error']}")
                break

            # if not fit_result['success']:
            #     print(f"Component {component_index + 1} fit failed: {fit_result['error']}")
            #     break

            fit_result['component_index'] = component_index
            fit_result['line_restframe_wavelength'] = line_restframe_wavelength
            fit_result['is_within_tolerance'] = self.check_line(
                line_restframe_wavelength,
                fit_result['parameters']['mean'],
                tolerance=tolerance
            )

            self.fit_results.append(fit_result)

            # self.print_fit_summary(fit_result, component_index)

            if plot_results:
                is_residual = component_index > 0
                self.plot_fit_result(
                    fit_result, component_index, is_residual=is_residual)

            self.spectrum.processing_flux = astropy.nddata.NDDataArray(
                data=self.spectrum.processing_flux.data -
                fit_result['fitted_curve'].data,
                unit=self.spectrum.processing_flux.unit
            )

            if plot_results and component_index < self.max_components - 1:
                plt.figure(figsize=(20, 10))
                plt.plot(self.spectrum.processing_wavelengths.data, self.spectrum.processing_flux.data,
                         label=f"Residual Spectrum after Component {component_index + 1}", color='blue', alpha=0.5)
                plt.xlabel("Wavelength (Angstrom)", fontsize=14)
                plt.ylabel("Flux (erg/cm^2/s/Angstrom)", fontsize=14)
                plt.title(
                    f"Residual Spectrum after Component {component_index + 1} Fitting", fontsize=16)
                plt.legend()
                plt.xlim(self.spectrum.processing_wavelengths.data.min(),
                         self.spectrum.processing_wavelengths.data.max())
                flux_margin = 0.1 * \
                    np.nanmax(self.spectrum.processing_flux.data)
                plt.ylim(np.nanmin(self.spectrum.processing_flux.data) - flux_margin,
                         np.nanmax(self.spectrum.processing_flux.data) + flux_margin)
                plt.grid()
                plt.show()

            if fit_result['is_within_tolerance']:
                # print(f"Component {component_index} is within tolerance for line at {line_restframe_wavelength:.3f}.")
                line_integrated_flux += fit_result['integrated_flux']
                break

        if len(self.fit_results) == 0:
            print("No successful fits were made.")
            return [], line_integrated_flux

    def iterative_gaussian_with_offset_fitting(self, line_restframe_wavelength, tolerance=10 * astropy.units.AA, plot_results=True):
        """
        Iteratively fits Gaussian components with an offset to a spectral line until the fit is successful or the maximum number of components is reached.
        Parameters
        ----------
        line_restframe_wavelength : astropy.units.Quantity
            The rest-frame wavelength of the spectral line to fit.
        tolerance : astropy.units.Quantity, optional
            The tolerance range for checking the fit (default is 10 * astropy.units.AA).
        plot_results : bool, optional
            If True, plots the fit results (default is True).

        Returns
        -------
        tuple
            A tuple containing:
            - fit_result: list of dict
                The fit results for each component, including the fitted parameters and the covariance matrix.
            - line_integrated_flux: astropy.units.Quantity
                The integrated flux of the spectral line.
        """
        self.fit_results = []
        line_integrated_flux = 0 * self.spectrum.processing_flux.unit * \
            self.spectrum.processing_wavelengths.unit

        if not isinstance(line_restframe_wavelength, astropy.units.Quantity):
            raise TypeError(
                "line_restframe_wavelength must be an astropy.units.Quantity object.")

        for component_index in range(self.max_components):

            if component_index == self.max_components - 1:
                indice = np.argmin(
                    (self.spectrum.processing_wavelengths.data - line_restframe_wavelength.value))
                initial_guess = [self.spectrum.processing_flux.data[indice],
                                 line_restframe_wavelength.value,
                                 2, np.median(self.spectrum.processing_flux.data)]

            else:
                initial_guess = None

            fit_result = self.fit_single_gaussian_with_offset(
                initial_guess=initial_guess)

            if not fit_result['success']:
                print(
                    f"Component {component_index + 1} fit failed: {fit_result['error']}")
                break

            fit_result['component_index'] = component_index
            fit_result['line_restframe_wavelength'] = line_restframe_wavelength
            fit_result['is_within_tolerance'] = self.check_line(
                line_restframe_wavelength,
                fit_result['parameters']['mean'],
                tolerance=tolerance
            )
            self.fit_results.append(fit_result)
            # self.print_fit_summary(fit_result, component_index)
            if plot_results:
                is_residual = component_index > 0
                self.plot_fit_result(
                    fit_result, component_index, is_residual=is_residual)

            self.spectrum.processing_flux = astropy.nddata.NDDataArray(
                data=self.spectrum.processing_flux.data -
                fit_result['fitted_curve'].data,
                unit=self.spectrum.processing_flux.unit
            )

            if plot_results and component_index < self.max_components - 1:
                plt.figure(figsize=(20, 10))
                plt.plot(self.spectrum.processing_wavelengths.data, self.spectrum.processing_flux.data,
                         label=f"Residual Spectrum after Component {component_index + 1}", color='blue', alpha=0.5)
                plt.xlabel("Wavelength (Angstrom)", fontsize=14)
                plt.ylabel("Flux (erg/cm^2/s/Angstrom)", fontsize=14)
                plt.title(
                    f"Residual Spectrum after Component {component_index + 1} Fitting", fontsize=16)
                plt.legend()
                plt.xlim(self.spectrum.processing_wavelengths.data.min(),
                         self.spectrum.processing_wavelengths.data.max())
                flux_margin = 0.1 * \
                    np.nanmax(self.spectrum.processing_flux.data)
                plt.ylim(np.nanmin(self.spectrum.processing_flux.data) - flux_margin,
                         np.nanmax(self.spectrum.processing_flux.data) + flux_margin)
                plt.grid()
                plt.show()

            if fit_result['is_within_tolerance']:
                # print(f"Component {component_index} is within tolerance for line at {line_restframe_wavelength:.3f}.")
                line_integrated_flux += fit_result['integrated_flux']
                break

        if len(self.fit_results) == 0:
            print("No successful fits were made.")
            return [], line_integrated_flux

    def get_line_components(self):
        """
        Returns the fitted components from the fit results.

        Returns
        -------
        Dict or None
            A dictionary containing the fitted components, or None if no fits were made.
        """

        for fit_result in self.fit_results:
            if fit_result.get('is_within_tolerance', True):
                return {
                    'component_index': fit_result['component_index'],
                    'line_restframe_wavelength': fit_result['line_restframe_wavelength'],
                    'parameters': fit_result['parameters'],
                    'integrated_flux': fit_result['integrated_flux'],
                    'integration_error': fit_result['integration_error']
                }
        return None

    def plot_final_decomposition(self, line_restframe_wavelength, figure_name=None):
        """
        Plots the final decomposition of the spectral line with all fitted components.

        Parameters
        ----------
        line_restframe_wavelength : astropy.units.Quantity
            The rest-frame wavelength of the spectral line to plot.
        tolerance : astropy.units.Quantity, optional
            The tolerance range for checking the fit (default is 10 * astropy.units.AA).
        """

        self.spectrum.processing_wavelengths = self.spectrum.processing_wavelengths.convert_unit_to(
            astropy.units.AA)
        self.spectrum.processing_flux = self.spectrum.processing_flux.convert_unit_to(
            astropy.units.erg / (astropy.units.cm**2 * astropy.units.s * astropy.units.AA))

        if not isinstance(self.spectrum.processing_wavelengths, astropy.nddata.NDDataArray):
            return None
        elif self.spectrum.processing_wavelengths.shape[0] == 0:
            print("No processing wavelengths available for plotting.")
            return None

        subindices = np.where(
            (self.spectrum.restframe_wavelengths.convert_unit_to(astropy.units.AA).data >= self.spectrum.processing_wavelengths.data.min()) &
            (self.spectrum.restframe_wavelengths.convert_unit_to(
                astropy.units.AA).data <= self.spectrum.processing_wavelengths.data.max())
        )[0]

        plt.figure(figsize=(20, 10))

        plt.step(self.spectrum.processing_wavelengths.data,
                 self.spectrum.observed_flux_lambda.data[subindices], label="Observed Spectrum", color='blue', alpha=0.5, linewidth=1.6)

        colors = ['red', 'blue', 'green', 'orange', 'purple']

        for i, fit_result in enumerate(self.fit_results):
            if fit_result['success']:

                params = [fit_result['parameters']['amplitude'].value,
                          fit_result['parameters']['mean'].value,
                          fit_result['parameters']['stddev'].value]

                component_flux = fit_result['fitted_curve'].data

                label = f"Component {fit_result['component_index']}"

                if fit_result['is_within_tolerance']:
                    label += f'(Line at {line_restframe_wavelength:.3f})'

                plt.plot(
                    self.spectrum.processing_wavelengths.data,
                    component_flux,
                    label=label,
                    color=colors[i % len(colors)],
                    alpha=0.7
                )

        plt.xlabel("Wavelength (Angstrom)", fontsize=14)
        plt.ylabel("Flux (erg/cm^2/s/Angstrom)", fontsize=14)
        plt.title(
            f"Final Decomposition for Line at {line_restframe_wavelength:.3f}", fontsize=16)
        plt.legend()
        plt.xlim(self.spectrum.processing_wavelengths.data.min(),
                 self.spectrum.processing_wavelengths.data.max())
        flux_margin = 0.1 * \
            np.nanmax(self.spectrum.observed_flux_lambda.data[subindices])
        plt.ylim(np.nanmin(self.spectrum.observed_flux_lambda.data[subindices]) - flux_margin, np.nanmax(
            self.spectrum.observed_flux_lambda.data[subindices]) + flux_margin)
        plt.grid()
        plt.savefig(
            f"{figure_name}.png" if figure_name else "final_decomposition.png")
        plt.close()


# class Spectrum_Catalog:
#     def __init__(self):
#         self.catalog = collections.defaultdict(lambda: {
#             'survey_id': None,
#             'prism_filepath': None,
#             'prism_redshift': None,
#             'determined_redshift': None,
#             'grating_filepaths': {},
#             'grating_redshifts': {},
#             'file_count': 0,
#             'available_filters': set(),
#             'properties': {}
#         })

#         self.filepath_pattern_re = r'^/([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^_]+)_([^_]+)_([a-zA-Z0-9_]+)\.spec.fits$'

#     def process_files(self, filepath_list, DJA_Catalog_DataFrame=None):
#         """
#         Process a list of file paths and populate the catalog with spectrum information.

#         Parameters
#         ----------
#         filepath_list : list of str
#             A list of file paths to process.

#         DJA_Catalog_DataFrame : pd.DataFrame, optional
#             A DataFrame containing the catalog data, used to load redshift information for prism spectra.
#         If provided, it will be used to load the redshift for prism spectra.
#         If None, the redshift will not be loaded for prism spectra.

#         Returns
#         -------
#         None
#         """

#         for filepath_str in tqdm.tqdm(filepath_list):
#             match = re.match(self.filepath_pattern_re, filepath_str)

#             if match:
#                 re_group = match.groups()
#                 survey_name = re_group[4]
#                 filter_name = re_group[5]
#                 survey_id_subid = f"{survey_name}_{re_group[6]}"

#                 entry = self.catalog[survey_id_subid]
#                 entry['survey_id'] = survey_name
#                 entry['id'] = survey_id_subid
#                 entry['file_count'] += 1
#                 entry['available_filters'].add(filter_name)

#                 if filter_name == 'prism-clear':
#                     entry['prism_filepath'] = filepath_str
#                     entry['prism_redshift'] = Load_Spectrum_Redshift(
#                         filepath_str, DJA_Catalog_DataFrame)
#                 else:
#                     entry['grating_filepaths'][filter_name] = filepath_str
#                     entry['grating_redshifts'][filter_name] = Load_Spectrum_Redshift(
#                         filepath_str, DJA_Catalog_DataFrame)

#     def load_spectrum_info(self, survey_id_subid):
#         """
#         Load the spectrum information for a given survey_id_subid.

#         Parameters
#         ----------
#         survey_id_subid : str
#             The survey_id_subid of the spectrum to retrieve.

#         Returns
#         -------
#         dict
#             A dictionary containing the spectrum information, or None if not found.
#         """
#         return dict(self.catalog.get(survey_id_subid, None))

#     def load_spectrums_with_prism(self):
#         """
#         Load a list of survey_id_subid that have prism spectra.

#         Returns
#         -------
#         dict
#             A dictionary with survey_id_subid as keys and their corresponding prism file paths as values.
#         """
#         return {survey_id_subid: catalog for survey_id_subid, catalog in self.catalog.items() if catalog['prism_filepath'] is not None}

#     def load_spectrums_with_grating(self):
#         """
#         Get a list of survey_id_subid that have grating spectra.

#         Returns
#         -------
#         dict
#             A dictionary with survey_id_subid as keys and their corresponding grating file paths as values.
#         """
#         return {survey_id_subid: catalog for survey_id_subid, catalog in self.catalog.items() if catalog['grating_filepaths']}

#     def load_spectrums_missing_prism(self):
#         """
#         Get a list of survey_id_subid that do not have prism spectra.

#         Returns
#         -------
#         dict
#             A dictionary with survey_id_subid as keys and their corresponding grating file paths as values.
#         """
#         return {survey_id_subid: catalog for survey_id_subid, catalog in self.catalog.items() if catalog['prism_filepath'] is None}

#     def get_summary_stats(self):
#         """
#         Get summary statistics of the catalog.

#         Returns
#         -------
#         dict
#             A dictionary containing the total number of spectra, number of unique objects, and number of available filters.
#         """
#         all_filters = set()

#         total_objects = len(self.catalog)
#         with_prism = len(self.load_spectrums_with_prism())
#         with_grating = len(self.load_spectrums_with_grating())
#         without_prism = len(self.load_spectrums_missing_prism())
#         total_spectra = sum(entry['file_count']
#                             for entry in self.catalog.values())
#         total_grating_spectra = sum(
#             len(entry['grating_filepaths']) for entry in self.catalog.values())
#         total_prism_spectra = sum(
#             1 for entry in self.catalog.values() if entry['prism_filepath'] is not None)

#         for entry in self.catalog.values():
#             all_filters.update(entry['available_filters'])

#         return {
#             'total_objects': total_objects,
#             'with_prism': with_prism,
#             'without_prism': without_prism,
#             'with_grating': with_grating,
#             'total_spectra': total_spectra,
#             'total_prism_spectra': total_prism_spectra,
#             'total_grating_spectra': total_grating_spectra
#         }

#     def to_dataframe(self):
#         """
#         Convert the catalog to a pandas DataFrame with dictionaries preserved.

#         Returns
#         -------
#         pd.DataFrame
#             A DataFrame containing the spectrum information with dictionaries and sets preserved.
#         """
#         data = []

#         for survey_id_subid, entry in self.catalog.items():
#             row = {
#                 'survey_id_subid': survey_id_subid,
#                 'survey_id': entry['survey_id'],
#                 'prism_filepath': entry['prism_filepath'],
#                 'prism_redshift': entry['prism_redshift'],
#                 # Keep as dict
#                 'grating_filepaths': entry['grating_filepaths'],
#                 # Keep as dict
#                 'grating_redshifts': entry['grating_redshifts'],
#                 'determined_redshift': entry['determined_redshift'],
#                 'file_count': entry['file_count'],
#                 'available_filters': entry['available_filters'],  # Keep as set
#                 'properties': entry['properties']  # Keep as dict
#             }
#             data.append(row)

#         return pd.DataFrame(data)

#     def save_catalog_to_pkl(self, filename):
#         """
#         Save the catalog to a pickle file.

#         Parameters
#         ----------
#         filename : str
#             The name of the file to save the catalog to.
#         """
#         df = self.to_dataframe()
#         df.to_pickle(filename)

#     def load_from_pkl(self, pkl_filepath):
#         """
#         Load catalog data from a pickle file.

#         Parameters
#         ----------
#         pkl_filepath : str
#             Path to the pickle file to load.

#         Returns
#         -------
#         None
#         """
#         if not os.path.exists(pkl_filepath):
#             raise FileNotFoundError(f"The file {pkl_filepath} does not exist.")
#         if not pkl_filepath.endswith('.pkl'):
#             raise ValueError(
#                 f"The file {pkl_filepath} is not a valid pickle file.")

#         df = pd.read_pickle(pkl_filepath)

#         # Clear existing catalog
#         self.catalog = collections.defaultdict(lambda: {
#             'survey_id': None,
#             'prism_filepath': None,
#             'prism_redshift': None,
#             'grating_filepaths': {},
#             'grating_redshifts': {},
#             'file_count': 0,
#             'available_filters': set(),
#             'properties': {},
#             'determined_redshift': None
#         })

#         for _, row in df.iterrows():
#             survey_id_subid = row['survey_id_subid']

#             # Basic information
#             entry = self.catalog[survey_id_subid]
#             entry['survey_id'] = row['survey_id']
#             entry['id'] = survey_id_subid
#             entry['prism_filepath'] = row['prism_filepath'] if pd.notna(
#                 row['prism_filepath']) else None
#             entry['prism_redshift'] = row['prism_redshift'] * \
#                 astropy.units.dimensionless_unscaled if pd.notna(
#                     row['prism_redshift']) else None
#             entry['file_count'] = int(row['file_count'])
#             entry['available_filters'] = set(filter for filter in row['available_filters']) if pd.notna(
#                 row['available_filters']) else set()
#             entry['properties'] = row['properties'] if pd.notna(
#                 row['properties']) else {}
#             entry['determined_redshift'] = row['determined_redshift'] if pd.notna(
#                 row['determined_redshift']) else None

#             # Handle grating_filepaths - keep as dict if already dict
#             if isinstance(row['available_filters'], set):
#                 for filter_name in row['available_filters']:
#                     if filter_name == 'prism-clear':
#                         continue
#                     entry['grating_filepaths'][filter_name] = row['grating_filepaths'].get(
#                         filter_name, None)
#                     entry['grating_redshifts'][filter_name] = row['grating_redshifts'].get(
#                         filter_name, None)

#     def find_complete_objects(self, required_filters=None):
#         """
#         Find objects that have all required filters and a prism file.

#         Parameters
#         ----------
#         required_filters : list of str, optional
#             A list of filter names that must be present for an object to be considered complete.

#         Returns
#         -------
#         dict
#             A dictionary with survey_id_subid as keys and their corresponding catalog entries as values.
#         """
#         if required_filters is None:
#             required_filters = set()
#         else:
#             required_filters = set(required_filters)

#         complete_objects = {}
#         for survey_id_subid, entry in self.catalog.items():
#             if (entry['prism_filepath'] is not None and required_filters.issubset(entry['available_filters'])):
#                 complete_objects[survey_id_subid] = entry

#         return complete_objects

#     def catalog_iterator(self,sample_num=None):
#         """
#         Returns an iterator over the catalog entries.

#         Yields
#         -------
#         tuple
#             A tuple containing the survey_id_subid and the corresponding catalog entry.
#         """
#         if sample_num is not None:
#             count = 0
#             for index in self.catalog.keys():
#                 if self.catalog[index]['properties'].get('Sample_Flag', False) is True:
#                     yield index, self.catalog[index]
#                     count += 1
#                     if count >= sample_num:
#                         break
#         else:
#             for index in self.catalog.keys():
#                 yield index, self.catalog[index]

#     def determine_redshift(self, survey_id_subid):
#         """
#         Determine the redshift for a given survey_id_subid. And fill the `determined_redshift` field in the catalog entry.
#         If two redshifts are available and within 5% of each other, it will use the average of the redshifts. If the difference is greater than 5%, it will use the prism redshift if available, or the grating redshift if only one is available.
#         If more than one grating redshift is available, it will choose the average of the two most similar redshifts.
#         If no redshift is available, it will return None.
#         If only one redshift is available, it will use that one.
#         Parameters
#         ----------
#         survey_id_subid : str
#             The survey_id_subid of the spectrum to determine the redshift for.
#         Returns
#         -------
#         astropy.units.Quantity or None
#             The determined redshift as an astropy.units.Quantity object, or None if no redshift could be determined.
#         """
#         if survey_id_subid not in self.catalog:
#             raise ValueError(
#                 f"Survey ID {survey_id_subid} not found in the catalog.")

#         entry = self.catalog[survey_id_subid]
#         entry['properties']['redshift_conflict'] = False

#         prism_redshift = entry['prism_redshift']
#         grating_redshifts = [
#             value for value in entry['grating_redshifts'].values() if not np.isnan(value)]

#         # No redshifts available
#         if prism_redshift is None and not grating_redshifts:
#             return None

#         # Only prism redshift available
#         if prism_redshift is not None and not grating_redshifts:
#             entry['determined_redshift'] = prism_redshift
#             return prism_redshift

#         # Only one grating redshift available, no prism
#         if prism_redshift is None and len(grating_redshifts) == 1:
#             entry['determined_redshift'] = grating_redshifts[0]
#             return grating_redshifts[0]

#         # One prism and one grating redshift
#         if prism_redshift is not None and len(grating_redshifts) == 1:
#             if abs(prism_redshift - grating_redshifts[0]) / prism_redshift < 0.05:
#                 entry['determined_redshift'] = (
#                     prism_redshift + grating_redshifts[0]) / 2
#                 return entry['determined_redshift']
#             else:
#                 entry['determined_redshift'] = prism_redshift
#                 entry['properties']['redshift_conflict'] = True
#                 return prism_redshift

#         # Multiple redshifts available (either prism + multiple grating, or just multiple grating)
#         all_redshifts = []
#         if prism_redshift is not None:
#             all_redshifts.append(prism_redshift)
#         all_redshifts.extend(grating_redshifts)

#         # Find the two most similar redshifts
#         min_diff = float('inf')
#         best_pair = None
#         for i in range(len(all_redshifts)):
#             for j in range(i + 1, len(all_redshifts)):
#                 diff = abs(all_redshifts[i] - all_redshifts[j])
#                 if diff < min_diff:
#                     min_diff = diff
#                     best_pair = (all_redshifts[i], all_redshifts[j])

#         if best_pair:
#             # Check if the two nearest redshifts are within 5%
#             relative_diff = abs(
#                 best_pair[0] - best_pair[1]) / min(best_pair[0], best_pair[1])
#             avg_redshift = (best_pair[0] + best_pair[1]) / 2

#             if relative_diff >= 0.05:
#                 entry['properties']['redshift_conflict'] = True

#             entry['determined_redshift'] = avg_redshift
#             return avg_redshift
#         else:
#             # Fallback (should not happen if we have redshifts)
#             entry['determined_redshift'] = all_redshifts[0]
#             return all_redshifts[0]

#     def update_catalog_item(self, id, catalog):
#         """
#         Update a catalog item with the given id and catalog data.

#         Parameters
#         ----------
#         id : str
#             The survey_id_subid of the spectrum to update.
#         catalog : dict
#             The catalog data to update the item with.
#         """
#         if id not in self.catalog:
#             raise ValueError(f"Survey ID {id} not found in the catalog.")

#         self.catalog[id].update(catalog)

#     def sample_num(self):
#         """
#         Returns the number of samples in the catalog.

#         Returns
#         -------
#         int
#             The number of samples in the catalog.
#         """
#         count = 0
#         for id, catalog in self.catalog.items():
#             if catalog['properties']['Sample_Flag'] is True:
#                 count += 1
#         return count

#     def __repr__(self):
#         """
#         Returns a panda DataFrame representation of the catalog.
#         """
#         df = self.to_dataframe()
#         if df.empty:
#             return "Spectrum_Catalog is empty."
#         else:
#             return df.to_string(index=False, max_rows=10, max_colwidth=50, justify='left') + "\n\n" + f"Total objects: {len(self.catalog)}"

# class Spectrum_Catalog:
#     def __init__(self):
#         # 初始化一个默认的字段集合
#         self.default_fields = {
#             'survey_id_subid': None,
#             'prism_filepath': None,
#             'prism_redshift': None,
#             'determined_redshift': None,
#             'grating_filepaths': {},
#             'grating_redshifts': {},
#             'file_count': 0,
#             'available_filters': set(),
#             'properties': {}
#         }
#         # 使用一个函数来创建新的条目，这个函数会用到当前的默认字段
#         self.catalog = collections.defaultdict(lambda: self.default_fields.copy())

#         self.filepath_pattern_re = r'^/([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^_]+)_([^_]+)_([a-zA-Z0-9_]+)\.spec.fits$'

#     def add_property_field(self, field_name, default_value=None):
#         """
#         为所有 catalog 条目动态添加一个新的顶级字段。

#         Parameters
#         ----------
#         field_name : str
#             要添加的新字段的名称。
#         default_value : any, optional
#             新字段的默认值, by default None
#         """
#         if field_name not in self.default_fields:
#             # 更新默认字段模板，以便新创建的条目也包含这个字段
#             self.default_fields[field_name] = default_value
#             # 遍历所有现有条目，为它们添加这个新字段
#             for survey_id in self.catalog:
#                 if field_name not in self.catalog[survey_id]:
#                     self.catalog[survey_id][field_name] = default_value

#     def update_entry_property(self, survey_id_subid, property_name, value):
#         """
#         更新指定条目的一个顶级属性。

#         如果该属性不存在，它会根据需要被添加。

#         Parameters
#         ----------
#         survey_id_subid : str
#             要更新的条目的 ID。
#         property_name : str
#             要更新的属性的名称。
#         value : any
#             要设置的新值。
#         """
#         if survey_id_subid not in self.catalog:
#             raise KeyError(f"ID '{survey_id_subid}' 在 catalog 中未找到。")

#         # 检查这个字段是否是已知的字段，如果不是，则将其添加到所有条目中
#         if property_name not in self.default_fields:
#             self.add_property_field(property_name)

#         self.catalog[survey_id_subid][property_name] = value


#     def process_files(self, filepath_list, DJA_Catalog_DataFrame=None):
#         """
#         Process a list of file paths and populate the catalog with spectrum information.

#         Parameters
#         ----------
#         filepath_list : list of str
#             A list of file paths to process.

#         DJA_Catalog_DataFrame : pd.DataFrame, optional
#             A DataFrame containing the catalog data, used to load redshift information for prism spectra.
#         If provided, it will be used to load the redshift for prism spectra.
#         If None, the redshift will not be loaded for prism spectra.

#         Returns
#         -------
#         None
#         """

#         for filepath_str in tqdm.tqdm(filepath_list):
#             match = re.match(self.filepath_pattern_re, filepath_str)

#             if match:
#                 re_group = match.groups()
#                 survey_name = re_group[4]
#                 filter_name = re_group[5]
#                 survey_id_subid = f"{survey_name}_{re_group[6]}"

#                 entry = self.catalog[survey_id_subid]
#                 entry['survey_id'] = survey_name
#                 #entry['survey_id_subid'] = survey_id_subid
#                 entry['file_count'] += 1
#                 entry['available_filters'].add(filter_name)

#                 if filter_name == 'prism-clear':
#                     entry['prism_filepath'] = filepath_str
#                     entry['prism_redshift'] = Load_Spectrum_Redshift(filepath_str, DJA_Catalog_DataFrame)
#                 else:
#                     entry['grating_filepaths'][filter_name] = filepath_str
#                     entry['grating_redshifts'][filter_name] = Load_Spectrum_Redshift(filepath_str, DJA_Catalog_DataFrame)

#     def load_spectrum_info(self, survey_id_subid):
#         """
#         Load the spectrum information for a given survey_id_subid.

#         Parameters
#         ----------
#         survey_id_subid : str
#             The survey_id_subid of the spectrum to retrieve.

#         Returns
#         -------
#         dict
#             A dictionary containing the spectrum information, or None if not found.
#         """
#         return dict(self.catalog.get(survey_id_subid, None))



#     def load_spectrums_with_prism(self):
#         """
#         Load a list of survey_id_subid that have prism spectra.

#         Returns
#         -------
#         dict
#             A dictionary with survey_id_subid as keys and their corresponding prism file paths as values.
#         """
#         return {survey_id_subid: catalog for survey_id_subid, catalog in self.catalog.items() if catalog['prism_filepath'] is not None}

#     def load_spectrums_with_grating(self):
#         """
#         Get a list of survey_id_subid that have grating spectra.

#         Returns
#         -------
#         dict
#             A dictionary with survey_id_subid as keys and their corresponding grating file paths as values.
#         """
#         return {survey_id_subid: catalog for survey_id_subid, catalog in self.catalog.items() if catalog['grating_filepaths']}

#     def load_spectrums_missing_prism(self):
#         """
#         Get a list of survey_id_subid that do not have prism spectra.

#         Returns
#         -------
#         dict
#             A dictionary with survey_id_subid as keys and their corresponding grating file paths as values.
#         """
#         return {survey_id_subid: catalog for survey_id_subid, catalog in self.catalog.items() if catalog['prism_filepath'] is None}

#     def get_summary_stats(self):
#         """
#         Get summary statistics of the catalog.

#         Returns
#         -------
#         dict
#             A dictionary containing the total number of spectra, number of unique objects, and number of available filters.
#         """
#         all_filters = set()

#         total_objects = len(self.catalog)
#         with_prism = len(self.load_spectrums_with_prism())
#         with_grating = len(self.load_spectrums_with_grating())
#         without_prism = len(self.load_spectrums_missing_prism())
#         total_spectra = sum(entry['file_count']
#                             for entry in self.catalog.values())
#         total_grating_spectra = sum(
#             len(entry['grating_filepaths']) for entry in self.catalog.values())
#         total_prism_spectra = sum(
#             1 for entry in self.catalog.values() if entry['prism_filepath'] is not None)

#         for entry in self.catalog.values():
#             all_filters.update(entry['available_filters'])

#         return {
#             'total_objects': total_objects,
#             'with_prism': with_prism,
#             'without_prism': without_prism,
#             'with_grating': with_grating,
#             'total_spectra': total_spectra,
#             'total_prism_spectra': total_prism_spectra,
#             'total_grating_spectra': total_grating_spectra
#         }

#     def to_dataframe(self):
#         """
#         Convert the catalog to a pandas DataFrame with dictionaries preserved.

#         Returns
#         -------
#         pd.DataFrame
#             A DataFrame containing the spectrum information with dictionaries and sets preserved.
#         """
#         # data = []

#         # for survey_id_subid, entry in self.catalog.items():
#         #     row = {
#         #         'survey_id_subid': survey_id_subid,
#         #         'survey_id': entry['survey_id'],
#         #         'prism_filepath': entry['prism_filepath'],
#         #         'prism_redshift': entry['prism_redshift'],
#         #         # Keep as dict
#         #         'grating_filepaths': entry['grating_filepaths'],
#         #         # Keep as dict
#         #         'grating_redshifts': entry['grating_redshifts'],
#         #         'determined_redshift': entry['determined_redshift'],
#         #         'file_count': entry['file_count'],
#         #         'available_filters': entry['available_filters'],  # Keep as set
#         #         'properties': entry['properties']  # Keep as dict
#         #     }
#         #     data.append(row)

#         if not self.catalog:
#             return pd.DataFrame()


#         df = pd.DataFrame.from_dict(self.catalog, orient='index')

#         df.reset_index(inplace=True)

#         df.rename(columns={'index': 'survey_id_subid'}, inplace=True)

#         return df


#     def save_catalog_to_pkl(self, filename):
#         """
#         Save the catalog to a pickle file.

#         Parameters
#         ----------
#         filename : str
#             The name of the file to save the catalog to.
#         """
#         df = self.to_dataframe()
#         df.to_pickle(filename)

#     def load_from_pkl(self, pkl_filepath):
#         """
#         Load catalog data from a pickle file.

#         Parameters
#         ----------
#         pkl_filepath : str
#             Path to the pickle file to load.

#         Returns
#         -------
#         None
#         """
#         if not os.path.exists(pkl_filepath):
#             raise FileNotFoundError(f"The file {pkl_filepath} does not exist.")
#         if not pkl_filepath.endswith('.pkl'):
#             raise ValueError(
#                 f"The file {pkl_filepath} is not a valid pickle file.")

#         df = pd.read_pickle(pkl_filepath)

#         # 从 DataFrame 的列中推断出所有的字段
#         self.default_fields = {col: None for col in df.columns if col != 'survey_id_subid'}
#         # 确保一些字段是正确的类型
#         self.default_fields['grating_filepaths'] = {}
#         self.default_fields['grating_redshifts'] = {}
#         self.default_fields['available_filters'] = set()
#         self.default_fields['properties'] = {}

#         self.catalog.clear()

#         df.set_index('survey_id_subid', inplace=True)

#         reconstructed_dict = df.to_dict(orient='index')

#         self.catalog.update(reconstructed_dict)

#     def find_complete_objects(self, required_filters=None):
#         """
#         Find objects that have all required filters and a prism file.

#         Parameters
#         ----------
#         required_filters : list of str, optional
#             A list of filter names that must be present for an object to be considered complete.

#         Returns
#         -------
#         dict
#             A dictionary with survey_id_subid as keys and their corresponding catalog entries as values.
#         """
#         if required_filters is None:
#             required_filters = set()
#         else:
#             required_filters = set(required_filters)

#         complete_objects = {}
#         for survey_id_subid, entry in self.catalog.items():
#             if (entry['prism_filepath'] is not None and required_filters.issubset(entry['available_filters'])):
#                 complete_objects[survey_id_subid] = entry

#         return complete_objects

#     def catalog_iterator(self, sample_num=None):
#         """
#         Returns an iterator over the catalog entries.

#         Yields
#         -------
#         tuple
#             A tuple containing the survey_id_subid and the corresponding catalog entry.
#         """
#         # 使用 .items() 方法可以高效地同时遍历字典的键和值
#         if sample_num is not None:
#             count = 0
#             # 遍历 catalog 中的每一对 (键, 值)
#             for survey_id_subid, entry in self.catalog.items():
#                 # 产生一个 (键, 值) 元组
#                 yield survey_id_subid, entry
#                 count += 1
#                 if count >= sample_num:
#                     break
#         else:
#             # 如果没有数量限制，直接遍历并产生所有的 (键, 值) 对
#             for survey_id_subid, entry in self.catalog.items():
#                 yield survey_id_subid, entry

#     def determine_redshift(self, survey_id_subid):
#         """
#         Determine the redshift for a given survey_id_subid. And fill the `determined_redshift` field in the catalog entry.
#         If two redshifts are available and within 5% of each other, it will use the average of the redshifts. If the difference is greater than 5%, it will use the prism redshift if available, or the grating redshift if only one is available.
#         If more than one grating redshift is available, it will choose the average of the two most similar redshifts.
#         If no redshift is available, it will return None.
#         If only one redshift is available, it will use that one.
#         Parameters
#         ----------
#         survey_id_subid : str
#             The survey_id_subid of the spectrum to determine the redshift for.
#         Returns
#         -------
#         astropy.units.Quantity or None
#             The determined redshift as an astropy.units.Quantity object, or None if no redshift could be determined.
#         """
#         if survey_id_subid not in self.catalog:
#             raise ValueError(
#                 f"Survey ID {survey_id_subid} not found in the catalog.")

#         entry = self.catalog[survey_id_subid]
#         entry['properties']['redshift_conflict'] = False

#         prism_redshift = entry['prism_redshift']
#         grating_redshifts = [
#             value for value in entry['grating_redshifts'].values() if not np.isnan(value)]

#         # No redshifts available
#         if prism_redshift is None and not grating_redshifts:
#             return None

#         # Only prism redshift available
#         if prism_redshift is not None and not grating_redshifts:
#             entry['determined_redshift'] = prism_redshift
#             return prism_redshift

#         # Only one grating redshift available, no prism
#         if prism_redshift is None and len(grating_redshifts) == 1:
#             entry['determined_redshift'] = grating_redshifts[0]
#             return grating_redshifts[0]

#         # One prism and one grating redshift
#         if prism_redshift is not None and len(grating_redshifts) == 1:
#             if abs(prism_redshift - grating_redshifts[0]) / prism_redshift < 0.05:
#                 entry['determined_redshift'] = (
#                     prism_redshift + grating_redshifts[0]) / 2
#                 return entry['determined_redshift']
#             else:
#                 entry['determined_redshift'] = prism_redshift
#                 entry['properties']['redshift_conflict'] = True
#                 return prism_redshift

#         # Multiple redshifts available (either prism + multiple grating, or just multiple grating)
#         all_redshifts = []
#         if prism_redshift is not None:
#             all_redshifts.append(prism_redshift)
#         all_redshifts.extend(grating_redshifts)

#         # Find the two most similar redshifts
#         min_diff = float('inf')
#         best_pair = None
#         for i in range(len(all_redshifts)):
#             for j in range(i + 1, len(all_redshifts)):
#                 diff = abs(all_redshifts[i] - all_redshifts[j])
#                 if diff < min_diff:
#                     min_diff = diff
#                     best_pair = (all_redshifts[i], all_redshifts[j])

#         if best_pair:
#             # Check if the two nearest redshifts are within 5%
#             relative_diff = abs(
#                 best_pair[0] - best_pair[1]) / min(best_pair[0], best_pair[1])
#             avg_redshift = (best_pair[0] + best_pair[1]) / 2

#             if relative_diff >= 0.05:
#                 entry['properties']['redshift_conflict'] = True

#             entry['determined_redshift'] = avg_redshift
#             return avg_redshift
#         else:
#             # Fallback (should not happen if we have redshifts)
#             entry['determined_redshift'] = all_redshifts[0]
#             return all_redshifts[0]

#     def update_catalog_item(self, id, catalog):
#         """
#         Update a catalog item with the given id and catalog data.

#         Parameters
#         ----------
#         id : str
#             The survey_id_subid of the spectrum to update.
#         catalog : dict
#             The catalog data to update the item with.
#         """
#         if id not in self.catalog:
#             raise ValueError(f"Survey ID {id} not found in the catalog.")

#         self.catalog[id].update(catalog)

#     def sample_num(self):
#         """
#         Returns the number of samples in the catalog.

#         Returns
#         -------
#         int
#             The number of samples in the catalog.
#         """
#         count = 0
#         for id, catalog in self.catalog.items():
#             if catalog.get('Sample_Flag') is True: # 使用 .get() 更安全
#                 count += 1
#         return count

#     def load_spectrum_filepaths_from_directory(self,directory_path, file_extension='.[sS][pP][eE][cC].[fF][iI][tT][sS]'):
#         """
#         Recursively finds files with a matching file extension in the specified directory.

#         Args:
#             directory_path (str or Path): The root directory path to search.
#             file_extension (str): The file ending string to look for, e.g., 'spec.fits'.

#         Returns:
#             list[str]: A list of all found file path strings.
#         """
#         # Convert the input path to a Path object, which is standard practice for pathlib
#         # and expand the tilde (~) to the user's home directory.
#         p = Path(directory_path).expanduser()

#         # Use rglob for a recursive search
#         # p.rglob('*<file_extension>') will find all files ending with <file_extension> in all subdirectories
#         filepaths = list(p.rglob(f'*{file_extension}'))

#         # Convert the Path objects back to strings and return them
#         return [str(fp) for fp in filepaths]


#     def __repr__(self):
#         """
#         Returns a panda DataFrame representation of the catalog.
#         """
#         df = self.to_dataframe()
#         if df.empty:
#             return "Spectrum_Catalog is empty."
#         else:
#             return df.to_string(index=False, max_rows=10, max_colwidth=50, justify='left') + "\n\n" + f"Total objects: {len(self.catalog)}"


import collections
import re
import tqdm
import pandas as pd
import numpy as np
import os
from pathlib import Path
from copy import deepcopy

import collections
import re
import tqdm
import pandas as pd
import numpy as np
import os
from pathlib import Path
from copy import deepcopy

class Spectrum_Catalog:
    def __init__(self):
        # 初始化一个默认的字段集合
        self.default_fields = {
            'survey_id_subid': None,
            'prism_filepath': None,
            'prism_redshift': None,
            'determined_redshift': None,
            'grating_filepaths': {},
            'grating_redshifts': {},
            'file_count': 0,
            'available_filters': set(),
            'properties': {}
        }
        # FIXED: Use a factory function that creates DEEP copies to avoid sharing mutable objects
        self.catalog = collections.defaultdict(self._create_default_entry)

        self.filepath_pattern_re = r'^/([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^_]+)_([^_]+)_([a-zA-Z0-9_]+)\.spec.fits$'

    def _create_default_entry(self):
        """
        Factory function to create a new default entry with deep copies of mutable objects.
        This ensures each entry has its own independent dictionaries and sets.
        """
        return {
            'survey_id_subid': None,
            'prism_filepath': None,
            'prism_redshift': None,
            'determined_redshift': None,
            'grating_filepaths': {},  # New empty dict for each entry
            'grating_redshifts': {},  # New empty dict for each entry
            'file_count': 0,
            'available_filters': set(),  # New empty set for each entry
            'properties': {}  # New empty dict for each entry
        }

    def add_property_field(self, field_name, default_value=None):
        """
        为所有 catalog 条目动态添加一个新的顶级字段。

        Parameters
        ----------
        field_name : str
            要添加的新字段的名称。
        default_value : any, optional
            新字段的默认值, by default None
        """
        if field_name not in self.default_fields:
            # 更新默认字段模板，以便新创建的条目也包含这个字段
            # FIXED: Use deepcopy for mutable default values
            if isinstance(default_value, (dict, set, list)):
                self.default_fields[field_name] = deepcopy(default_value)
            else:
                self.default_fields[field_name] = default_value

            # 遍历所有现有条目，为它们添加这个新字段
            for survey_id in self.catalog:
                if field_name not in self.catalog[survey_id]:
                    # FIXED: Use deepcopy for mutable values
                    if isinstance(default_value, (dict, set, list)):
                        self.catalog[survey_id][field_name] = deepcopy(default_value)
                    else:
                        self.catalog[survey_id][field_name] = default_value

    def update_entry_property(self, survey_id_subid, property_name, value):
        """
        更新指定条目的一个顶级属性。

        如果该属性不存在，它会根据需要被添加。

        Parameters
        ----------
        survey_id_subid : str
            要更新的条目的 ID。
        property_name : str
            要更新的属性的名称。
        value : any
            要设置的新值。
        """
        if survey_id_subid not in self.catalog:
            raise KeyError(f"ID '{survey_id_subid}' 在 catalog 中未找到。")

        # 检查这个字段是否是已知的字段，如果不是，则将其添加到所有条目中
        if property_name not in self.default_fields:
            self.add_property_field(property_name)

        self.catalog[survey_id_subid][property_name] = value


    def remove_property_field(self, field_name):
        """
        从所有 catalog 条目中动态删除一个顶级字段。

        Parameters
        ----------
        field_name : str
            要删除的字段的名称。
        """
        if field_name in self.default_fields:
            del self.default_fields[field_name]

        for survey_id in self.catalog:
            if field_name in self.catalog[survey_id]:
                del self.catalog[survey_id][field_name]

    def remove_entry_property(self, survey_id_subid, property_name):
        """
        从指定的单个条目中删除一个顶级属性。

        Parameters
        ----------
        survey_id_subid : str
            要修改的条目的 ID。
        property_name : str
            要删除的属性的名称。
        """
        if survey_id_subid not in self.catalog:
            raise KeyError(f"ID '{survey_id_subid}' 在 catalog 中未找到。")

        if property_name not in self.catalog[survey_id_subid]:
            raise KeyError(f"属性 '{property_name}' 在 ID '{survey_id_subid}' 中未找到。")

        del self.catalog[survey_id_subid][property_name]

    def process_files(self, filepath_list, DJA_Catalog_DataFrame=None):
        """
        Process a list of file paths and populate the catalog with spectrum information.

        Parameters
        ----------
        filepath_list : list of str
            A list of file paths to process.

        DJA_Catalog_DataFrame : pd.DataFrame, optional
            A DataFrame containing the catalog data, used to load redshift information for prism spectra.
        If provided, it will be used to load the redshift for prism spectra.
        If None, the redshift will not be loaded for prism spectra.

        Returns
        -------
        None
        """

        for filepath_str in tqdm.tqdm(filepath_list):
            match = re.match(self.filepath_pattern_re, filepath_str)

            if match:
                re_group = match.groups()
                survey_name = re_group[4]
                filter_name = re_group[5]
                survey_id_subid = f"{survey_name}_{re_group[6]}"

                # Access entry - this will create a new one if it doesn't exist
                entry = self.catalog[survey_id_subid]
                entry['survey_id'] = survey_name
                entry['survey_id_subid'] = survey_id_subid  # FIXED: Store the ID in the entry
                entry['file_count'] += 1
                entry['available_filters'].add(filter_name)

                if filter_name == 'prism-clear':
                    entry['prism_filepath'] = filepath_str
                    entry['prism_redshift'] = Load_Spectrum_Redshift(filepath_str, DJA_Catalog_DataFrame)
                else:
                    entry['grating_filepaths'][filter_name] = filepath_str
                    entry['grating_redshifts'][filter_name] = Load_Spectrum_Redshift(filepath_str, DJA_Catalog_DataFrame)

    def load_spectrum_info(self, survey_id_subid):
        """
        Load the spectrum information for a given survey_id_subid.

        Parameters
        ----------
        survey_id_subid : str
            The survey_id_subid of the spectrum to retrieve.

        Returns
        -------
        dict
            A dictionary containing the spectrum information, or None if not found.
        """
        result = self.catalog.get(survey_id_subid, None)
        if result is not None:
            return dict(result)
        return None

    def load_spectrums_with_prism(self):
        """
        Load a list of survey_id_subid that have prism spectra.

        Returns
        -------
        dict
            A dictionary with survey_id_subid as keys and their corresponding prism file paths as values.
        """
        return {survey_id_subid: catalog for survey_id_subid, catalog in self.catalog.items() if catalog['prism_filepath'] is not None}

    def load_spectrums_with_grating(self):
        """
        Get a list of survey_id_subid that have grating spectra.

        Returns
        -------
        dict
            A dictionary with survey_id_subid as keys and their corresponding grating file paths as values.
        """
        return {survey_id_subid: catalog for survey_id_subid, catalog in self.catalog.items() if catalog['grating_filepaths']}

    def load_spectrums_missing_prism(self):
        """
        Get a list of survey_id_subid that do not have prism spectra.

        Returns
        -------
        dict
            A dictionary with survey_id_subid as keys and their corresponding grating file paths as values.
        """
        return {survey_id_subid: catalog for survey_id_subid, catalog in self.catalog.items() if catalog['prism_filepath'] is None}

    def get_summary_stats(self):
        """
        Get summary statistics of the catalog.

        Returns
        -------
        dict
            A dictionary containing the total number of spectra, number of unique objects, and number of available filters.
        """
        all_filters = set()

        total_objects = len(self.catalog)
        with_prism = len(self.load_spectrums_with_prism())
        with_grating = len(self.load_spectrums_with_grating())
        without_prism = len(self.load_spectrums_missing_prism())
        total_spectra = sum(entry['file_count']
                            for entry in self.catalog.values())
        total_grating_spectra = sum(
            len(entry['grating_filepaths']) for entry in self.catalog.values())
        total_prism_spectra = sum(
            1 for entry in self.catalog.values() if entry['prism_filepath'] is not None)

        for entry in self.catalog.values():
            all_filters.update(entry['available_filters'])

        return {
            'total_objects': total_objects,
            'with_prism': with_prism,
            'without_prism': without_prism,
            'with_grating': with_grating,
            'total_spectra': total_spectra,
            'total_prism_spectra': total_prism_spectra,
            'total_grating_spectra': total_grating_spectra
        }

    def to_dataframe(self):
        """
        Convert the catalog to a pandas DataFrame with dictionaries preserved.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the spectrum information with dictionaries and sets preserved.
        """
        if not self.catalog:
            return pd.DataFrame()

        # FIXED: Create DataFrame from the dictionary
        df = pd.DataFrame.from_dict(self.catalog, orient='index')

        # Reset index to make survey_id_subid a column
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'survey_id_subid'}, inplace=True)

        # Remove duplicate survey_id_subid column if it exists in the data
        # (This can happen if entries have 'survey_id_subid' as a field)
        if 'survey_id_subid' in df.columns:
            # Count how many survey_id_subid columns exist
            survey_id_cols = [col for col in df.columns if col == 'survey_id_subid']
            if len(survey_id_cols) > 1:
                # Keep only the first one (from index)
                df = df.loc[:, ~df.columns.duplicated()]

        # Ensure survey_id_subid is in the first column (only if it exists)
        if 'survey_id_subid' in df.columns:
            cols = ['survey_id_subid'] + [col for col in df.columns if col != 'survey_id_subid']
            df = df[cols]

        return df

    def save_catalog_to_pkl(self, filename):
        """
        Save the catalog to a pickle file.

        Parameters
        ----------
        filename : str
            The name of the file to save the catalog to.
        """
        df = self.to_dataframe()
        df.to_pickle(filename)

    def load_from_pkl(self, pkl_filepath):
        """
        Load catalog data from a pickle file.

        Parameters
        ----------
        pkl_filepath : str
            Path to the pickle file to load.

        Returns
        -------
        None
        """
        if not os.path.exists(pkl_filepath):
            raise FileNotFoundError(f"The file {pkl_filepath} does not exist.")
        if not pkl_filepath.endswith('.pkl'):
            raise ValueError(
                f"The file {pkl_filepath} is not a valid pickle file.")

        df = pd.read_pickle(pkl_filepath)

        # FIXED: Handle both cases - when survey_id_subid is a column or index
        # If survey_id_subid is in the index, reset it to a column
        if df.index.name == 'survey_id_subid' or 'survey_id_subid' not in df.columns:
            df.reset_index(inplace=True)
            if 'index' in df.columns:
                df.rename(columns={'index': 'survey_id_subid'}, inplace=True)

        # 从 DataFrame 的列中推断出所有的字段
        self.default_fields = {col: None for col in df.columns if col != 'survey_id_subid'}
        # 确保一些字段是正确的类型（使用空实例而非共享引用）
        self.default_fields['grating_filepaths'] = {}
        self.default_fields['grating_redshifts'] = {}
        self.default_fields['available_filters'] = set()
        self.default_fields['properties'] = {}

        # Clear and rebuild catalog
        self.catalog.clear()

        # Set survey_id_subid as index for easier dictionary conversion
        df.set_index('survey_id_subid', inplace=True)

        # Convert to dictionary and ensure each entry has independent mutable objects
        reconstructed_dict = df.to_dict(orient='index')

        # Deep copy mutable objects to avoid sharing references
        for survey_id, entry in reconstructed_dict.items():
            # Remove survey_id_subid from entry if it exists (it's already the key)
            if 'survey_id_subid' in entry:
                del entry['survey_id_subid']

            # Ensure mutable objects are independent copies
            if 'grating_filepaths' in entry and isinstance(entry['grating_filepaths'], dict):
                entry['grating_filepaths'] = dict(entry['grating_filepaths'])
            if 'grating_redshifts' in entry and isinstance(entry['grating_redshifts'], dict):
                entry['grating_redshifts'] = dict(entry['grating_redshifts'])
            if 'available_filters' in entry:
                if isinstance(entry['available_filters'], set):
                    entry['available_filters'] = set(entry['available_filters'])
                elif isinstance(entry['available_filters'], (list, tuple)):
                    entry['available_filters'] = set(entry['available_filters'])
            if 'properties' in entry and isinstance(entry['properties'], dict):
                entry['properties'] = dict(entry['properties'])

            self.catalog[survey_id] = entry

    def find_complete_objects(self, required_filters=None):
        """
        Find objects that have all required filters and a prism file.

        Parameters
        ----------
        required_filters : list of str, optional
            A list of filter names that must be present for an object to be considered complete.

        Returns
        -------
        dict
            A dictionary with survey_id_subid as keys and their corresponding catalog entries as values.
        """
        if required_filters is None:
            required_filters = set()
        else:
            required_filters = set(required_filters)

        complete_objects = {}
        for survey_id_subid, entry in self.catalog.items():
            if (entry['prism_filepath'] is not None and required_filters.issubset(entry['available_filters'])):
                complete_objects[survey_id_subid] = entry

        return complete_objects

    def catalog_iterator(self, sample_num=None):
        """
        Returns an iterator over the catalog entries.

        Parameters
        ----------
        sample_num : int, optional
            If provided, only iterate over the first sample_num entries.

        Yields
        -------
        tuple
            A tuple containing the survey_id_subid and the corresponding catalog entry.
        """
        if sample_num is not None:
            count = 0
            for survey_id_subid, entry in self.catalog.items():
                yield survey_id_subid, entry
                count += 1
                if count >= sample_num:
                    break
        else:
            for survey_id_subid, entry in self.catalog.items():
                yield survey_id_subid, entry

    def determine_redshift(self, survey_id_subid):
        """
        Determine the redshift for a given survey_id_subid. And fill the `determined_redshift` field in the catalog entry.
        If two redshifts are available and within 5% of each other, it will use the average of the redshifts. If the difference is greater than 5%, it will use the prism redshift if available, or the grating redshift if only one is available.
        If more than one grating redshift is available, it will choose the average of the two most similar redshifts.
        If no redshift is available, it will return None.
        If only one redshift is available, it will use that one.
        Parameters
        ----------
        survey_id_subid : str
            The survey_id_subid of the spectrum to determine the redshift for.
        Returns
        -------
        astropy.units.Quantity or None
            The determined redshift as an astropy.units.Quantity object, or None if no redshift could be determined.
        """
        if survey_id_subid not in self.catalog:
            raise ValueError(
                f"Survey ID {survey_id_subid} not found in the catalog.")

        entry = self.catalog[survey_id_subid]

        # FIXED: Initialize properties dict if it doesn't exist
        if 'properties' not in entry or entry['properties'] is None:
            entry['properties'] = {}

        entry['properties']['redshift_conflict'] = False

        prism_redshift = entry['prism_redshift']
        grating_redshifts = [
            value for value in entry['grating_redshifts'].values() if not np.isnan(value)]

        # No redshifts available
        if prism_redshift is None and not grating_redshifts:
            return None

        # Only prism redshift available
        if prism_redshift is not None and not grating_redshifts:
            entry['determined_redshift'] = prism_redshift
            return prism_redshift

        # Only one grating redshift available, no prism
        if prism_redshift is None and len(grating_redshifts) == 1:
            entry['determined_redshift'] = grating_redshifts[0]
            return grating_redshifts[0]

        # One prism and one grating redshift
        if prism_redshift is not None and len(grating_redshifts) == 1:
            if abs(prism_redshift - grating_redshifts[0]) / prism_redshift < 0.1:
                entry['determined_redshift'] = grating_redshifts[0]
                return entry['determined_redshift']
            else:
                entry['determined_redshift'] = prism_redshift
                entry['properties']['redshift_conflict'] = True
                return prism_redshift

        # Multiple redshifts available (either prism + multiple grating, or just multiple grating)
        all_redshifts = []
        if prism_redshift is not None:
            all_redshifts.append(prism_redshift)
        all_redshifts.extend(grating_redshifts)

        # Find the two most similar redshifts
        min_diff = float('inf')
        best_pair = None
        for i in range(len(all_redshifts)):
            for j in range(i + 1, len(all_redshifts)):
                diff = abs(all_redshifts[i] - all_redshifts[j])
                if diff < min_diff:
                    min_diff = diff
                    best_pair = (all_redshifts[i], all_redshifts[j])

        if best_pair:
            # Check if the two nearest redshifts are within 5%
            relative_diff = abs(
                best_pair[0] - best_pair[1]) / min(best_pair[0], best_pair[1])
            avg_redshift = (best_pair[0] + best_pair[1]) / 2

            if relative_diff >= 0.1:
                entry['properties']['redshift_conflict'] = True

            entry['determined_redshift'] = avg_redshift
            return avg_redshift
        else:
            # Fallback (should not happen if we have redshifts)
            entry['determined_redshift'] = all_redshifts[0]
            return all_redshifts[0]

    def update_catalog_item(self, id, catalog):
        """
        Update a catalog item with the given id and catalog data.

        Parameters
        ----------
        id : str
            The survey_id_subid of the spectrum to update.
        catalog : dict
            The catalog data to update the item with.
        """
        if id not in self.catalog:
            raise ValueError(f"Survey ID {id} not found in the catalog.")

        self.catalog[id].update(catalog)

    def sample_num(self):
        """
        Returns the number of samples in the catalog.

        Returns
        -------
        int
            The number of samples in the catalog.
        """
        count = 0
        for id, catalog in self.catalog.items():
            if catalog.get('sample_flag') is True:
                count += 1
        return count

    def load_spectrum_filepaths_from_directory(self, directory_path, file_extension='.[sS][pP][eE][cC].[fF][iI][tT][sS]'):
        """
        Recursively finds files with a matching file extension in the specified directory.

        Args:
            directory_path (str or Path): The root directory path to search.
            file_extension (str): The file ending string to look for, e.g., 'spec.fits'.

        Returns:
            list[str]: A list of all found file path strings.
        """
        p = Path(directory_path).expanduser()
        filepaths = list(p.rglob(f'*{file_extension}'))
        return [str(fp) for fp in filepaths]

    def __repr__(self):
        """
        Returns a concise summary of the catalog's contents.
        """
        if not self.catalog:
            return "<Spectrum_Catalog (empty)>"

        stats = self.get_summary_stats()
        return (
            f"<Spectrum_Catalog with {stats['total_objects']} objects>\n"
            f"  Total spectra: {stats['total_spectra']}\n"
            f"  - Prism: {stats['total_prism_spectra']}\n"
            f"  - Grating: {stats['total_grating_spectra']}\n"
            f"  Objects with prism: {stats['with_prism']}\n"
            f"  Objects with grating: {stats['with_grating']}"
        )


def estimate_snr_error(spectrum_1d,fitter_fit_result, line_center_wavelength,continuum_width=150*astropy.units.AA, measureLengthInFWHM=3, n_random_samples=1000, plot_diagnostics=False):
    """
    Estimate the continuum uncertainty using the linefitter object.

    Parameters
    ----------
    linefitter : LineFitter
        The linefitter object containing the spectrum and fitted lines.
    continuum_width : astropy.units.Quantity, optional
        The width of the continuum region to use for SNR estimation, by default 150 * astropy.units.AA
    sigma_range : int, optional
        The number of standard deviations to consider for the noise estimation, by default 3
    n_random_samples : int, optional
        The number of random samples to use for noise estimation, by default 1000
    plot_diagnostics : bool, optional
        Whether to plot diagnostic figures, by default False

    Returns
    -------
    dict
        A dictionary containing the estimated SNR error and other relevant information.
        - 'continuum_std': The standard deviation of the continuum region.
        - 'random_integrals': The random integrals used for uncertainty estimation.
        - 'snr_error': The estimated SNR error.
        - 'continuum_region': The wavelength range of the continuum region.
    """

    continuum_width=continuum_width.to(astropy.units.AA)
    line_center_wavelength = line_center_wavelength.to(astropy.units.AA)

    line_center_wavelength= fitter_fit_result['parameters']['mean'].to(astropy.units.AA)
    fitted_curve= fitter_fit_result['fitted_curve']
    sigma= fitter_fit_result['parameters']['stddev'].to(astropy.units.AA)

    measure_range= measureLengthInFWHM * 2.355*sigma

    half_width= continuum_width / 2
    continuum_start_region = (line_center_wavelength - half_width, line_center_wavelength + half_width- measure_range)

    #spectrum_1d.set_boundarys(continuum_start_region[0], continuum_start_region[1]+ measure_range)

    continuum_sampling_region = spectrum_1d.dual_boundarys(if_process=True,unit=astropy.units.AA)

    indices = (spectrum_1d.restframe_wavelengths.convert_unit_to(astropy.units.AA).data >= continuum_start_region[0].value) & (spectrum_1d.restframe_wavelengths.convert_unit_to(astropy.units.AA).data <= continuum_start_region[1].value)

    continuumStartRegionWavelengths = spectrum_1d.restframe_wavelengths[indices]

    continuumStartRegionWavelengths = continuumStartRegionWavelengths.convert_unit_to(astropy.units.AA)

    residual_flux=fitter_fit_result['residual_flux'].data

    #plt.plot(spectrum_1d.processing_wavelengths.data, residual_flux)

    np.random.seed(42)  # For reproducibility
    random_integrals = []
    for _ in range(n_random_samples):
        start_idx=np.random.randint(0, (continuumStartRegionWavelengths).data.shape[0]-1)

        random_start = continuumStartRegionWavelengths.data[start_idx]
        random_end = random_start + continuum_width.value
        random_integrals.append(
            np.trapz(
                residual_flux[
                    (spectrum_1d.processing_wavelengths.convert_unit_to(astropy.units.AA).data >= random_start) &
                    (spectrum_1d.processing_wavelengths.convert_unit_to(astropy.units.AA).data <= random_end)
                ],
                x=spectrum_1d.processing_wavelengths[
                    (spectrum_1d.processing_wavelengths.convert_unit_to(astropy.units.AA).data >= random_start) &
                    (spectrum_1d.processing_wavelengths.convert_unit_to(astropy.units.AA).data <= random_end)
                ]
            )
        )


    return random_integrals



def fitting_uncertainty(fit_result):
    covariance = fit_result['covariance']

# 提取相关参数的不确定性
    var_A = covariance[0,0]         # = 3.10e-35
    var_sigma = covariance[2,2]     # = 9.98e-01
    cov_A_sigma = covariance[0,2]   # = -3.05e-18

    sigma_A = np.sqrt(var_A)        # = 5.59e-18
    sigma_sigma = np.sqrt(var_sigma) # = 0.999


# 高斯积分流量：F = A * sigma * sqrt(2*pi)
    Amp = fit_result['parameters']['amplitude'].value
    sigma = fit_result['parameters']['stddev'].value
    sqrt_2pi = np.sqrt(2 * np.pi)

# 验证积分流量
    flux_calculated = Amp * sigma * sqrt_2pi
#  print(f"Integrated Flux: {flux_calculated:.3e}")

# 误差传播公式：σ_F² = (∂F/∂A)²σ_A² + (∂F/∂σ)²σ_σ² + 2(∂F/∂A)(∂F/∂σ)cov_A_σ
    dF_dA = sigma * sqrt_2pi
    dF_dsigma = Amp * sqrt_2pi

    flux_variance = (dF_dA**2 * var_A +
                    dF_dsigma**2 * var_sigma +
                    2 * dF_dA * dF_dsigma * cov_A_sigma)

    fitting_uncertainty = np.sqrt(flux_variance)

    return fitting_uncertainty, flux_calculated


def calculate_err_at_given_wavelength(Spectrum_1d, RestFrameWavelength):

    Spectrum_1d.set_boundarys(RestFrameWavelength-75*astropy.units.AA, RestFrameWavelength+75*astropy.units.AA)

    spectralLineFitter=SpectralLineFitter(Spectrum_1d,RestFrameWavelength)

    fit_result=spectralLineFitter.fit_single_gaussian_with_offset()

    fitting_uncertainty_value, flux_calculated=fitting_uncertainty(fit_result)

    error_array=estimate_snr_error(Spectrum_1d, fit_result, fit_result['parameters']['mean'].to(astropy.units.AA), continuum_width=150*astropy.units.AA, measureLengthInFWHM=2, n_random_samples=20)

    err=np.std(error_array)



    return err, flux_calculated



    return np.sqrt(fitting_uncertainty_value**2 + err**2), flux_calculated


def BalmerDecrementUncertainty(alphaValve, alphaError, betaValve, betaError):
    err_ratio=alphaError**2/betaValve**2 + betaError**2*alphaValve**2/betaValve**4
    return np.sqrt(err_ratio)






class SpectrumAverager:
    def __init__(self):
        """
        Initialize the SpectrumAverager class.
        """
        self.spectra_data = []
        self.common_wavelength = None
        self.interpolated_fluxes = []

        # 归一化数据
        self.normalization_factors = []
        self.normalized_fluxes = None

        # 平均谱相关变量
        self.average_flux = None          # 当前主要使用的平均谱 (可能是Raw也可能是Masked)
        self.average_flux_err = None

        # Masking 相关存储
        self.average_flux_raw = None      # 永远存储未Mask的原始平均谱
        self.average_flux_masked = None   # 存储Mask发射线后的平均谱

    def load_spectrum_catalog(self, catalog_entry_key_list, Catalog, load_func):
        """
        Load spectra from a list of catalog entries.

        Parameters:
        -----------
        catalog_entry_key_list : list
            List of catalog entries containing spectra.
        Catalog : object
            The catalog object containing entry details.
        load_func : function
            A function that takes (filepath, redshift) and returns a spectrum object.
            This decouples the class from external specific functions.
        """
        initial_count = len(self.spectra_data)
        successful_loads = 0

        for i, entry_key in enumerate(catalog_entry_key_list):
            try:
                entry = Catalog.catalog[entry_key]
                filepath = entry.get('prism_filepath', None)
                if not filepath:
                    print(f"Entry {entry_key}: Missing prism_filepath. Skipping.")
                    continue

                # 使用传入的函数加载
                prism_spectrum = load_func(filepath, entry.get('prism_redshift', 0))

                # 数据完整性检查
                wavelengths = prism_spectrum.processing_wavelengths.data
                fluxes = prism_spectrum.processing_flux_lambda.data

                if len(wavelengths) == 0 or len(fluxes) == 0:
                    continue

                if len(wavelengths) != len(fluxes):
                    print(f"Entry {entry_key}: Mismatched data lengths. Skipping.")
                    continue

                # 过滤无效值
                mask = ~(np.isnan(wavelengths) | np.isnan(fluxes) | (wavelengths <= 0))
                if np.sum(mask) < 2:
                    continue

                self.spectra_data.append({
                    'index': initial_count + i,
                    'key': entry_key,
                    'wavelengths': wavelengths[mask],
                    'fluxes': fluxes[mask]
                })
                successful_loads += 1

            except Exception as e:
                print(f"Failed to load spectrum for entry {entry_key}: {e}")
                continue

        return successful_loads > 0

    def create_common_wavelength_grid(self, wavelength_range=None, grid_size=None):
        """
        Create a common wavelength grid for averaging.
        """
        if not self.spectra_data:
            raise ValueError("No spectra data loaded.")

        all_min = [s['wavelengths'].min() for s in self.spectra_data]
        all_max = [s['wavelengths'].max() for s in self.spectra_data]

        if wavelength_range is None:
            common_min = max(all_min)
            common_max = min(all_max)
        else:
            common_min, common_max = wavelength_range

        if common_min >= common_max:
            warnings.warn("No overlapping range found using intersection. Checking full range union.")
            common_min = min(all_min)
            common_max = max(all_max)
            if common_min >= common_max:
                raise ValueError("Invalid wavelength range.")

        if grid_size is None:
            max_points = max([len(s['wavelengths']) for s in self.spectra_data])
            num_points = max_points
        else:
            num_points = int((common_max - common_min) / grid_size) + 1

        self.common_wavelength = np.linspace(common_min, common_max, int(num_points))
        return self.common_wavelength

    def interpolate_spectra(self, interpolation_method='linear'):
        """
        Interpolate all loaded spectra onto the common wavelength grid.
        """
        if self.common_wavelength is None:
            self.create_common_wavelength_grid()

        self.interpolated_fluxes = []
        successful_interpolations = 0

        for spec in self.spectra_data:
            try:
                w_clean = spec['wavelengths']
                f_clean = spec['fluxes']

                interp_func = scipy.interpolate.interp1d(
                    w_clean, f_clean,
                    kind=interpolation_method,
                    bounds_error=False,
                    fill_value=np.nan
                )

                interp_flux = interp_func(self.common_wavelength)

                # 即使全是NaN也保留占位，保持索引对齐
                self.interpolated_fluxes.append(interp_flux)
                if not np.all(np.isnan(interp_flux)):
                    successful_interpolations += 1

            except Exception as e:
                print(f"Failed to interpolate spectrum {spec['index']}: {e}")
                self.interpolated_fluxes.append(np.full_like(self.common_wavelength, np.nan))
                continue

        self.interpolated_fluxes = np.array(self.interpolated_fluxes)
        return successful_interpolations > 0

    def spectrum_normalization(self, reference_wavelength=5500.0, target_flux=1e-19, window_width=10.0):
        """
        Normalize spectra at a reference wavelength using a small window to reduce noise.
        """
        if len(self.interpolated_fluxes) == 0:
            raise ValueError("No interpolated spectra available.")

        if (reference_wavelength < self.common_wavelength.min()) or (reference_wavelength > self.common_wavelength.max()):
            raise ValueError(f"Reference wavelength {reference_wavelength} out of range.")

        self.normalization_factors = []
        self.normalized_fluxes = []

        window_mask = (self.common_wavelength >= (reference_wavelength - window_width/2)) & \
                      (self.common_wavelength <= (reference_wavelength + window_width/2))

        if np.sum(window_mask) == 0:
            ref_index = np.argmin(np.abs(self.common_wavelength - reference_wavelength))
            window_mask[ref_index] = True

        successful = 0
        for i, flux in enumerate(self.interpolated_fluxes):
            flux_in_window = flux[window_mask]
            mean_flux = np.nanmedian(flux_in_window)

            if np.isnan(mean_flux) or mean_flux <= 0:
                self.normalization_factors.append(np.nan)
                self.normalized_fluxes.append(np.full_like(flux, np.nan))
                continue

            factor = target_flux / mean_flux
            self.normalization_factors.append(factor)
            self.normalized_fluxes.append(flux * factor)
            successful += 1

        self.normalized_fluxes = np.array(self.normalized_fluxes)
        self.normalization_factors = np.array(self.normalization_factors)
        return successful > 0

    def spectrum_normalization_within_range(self, wavelength_range=(5000.0, 6000.0), target_flux=None):
        """
        Normalize based on the median flux within a wavelength range.
        """
        if len(self.interpolated_fluxes) == 0:
            raise ValueError("No interpolated spectra available.")

        w_mask = (self.common_wavelength >= wavelength_range[0]) & \
                 (self.common_wavelength <= wavelength_range[1])

        if np.sum(w_mask) == 0:
            raise ValueError("Normalization range contains no data points.")

        flux_subset = self.interpolated_fluxes[:, w_mask]
        median_fluxes = np.nanmedian(flux_subset, axis=1)

        if target_flux is None:
            valid_medians = median_fluxes[~np.isnan(median_fluxes)]
            normalization_base = np.median(valid_medians) if len(valid_medians) > 0 else 1.0
        else:
            normalization_base = target_flux

        # 防止除以0或NaN
        with np.errstate(divide='ignore', invalid='ignore'):
            self.normalization_factors = normalization_base / median_fluxes

        # 处理无效的 normalization factors
        self.normalization_factors[~np.isfinite(self.normalization_factors)] = np.nan

        # 应用归一化
        factors_reshaped = self.normalization_factors[:, np.newaxis]
        self.normalized_fluxes = self.interpolated_fluxes * factors_reshaped

        return True

    def compute_average_spectrum(self, method='mean', ignore_nan=True, use_percentile_filter=False, lower_percentile=16, upper_percentile=84):
        """
        Compute average spectrum using vectorized operations.
        Automatically backs up result to self.average_flux_raw.
        """
        if self.normalized_fluxes is None:
            raise ValueError("No normalized spectra available.")

        flux_matrix = self.normalized_fluxes

        if use_percentile_filter:
            avg_flux, avg_err = self._compute_percentile_filtered_average_vectorized(
                flux_matrix, method, lower_percentile, upper_percentile
            )
        else:
            nan_policy = 'omit' if ignore_nan else 'propagate'
            if method == 'mean':
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    avg_flux = np.nanmean(flux_matrix, axis=0)
                    avg_err = scipy.stats.sem(flux_matrix, axis=0, nan_policy=nan_policy)
            elif method == 'median':
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    avg_flux = np.nanmedian(flux_matrix, axis=0)
                    avg_err = 1.2533 * scipy.stats.sem(flux_matrix, axis=0, nan_policy=nan_policy)
            else:
                raise ValueError("Method must be 'mean' or 'median'")

        valid_counts = np.sum(~np.isnan(flux_matrix), axis=0)

        # 保存结果
        self.average_flux = avg_flux
        self.average_flux_err = avg_err

        # 备份到 RAW
        self.average_flux_raw = avg_flux.copy()
        # 重置 Masked 数据，因为 Raw 变了，Masked 需要重新计算
        self.average_flux_masked = None

        return self.common_wavelength, self.average_flux, self.average_flux_err, valid_counts

    def _compute_percentile_filtered_average_vectorized(self, flux_matrix, method, lower_p, upper_p):
        """
        Helper for fast percentile filtering.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            limit_low = np.nanpercentile(flux_matrix, lower_p, axis=0)
            limit_high = np.nanpercentile(flux_matrix, upper_p, axis=0)

        # 使用广播机制创建 Mask
        mask = (flux_matrix >= limit_low[np.newaxis, :]) & (flux_matrix <= limit_high[np.newaxis, :])
        filtered_matrix = np.where(mask, flux_matrix, np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if method == 'median':
                avg = np.nanmedian(filtered_matrix, axis=0)
                err = 1.2533 * scipy.stats.sem(filtered_matrix, axis=0, nan_policy='omit')
            else:
                avg = np.nanmean(filtered_matrix, axis=0)
                err = scipy.stats.sem(filtered_matrix, axis=0, nan_policy='omit')
        return avg, err

    def apply_emission_line_mask(self, mask_regions):
        """
        Mask emission lines by setting flux values to NaN within specified regions.
        Stores result in self.average_flux_masked.

        Parameters:
        -----------
        mask_regions : list of tuples
            [(start_wave, end_wave), ...]
        """
        if self.average_flux_raw is None:
            raise ValueError("Average spectrum has not been computed yet.")

        if self.common_wavelength is None:
            raise ValueError("Common wavelength grid is missing.")

        # 从 Raw 数据开始 Mask
        masked_flux = self.average_flux_raw.copy()

        for (start_wave, end_wave) in mask_regions:
            mask_indices = (self.common_wavelength >= start_wave) & (self.common_wavelength <= end_wave)
            masked_flux[mask_indices] = np.nan

        self.average_flux_masked = masked_flux

        # 将主指针指向 masked 版本 (可选)
        # self.average_flux = self.average_flux_masked

        print(f"Applied masks to {len(mask_regions)} regions.")
        return self.average_flux_masked

    def plot_average_spectrum(self, show_individual=True, show_average=True, show_error=True,
                              show_masked_comparison=True, figsize=(15, 8), alpha=0.1,
                              show_normalization_point=True, reference_wavelength=5500.0,
                              ylim=None, if_show=True):
        """
        Plot average spectrum. Supports visualizing masked vs unmasked data.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # 1. 绘制个体光谱
        if show_individual and self.normalized_fluxes is not None:
            has_label = False
            for flux in self.normalized_fluxes:
                if np.all(np.isnan(flux)): continue
                label = 'Individual Spectra' if not has_label else None
                ax.plot(self.common_wavelength, flux, color='gray', alpha=alpha, label=label, lw=0.5, zorder=1)
                has_label = True

        # 2. 绘制平均光谱
        if show_average:
            # 存在 Masked 数据且要求对比显示
            if self.average_flux_masked is not None and show_masked_comparison:
                # 原始谱 (背景)
                ax.plot(self.common_wavelength, self.average_flux_raw, color='orange', alpha=0.6,
                        linewidth=1.5, label='Original (with Emission)', zorder=2)
                # Masked 谱 (前景，黑色)
                ax.plot(self.common_wavelength, self.average_flux_masked, color='k',
                        linewidth=2, label='Masked Continuum', zorder=3)

            # 仅显示当前的 average_flux
            elif self.average_flux is not None:
                ax.plot(self.common_wavelength, self.average_flux, color='k',
                        linewidth=2, label='Average Spectrum', zorder=3)

            # 绘制误差
            if show_error and self.average_flux_err is not None:
                curr_flux = self.average_flux_raw if self.average_flux_raw is not None else self.average_flux
                if curr_flux is not None:
                    ax.fill_between(self.common_wavelength,
                                    curr_flux - self.average_flux_err,
                                    curr_flux + self.average_flux_err,
                                    color='red', alpha=0.2, label='Standard Error', zorder=2)

        # 3. 绘制归一化点 (使用 Raw 数据定位)
        if show_normalization_point:
            ref_flux = self.average_flux_raw if self.average_flux_raw is not None else self.average_flux
            if ref_flux is not None:
                idx = np.argmin(np.abs(self.common_wavelength - reference_wavelength))
                if idx < len(ref_flux) and not np.isnan(ref_flux[idx]):
                    ax.scatter(self.common_wavelength[idx], ref_flux[idx],
                               color='blue', s=100, zorder=10, marker='X', label='Norm Point')

        ax.set_xlabel('Wavelength ($\AA$)', fontsize=16)
        ax.set_ylabel('Normalized Flux', fontsize=16)
        ax.set_title('Average Spectrum', fontsize=18)

        if ylim:
            ax.set_ylim(ylim)
        else:
            # 自动缩放
            target = self.average_flux_raw if self.average_flux_raw is not None else self.average_flux
            if target is not None:
                valid = target[~np.isnan(target)]
                if len(valid) > 0:
                    ymax = np.percentile(valid, 99) * 1.5
                    ax.set_ylim(0, ymax)

        ax.legend(loc='best', fontsize=12)
        plt.tight_layout()

        if if_show:
            plt.show()
        return fig, ax

    def get_normalization_factors(self):
        """
        Get normalization statistics.
        """
        if len(self.normalization_factors) == 0:
            return None
        valid_factors = self.normalization_factors[~np.isnan(self.normalization_factors)]
        return {
            'factors': self.normalization_factors,
            'count': len(valid_factors),
            'mean': np.mean(valid_factors) if len(valid_factors) > 0 else None,
            'std': np.std(valid_factors) if len(valid_factors) > 0 else None
        }

    def save_average_spectrum_to_file(self, filepath):
        """
        Save the average spectrum data to a file.
        Saves 'flux_raw' and optionally 'flux_masked' if available.
        """
        if self.average_flux_raw is None and self.average_flux is None:
            raise ValueError("No average spectrum data to save.")

        # 优先使用 Raw，如果没有则使用普通 average
        base_flux = self.average_flux_raw if self.average_flux_raw is not None else self.average_flux

        data_to_save = {
            'common_wavelength': self.common_wavelength,
            'average_flux': base_flux, # 兼容旧读取习惯
            'flux_raw': base_flux,
            'average_flux_err': self.average_flux_err
        }

        # 如果有 Masked 数据，额外保存
        if self.average_flux_masked is not None:
            data_to_save['flux_masked'] = self.average_flux_masked
        else:
            # 为了数据结构一致性，如果没有masked，副本一份 raw
            data_to_save['flux_masked'] = base_flux

        np.savez(filepath, **data_to_save)
        print(f"Saved spectrum data to {filepath}")

    def load_average_spectrum_from_file(self, filepath):
        """
        Load the average spectrum data.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File {filepath} not found.")

        loaded_data = np.load(filepath)
        self.common_wavelength = loaded_data['common_wavelength']
        self.average_flux_err = loaded_data['average_flux_err']

        # 尝试读取新结构，兼容旧结构
        if 'flux_raw' in loaded_data:
            self.average_flux_raw = loaded_data['flux_raw']
        else:
            self.average_flux_raw = loaded_data['average_flux']

        if 'flux_masked' in loaded_data:
            self.average_flux_masked = loaded_data['flux_masked']
            self.average_flux = self.average_flux_masked # 默认加载后主要指引为 masked
        else:
            self.average_flux = self.average_flux_raw
            self.average_flux_masked = None

        return True







class CalzettiCorrector:

    """
    A class to apply the Calzetti et al. (2000) dust attenuation law to a spectrum_1d.

    """

    def __init__(self, Rv=4.05):
        """
        Initialize the CalzettiCorrector with a specified Rv value.

        Parameters
        ----------
        Rv : float, optional
            The total-to-selective extinction ratio. Default is 4.05.
        """
        self.Rv = Rv

    def _get_k_lambda(self, wavelength_micron):
        """
        Calculate the attenuation curve k(λ) based on the Calzetti et al. (2000) law.

        Parameters
        ----------
        wavelength_micron : array-like
            Wavelengths in Angstroms.

        Returns
        -------
        k_lambda : array-like
            The attenuation curve values at the given wavelengths.
        """
        k_lambda = np.zeros_like(wavelength_micron)
        if isinstance(wavelength_micron, (int, float)):
            wavelength_micron = np.array([wavelength_micron])

        if isinstance(wavelength_micron, astropy.units.Quantity):
            if not wavelength_micron.unit.is_equivalent(astropy.units.micron):
                raise ValueError("Input wavelength must have units of microns.")
            wavelength_micron = (wavelength_micron.to(astropy.units.micron)).value

        # Apply the Calzetti law piecewise
        mask1 = (wavelength_micron >= 0.12) & (wavelength_micron < 0.63)

        if np.any(mask1):
            x=1./wavelength_micron[mask1]
            k_lambda[mask1] = (2.659 * (-2.156 + 1.509 * x - 0.198 * (x**2) + 0.011 * (x**3)) + self.Rv)

        mask2 = (wavelength_micron >= 0.63) & (wavelength_micron <= 2.2)
        if np.any(mask2):
            x=1. / wavelength_micron[mask2]
            k_lambda[mask2] = (2.659 * (-1.857 + 1.040 * x) + self.Rv)
            print(x)

        mask_out=(~mask1) & (~mask2)
        if np.any(mask_out):
            k_lambda[mask_out] = np.nan

        return k_lambda



    def calculate_EBV_stars(self, ebv_gas=None, Halpha_flux=None, Hbeta_flux=None):
        """
        Calculate the stellar color excess E(B-V)_stars from the gas color excess E(B-V)_gas.

        Parameters
        ----------
        ebv_gas : float, optional
            The gas color excess E(B-V)_gas.
        Halpha_flux : float, optional
            The flux of the Hα emission line.
        Hbeta_flux : float, optional
            The flux of the Hβ emission line.

        Returns
        -------
        ebv_stars : float
            The stellar color excess E(B-V)_stars.
        """
        if ebv_gas is not None:
            return 0.44 * ebv_gas

        elif Halpha_flux is not None and Hbeta_flux is not None:

            F_Ha=Halpha_flux.value if isinstance(Halpha_flux, astropy.units.Quantity) else Halpha_flux
            F_Hb=Hbeta_flux.value if isinstance(Hbeta_flux, astropy.units.Quantity) else Hbeta_flux

            if F_Hb <= 0 or F_Ha <= 0:
                raise ValueError("Hα and Hβ fluxes must be positive values.")

            balmer_decrement = F_Ha / F_Hb
            intrinsic_balmer_decrement = 2.86  # Case B recombination at T=10,000 K

            if balmer_decrement < intrinsic_balmer_decrement:
                warnings.warn("Observed Balmer decrement is less than intrinsic value. Setting E(B-V)_gas to 0.")
                ebv_gas = 0.0
                return 0.0

            EBV_stars=0.3762 * np.log(balmer_decrement / intrinsic_balmer_decrement)
            return EBV_stars

        else:
            raise ValueError("Either ebv_gas or both Halpha_flux and Hbeta_flux must be provided.")


    def deredden_flux(self, spectrum, ebv_star=None, ebv_gas=None, ha_flux=None, hb_flux=None):
        """
        Apply dust correction to a Spectrum_1d object.
        """
        # 1. Determine E(B-V)_star
        if ebv_star is None:
            ebv_star = self.calculate_EBV_stars(ebv_gas, ha_flux, hb_flux)

        # print(f"Applying Calzetti Correction with E(B-V)_star = {ebv_star:.4f}")

        # 2. Create a deep copy of the spectrum to avoid modifying the original
        spec_new = copy.deepcopy(spectrum)

        # 3. Get Rest-frame Wavelengths in Microns for calculation
        wave_quantity = spec_new.restframe_wavelengths.data
        if not isinstance(wave_quantity, astropy.units.Quantity):
            wave_quantity = wave_quantity * spec_new.restframe_wavelengths.unit

        wave_micron = wave_quantity.to(astropy.units.micron).value

        # 4. Calculate A_lambda & Correction Factor
        k_lambda_vals = self._get_k_lambda(wave_micron)
        A_lambda = k_lambda_vals * ebv_star
        correction_factor = 10**(0.4 * A_lambda)

        # 5. Apply to Fluxes (both lambda and nu) safely

        # --- Update restframe_flux_lambda ---
        if spec_new.restframe_flux_lambda is not None:

            new_data_lambda = spec_new.restframe_flux_lambda.data * correction_factor

            new_unc_lambda = None
            if spec_new.restframe_flux_lambda.uncertainty is not None:
                unc_type = type(spec_new.restframe_flux_lambda.uncertainty)
                new_unc_vals = spec_new.restframe_flux_lambda.uncertainty.array * correction_factor
                new_unc_lambda = unc_type(new_unc_vals)

            spec_new.restframe_flux_lambda = astropy.nddata.NDDataArray(
                data=new_data_lambda,
                uncertainty=new_unc_lambda,
                unit=spec_new.restframe_flux_lambda.unit
            )

        # --- Update restframe_flux_nu ---
        if spec_new.restframe_flux_nu is not None:

            new_data_nu = spec_new.restframe_flux_nu.data * correction_factor


            new_unc_nu = None
            if spec_new.restframe_flux_nu.uncertainty is not None:
                unc_type = type(spec_new.restframe_flux_nu.uncertainty)
                new_unc_vals = spec_new.restframe_flux_nu.uncertainty.array * correction_factor
                new_unc_nu = unc_type(new_unc_vals)


            spec_new.restframe_flux_nu = astropy.nddata.NDDataArray(
                data=new_data_nu,
                uncertainty=new_unc_nu,
                unit=spec_new.restframe_flux_nu.unit
            )

        # 6. Reset processing arrays
        spec_new.processing_wavelengths = spec_new.restframe_wavelengths
        spec_new.processing_flux_lambda = spec_new.restframe_flux_lambda
        spec_new.processing_flux_nu = spec_new.restframe_flux_nu

        # Ensure processing wavelengths are in Angstroms
        if spec_new.processing_wavelengths.unit != astropy.units.AA:
             spec_new.processing_wavelengths = spec_new.processing_wavelengths.convert_unit_to(astropy.units.AA)

        return spec_new

def flux_density(wavelength, flux_lambda, target_region):
    mask = (wavelength >= target_region[0]) & (wavelength <= target_region[1])
    if np.sum(mask) == 0:
        return np.nan
    if np.isnan(flux_lambda[mask]).any():
        return np.nan

    return np.average(flux_lambda[mask],weights=wavelength[mask])



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import savgol_filter
from collections import defaultdict
import warnings

# 假设 FunctionLib 就在当前路径或环境变量中
import FunctionLib as FL

# --- 全局绘图样式配置 ---
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
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
}
mpl.rcParams.update(config)

class DustAttenuationAnalyst:
    def __init__(self, catalog_obj, wavelength_grid=(1350, 8000), grid_step=5.0):
        """
        初始化分析器。

        Parameters
        ----------
        catalog_obj : Spectrum_Catalog
            包含元数据的 Catalog 对象。
        wavelength_grid : tuple
            (min_wave, max_wave) 单位 Å。
        grid_step : float
            网格步长。
        """
        self.catalog = catalog_obj
        self.wavelength_grid = wavelength_grid
        self.grid_step = grid_step

        # 巴耳末递减分组条件
        self.balmer_conditions = [
            (-0.05, 0.02), (0.02, 0.10), (0.10, 0.20),
            (0.20, 0.30), (0.30, 0.45), (0.45, 0.60), (0.60, 1.00)
        ]

        # 发射线 Mask 区域
        self.mask_regions = [
            (3675, 4150), (4300, 4400), (4800, 5100),
            (5850, 5900), (6200, 6325), (6500, 6775),
            (7000, 7150), (7300, 7500)
        ]

        # 数据存储容器
        self.groups = defaultdict(list)
        self.group_mean_decrements = {}
        self.averagers = {}
        self.reference_smooth_flux = None

        # 计算结果存储
        self.attenuation_curves = {}     # 每个组的 Q_lambda
        self.effective_curve = None      # 加权平均后的 Q_lambda
        self.effective_curve_wave = None
        self.curve_mode = None           # 记录当前使用的是 smoothed 还是 raw

        # 拟合结果存储
        self.poly_coeffs = None          # 多项式系数 (针对 1/lambda)
        self.scaling_factor = None       # 归一化因子 S
        self.derived_Rv = None           # 推导出的 Rv
        self.intercept_val = 0.0         # 截距 (默认为0)

    # =========================================================================
    # 1. 数据加载与处理
    # =========================================================================

    def _calculate_balmer_decrement(self, halpha, hbeta):
        if hbeta <= 0 or halpha <= 0:
            return None
        return np.log(halpha / hbeta / 2.86)

    def _get_group_index(self, value):
        for i, (low, high) in enumerate(self.balmer_conditions):
            if low <= value < high:
                return i
        return -1

    def load_and_group_objects(self, target_ids):
        """第一步：加载 ID 并按巴耳末递减分组"""
        print(f"Categorizing {len(target_ids)} objects...")
        self.groups.clear()
        self.group_mean_decrements.clear()

        group_values = defaultdict(list)

        for survey_id in target_ids:
            entry = self.catalog.load_spectrum_info(survey_id)
            if not entry: continue

            try:
                params = entry.get('dust_curve_parameters', {})
                line_params = params.get('Best_Balmer_Decrement_Line_Parameters', {})

                ha = line_params.get('halpha_flux', 0)
                hb = line_params.get('hbeta_flux', 0)

                bd_val = self._calculate_balmer_decrement(ha, hb)

                if bd_val is not None:
                    g_idx = self._get_group_index(bd_val)
                    if g_idx != -1:
                        self.groups[g_idx].append(survey_id)
                        group_values[g_idx].append(bd_val)
            except Exception:
                continue

        for idx in sorted(self.groups.keys()):
            vals = group_values[idx]
            mean_val = np.mean(vals)
            if idx == 0: mean_val = 0.0 # 强制参考组为 0
            self.group_mean_decrements[idx] = mean_val
            print(f"Group {idx}: {len(vals)} objects, Mean BD={mean_val:.3f}")

    def compute_average_spectra(self):
        """第二步：堆叠光谱"""
        print("\nComputing average spectra...")
        self.averagers = {}

        for g_idx in sorted(self.groups.keys()):
            ids = self.groups[g_idx]
            if not ids: continue

            averager = FL.SpectrumAverager()
            success = averager.load_spectrum_catalog(ids, self.catalog, FL.Load_Spectrum_From_Fits)

            if not success:
                print(f"  Group {g_idx}: Failed to load spectra.")
                continue

            averager.create_common_wavelength_grid(self.wavelength_grid, self.grid_step)
            averager.interpolate_spectra(interpolation_method='linear')
            averager.spectrum_normalization_within_range((5400, 5600), target_flux=1.e-19)

            # 使用中值堆叠 + 百分位滤波
            averager.compute_average_spectrum(
                method='median', ignore_nan=True, use_percentile_filter=True,
                lower_percentile=16, upper_percentile=84
            )

            # 应用发射线 Mask
            averager.apply_emission_line_mask(self.mask_regions)
            self.averagers[g_idx] = averager
            print(f"  Group {g_idx}: Processed.")

    def compute_smooth_reference(self, window_len=101, poly_order=3):
        """第三步：平滑参考谱 (Group 0)"""
        if 0 not in self.averagers:
            raise ValueError("Group 0 (Reference) is missing.")

        ref_avg = self.averagers[0]
        # 使用 masked 数据作为起点
        flux_data = ref_avg.average_flux_masked.copy()

        # 线性插值填补 Mask 掉的 NaN，以便进行平滑滤波
        series = pd.Series(flux_data)
        flux_interp = series.interpolate(method='linear', limit_direction='both').to_numpy()

        self.reference_smooth_flux = savgol_filter(flux_interp, window_length=window_len, polyorder=poly_order)
        print("Reference spectrum smoothed.")

    # =========================================================================
    # 2. 核心计算：Attenuation Curves (Q_lambda)
    # =========================================================================

    # def compute_attenuation_curves(self, mode='smoothed'):
    #     """
    #     第四步：计算有效衰减曲线 Q_lambda

    #     Parameters
    #     ----------
    #     mode : str
    #         'smoothed' : 使用平滑后的 Group 0 作为参考 (推荐，减少噪声)。
    #         'raw'      : 使用原始堆叠的 Group 0 作为参考。
    #     """
    #     if mode == 'smoothed':
    #         if self.reference_smooth_flux is None:
    #             self.compute_smooth_reference()
    #         ref_flux_used = self.reference_smooth_flux
    #     elif mode == 'raw':
    #         if 0 not in self.averagers:
    #              raise ValueError("Group 0 not processed.")
    #         ref_flux_used = self.averagers[0].average_flux_masked
    #     else:
    #         raise ValueError("mode must be 'smoothed' or 'raw'")

    #     self.curve_mode = mode
    #     ref_tau = self.group_mean_decrements[0]
    #     self.attenuation_curves = {}

    #     # 计算每个组的 Q 曲线
    #     for g_idx, averager in self.averagers.items():
    #         if g_idx == 0: continue

    #         target_flux = averager.average_flux_masked
    #         target_tau = self.group_mean_decrements[g_idx]
    #         delta_tau = target_tau - ref_tau

    #         with np.errstate(divide='ignore', invalid='ignore'):
    #             ratio = target_flux / ref_flux_used
    #             ratio[ratio <= 0] = np.nan
    #             q_curve = -np.log(ratio) / delta_tau

    #         self.attenuation_curves[g_idx] = {
    #             'wavelength': averager.common_wavelength,
    #             'Q_lambda': q_curve,
    #             'delta_tau': delta_tau
    #         }

    #     # 计算加权平均 (Effective Curve)
    #     if self.attenuation_curves:
    #         q_list = [d['Q_lambda'] for d in self.attenuation_curves.values()]
    #         w_list = [d['delta_tau'] for d in self.attenuation_curves.values()]

    #         # 关键修复：确保是数值数组而非字典列表
    #         stack_q = np.ma.masked_invalid(np.array(q_list))
    #         self.effective_curve = np.ma.average(stack_q, axis=0, weights=w_list).filled(np.nan)
    #         self.effective_curve_wave = list(self.attenuation_curves.values())[0]['wavelength']
    #         print(f"Computed effective curve using {mode} reference.")
    #     else:
    #         print("No groups available to compute curves.")

    def compute_attenuation_curves(self, mode='smoothed'):
        """第四步：计算有效衰减曲线及其误差"""
        # ... (前部分代码保持不变: 确定 ref_flux_used) ...
        if mode == 'smoothed':
            if self.reference_smooth_flux is None: self.compute_smooth_reference()
            ref_flux_used = self.reference_smooth_flux
        elif mode == 'raw':
            if 0 not in self.averagers: raise ValueError("Group 0 missing.")
            ref_flux_used = self.averagers[0].average_flux_masked
        else:
            raise ValueError("mode error")

        self.curve_mode = mode
        ref_tau = self.group_mean_decrements[0]
        self.attenuation_curves = {}

        for g_idx, averager in self.averagers.items():
            if g_idx == 0: continue

            target_flux = averager.average_flux_masked
            # 获取 Flux Error (SEM)
            target_flux_err = averager.average_flux_err
            if target_flux_err is None:
                # 如果没有误差，给一个极小值避免除以0 (虽不应该发生)
                target_flux_err = np.zeros_like(target_flux) + 1e-18

            target_tau = self.group_mean_decrements[g_idx]
            delta_tau = target_tau - ref_tau

            # --- Q 计算 ---
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = target_flux / ref_flux_used
                ratio[ratio <= 0] = np.nan
                q_curve = -np.log(ratio) / delta_tau

                # --- Error Propagation ---
                # sigma_Q = (1/delta_tau) * (sigma_F / F)
                # 注意：ref_flux_used 视为无误差常量
                q_err = (1.0 / np.abs(delta_tau)) * (target_flux_err / np.abs(target_flux))

            self.attenuation_curves[g_idx] = {
                'wavelength': averager.common_wavelength,
                'Q_lambda': q_curve,
                'Q_err': q_err,      # <--- 新增
                'delta_tau': delta_tau
            }

        # --- 计算 Effective Curve 及其误差 ---
        if self.attenuation_curves:
            q_list = [d['Q_lambda'] for d in self.attenuation_curves.values()]
            err_list = [d['Q_err'] for d in self.attenuation_curves.values()]
            w_list = [d['delta_tau'] for d in self.attenuation_curves.values()]

            # 转换为 Masked Array
            stack_q = np.ma.masked_invalid(np.array(q_list))
            stack_err = np.ma.masked_invalid(np.array(err_list))
            weights = np.array(w_list)

            # 1. 加权平均值
            self.effective_curve = np.ma.average(stack_q, axis=0, weights=weights).filled(np.nan)

            # 2. 误差合成
            # 公式: sigma_eff = sqrt( sum( (w_i * sigma_i)^2 ) ) / sum(w_i)
            # 注意：这是假设各组误差相互独立
            w_sum = np.sum(weights)
            weighted_var_sum = np.sum((stack_err * weights[:, np.newaxis])**2, axis=0)
            self.effective_curve_err = (np.sqrt(weighted_var_sum) / w_sum).filled(np.nan) # <--- 新增结果

            self.effective_curve_wave = list(self.attenuation_curves.values())[0]['wavelength']
            print(f"Computed effective curve and errors using {mode} reference.")
        else:
            print("No groups available.")

    # =========================================================================
    # 3. 拟合与推导 (Analysis)
    # =========================================================================
    def fit_polynomial(self, order=3):
        """
        对 Effective Curve 进行多项式拟合 (vs 1/lambda)。

        改进点：
        如果存在误差数组 (effective_curve_err)，则执行加权最小二乘拟合 (Weighted Least Squares)。
        权重设为 w = 1 / sigma。
        """
        if self.effective_curve is None:
            raise ValueError("Run compute_attenuation_curves first.")

        # 1. 准备数据
        wave_ang = self.effective_curve_wave
        y_data = self.effective_curve
        y_err = self.effective_curve_err

        # 转换为 1/μm
        wave_microns = wave_ang / 10000.0
        x_inverse = 1.0 / wave_microns

        # 2. 构建 Mask (剔除 NaN, Inf)
        # 基础 Mask：x 和 y 必须有效
        valid_mask = np.isfinite(y_data) & np.isfinite(x_inverse)

        # 3. 拟合逻辑
        if y_err is not None:
            # 如果有误差，还需要剔除误差无效或为0的点
            valid_mask = valid_mask & np.isfinite(y_err) & (y_err > 0)

            x_fit = x_inverse[valid_mask]
            y_fit = y_data[valid_mask]
            err_fit = y_err[valid_mask]

            # 定义权重: w = 1/sigma
            weights = 1.0 / err_fit

            print(f"Fitting polynomial (Weighted Least Squares) with {len(x_fit)} points...")

            # cov=True 会返回协方差矩阵，可以用来估算系数的误差
            coeffs, cov_matrix = np.polyfit(x_fit, y_fit, order, w=weights, cov=True)

            # 计算系数的标准误差 (diag(cov) 的平方根)
            perr = np.sqrt(np.diag(cov_matrix))
            print("Polynomial Coefficients errors:", perr)

        else:
            # 如果没有误差，执行普通最小二乘法
            print("Fitting polynomial (Ordinary Least Squares, no errors)...")
            x_fit = x_inverse[valid_mask]
            y_fit = y_data[valid_mask]
            coeffs = np.polyfit(x_fit, y_fit, order)

        self.poly_coeffs = coeffs
        print("Polynomial Fit Coefficients (highest power first):", self.poly_coeffs)

        return self.poly_coeffs

    def _poly_shape(self, x):
        """根据拟合系数计算多项式值"""
        return np.polyval(self.poly_coeffs, x)

    def _poly_shape_cubic_only(self, x):
        """仅包含高阶项 (去掉常数项)，用于推导形状"""
        # coeffs 顺序: [c3, c2, c1, c0]
        c3, c2, c1 = self.poly_coeffs[0], self.poly_coeffs[1], self.poly_coeffs[2]
        return c3 * x**3 + c2 * x**2 + c1 * x

    def derive_k_curve_parameters(self, intercept_val=None):
        """
        推导归一化因子 S 和 Rv。

        Parameters
        ----------
        intercept_val : float, optional
            设定无穷远处的截距值 (例如 Calzetti 曲线在 1/lambda=0 处的推断值)。
            如果为 None，则假设 k(0) = 0 (即只使用多项式的高阶项进行归一化)。
        """
        if self.poly_coeffs is None:
            self.fit_polynomial()

        x_B = 1.0 / 0.44
        x_V = 1.0 / 0.55

        # 如果指定了截距，我们需要调整公式 k(x) = S * P_cubic(x) + intercept
        # 归一化条件 k(B) - k(V) = 1
        # => S * (P(B) - P(V)) = 1  (截距相减抵消)

        val_B = self._poly_shape_cubic_only(x_B)
        val_V = self._poly_shape_cubic_only(x_V)

        self.scaling_factor = 1.0 / (val_B - val_V)

        if intercept_val is not None:
            self.intercept_val = intercept_val
        else:
            self.intercept_val = 0.0 # 或者基于拟合的 c0，取决于物理假设

        # 计算 Rv = k(V)
        k_V = self.scaling_factor * val_V + self.intercept_val
        self.derived_Rv = k_V

        print(f"Derived Parameters: S={self.scaling_factor:.4f}, Intercept={self.intercept_val}, Rv={self.derived_Rv:.3f}")

    # =========================================================================
    # 4. 绘图方法 (Visualization)
    # =========================================================================

    def plot_fit_vs_calzetti(self):
        """绘制图表 1：Effective Q_lambda 数据 vs 多项式拟合 vs Calzetti"""
        if self.poly_coeffs is None: self.fit_polynomial()

        wave = self.effective_curve_wave
        x = 10000.0 / wave
        y_fit = self._poly_shape(x)

        fig, ax = plt.subplots()

        # 原始数据
        ax.plot(wave, self.effective_curve, color='blue', lw=3, alpha=0.4,
                label=f'Calculated Curve ({self.curve_mode} ref)')
        ax.errorbar(wave, self.effective_curve, yerr=self.effective_curve_err,
                    fmt='o', color='blue', alpha=0.2, markersize=4, capsize=2)

        # 拟合曲线
        ax.plot(wave, y_fit, color='black', lw=3, ls='--',
                label=r'Polynomial Fit ($x=1/\lambda$)')

        # Calzetti 参考
        calz = self.get_calzetti_law(wave)
        ax.plot(wave, calz, color='red', lw=3, label='Calzetti (2000)')

        ax.set_xlabel('Wavelength (Å)')
        ax.set_ylabel(r'$Q_\lambda$ (Attenuation Curve)')
        ax.set_title(r'Effective Attenuation Curve: Fit vs Calzetti')
        ax.set_xlim(1500, 8000)
        ax.set_ylim(-2, 7)
        ax.legend()
        ax.grid(True, alpha=0.2)
        plt.show()

    def plot_derived_k_curve(self):
        """绘制图表 2：推导出的归一化 Extinction Curve k(lambda)"""
        if self.scaling_factor is None: self.derive_k_curve_parameters()

        wave = np.linspace(1500, 8000, 1000)
        x = 10000.0 / wave

        # 计算 k_curve
        shape_val = self._poly_shape_cubic_only(x)
        k_curve = self.scaling_factor * shape_val + self.intercept_val

        # Calzetti 参考 (Rv=4.05)
        k_calz = 2.659 * self.get_calzetti_law(wave, Rv=4.05) + 4.05

        fig, ax = plt.subplots()
        ax.plot(wave, k_curve, color='blue', lw=4,
                label=f'Derived Curve ($R_V={self.derived_Rv:.2f}$)')
        ax.plot(wave, k_calz, color='red', lw=3, ls='--',
                label='Calzetti (2000) ($R_V=4.05$)')

        # 标注 Rv 点
        ax.scatter([5500], [self.derived_Rv], color='blue', s=200, zorder=10)
        ax.annotate(f'{self.derived_Rv:.2f}', xy=(5500, self.derived_Rv),
                    xytext=(5600, self.derived_Rv-0.5), fontsize=24, color='blue')

        # 辅助线
        ax.axvline(5500, color='gray', ls=':', alpha=0.5)
        ax.axvline(4400, color='gray', ls=':', alpha=0.5)
        ax.text(4400, max(k_curve)*0.9, 'B', ha='center', color='gray')
        ax.text(5500, max(k_curve)*0.9, 'V', ha='center', color='gray')

        ax.set_xlabel('Wavelength (Å)')
        ax.set_ylabel(r'$k(\lambda)$ (Normalized Extinction)')
        ax.set_title('Derived Attenuation Curve vs. Calzetti')
        ax.set_xlim(1500, 8000)
        ax.set_ylim(0, max(k_curve.max(), k_calz.max())*1.1)
        ax.legend()
        ax.grid(True, alpha=0.2)
        plt.show()


    # def plot_reddening_curve_comparison(self,
    #                                     if_show=True,
    #                                     if_input=False,
    #                                     fig=None,
    #                                     ax=None,
    #                                     curve_label='Derived Curve (Data)',  # <--- 在这里输入你的 Label
    #                                     curve_color='blue',                  # <--- 在这里输入你的颜色
    #                                     draw_background=True):               # <--- 控制是否画背景参考线
    #     """
    #     绘制图表 3：去 Rv 化的红化对比 E(l-V)/E(B-V)

    #     Parameters
    #     ----------
    #     curve_label : str
    #         你的数据曲线在图例中的名称。
    #     curve_color : str
    #         你的数据曲线的颜色。
    #     draw_background : bool
    #         是否绘制 MW/SMC/Calzetti 参考线及坐标轴装饰。
    #         (叠加绘图时，建议第一次设为 True，之后设为 False)
    #     """
    #     if self.scaling_factor is None: self.derive_k_curve_parameters()

    #     if if_input:
    #         if fig is None or ax is None:
    #             raise ValueError("If if_input is True, fig and ax must be provided.")
    #     else:
    #         fig, ax = plt.subplots(figsize=(12, 10))
    #         # 如果是新图，强制画背景
    #         draw_background = True

    #     x_grid = np.linspace(0.5, 8.2, 1000) # 1/um

    #     # 1. 计算 Derived Curve 的 E(l-V)/E(B-V)
    #     shape_grid = self._poly_shape_cubic_only(x_grid)
    #     shape_V = self._poly_shape_cubic_only(1.0/0.55)
    #     shape_B = self._poly_shape_cubic_only(1.0/0.44)
    #     y_derived = (shape_grid - shape_V) / (shape_B - shape_V)

    #     # 2. 绘制背景参考线 (仅当 draw_background=True 时)
    #     if draw_background:
    #         y_calz = self._get_normalized_reddening(x_grid, 'calzetti')
    #         y_mw = self._get_normalized_reddening(x_grid, 'mw')
    #         y_smc = self._get_normalized_reddening(x_grid, 'smc')

    #         ax.plot(x_grid, y_mw, color='green', ls=':', lw=2, label='Milky Way (CCM89)')
    #         ax.plot(x_grid, y_smc, color='purple', ls='-.', lw=2, label='SMC (Pei92)')
    #         ax.plot(x_grid, y_calz, color='black', ls='--', lw=3, label='Calzetti (2000)')

    #         # 绘制辅助线和文本
    #         x_V, x_B = 1.0/0.55, 1.0/0.44
    #         ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    #         ax.axvline(x_V, color='gray', linestyle='-', alpha=0.3)
    #         ax.scatter([x_V], [0], color='black', s=80, zorder=10)
    #         ax.text(x_V, 0.5, 'V (5500$\AA$)\nNormalized to 0', rotation=90, color='gray', fontsize=14, ha='right')

    #         ax.axhline(1, color='gray', linestyle=':', alpha=0.3)
    #         ax.axvline(x_B, color='gray', linestyle=':', alpha=0.3)
    #         ax.scatter([x_B], [1], color='black', s=80, zorder=10)
    #         ax.text(x_B, 1.5, 'B (4400$\AA$)\nNormalized to 1', rotation=90, color='gray', fontsize=14, ha='right')

    #         # 设置轴标签和标题
    #         ax.set_xlabel(r'Inverse Wavelength $\lambda^{-1}$ ($\mu m^{-1}$)', fontsize=20)
    #         ax.set_ylabel(r'$E(\lambda - V) / E(B - V)$', fontsize=20)
    #         ax.set_title(r'Reddening Curves Comparison (Independent of $R_V$)', fontsize=24)
    #         ax.set_xlim(0.5, 8.2)
    #         ax.set_ylim(-2, 12)

    #         # 顶部波长轴
    #         ax2 = ax.twiny()
    #         ticks = [10000, 5500, 4400, 3000, 2000, 1500]
    #         ax2.set_xlim(ax.get_xlim())
    #         ax2.set_xticks([10000.0/t for t in ticks])
    #         ax2.set_xticklabels([str(t) for t in ticks], fontsize=14)
    #         ax2.set_xlabel(r'Wavelength ($\AA$)', fontsize=18)

    #         ax.grid(True, alpha=0.2)

    #     # 3. 绘制你的数据 (使用自定义的 label 和 color)
    #     ax.plot(x_grid, y_derived, color=curve_color, lw=4, label=curve_label)

    #     # 4. 图例去重处理 (防止多次调用导致图例重复)
    #     handles, labels = ax.get_legend_handles_labels()
    #     # 使用字典保持插入顺序并去重
    #     by_label = dict(zip(labels, handles))
    #     ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=16)

    #     plt.tight_layout()
    #     if if_show:
    #         plt.show()
    #     return fig, ax

    def plot_reddening_curve_comparison(self,
                                        if_show=True,
                                        if_input=False,
                                        fig=None,
                                        ax=None,
                                        curve_label='Derived Curve (Data)',
                                        curve_color='blue',
                                        draw_background=True,
                                        plot_mcmc_samples=False,  # <--- 新接口
                                        n_samples=100):           # <--- 新接口
        """
        绘制去 Rv 化的红化对比图，支持叠加 MCMC 样本线。
        """
        if self.scaling_factor is None: self.derive_k_curve_parameters()

        if if_input:
            if fig is None or ax is None:
                raise ValueError("If if_input is True, fig and ax must be provided.")
        else:
            fig, ax = plt.subplots(figsize=(12, 10))
            draw_background = True

        x_grid = np.linspace(0.5, 8.2, 1000) # 1/um

        # ==========================================
        # 1. 准备 MCMC 样本线 (如果在 MCMC 之后)
        # ==========================================
        sample_lines_y = []
        if plot_mcmc_samples and hasattr(self, 'mcmc_samples'):
            # 定义 B 和 V 的频率
            x_B, x_V = 1.0/0.44, 1.0/0.55

            # 随机抽取样本
            n_total = len(self.mcmc_samples)
            inds = np.random.randint(n_total, size=n_samples)

            for ind in inds:
                coeffs = self.mcmc_samples[ind]
                # 提取形状部分 (假设只用高阶项决定形状，如果拟合包含c0但假设c0不影响形状)
                # 注意：这里的逻辑必须与 _poly_shape_cubic_only 保持一致
                # theta = [c3, c2, c1, c0]
                c3, c2, c1 = coeffs[0], coeffs[1], coeffs[2]

                # 局部函数计算该样本的形状值
                def shape_func(x):
                    return c3 * x**3 + c2 * x**2 + c1 * x

                val_grid = shape_func(x_grid)
                val_B = shape_func(x_B)
                val_V = shape_func(x_V)

                # 计算归一化曲线
                y_sample = (val_grid - val_V) / (val_B - val_V)
                sample_lines_y.append(y_sample)

        # ==========================================
        # 2. 计算最佳拟合曲线 (Best Fit)
        # ==========================================
        shape_grid = self._poly_shape_cubic_only(x_grid)
        shape_V = self._poly_shape_cubic_only(1.0/0.55)
        shape_B = self._poly_shape_cubic_only(1.0/0.44)
        y_derived = (shape_grid - shape_V) / (shape_B - shape_V)

        # ==========================================
        # 3. 绘制背景参考线
        # ==========================================
        if draw_background:
            y_calz = self._get_normalized_reddening(x_grid, 'calzetti')
            y_mw = self._get_normalized_reddening(x_grid, 'mw')
            y_smc = self._get_normalized_reddening(x_grid, 'smc')

            ax.plot(x_grid, y_mw, color='green', ls=':', lw=2, label='Milky Way (CCM89)')
            ax.plot(x_grid, y_smc, color='purple', ls='-.', lw=2, label='SMC (Pei92)')
            ax.plot(x_grid, y_calz, color='black', ls='--', lw=3, label='Calzetti (2000)')

            # 辅助线
            x_V, x_B = 1.0/0.55, 1.0/0.44
            ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
            ax.axvline(x_V, color='gray', linestyle='-', alpha=0.3)
            ax.scatter([x_V], [0], color='black', s=80, zorder=10)
            ax.text(x_V, 0.5, 'V (5500$\AA$)\nNormalized to 0', rotation=90, color='gray', fontsize=14, ha='right')

            ax.axhline(1, color='gray', linestyle=':', alpha=0.3)
            ax.axvline(x_B, color='gray', linestyle=':', alpha=0.3)
            ax.scatter([x_B], [1], color='black', s=80, zorder=10)
            ax.text(x_B, 1.5, 'B (4400$\AA$)\nNormalized to 1', rotation=90, color='gray', fontsize=14, ha='right')

            ax.set_xlabel(r'Inverse Wavelength $\lambda^{-1}$ ($\mu m^{-1}$)', fontsize=20)
            ax.set_ylabel(r'$E(\lambda - V) / E(B - V)$', fontsize=20)
            ax.set_title(r'Reddening Curves Comparison (Independent of $R_V$)', fontsize=24)
            ax.set_xlim(0.5, 8.2)
            ax.set_ylim(-2, 12)

            # 顶部波长轴
            ax2 = ax.twiny()
            ticks = [10000, 5500, 4400, 3000, 2000, 1500]
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks([10000.0/t for t in ticks])
            ax2.set_xticklabels([str(t) for t in ticks], fontsize=14)
            ax2.set_xlabel(r'Wavelength ($\AA$)', fontsize=18)
            ax.grid(True, alpha=0.2)

        # ==========================================
        # 4. 绘制 MCMC 样本线 (在最佳拟合线下面绘制)
        # ==========================================
        if sample_lines_y:
            for y_samp in sample_lines_y:
                # 使用与主曲线相同的颜色，但透明度极低
                ax.plot(x_grid, y_samp, color=curve_color, alpha=0.05, lw=1)

        # ==========================================
        # 5. 绘制最佳拟合线
        # ==========================================
        ax.plot(x_grid, y_derived, color=curve_color, lw=4, label=curve_label)

        # 图例去重
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=16)

        plt.tight_layout()
        if if_show:
            plt.show()
        return fig, ax

    # =========================================================================
    # 5. 静态辅助方法 (物理定律)
    # =========================================================================

    @staticmethod
    def get_calzetti_law(wavelength_angstrom, Rv=4.05):
        """返回 Calzetti k(lambda) 其中的形状部分 (normalized so roughly Q) 或完整公式"""
        # 注意：这里返回的是用于 Q_lambda 对比的形状 (-2.156... 那部分)
        # 如果需要完整 k(lambda)，需要乘以 2.659 并加 Rv
        microns = wavelength_angstrom / 10000.0
        k_lambda = np.zeros_like(microns)

        mask1 = (microns >= 0.12) & (microns < 0.63)
        w1 = microns[mask1]
        k_lambda[mask1] = (-2.156 + 1.509/w1 - 0.198/(w1**2) + 0.011/(w1**3))

        mask2 = (microns >= 0.63) & (microns <= 2.20)
        w2 = microns[mask2]
        k_lambda[mask2] = (-1.857 + 1.040/w2)

        return k_lambda
    @staticmethod
    def _get_normalized_reddening(x_grid, type_name):
        """
        Calculate the reddening curve normalized as E(lambda-V) / E(B-V).
        Formula: (A_lambda - A_V) / (A_B - A_V)

        This shape is independent of Rv for the target (though internal models use standard Rvs).
        """
        x_grid = np.array(x_grid)

        # --- Internal Helper: MW (Cardelli, Clayton, & Mathis 1989) ---
        def get_mw_ccm89_Av(x, Rv=3.1):
            x = np.array(x)
            a = np.zeros_like(x)
            b = np.zeros_like(x)

            # 1. Infrared (0.3 <= x < 1.1)
            mask_ir = (x >= 0.3) & (x < 1.1)
            if np.any(mask_ir):
                a[mask_ir] = 0.574 * x[mask_ir]**1.61
                b[mask_ir] = -0.527 * x[mask_ir]**1.61

            # 2. Optical/NIR (1.1 <= x < 3.3)
            mask_opt = (x >= 1.1) & (x < 3.3)
            if np.any(mask_opt):
                y = x[mask_opt] - 1.82
                a[mask_opt] = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
                b[mask_opt] = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7

            # 3. UV (3.3 <= x <= 8)
            mask_uv = (x >= 3.3)
            if np.any(mask_uv):
                X = x[mask_uv]
                F_a = -0.04473 * (X - 5.9)**2 - 0.009779 * (X - 5.9)**3
                F_b = 0.2130 * (X - 5.9)**2 + 0.1207 * (X - 5.9)**3
                # Only apply F_a, F_b for x >= 5.9
                low_uv_mask = X < 5.9
                F_a[low_uv_mask] = 0
                F_b[low_uv_mask] = 0

                a[mask_uv] = 1.752 - 0.316*X - 0.104/((X - 4.67)**2 + 0.341) + F_a
                b[mask_uv] = -3.090 + 1.825*X + 1.206/((X - 4.67)**2 + 0.263) + F_b

            return a + b / Rv

        # --- Internal Helper: SMC (Pei 1992) ---
        def get_smc_pei92_Av(x):
            a_params = [185, 27, 0.005, 0.010, 0.012, 0.030]
            w_params = [0.042, 0.08, 0.22, 9.7, 18, 25] # lambda_i in microns
            b_params = [90, 5.5, -1.95, -1.95, -1.80, 0.0]
            n_params = [2.0, 4.0, 2.0, 2.0, 2.0, 2.0]

            lam = 1.0 / x # microns
            ext_sum = np.zeros_like(x)

            for i in range(6):
                term = a_params[i] / ((lam/w_params[i])**n_params[i] + (w_params[i]/lam)**n_params[i] + b_params[i])
                ext_sum += term

            # Calculate normalization factor at V-band (0.55 um => x=1.818)
            lam_V = 0.55
            val_V = 0
            for i in range(6):
                val_V += a_params[i] / ((lam_V/w_params[i])**n_params[i] + (w_params[i]/lam_V)**n_params[i] + b_params[i])

            return ext_sum / val_V

        # --- Internal Helper: Calzetti 2000 ---
        def get_calzetti_curve_Av(x, Rv=4.05):
            lam_um = 1.0 / x
            k = np.zeros_like(x)

            # Range 1: 0.12um to 0.63um
            mask1 = (lam_um >= 0.12) & (lam_um < 0.63)
            if np.any(mask1):
                l = lam_um[mask1]
                k[mask1] = 2.659 * (-2.156 + 1.509/l - 0.198/l**2 + 0.011/l**3) + Rv

            # Range 2: 0.63um to 2.2um
            mask2 = (lam_um >= 0.63) & (lam_um <= 2.2)
            if np.any(mask2):
                l = lam_um[mask2]
                k[mask2] = 2.659 * (-1.857 + 1.040/l) + Rv

            # Returns A_lambda / A_V
            return k / Rv

        # --- Selector Logic ---
        if type_name == 'mw':
            y_vals = get_mw_ccm89_Av(x_grid, Rv=3.1)
        elif type_name == 'smc':
            y_vals = get_smc_pei92_Av(x_grid)
        elif type_name == 'calzetti':
            y_vals = get_calzetti_curve_Av(x_grid, Rv=4.05)
        else:
            return np.zeros_like(x_grid)

        # --- Normalization ---
        # Calculate B and V band values for the chosen model
        x_B = 1.0 / 0.44  # B-band inverse micron
        x_V = 1.0 / 0.55  # V-band inverse micron

        # Helper to compute single points without array shape issues
        def get_pt(x_pt):
            if type_name == 'mw': return get_mw_ccm89_Av(np.array([x_pt]), Rv=3.1)[0]
            if type_name == 'smc': return get_smc_pei92_Av(np.array([x_pt]))[0]
            if type_name == 'calzetti': return get_calzetti_curve_Av(np.array([x_pt]), Rv=4.05)[0]
            return 0.0

        y_B = get_pt(x_B)
        y_V = get_pt(x_V)

        # Formula: (A_lambda - A_V) / (A_B - A_V)
        # y_vals corresponds to A_lambda / A_V
        # Therefore: (y_vals * Av - Av) / (y_B * Av - Av) = (y_vals - 1) / (y_B - 1)
        # Note: If y_V is exactly 1.0, this simplifies to (y_vals - 1)/(y_B - 1).
        # We use y_V variable to be numerically safe.

        return (y_vals - y_V) / (y_B - y_V)

    def plot_extinction_curves_comparison(self):
            """
            绘制图表 3：Extinction Curves Comparison (A_lambda / A_V vs 1/lambda)

            对比以下曲线：
            1. Milky Way (Cardelli et al. 1989), Rv=3.1
            2. SMC (Pei 1992)
            3. Calzetti (2000), Rv=4.05
            4. Derived Curve (假设 k(0)=0)
            5. Derived Curve (假设 k(0)=intercept)
            """

            # 确保前置计算已完成
            if self.poly_coeffs is None:
                self.fit_polynomial()
            if self.scaling_factor is None:
                self.derive_k_curve_parameters()

            # 定义 X 轴网格: 1/lambda [um^-1]
            x_grid = np.linspace(0.5, 8.2, 1000)

            # =========================================================
            # 内部核心函数：计算参考模型的 A_lambda / A_V
            # =========================================================
            def get_ref_Av(x, model):
                x = np.array(x)

                # --- 1. Calzetti (2000) Starburst ---
                if model == 'calzetti':
                    Rv = 4.05
                    lam = 1.0 / x  # microns
                    k = np.zeros_like(x)

                    # Range 1: 0.12 to 0.63 microns
                    mask1 = (lam >= 0.12) & (lam < 0.63)
                    if np.any(mask1):
                        l = lam[mask1]
                        # k(lambda) formula
                        k[mask1] = 2.659 * (-2.156 + 1.509/l - 0.198/l**2 + 0.011/l**3) + Rv

                    # Range 2: 0.63 to 2.20 microns
                    mask2 = (lam >= 0.63) & (lam <= 2.2)
                    if np.any(mask2):
                        l = lam[mask2]
                        k[mask2] = 2.659 * (-1.857 + 1.040/l) + Rv

                    # Return A_lambda / A_V = k(lambda) / Rv
                    return k / Rv

                # --- 2. Milky Way (CCM 1989) ---
                elif model == 'mw':
                    Rv = 3.1
                    a = np.zeros_like(x)
                    b = np.zeros_like(x)

                    # Infrared (0.3 <= x < 1.1)
                    mask_ir = (x >= 0.3) & (x < 1.1)
                    if np.any(mask_ir):
                        xi = x[mask_ir]
                        a[mask_ir] = 0.574 * xi**1.61
                        b[mask_ir] = -0.527 * xi**1.61

                    # Optical/NIR (1.1 <= x < 3.3)
                    mask_opt = (x >= 1.1) & (x < 3.3)
                    if np.any(mask_opt):
                        y = x[mask_opt] - 1.82
                        a[mask_opt] = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
                        b[mask_opt] = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7

                    # UV (3.3 <= x <= 8)
                    mask_uv = (x >= 3.3)
                    if np.any(mask_uv):
                        X = x[mask_uv]
                        Fa = -0.04473 * (X - 5.9)**2 - 0.009779 * (X - 5.9)**3
                        Fb = 0.2130 * (X - 5.9)**2 + 0.1207 * (X - 5.9)**3

                        # Fa, Fb only active above 5.9 um^-1
                        low_uv = X < 5.9
                        Fa[low_uv] = 0
                        Fb[low_uv] = 0

                        a[mask_uv] = 1.752 - 0.316*X - 0.104/((X - 4.67)**2 + 0.341) + Fa
                        b[mask_uv] = -3.090 + 1.825*X + 1.206/((X - 4.67)**2 + 0.263) + Fb

                    # A_lambda / A_V = a(x) + b(x) / Rv
                    return a + b / Rv

                # --- 3. SMC (Pei 1992) ---
                elif model == 'smc':
                    # Coefficients from Pei (1992) Table 2 (SMC column)
                    a_params = [185, 27, 0.005, 0.010, 0.012, 0.030]
                    w_params = [0.042, 0.08, 0.22, 9.7, 18, 25] # microns
                    b_params = [90, 5.5, -1.95, -1.95, -1.80, 0.0]
                    n_params = [2.0, 4.0, 2.0, 2.0, 2.0, 2.0]

                    lam = 1.0 / x # microns
                    ext_sum = np.zeros_like(x)

                    # Sum the 6 terms
                    for i in range(6):
                        term = a_params[i] / ((lam/w_params[i])**n_params[i] + (w_params[i]/lam)**n_params[i] + b_params[i])
                        ext_sum += term

                    # Normalize at V-band (0.55 um -> x=1.818)
                    lam_V = 0.55
                    val_V = 0
                    for i in range(6):
                        val_V += a_params[i] / ((lam_V/w_params[i])**n_params[i] + (w_params[i]/lam_V)**n_params[i] + b_params[i])

                    # Return Normalized A_lambda / A_V
                    return ext_sum / val_V

                return np.zeros_like(x)

            # =========================================================
            # 2. 计算数据
            # =========================================================

            # A. 参考曲线
            y_calzetti = get_ref_Av(x_grid, 'calzetti')
            y_mw = get_ref_Av(x_grid, 'mw')
            y_smc = get_ref_Av(x_grid, 'smc')

            # B. 你的 Derived Curves (根据之前计算的多项式和归一化因子)
            # k(x) = S * P_cubic(x) + intercept
            # A/Av = k(x) / Rv

            shape_grid = self._poly_shape_cubic_only(x_grid)
            shape_V = self._poly_shape_cubic_only(1.0/0.55) # V-band value of shape

            # --- Case 1: 假设 k(0) = 0 ---
            # 此时 intercept = 0
            # S_zero 由 k(B)-k(V)=1 决定，这已经是 self.scaling_factor 了
            # Rv_zero = k(V) = S * shape(V)
            k_zero_curve = self.scaling_factor * shape_grid
            Rv_zero = self.scaling_factor * shape_V
            y_derived_zero = k_zero_curve / Rv_zero

            # --- Case 2: 假设 k(0) = intercept (Shifted) ---
            # k(x) = S * shape(x) + intercept
            # Rv_shifted = k(V) = S * shape(V) + intercept
            k_shifted_curve = self.scaling_factor * shape_grid + self.intercept_val
            Rv_shifted = self.scaling_factor * shape_V + self.intercept_val # 应该等于 self.derived_Rv
            y_derived_shifted = k_shifted_curve / Rv_shifted

            # =========================================================
            # 3. 绘图
            # =========================================================
            fig, ax = plt.subplots(figsize=(14, 10))

            # 绘制参考模型
            ax.plot(x_grid, y_mw, color='green', ls=':', lw=2, label='Milky Way (CCM89, $R_V=3.1$)')
            ax.plot(x_grid, y_smc, color='purple', ls='-.', lw=2, label='SMC (Pei92)')
            ax.plot(x_grid, y_calzetti, color='black', ls='--', lw=3, label='Calzetti (2000, $R_V=4.05$)')

            # 绘制推导出的曲线
            ax.plot(x_grid, y_derived_zero, color='dodgerblue', lw=3, alpha=0.8,
                    label=f'Derived (Assume $k(0)=0$, $R_V={Rv_zero:.2f}$)')

            ax.plot(x_grid, y_derived_shifted, color='firebrick', lw=3,
                    label=f'Derived (Assume $k(0)={self.intercept_val:.2f}$, $R_V={Rv_shifted:.2f}$)')

            # 辅助标记 V 和 B 波段
            x_V = 1.0/0.55
            x_B = 1.0/0.44

            ax.axvline(x_V, color='gray', ls='-', alpha=0.3)
            ax.scatter([x_V]*2, [1]*2, color='gray', s=50, zorder=10) # A/Av 在 V 波段恒为 1
            ax.text(x_V, 0.2, 'V (5500$\AA$)', rotation=90, color='gray', fontsize=16, ha='right')

            ax.axvline(x_B, color='gray', ls='-', alpha=0.3)
            ax.text(x_B, 0.2, 'B (4400$\AA$)', rotation=90, color='gray', fontsize=16, ha='right')

            # 坐标轴设置
            ax.set_xlabel(r'Inverse Wavelength $\lambda^{-1}$ ($\mu m^{-1}$)', fontsize=20)
            ax.set_ylabel(r'$A_\lambda / A_V$', fontsize=20)
            ax.set_title('Extinction Curves Comparison', fontsize=24)
            ax.set_xlim(0.5, 8.2)
            ax.set_ylim(0, 9) # 根据 UV 上升幅度调整

            # 顶部波长刻度
            ax2 = ax.twiny()
            ticks = [10000, 5500, 4400, 3000, 2000, 1500, 1250]
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks([10000.0/t for t in ticks])
            ax2.set_xticklabels([str(t) for t in ticks], fontsize=14)
            ax2.set_xlabel(r'Wavelength ($\AA$)', fontsize=18)

            ax.legend(loc='upper left', fontsize=16)
            ax.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.show()

    def run_mcmc_fitting(self, nwalkers=32, nsteps=2000, poly_order=3):
        """
        使用 MCMC (emcee) 拟合，并强制 UV 端单调递增 (Derivative >= 0)。
        """
        import emcee

        if self.effective_curve is None or self.effective_curve_err is None:
            raise ValueError("Data or Error missing. Run compute_attenuation_curves first.")

        # 1. 准备数据 (x = 1/lambda)
        wave_um = self.effective_curve_wave / 10000.0
        x_data = 1.0 / wave_um
        y_data = self.effective_curve
        y_err = self.effective_curve_err

        # 剔除 NaN 和 Inf
        mask = np.isfinite(x_data) & np.isfinite(y_data) & np.isfinite(y_err) & (y_err > 0)
        x_fit, y_fit, err_fit = x_data[mask], y_data[mask], y_err[mask]

        # -----------------------------------------------------------------
        # 定义内部函数
        # -----------------------------------------------------------------

        # 模型: c3*x^3 + c2*x^2 + c1*x + c0
        def model(theta, x):
            return np.polyval(theta, x)

        def log_likelihood(theta, x, y, yerr):
            model_y = model(theta, x)
            sigma2 = yerr ** 2
            return -0.5 * np.sum((y - model_y) ** 2 / sigma2 + np.log(sigma2))

        def log_prior(theta):
            # 1. 基础参数范围限制 (防止跑飞)
            # theta 顺序是 [c3, c2, c1, c0] (最高次幂在前)
            c3, c2, c1, c0 = theta
            if not (-10 < c3 < 10 and -20 < c2 < 20 and -20 < c1 < 20 and -10 < c0 < 10):
                return -np.inf

            # 2. 【核心修改】强制 UV 端单调递增 (Derivative >= 0)
            # 我们检查 x 在 [2.5, 8.5] 范围内 (即 UV 到 远UV)
            # 多项式 P(x) = c3*x^3 + c2*x^2 + c1*x + c0
            # 导数 P'(x) = 3*c3*x^2 + 2*c2*x + c1

            x_check = np.linspace(2.5, 8.5, 20) # 检查点的范围
            deriv = 3 * c3 * (x_check**2) + 2 * c2 * x_check + c1

            # 如果在检查范围内任何一点导数小于 0 (曲线下降)，则拒绝该参数
            if np.any(deriv < 0):
                return -np.inf

            return 0.0

        def log_probability(theta, x, y, yerr):
            lp = log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(theta, x, y, yerr)

        # -----------------------------------------------------------------
        # 运行 MCMC
        # -----------------------------------------------------------------

        # 初始猜测
        initial_coeffs = np.polyfit(x_fit, y_fit, poly_order)
        ndim = len(initial_coeffs)
        pos = initial_coeffs + 1e-4 * np.random.randn(nwalkers, ndim)

        print(f"Running MCMC with {nwalkers} walkers for {nsteps} steps (Monotonicity Constrained)...")
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x_fit, y_fit, err_fit))
        sampler.run_mcmc(pos, nsteps, progress=True)

        discard = int(nsteps * 0.25)
        self.mcmc_samples = sampler.get_chain(discard=discard, thin=15, flat=True)

        # 更新最佳系数
        self.poly_coeffs = np.median(self.mcmc_samples, axis=0)
        print("MCMC Best Fit Coeffs:", self.poly_coeffs)

        return self.mcmc_samples

    def plot_mcmc_results(self, plot_samples=True, n_samples=100):
        """
        绘制 MCMC 拟合结果：包含数据点误差棒、最佳拟合线、误差带以及(可选)随机样本猜测线。

        Parameters
        ----------
        plot_samples : bool
            是否画出从 MCMC 链中随机抽取的样本线 (Spaghetti plot)。
        n_samples : int
            画多少条样本线。
        """
        if not hasattr(self, 'mcmc_samples'):
            print("Please run run_mcmc_fitting first.")
            return

        # 定义绘图网格
        x_grid = 1.0 / (np.linspace(1500, 8000, 500) / 10000.0) # 1/um
        wave_grid = 10000.0 / x_grid

        # 1. 抽取随机样本
        # 如果样本数足够，随机抽取；否则全部使用
        n_total = len(self.mcmc_samples)
        if n_samples > n_total: n_samples = n_total
        inds = np.random.randint(n_total, size=n_samples)

        sample_curves = []
        for ind in inds:
            sample = self.mcmc_samples[ind]
            # 计算该样本对应的曲线值
            sample_curves.append(np.polyval(sample, x_grid))

        sample_curves = np.array(sample_curves)

        # 计算统计量用于误差带
        lower_bound = np.percentile(sample_curves, 2.5, axis=0)
        upper_bound = np.percentile(sample_curves, 97.5, axis=0)

        # 最佳拟合 (使用所有样本的中位数系数，或者当前存储的 poly_coeffs)
        best_fit_curve = self._poly_shape(x_grid)

        # 2. 绘图
        fig, ax = plt.subplots(figsize=(12, 8))

        # A. 绘制随机样本猜测线 (The "Guesses")
        if plot_samples:
            # 同样是画 x_grid 对应的曲线
            # 颜色使用 alpha=0.1 变淡
            for i in range(len(sample_curves)):
                ax.plot(wave_grid, sample_curves[i], color='dodgerblue', alpha=0.05, lw=1)

        # B. 原始数据带误差棒
        wave_data = self.effective_curve_wave
        y_data = self.effective_curve
        y_err = self.effective_curve_err
        step = 10
        ax.errorbar(wave_data[::step], y_data[::step], yerr=y_err[::step],
                    fmt='o', color='black', alpha=0.6, label='Data (with SEM)', markersize=4, capsize=0)

        # C. 拟合带 (95% CI)
        # 如果画了 sample lines，fill_between 可以选择不画或者颜色淡一点
        ax.fill_between(wave_grid, lower_bound, upper_bound, color='gray', alpha=0.2, label='95% Credible Interval')

        # D. 最佳拟合线
        ax.plot(wave_grid, best_fit_curve, color='blue', lw=2, label='MCMC Best Fit')

        ax.set_xlabel('Wavelength (Å)')
        ax.set_ylabel(r'$Q_\lambda$')
        ax.set_title('MCMC Polynomial Fit with Posterior Samples')
        ax.set_xlim(1500, 8000)
        ax.set_ylim(-2, 8)

        # 处理 Legend，避免重复
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best')

        plt.show()
