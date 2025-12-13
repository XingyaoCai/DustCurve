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
        self.spectra_data=[]
        self.common_wavelength=None
        self.interpolated_fluxes=[]
        self.average_flux=None
        self.average_flux_err=None
        self.normalization_factors=[]
        self.normalized_fluxes=None

    def load_spectrum_catalog(self, catalog_entry_key_list,Catalog):
        """
        Load spectra from a list of catalog entries.

        Parameters:
        catalog_entry_key_list (list): List of catalog entries containing spectra.
        """
        spectra_data_len=len(self.spectra_data)
        successful_loads=0

        for i,entry_key in enumerate(catalog_entry_key_list):
            try:
                entry=Catalog.catalog[entry_key]
                if not entry.get('prism_filepath', None):
                    print(f"Entry {entry_key} does not have prism_filepath attribute. Skipping.")
                    continue

                prism_spectrum=Load_Spectrum_From_Fits(entry['prism_filepath'], entry['prism_redshift'])

                # prism_spectrum.set_boundarys(1250*u.AA, 8000*u.AA)


                if len(prism_spectrum.processing_wavelengths.data)==0 or len(prism_spectrum.processing_flux_lambda.data)==0:
                    print(f"Entry {entry_key} has empty wavelength or flux data. Skipping.")
                    continue

                if len(prism_spectrum.processing_wavelengths.data)!=len(prism_spectrum.processing_flux_lambda.data):
                    print(f"Entry {entry_key} has mismatched wavelength and flux lengths. Skipping.")
                    continue

                mask=~(np.isnan(prism_spectrum.processing_wavelengths.data) | np.isnan(prism_spectrum.processing_flux_lambda.data)|(prism_spectrum.processing_wavelengths.data==0))
                if np.sum(mask)<2:
                    print(f"Entry {entry_key} has insufficient valid data points. Skipping.")
                    continue

                self.spectra_data.append({
                    'index':spectra_data_len+i,
                    'spectrum1d':prism_spectrum,
                    'wavelengths':prism_spectrum.processing_wavelengths.data[mask],
                    'fluxes':prism_spectrum.processing_flux_lambda.data[mask]
                })
                successful_loads+=1

            except Exception as e:
                print(f"Failed to load spectrum for entry {entry_key}: {e}")
                continue

        return successful_loads > 0


    def create_common_wavelength_grid(self, wavelength_range=None, grid_size=None):
        """
        Create a common wavelength grid for averaging.

        Parameters:
        -------
        wavelength_range: tuple, optional
            Tuple specifying the (min, max) wavelength range, if None then automatically use the overlapping range of all spectra.
        grid_size: float, optional
            Desired wavelength grid size in Angstroms. If None, it will be determined based on the maximum number of data points among the spectra.
        """

        if not self.spectra_data:
            raise ValueError("No spectra data loaded. Please load spectra before creating a common wavelength grid.")

        all_min_wavelengths = [spectrum_data['wavelengths'].min() for spectrum_data in self.spectra_data]
        all_max_wavelengths = [spectrum_data['wavelengths'].max() for spectrum_data in self.spectra_data]

        if wavelength_range is None:
            common_min_wavelength = max(all_min_wavelengths)
            common_max_wavelength = min(all_max_wavelengths)
        else:
            common_min_wavelength, common_max_wavelength = wavelength_range

        if common_min_wavelength >= common_max_wavelength:
            raise ValueError("No overlapping wavelength range found among the spectra.")

        if grid_size is None:
            spectrum_data_points=[]
            for spec in self.spectra_data:
                mask=(spec['wavelengths']>=common_min_wavelength) & (spec['wavelengths']<=common_max_wavelength)
                valid_datapoints=np.sum(mask)
                if valid_datapoints>1:
                    spectrum_data_points.append(valid_datapoints)

            if spectrum_data_points:
                max_data_points = max(spectrum_data_points)
                num_points = max_data_points
            else:
                num_points=1000
        else:
            num_points = int((common_max_wavelength - common_min_wavelength) / grid_size) + 1

        self.common_wavelength = np.linspace(common_min_wavelength, common_max_wavelength, num_points)

        return self.common_wavelength

    def interpolate_spectra(self, interpolation_method='linear'):
        """
        Interpolate all loaded spectra onto the common wavelength grid.

        Parameters:
        ------
        interpolation_method: str, optional
            Interpolation method to use. Default is 'linear', can also be 'nearest', 'cubic' and 'quadratic'.
        """

        if self.common_wavelength is None:
            self.create_common_wavelength_grid()

        self.interpolated_fluxes = []
        successful_interpolations=0

        for spectrum_packed_data in self.spectra_data:
            try:
                wavelength_data = spectrum_packed_data['wavelengths']
                flux_data = spectrum_packed_data['fluxes']

                mask=(wavelength_data>=self.common_wavelength.min()) & (wavelength_data<=self.common_wavelength.max())

                if np.sum(mask)<2:
                    print(f"Spectrum index {spectrum_packed_data['index']} has insufficient data in the common wavelength range. Skipping.")
                    continue

                wavelength_data_valid = wavelength_data[mask]
                flux_data_valid = flux_data[mask]

                interpolation_function = scipy.interpolate.interp1d(wavelength_data_valid, flux_data_valid, kind=interpolation_method,bounds_error=False, fill_value=np.nan)

                interpolated_flux = interpolation_function(self.common_wavelength)

                if np.all(np.isnan(interpolated_flux)):
                    print(f"Spectrum index {spectrum_packed_data['index']} interpolation resulted in all NaN values. Skipping.")
                    continue

                self.interpolated_fluxes.append(interpolated_flux)
                successful_interpolations+=1

            except Exception as e:
                print(f"Failed to interpolate spectrum index {spectrum_packed_data['index']}: {e}")
                continue

        return successful_interpolations > 0


    def spectrum_normalization(self, reference_wavelength=5500.0, target_flux=1e-19):
        """
        Normalize the interpolated spectra at a reference wavelength.

        Parameters:
        ----------
        reference_wavelength: float
            Wavelength at which to normalize the spectra. Default is 5500.0 Angstroms in V-band.
        target_flux: float
            Target flux value for normalization. Default is 1e-19 erg/s/cm^2/Angstrom.
        """

        if not self.interpolated_fluxes:
            raise ValueError("No interpolated spectra available. Please interpolate spectra before normalization.")

        if reference_wavelength < self.common_wavelength.min() or reference_wavelength > self.common_wavelength.max():
            raise ValueError(f"Reference wavelength {reference_wavelength} is out of the common wavelength range [{self.common_wavelength.min()}, {self.common_wavelength.max()}].")

        ref_index=np.argmin(np.abs(self.common_wavelength - reference_wavelength))
        actual_ref_wavelength=self.common_wavelength[ref_index]

        self.normalization_factors = []
        self.normalized_fluxes = []
        successful_normalizations=0

        for i,interpolated_flux in enumerate(self.interpolated_fluxes):
            try:
                flux_at_ref = interpolated_flux[ref_index]

                if np.isnan(flux_at_ref) or flux_at_ref == 0:
                    print(f"Spectrum index {i} has invalid flux at reference wavelength {actual_ref_wavelength}. Skipping.")
                    continue

                normalization_factor = target_flux / flux_at_ref
                normalized_flux = interpolated_flux * normalization_factor

                self.normalization_factors.append(normalization_factor)
                self.normalized_fluxes.append(normalized_flux)
                successful_normalizations+=1

            except Exception as e:
                print(f"Failed to normalize spectrum index {i}: {e}")
                continue

        if self.normalized_fluxes:
            self.normalized_fluxes = np.array(self.normalized_fluxes).copy()

        return successful_normalizations > 0

    def spectrum_normalization_within_range(self, wavelength_range=(5000.0, 6000.0), target_flux=None):
        """
        Normalize the interpolated spectra within a specified wavelength range.

        Parameters:
        ----------
        wavelength_range: tuple
            Wavelength range (min, max) within which to normalize the spectra.
        target_flux: float, optional
            Target flux value for normalization. If None, use the median flux within the range.
        """

        if not self.interpolated_fluxes:
            raise ValueError("No interpolated spectra available. Please interpolate spectra before normalization.")

        min_wavelength, max_wavelength = wavelength_range

        if min_wavelength < self.common_wavelength.min():
            pass

    def compute_average_spectrum(self, method='mean',ignore_nan=True, use_percentile_filter=False, lower_percentile=16, upper_percentile=84):
        """
        Compute the average spectrum from the normalized spectra on the common wavelength grid.

        Parameters:
        ------
        method: str
            Method to compute the average spectrum. Options are 'mean' or 'median'. Default is 'mean'.
        ignore_nan: bool
            Whether to ignore NaN values in the computation. Default is True.
        use_percentile_filter: bool
            Whether to use percentile filtering to exclude outliers. Default is False.
        lower_percentile: float
            If use_percentile_filter is True, the lower percentile to exclude. Default is 16.
        upper_percentile: float
            If use_percentile_filter is True, the upper percentile to exclude. Default is 84.
        """

        if not hasattr(self, 'normalized_fluxes') or self.normalized_fluxes is None:
            raise ValueError("No normalized spectra available. Please normalize spectra before computing the average.")

        flux_matrix=np.array(self.normalized_fluxes)

        if use_percentile_filter:
            self.average_flux, self.average_flux_err = self._compute_percentile_filtered_average(
                flux_matrix, method, lower_percentile, upper_percentile, ignore_nan)

        else:
            if ignore_nan:
                if method=='mean':
                    self.average_flux = np.nanmean(flux_matrix, axis=0)
                    self.average_flux_err = scipy.stats.sem(flux_matrix, axis=0, nan_policy='omit')
                elif method=='median':
                    self.average_flux = np.nanmedian(flux_matrix, axis=0)
                    self.average_flux_err = scipy.stats.sem(flux_matrix, axis=0, nan_policy='omit')
                else:
                    raise ValueError(f"Unknown method '{method}'. Use 'mean' or 'median'.")

            else:
                if method=='mean':
                    self.average_flux = np.mean(flux_matrix, axis=0)
                    self.average_flux_err = scipy.stats.sem(flux_matrix, axis=0)
                elif method=='median':
                    self.average_flux = np.median(flux_matrix, axis=0)
                    self.average_flux_err = scipy.stats.sem(flux_matrix, axis=0)
                else:
                    raise ValueError(f"Unknown method '{method}'. Use 'mean' or 'median'.")

        valid_counts=np.sum(~np.isnan(flux_matrix), axis=0) if ignore_nan else flux_matrix.shape[0]

        return self.common_wavelength, self.average_flux, self.average_flux_err, valid_counts

    def _compute_percentile_filtered_average(self, flux_matrix, method, lower_percentile, upper_percentile, ignore_nan):
        """
        Helper function to compute average spectrum with percentile filtering.

        Parameters:
        ----------
        flux_matrix: np.ndarray
            2D array of shape (num_spectra, num_wavelengths) containing normalized fluxes.
        method: str
            Method to compute the average spectrum. Options are 'mean' or 'median'.
        lower_percentile: float
            Lower percentile to exclude.
        upper_percentile: float
            Upper percentile to exclude.
        ignore_nan: bool
            Whether to ignore NaN values in the computation.

        Returns:
        -------
        average_flux: np.ndarray
            1D array of average flux values.
        average_flux_err: np.ndarray
            1D array of standard error of the mean for the average flux.

        """
        num_wavelengths = flux_matrix.shape[1]
        average_flux = np.empty(num_wavelengths)
        average_flux_err = np.empty(num_wavelengths)

        total_used_points=0
        total_original_points=0

        for index in range(num_wavelengths):
            wavelength_fluxes = flux_matrix[:, index]

            if ignore_nan:
                valid_fluxes = wavelength_fluxes[~np.isnan(wavelength_fluxes)]
            else:
                valid_fluxes = wavelength_fluxes

            total_original_points+=len(valid_fluxes)

            if len(valid_fluxes) == 0:
                average_flux[index] = np.nan
                average_flux_err[index] = np.nan
                continue

            try:
                lower_bound = np.percentile(valid_fluxes, lower_percentile)
                upper_bound = np.percentile(valid_fluxes, upper_percentile)

                filtered_fluxes = valid_fluxes[(valid_fluxes >= lower_bound) & (valid_fluxes <= upper_bound)]

                if len(filtered_fluxes) == 0:
                    filtered_fluxes = valid_fluxes

                total_used_points+=len(filtered_fluxes)

                if method == 'mean':
                    average_flux[index] = np.mean(filtered_fluxes)
                    average_flux_err[index] = scipy.stats.sem(filtered_fluxes)
                elif method == 'median':
                    average_flux[index] = np.median(filtered_fluxes)
                    average_flux_err[index] = scipy.stats.sem(filtered_fluxes)
                else:
                    raise ValueError(f"Unknown method '{method}'. Use 'mean' or 'median'.")

            except Exception as e:
                continue


        return average_flux, average_flux_err

    def plot_average_spectrum(self, show_individual=True, show_average=True, show_error=True,figsize=(25,14), alpha=0.3, show_normalization_point=True, reference_wavelength=5500.0, ylim=None,if_show=True):
        """
        Plot the average spectrum along with individual normalized spectra.

        Parameters:
        ----------
        show_individual: bool
            Whether to plot individual normalized spectra. Default is True.
        show_average: bool
            Whether to plot the average spectrum. Default is True.
        show_error: bool
            Whether to plot error bars for the average spectrum. Default is True.
        figsize: tuple
            Figure size for the plot. Default is (25, 14).
        alpha: float
            Transparency level for individual spectra. Default is 0.3.
        show_normalization_point: bool
            Whether to highlight the normalization point. Default is True.
        reference_wavelength: float
            Wavelength used for normalization. Default is 5500.0 Angstroms.
        ylim: tuple or None
            Y-axis limits for the plot. If None, it will be set automatically.
        """

        fig, ax= plt.subplots(figsize=figsize)
        if show_individual and self.normalized_fluxes is not None:
            for normalized_flux in self.normalized_fluxes:
                ax.plot(self.common_wavelength, normalized_flux, color='gray', alpha=alpha, label='Individual Normalized Spectra' if 'Individual Normalized Spectra' not in ax.get_legend_handles_labels()[1] else "")
        if show_average and self.average_flux is not None:
            ax.plot(self.common_wavelength, self.average_flux, color='k',linewidth=2, label='Average Spectrum')
            if show_error and self.average_flux_err is not None:
                ax.fill_between(self.common_wavelength, self.average_flux - self.average_flux_err, self.average_flux + self.average_flux_err, color='k', alpha=0.2, label='±1σ Error')


        if show_normalization_point and self.normalized_fluxes is not None:
            ref_index=np.argmin(np.abs(self.common_wavelength - reference_wavelength))
            actual_ref_wavelength=self.common_wavelength[ref_index]
            avg_flux_at_ref=self.average_flux[ref_index] if self.average_flux is not None else None
            if avg_flux_at_ref is not None:
                ax.scatter([actual_ref_wavelength], [avg_flux_at_ref], color='red', s=100, zorder=5, label='Normalization Point')

        ax.set_xlabel('Wavelength (Angstrom)', fontsize=24)
        ax.set_ylabel('Normalized Flux (erg/s/cm²/Angstrom)', fontsize=24)
        ax.set_title('Average Spectrum with Individual Normalized Spectra', fontsize=28)
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(0,np.nanmax(self.average_flux)*1.2 if self.average_flux is not None else None)



        ax.spines['bottom'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.xaxis.set_tick_params(width=2, direction='in', which='both', labelsize=18)
        ax.yaxis.set_tick_params(width=2, direction='in', which='both', labelsize=18)
        ax.xaxis.set_tick_params(length=6, which='major')
        ax.xaxis.set_tick_params(length=3, which='minor')
        ax.yaxis.set_tick_params(length=6, which='major')
        ax.yaxis.set_tick_params(length=3, which='minor')
        ax.legend(fontsize=20, loc='best')
        if if_show:
            plt.show()
        return fig, ax

    def get_normalization_factors(self):
        """
        Get the normalization factors used for each spectrum.

        Returns:
        -------
        normalization_dict: dict
            Dictionary mapping spectrum indices to their normalization factors.
        """
        return {'normalization_factors': self.normalization_factors.copy() if self.normalization_factors else None,
                'num_normalized_spectra': len(self.normalization_factors) if self.normalization_factors else 0,
                'factor_mean': np.mean(self.normalization_factors) if self.normalization_factors else None,
                'factor_std': np.std(self.normalization_factors) if self.normalization_factors else None
                }

    def get_average_spectrum_data(self):
        """
        Get the computed average spectrum data.

        Returns:
        -------
        average_spectrum_dict: dict
            Dictionary containing common wavelengths, average flux, and average flux error.
        """
        return {'common_wavelength': self.common_wavelength.copy() if self.common_wavelength is not None else None,
                'average_flux': self.average_flux.copy() if self.average_flux is not None else None,
                'average_flux_err': self.average_flux_err.copy() if self.average_flux_err is not None else None,
                'normalization_info': self.get_normalization_factors()
                }

    def save_average_spectrum_to_file(self, filepath):
        """
        Save the average spectrum data to a file.

        Parameters:
        ----------
        filepath: str
            Path to the file where the average spectrum data will be saved.
        """

        if self.average_flux is None or self.common_wavelength is None:
            raise ValueError("Average spectrum data is not available. Please compute the average spectrum before saving.")

        data_to_save = {
            'common_wavelength': self.common_wavelength,
            'average_flux': self.average_flux,
            'average_flux_err': self.average_flux_err
        }
        np.savez(filepath, **data_to_save)

    def load_average_spectrum_from_file(self, filepath):
        """
        Load the average spectrum data from a file saved via save_average_spectrum_to_file.

        Args:
            filepath (str): Path to the file from which the average spectrum data will be loaded.
        """

        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"The file {filepath} does not exist.")

        loaded_data = np.load(filepath)

        self.common_wavelength = loaded_data['common_wavelength']
        self.average_flux = loaded_data['average_flux']
        self.average_flux_err = loaded_data['average_flux_err']

        return True
