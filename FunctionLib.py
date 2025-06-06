import wave
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
    redshift : astropy.units.Quantity
        The redshift value for the specified file.

    Errors
    -------
    Returns the error message if the file is not found in the dataframe.
    """
    try:
        index = int(np.where(Pandas_Dataframe.file == File_Name)[0][0])
        redshift = Pandas_Dataframe.iloc[index]['z']
        return np.float32(redshift)* units.dimensionless_unscaled
    except Exception as e:
        return e

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
    with io.fits.open(Fits_FilePath) as hdul:

      try:
        Spectra_Data = hdul[1].data

        Wavelength = Spectra_Data['wave']*units.micron
        Flux = Spectra_Data['flux']*units.uJy
        Error= Spectra_Data['err']*units.uJy

        Flux_Lambda = Flux.to(units.erg / (units.cm**2 * units.s * units.AA), equivalencies=units.spectral_density(Wavelength))
        Error_Lambda = Error.to(units.erg / (units.cm**2 * units.s * units.AA), equivalencies=units.spectral_density(Wavelength))
        Error_Lambda = astropy.nddata.StdDevUncertainty(Error_Lambda)
        Error=astropy.nddata.StdDevUncertainty(Error)


        return astropy.nddata.NDDataArray(Wavelength),astropy.nddata.NDDataArray(Flux_Lambda, uncertainty=Error_Lambda),astropy.nddata.NDDataArray(Flux, uncertainty=Error)

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

import astropy.units as u
from astropy.nddata import NDDataArray
import numpy as np

import astropy.units as u
from astropy.nddata import NDDataArray
from astropy.constants import c
import numpy as np

class Spectrum_1d:
    """
    A class to store and manipulate a 1D spectrum. The initialization requires the observed
    wavelengths, fluxes in F_nu or F_lambda units, and the redshift value. The initialization
    will automatically calculate the rest frame wavelengths based on the observed wavelengths
    and redshift and store them as astropy.nddata.NDDataArray objects.

    Attributes
    ----------
    observed_wavelengths : astropy.nddata.NDDataArray
        The observed wavelengths of the spectrum, should contain the unit.
    observed_flux_nu : astropy.nddata.NDDataArray
        The observed flux values of the spectrum in F_nu units, should contain the unit and uncertainty.
    observed_flux_lambda : astropy.nddata.NDDataArray
        The observed flux values of the spectrum in F_lambda units, should contain the unit and uncertainty.
    redshift : astropy.units.Quantity
        The redshift value of the spectrum.
    restframe_wavelengths : astropy.nddata.NDDataArray
        The rest frame wavelengths of the spectrum, calculated from the observed wavelengths and redshift.
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
            raise ValueError("At least one of observed_flux_nu or observed_flux_lambda must be provided.")

        # Handle wavelengths - allow both NDDataArray and Quantity
        if isinstance(observed_wavelengths, astropy.nddata.NDDataArray):
            self.observed_wavelengths = observed_wavelengths
        elif isinstance(observed_wavelengths, u.Quantity):
            # If input is a Quantity, NDDataArray stores it in its .data attribute
            self.observed_wavelengths = astropy.nddata.NDDataArray(observed_wavelengths)
        else:
            raise TypeError("observed_wavelengths must be an astropy.nddata.NDDataArray or astropy.units.Quantity object.")

        # Handle redshift
        if isinstance(redshift, (float, int)):
            self.redshift = redshift * units.dimensionless_unscaled
        elif isinstance(redshift, units.Quantity):
            if redshift.unit.is_equivalent(units.dimensionless_unscaled):
                self.redshift = redshift
            else:
                raise ValueError("Redshift must be dimensionless.")
        else:
            raise TypeError("Redshift must be a float, int, or dimensionless astropy.units.Quantity.")

        # Calculate rest-frame wavelengths
        # Get the observed wavelength data, which might be a Quantity or ndarray
        obs_wave_data_attr = self.observed_wavelengths.data
        if isinstance(obs_wave_data_attr, units.Quantity):
            obs_wave_values = obs_wave_data_attr.value
        else:
            obs_wave_values = obs_wave_data_attr # Assuming it's a numpy array

        rest_wave_data = obs_wave_values / (1 + self.redshift.value)

        rest_wave_uncertainty_values = None
        if self.observed_wavelengths.uncertainty is not None:
            # Assuming uncertainty.array gives the numerical values of uncertainty
            # and it needs to be scaled like the data.
            rest_wave_uncertainty_values = self.observed_wavelengths.uncertainty.array / (1 + self.redshift.value)
            # Reconstruct the uncertainty object with the new values
            rest_wave_uncertainty = type(self.observed_wavelengths.uncertainty)(rest_wave_uncertainty_values)
        else:
            rest_wave_uncertainty = None

        self.restframe_wavelengths = astropy.nddata.NDDataArray(
            data=rest_wave_data, # This data is a numpy array
            uncertainty=rest_wave_uncertainty,
            unit=self.observed_wavelengths.unit # Unit is preserved
        )

        # Handle F_nu flux
        if observed_flux_nu is not None:
            if isinstance(observed_flux_nu, astropy.nddata.NDDataArray):
                self.observed_flux_nu = observed_flux_nu
            elif isinstance(observed_flux_nu, astropy.units.Quantity):
                self.observed_flux_nu = astropy.nddata.NDDataArray(observed_flux_nu) # .data will be the Quantity
            else:
                raise TypeError("observed_flux_nu must be an astropy.nddata.NDDataArray or astropy.units.Quantity object.")

            target_unit_nu = astropy.units.erg / (astropy.units.cm**2 * astropy.units.s * astropy.units.Hz)
            # Check unit equivalency using the actual quantity
            flux_nu_data_attr = self.observed_flux_nu.data
            if isinstance(flux_nu_data_attr, astropy.units.Quantity):
                current_flux_nu_q = flux_nu_data_attr
            else:
                current_flux_nu_q = flux_nu_data_attr * self.observed_flux_nu.unit

            if not current_flux_nu_q.unit.is_equivalent(target_unit_nu):
                raise ValueError(f"observed_flux_nu units ({current_flux_nu_q.unit}) must be equivalent to {target_unit_nu}")
        else:
            self.observed_flux_nu = None

        # Handle F_lambda flux
        if observed_flux_lambda is not None:
            if isinstance(observed_flux_lambda, astropy.nddata.NDDataArray):
                self.observed_flux_lambda = observed_flux_lambda
            elif isinstance(observed_flux_lambda, astropy.units.Quantity):
                self.observed_flux_lambda = astropy.nddata.NDDataArray(observed_flux_lambda) # .data will be the Quantity
            else:
                raise TypeError("observed_flux_lambda must be an astropy.nddata.NDDataArray or astropy.units.Quantity object.")

            target_unit_lambda = astropy.units.erg / (astropy.units.cm**2 * astropy.units.s * astropy.units.AA)
            # Check unit equivalency
            flux_lambda_data_attr = self.observed_flux_lambda.data
            if isinstance(flux_lambda_data_attr, astropy.units.Quantity):
                current_flux_lambda_q = flux_lambda_data_attr
            else:
                current_flux_lambda_q = flux_lambda_data_attr * self.observed_flux_lambda.unit

            if not current_flux_lambda_q.unit.is_equivalent(target_unit_lambda):
                raise ValueError(f"observed_flux_lambda units ({current_flux_lambda_q.unit}) must be equivalent to {target_unit_lambda}")
        else:
            self.observed_flux_lambda = None

        # Auto-convert between flux types if only one is provided
        if self.observed_flux_nu is None and self.observed_flux_lambda is not None:
            self._convert_lambda_to_nu()
        elif self.observed_flux_lambda is None and self.observed_flux_nu is not None:
            self._convert_nu_to_lambda()

    def _get_quantity_from_nddata(self, nddata_array):
        """Helper to reliably get an astropy.units.Quantity from an NDDataArray."""
        if nddata_array is None:
            return None
        data_attr = nddata_array.data
        if isinstance(data_attr, astropy.units.Quantity):
            return data_attr
        elif nddata_array.unit is not None:
            return data_attr * nddata_array.unit
        else: # Should not happen if units are always present as per design
            raise ValueError("NDDataArray is missing unit information for quantity conversion.")


    def _convert_lambda_to_nu(self):
        """Convert F_lambda to F_nu using astropy.units.spectral_density and rest-frame wavelength."""
        if self.observed_flux_lambda is None:
            raise ValueError("Cannot convert from F_lambda: observed_flux_lambda is None.")

        F_lambda_quantity_to_convert = self._get_quantity_from_nddata(self.observed_flux_lambda)

        # For spectral_density, the wavelength needs to be a Quantity.
        # self.restframe_wavelengths.data is a numpy array, .unit is the unit.
        rest_wave_quantity = self.restframe_wavelengths.data * self.restframe_wavelengths.unit

        target_F_nu_unit = astropy.units.erg / (astropy.units.cm**2 * astropy.units.s * astropy.units.Hz)

        F_nu_converted = F_lambda_quantity_to_convert.to(
            target_F_nu_unit,
            equivalencies=astropy.units.spectral_density(rest_wave_quantity)
        )

        flux_nu_data = F_nu_converted.value

        uncertainty_nu_obj = None
        if self.observed_flux_lambda.uncertainty is not None:
            uncertainty_F_lambda_values = self.observed_flux_lambda.uncertainty.array
            # Assume uncertainty has the same unit as the flux data
            uncertainty_F_lambda_quantity = uncertainty_F_lambda_values * self.observed_flux_lambda.unit

            uncertainty_F_nu_converted = uncertainty_F_lambda_quantity.to(
                target_F_nu_unit,
                equivalencies=units.spectral_density(rest_wave_quantity)
            )
            uncertainty_nu_data = uncertainty_F_nu_converted.value
            uncertainty_nu_obj = type(self.observed_flux_lambda.uncertainty)(uncertainty_nu_data)

        self.observed_flux_nu = astropy.nddata.NDDataArray(
            data=flux_nu_data, # flux_nu_data is now a numpy array
            uncertainty=uncertainty_nu_obj,
            unit=target_F_nu_unit
        )

    def _convert_nu_to_lambda(self):
        """Convert F_nu to F_lambda using astropy.units.spectral_density and rest-frame wavelength."""
        if self.observed_flux_nu is None:
            raise ValueError("Cannot convert from F_nu: observed_flux_nu is None.")

        F_nu_quantity_to_convert = self._get_quantity_from_nddata(self.observed_flux_nu)

        # For spectral_density, the wavelength needs to be a Quantity.
        rest_wave_quantity = self.restframe_wavelengths.data * self.restframe_wavelengths.unit


        target_F_lambda_unit = astropy.units.erg / (astropy.units.cm**2 * astropy.units.s * astropy.units.AA)

        F_lambda_converted = F_nu_quantity_to_convert.to(
            target_F_lambda_unit,
            equivalencies=astropy.units.spectral_density(rest_wave_quantity)
        )

        flux_lambda_data = F_lambda_converted.value

        uncertainty_lambda_obj = None
        if self.observed_flux_nu.uncertainty is not None:
            uncertainty_F_nu_values = self.observed_flux_nu.uncertainty.array
            # Assume uncertainty has the same unit as the flux data
            uncertainty_F_nu_quantity = uncertainty_F_nu_values * self.observed_flux_nu.unit

            uncertainty_F_lambda_converted = uncertainty_F_nu_quantity.to(
                target_F_lambda_unit,
                equivalencies=astropy.units.spectral_density(rest_wave_quantity)
            )
            uncertainty_lambda_data = uncertainty_F_lambda_converted.value
            uncertainty_lambda_obj = type(self.observed_flux_nu.uncertainty)(uncertainty_lambda_data)

        self.observed_flux_lambda = astropy.nddata.NDDataArray(
            data=flux_lambda_data, # flux_lambda_data is now a numpy array
            uncertainty=uncertainty_lambda_obj,
            unit=target_F_lambda_unit
        )

    def __repr__(self):
        """String representation of the spectrum object."""
        wave_data_repr = "N/A"
        num_points_str = "N/A"

        # Check observed_wavelengths and its data
        if self.observed_wavelengths is not None and self.observed_wavelengths.data is not None:
            # Determine if data is a Quantity or ndarray for min/max operations
            obs_wave_values = self.observed_wavelengths.data
            if isinstance(obs_wave_values, astropy.units.Quantity):
                obs_wave_values = obs_wave_values.value # Get numpy array for min/max

            if obs_wave_values is not None and len(obs_wave_values) > 0: # Check if array is not empty
                 wave_data_repr = f"{obs_wave_values.min():.2f} - {obs_wave_values.max():.2f} {self.observed_wavelengths.unit}"
                 num_points_str = str(len(obs_wave_values))
            else: # Handle empty array case
                wave_data_repr = f"Empty {self.observed_wavelengths.unit}"
                num_points_str = "0"


        return f"Spectrum_1d(z={self.redshift.value:.3f}, λ_obs={wave_data_repr}, {num_points_str} points)"


class SpectralLineFitter:
    """
    A class to fit Gaussian components to spectral lines in a spectrum.

    Attributes
    ----------
    redshift : float
        The redshift value to use for fitting.
    observed_wavelengths : astropy.nddata.NDDataArray
        The observed wavelengths of the spectrum.
    observed_flux_mu : astropy.nddata.NDDataArray
        The observed flux values of the spectrum in F_mu units, containing the error as uncertainty.
    observed_flux_lambda : astropy.nddata.NDDataArray
        The observed flux values of the spectrum in F_lambda units, containing the error as uncertainty.
    restframe_wavelengths : astropy.nddata.NDDataArray
        The rest frame wavelengths of the spectrum, calculated from the observed wavelengths and redshift.
    line_restframe_wavelength : float
        The restframe wavelength of the spectral line to fit.
    max_components : int
        The maximum number of Gaussian components to fit.
    max_iterations : int
        The maximum number of iterations for the fitting process.
    fit_results : list
        A list to store the results of the fitting process for each component.

    Methods
    -------
    gaussian(x, amplitude, mean, stddev):
        Returns the value of a Gaussian function at a given x.

    """

    def __init__(self, fits_FilePath,spectrum_load_function, redshift, line_restframe_wavelength, max_components=8, max_iterations=100000):
        """
        Initializes the SpectralLineFitter with the given parameters.

        Parameters
        ----------
        fits_FilePath : str
            The path to the FITS file containing the spectrum data.
        spectrum_load_function : callable
            A function to load the spectrum data from the FITS file, outputting the astropy.nddata.NDDataArray object with the observed wavelengths and fluxes in F_nu and F_lambda, respectively.
        redshift : float
            The redshift value to use for fitting.
        line_restframe_wavelength : float
            The restframe wavelength of the spectral line to fit.
        max_components : int, optional
            The maximum number of Gaussian components to fit (default is 8).
        max_iterations : int, optional
            The maximum number of iterations for the fitting process (default is 100000).
        """
        observed_wavelength, observed_flux_lambda, observed_flux_mu = spectrum_load_function(fits_FilePath)

        self.observed_wavelengths = observed_wavelength
        self.observed_flux_lambda = observed_flux_lambda
        self.observed_flux_mu = observed_flux_mu
        self.redshift = redshift
        self.restframe_wavelengths = self.observed_wavelengths / (1 + redshift)

        self.line_restframe_wavelength = line_restframe_wavelength
        self.max_components = max_components
        self.max_iterations = max_iterations
        self.fit_results = list()

    @staticmethod
    def gaussian(x, amplitude, mean, stddev):
        """
        Returns the value of a Gaussian function at a given x. Here used as the model function for the fitting process.

        Parameters
        ----------
        x : float, np.ndarray or astropy.nddata.NDDataArray
            The input value(s) at which to evaluate the Gaussian function.
        amplitude : float
            The amplitude of the Gaussian.
        mean : float
            The mean (center) of the Gaussian.
        stddev : float
            The standard deviation (width) of the Gaussian.

        Returns
        -------
        Astropy.nddata.NDDataArray or np.ndarray
            The value of the Gaussian function evaluated at x.
        """
        if isinstance(x, astropy.nddata.NDDataArray):
            x = x.data
            return astropy.nddata.NDDataArray(amplitude * np.exp(-0.5 * ((x - mean) / stddev) ** 2), unit=x.unit)
        elif isinstance(x, astropy.nddata.NDData):
            x = x.data
            return astropy.nddata.NDDataArray(amplitude * np.exp(-0.5 * ((x - mean) / stddev) ** 2), unit=x.unit)
        elif isinstance(x, astropy.units.Quantity):
            x = x.value
            return amplitude * np.exp(-0.5 * ((x - mean) / stddev) ** 2) * x.unit
        else:
            return amplitude * np.exp(-0.5 * ((x - mean) / stddev) ** 2)


    def fit_single_gaussian(self, wavelength, flux,uncertainty=None, initial_guess=None):
        """
        Fit a single Gaussian to the spectral data provided, if the initial guess is not provided, it will be automatically generated based on the data.

        Parameters
        ----------
        wavelength : astropy.nddata.NDDataArray
            The wavelengths of the spectrum to fit， should be in the rest frame and containing the unit.

        flux : astropy.nddata.NDDataArray
            The flux values of the spectrum to fit, should be in F_lambda and containing the unit.

        initial_guess : list, optional
            Initial guess for the Gaussian parameters [amplitude, mean, stddev]. If not provided, it will be automatically generated based on the data.

        Returns
        -------
        dict
            A dictionary containing the fit results, including the fitted parameters and the covariance matrix.
        """
        try:
            wavelength=wavelength.convert_unit_to(units.AA)
            flux=flux.convert_unit_to(units.erg / (units.cm**2 * units.s * units.AA), equivalencies=units.spectral_density(wavelength))
            if uncertainty is not None:
                uncertainty = uncertainty.convert_unit_to(units.erg / (units.cm**2 * units.s * units.AA), equivalencies=units.spectral_density(wavelength))
                uncertainty = astropy.nddata.StdDevUncertainty(uncertainty)
            # Check if the initial guess is provided


            if initial_guess is None:
                # Automatically generate an initial guess based on the data
                amplitude = np.max(flux.value)* flux.unit
                mean = np.argmax(flux.value) * wavelength.unit
                stddev = 10.*units.AA
                initial_guess = [amplitude, mean, stddev]
            else:
                # Ensure the initial guess is in the correct format
                if not isinstance(initial_guess, list) or len(initial_guess) != 3:
                    raise ValueError("Initial guess must be a list of three values: [amplitude, mean, stddev]")
                initial_guess = [initial_guess[0] * flux.unit, initial_guess[1] * wavelength.unit, initial_guess[2] * wavelength.unit]
            # Fit the Gaussian using the initial guess

            popt, pcov = scipy.optimize.curve_fit(
                self.gaussian,
                wavelength.value,
                flux.value,
                p0=initial_guess,
                maxfev=self.max_iterations)

            y_fit= self.gaussian(wavelength, *popt)

            integrated_flux, integration_error = scipy.integrate.quad(self.gaussian, wavelength.value.min(), wavelength.value.max(), args=tuple(popt))* flux.unit

            return {
                'success': True,
                'parameters': {
                    'amplitude': popt[0],
                    'mean': popt[1],
                    'stddev': popt[2]
                },
                'fitted_curve': y_fit,
                'integrated_flux': integrated_flux,
                'integration_error': integration_error,
                'covariance': pcov
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


    def check_line(self, line_restframe_wavelength, mean_fit, tolerance=10*units.AA):
        """
        Check if the fitted mean wavelength is within the specified tolerance of the line restframe wavelength.

        Parameters
        ----------
        line_restframe_wavelength : astropy.units.Quantity
            The restframe wavelength of the spectral line to check.
        mean_fit : astropy.units.Quantity
            The mean wavelength from the Gaussian fit.
        tolerance : astropy.units.Quantity, optional
            The tolerance within which the mean wavelength should be to be considered a match (default is 10 Angstroms).

        Returns
        -------
        bool
            True if the mean wavelength is within the tolerance of the line restframe wavelength, False otherwise.
        """

        if not isinstance(line_restframe_wavelength, u.Quantity):
            raise TypeError("line_restframe_wavelength must be an astropy.units.Quantity object.")
        if not isinstance(mean_fit, u.Quantity):
            raise TypeError("mean_fit must be an astropy.units.Quantity object.")
        if not isinstance(tolerance, u.Quantity):
            raise TypeError("tolerance must be an astropy.units.Quantity object.")

        return abs(line_restframe_wavelength - mean_fit) <= tolerance

    def plot_fit_result(self, wavelength, flux, fit_result, component_index, is_residual=False):
        """
        Plot the fit result with Gaussian components fit overlay.

        Parameters
        ----------
        wavelength : astropy.nddata.NDDataArray or astropy.units.Quantity
            The wavelengths of the spectrum to plot, should be in the rest frame and containing the unit.
        flux : astropy.nddata.NDDataArray or astropy.units.Quantity
            The flux values of the spectrum to plot, should be in F_lambda and containing the unit.
        fit_result : dict
            The fit result dictionary containing the fitted parameters and the fitted curve.
        component_index : int
            The component number for the current fit result.
        is_residual : bool, optional
            If True, plot the residuals instead of the fit result (default is False).
        """

        plt.figure(figsize=(20,10))

        if not isinstance(wavelength, astropy.nddata.NDDataArray) and not isinstance(wavelength, u.Quantity):
            raise TypeError("wavelength must be an astropy.nddata.NDDataArray or astropy.units.Quantity object.")

        if not isinstance(flux, astropy.nddata.NDDataArray) and not isinstance(flux, units.Quantity):
            raise TypeError("flux must be an astropy.nddata.NDDataArray or astropy.units.Quantity object.")
        if not isinstance(wavelength, astropy.nddata.NDDataArray) and not isinstance(wavelength, units.Quantity):
            raise TypeError("wavelength must be an astropy.nddata.NDDataArray or astropy.units.Quantity object.")

        if isinstance(wavelength, astropy.nddata.NDDataArray):
            wavelength = wavelength.convert_unit_to(units.AA)
            flux = flux.convert_unit_to(units.erg / (units.cm**2 * units.s * units.AA), equivalencies=units.spectral_density(wavelength))
        elif isinstance(wavelength, u.Quantity):
            wavelength = wavelength.to(units.AA)
            flux = flux.to(units.erg / (units.cm**2 * units.s * units.AA), equivalencies=units.spectral_density(wavelength))

        if is_residual:
            plt.plot(wavelength.value, flux.value,
                     label=f'Residual Spectrum {component_index}',color='blue')
            title = f'Component {component_index} - Gaussian Fit Residuals'
        else:
            plt.plot(wavelength.value, flux.value,
                     label=f'Observed Spectrum {component_index}', color='blue')
            title = f'Component {component_index} - Gaussian Fit Result'

        if fit_result['success']:
            plt.plot(wavelength.value, fit_result['fitted_curve'].value,label=f'Gaussian Fit {component_index}', color='red')

        plt.xlabel('Wavelength (Angstroms)')
        plt.ylabel('Flux (erg / (cm^2 s Angstrom))')
        plt.title(title)
        plt.legend()
        plt.xlim(wavelength.value.min(), wavelength.value.max())

        if not is_residual:
            flux_margin = 0.1 * np.max(flux.value)
            plt.ylim(np.min(flux.value) - flux_margin, np.max(flux.value) + flux_margin)

        plt.show()

    def print_fit_summary(self, fit_result, component_index):
        """
        Print a summary of the fit results for a specific component.

        Parameters
        ----------
        fit_result : dict
            The fit result dictionary containing the fitted parameters and the covariance matrix.
        component_index : int
            The component number for the current fit result.
        """
        if fit_result['success']:
            print(f"\n{'='*60}")
            params= fit_result['parameters']
            print(f"\nComponent {component_index} Fit Parameters:")
            print(f"Amplitude: {params['amplitude']:.3e}")
            print(f"Mean (Restframe Wavelength): {params['mean']:.3f} {self.restframe_wavelengths.unit}")
            print(f"Standard Deviation: {params['stddev']:.3f} {self.restframe_wavelengths.unit}")
            print(f"Integrated Flux: {fit_result['integrated_flux']:.3e} erg/cm^2")
            print(f"Integration Error: {fit_result['integration_error']:.3e} erg/cm^2")
            print(f"Covariance Matrix:\n{fit_result['covariance']}")
            print(f"\n{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"\nComponent {component_index} Fit Failed: {fit_result['error']}")
            print(f"\n{'='*60}")

    def iterative_gaussian_fitting(self,line_restframe_wavelength,uncertainty=None, plot_results=True):
        """
        Perform iterative Gaussian fitting on the spectral data, fitting until the mean wavelength of the fitted Gaussian is within a specified tolerance of the line restframe wavelength or the maximum number of components is reached.

        Parameters
        ----------
        line_restframe_wavelength : astropy.units.Quantity
            The restframe wavelength of the spectral line to fit.
            Should be a single value with the appropriate unit.
        uncertainty : astropy.nddata.NDDataArray or astropy.units.Quantity, optional
            The uncertainty in the flux values, should be in the same unit as the flux values.
            If not provided, no uncertainty will be used in the fitting process.
        plot_results : bool, optional
            If True, plot the fit results for each component (default is True).

        Returns
        -------
        tuple
            A tuple containing:
            - fit_results: list of dictionaries with the fit results for each component.
            - line_integrated_flux: astropy.units.Quantity
                The total integrated flux of the line after all components have been fitted.
        """

        self.fit_results = []  # Reset fit results for each call
        current_flux= self.observed_flux_lambda# Start with the original flux
        line_integrated_flux = 0 * units.erg / units.cm**2

        for component_index in range(0,self.max_components):
            if not isinstance(self.restframe_wavelengths, astropy.nddata.NDDataArray) and not isinstance(self.restframe_wavelengths, u.Quantity):
                raise TypeError("wavelength must be an astropy.nddata.NDDataArray or astropy.units.Quantity object.")

            if not isinstance(current_flux, astropy.nddata.NDDataArray) and not isinstance(current_flux, units.Quantity):
                raise TypeError("flux must be an astropy.nddata.NDDataArray or astropy.units.Quantity object.")

            print(f"\n{'='*60}")
            print(f"Fitting Component {component_index + 1} of {self.max_components}...")
            print(f"\n{'='*60}")
            print

            fit_result= self.fit_single_gaussian(
                wavelength=self.restframe_wavelengths,
                flux=current_flux
                #uncertainty=uncertainty,
                #initial_guess=None
            )

            if not fit_result['success']:
                print(f"Fit failed for component {component_index + 1}: {fit_result['error']}")
                break

            fit_result['component_index'] = component_index + 1
            fit_result['line_restframe_wavelength'] = line_restframe_wavelength
            fit_result['is_within_tolerance'] = self.check_line(
                line_restframe_wavelength,
                fit_result['parameters']['mean'],
                tolerance=10 * units.AA
            )

            self.fit_results.append(fit_result)

            self.print_fit_summary(fit_result, component_index + 1)

            if plot_results:

                is_residual= component_index > 0

                self.plot_fit_result(
                    wavelength=self.restframe_wavelengths,
                    flux=current_flux,
                    fit_result=fit_result,
                    component_index=component_index + 1,
                    is_residual=is_residual
                )

            if fit_result['is_within_tolerance']:
                print(f"Component {component_index + 1} mean wavelength is within tolerance of the line restframe wavelength.")
                line_integrated_flux += fit_result['integrated_flux']
                print(f"Total integrated flux for the line so far: {line_integrated_flux:.3e} erg/cm^2")
                break

            current_flux = current_flux - fit_result['fitted_curve']

            if plot_results and component_index < self.max_components - 1:
                plt.figure(figsize=(20,10))
                plt.plot(self.restframe_wavelengths.value, current_flux.value, label=f'Residual Spectrum after Component {component_index + 1}', color='green')
                plt.xlabel('Wavelength (Angstroms)')
                plt.ylabel('Flux (erg / (cm^2 s Angstrom))')
                plt.title(f'Residual Spectrum after Component {component_index + 1}')
                plt.legend()
                plt.xlim(self.restframe_wavelengths.value.min(), self.restframe_wavelengths.value.max())
                flux_margin = 0.1 * np.max(current_flux.value)
                plt.ylim(np.min(current_flux.value) - flux_margin, np.max(current_flux.value) + flux_margin)
                plt.show()

        if len(self.fit_results) == 0:
            print("No components were fitted successfully.")
            return [], 0 * units.erg / units.cm**2

        return self.fit_results, line_integrated_flux

    def get_line_components(self):
        """
        Get the fitted line components from the fit results.

        Returns
        -------
        Dict or None
            A dictionary containing the fitted line components, where the keys are the component indices and the values are dictionaries with the fitted parameters.
            If no components were fitted successfully, returns None.
        """

        for fit_result in self.fit_results:
            if fit_result.get('is_within_tolerance', True):
                return {
                    fit_result['component_index']: {
                        'amplitude': fit_result['parameters']['amplitude'],
                        'mean': fit_result['parameters']['mean'],
                        'stddev': fit_result['parameters']['stddev'],
                        'integrated_flux': fit_result['integrated_flux'],
                        'integration_error': fit_result['integration_error']
                    }
                }
        return None

    def plot_final_decomposition(self, line_restframe_wavelength, uncertainty=None):
        """
        Plot the final decomposition of the spectrum with all fitted components.

        Parameters
        ----------
        line_restframe_wavelength : astropy.units.Quantity
            The restframe wavelength of the spectral line to plot.
        uncertainty : astropy.nddata.NDDataArray or astropy.units.Quantity, optional
            The uncertainty in the flux values, should be in the same unit as the flux values.
            If not provided, no uncertainty will be used in the plotting process.
        uncertainty : astropy.nddata.NDDataArray or astropy.units.Quantity, optional
            The uncertainty in the flux values, should be in the same unit as the flux values.
            If not provided, no uncertainty will be used in the plotting process.
        """

        if not isinstance(line_restframe_wavelength, astropy.nddata.NDDataArray) and not isinstance(line_restframe_wavelength, u.Quantity):
            raise TypeError("line_restframe_wavelength must be an astropy.nddata.NDDataArray or astropy.units.Quantity object.")

        #if not isinstance(uncertainty, astropy.nddata.NDDataArray) and not isinstance(uncertainty, units.Quantity):
            #raise TypeError("uncertainty must be an astropy.nddata.NDDataArray or astropy.units.Quantity object.")

        plt.figure(figsize=(20,10))
        plt.plot(self.restframe_wavelengths.data,self.observed_flux_lambda.data,label='Observed Spectrum', color='black', alpha=0.7)

        colors = ['red', 'blue', 'green', 'orange', 'purple']

        for i, fit_result in enumerate(self.fit_results):
            if fit_result['success']:
                params= fit_result['parameters']

                component_flux=self.gaussian(self.restframe_wavelengths.data, **params)

                label=f'Component {fit_result["component_index"]}'

                if fit_result['is_within_tolerance']:
                    label += f' (Line at {line_restframe_wavelength:.3f} {self.restframe_wavelengths.unit})'

                plt.plot(self.restframe_wavelengths.data, component_flux.data, label=label, color=colors[i % len(colors)], alpha=0.7,linewidth=1.5)

        plt.xlabel('Wavelength (Angstroms)')
        plt.ylabel('Flux (erg / (cm^2 s Angstrom))')
        plt.title('Final Spectrum Decomposition with Fitted Components')
        plt.legend()
        plt.xlim(self.restframe_wavelengths.data.min(), self.restframe_wavelengths.data.max())
        plt.grid(True,alpha=0.3)
        plt.show()
#uncertainty handling is not implemented in this function, but can be added if needed.

