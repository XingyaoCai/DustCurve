import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import scipy
import astropy


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
        return np.float32(redshift)* astropy.units.dimensionless_unscaled
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
    with astropy.io.fits.open(Fits_FilePath) as hdul:

      try:
        Spectra_Data = hdul[1].data

        Wavelength = Spectra_Data['wave']*astropy.units.micron
        Flux = Spectra_Data['flux']*astropy.units.uJy
        Error= Spectra_Data['err']*astropy.units.uJy

        Flux_Lambda = Flux.to(astropy.units.erg / (astropy.units.cm**2 * astropy.units.s * astropy.units.AA), equivalencies=astropy.units.spectral_density(Wavelength))
        Error_Lambda = Error.to(astropy.units.erg / (astropy.units.cm**2 * astropy.units.s * astropy.units.AA), equivalencies=astropy.units.spectral_density(Wavelength))
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
    processing_wavelengths : astropy.nddata.NDDataArray
        The processing wavelengths of the spectrum, which can be either observed or rest frame depending on the input. Here used to store the wavelengths that are currently being processed or analyzed.
    processing_flux: astropy.nddata.NDDataArray
        The processing flux values of the spectrum, which can be either F_nu or F_lambda depending on the input. Here used to store the flux that is currently being processed or analyzed.
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
        elif isinstance(observed_wavelengths, astropy.units.Quantity):
            # If input is a Quantity, NDDataArray stores it in its .data attribute
            self.observed_wavelengths = astropy.nddata.NDDataArray(observed_wavelengths)
        else:
            raise TypeError("observed_wavelengths must be an astropy.nddata.NDDataArray or astropy.units.Quantity object.")

        # Handle redshift
        if isinstance(redshift, (float, int)):
            self.redshift = redshift * astropy.units.dimensionless_unscaled
        elif isinstance(redshift, astropy.units.Quantity):
            if redshift.unit.is_equivalent(astropy.units.dimensionless_unscaled):
                self.redshift = redshift
            else:
                raise ValueError("Redshift must be dimensionless.")
        else:
            raise TypeError("Redshift must be a float, int, or dimensionless astropy.units.Quantity.")

        # Calculate rest-frame wavelengths
        # Get the observed wavelength data, which might be a Quantity or ndarray
        obs_wave_data_attr = self.observed_wavelengths.data
        if isinstance(obs_wave_data_attr, astropy.units.Quantity):
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

            # if not current_flux_nu_q.unit.is_equivalent(target_unit_nu):
            #     raise ValueError(f"observed_flux_nu units ({current_flux_nu_q.unit}) must be equivalent to {target_unit_nu}")
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

            # if not current_flux_lambda_q.unit.is_equivalent(target_unit_lambda):
            #     raise ValueError(f"observed_flux_lambda units ({current_flux_lambda_q.unit}) must be equivalent to {target_unit_lambda}")
        else:
            self.observed_flux_lambda = None

        # Auto-convert between flux types if only one is provided
        if self.observed_flux_nu is None and self.observed_flux_lambda is not None:
            self._convert_lambda_to_nu()
        elif self.observed_flux_lambda is None and self.observed_flux_nu is not None:
            self._convert_nu_to_lambda()

        self.processing_flux = self.observed_flux_lambda if self.observed_flux_lambda is not None else self.observed_flux_nu
        self.processing_wavelengths = self.restframe_wavelengths if self.restframe_wavelengths is not None else self.observed_wavelengths

        self.processing_wavelengths = self.processing_wavelengths.convert_unit_to(
            astropy.units.AA)

        # Handle NaN values by converting them to 0
        self._handle_nan_values()

    def _handle_nan_values(self):
        """Convert NaN values to 0 in wavelengths and fluxes."""

        # Handle observed wavelengths
        if self.observed_wavelengths is not None:
            obs_wave_data = self.observed_wavelengths.data
            if isinstance(obs_wave_data, astropy.units.Quantity):
                nan_mask = np.isnan(obs_wave_data.value)
                if np.any(nan_mask):
                    new_values = obs_wave_data.value.copy()
                    new_values[nan_mask] = 0
                    self.observed_wavelengths.data = new_values * obs_wave_data.unit
            else:
                nan_mask = np.isnan(obs_wave_data)
                if np.any(nan_mask):
                    self.observed_wavelengths.data[nan_mask] = 0

        # Handle rest-frame wavelengths
        if self.restframe_wavelengths is not None:
            rest_wave_data = self.restframe_wavelengths.data
            if isinstance(rest_wave_data, astropy.units.Quantity):
                nan_mask = np.isnan(rest_wave_data.value)
                if np.any(nan_mask):
                    new_values = rest_wave_data.value.copy()
                    new_values[nan_mask] = 0
                    self.restframe_wavelengths.data = new_values * rest_wave_data.unit
            else:
                nan_mask = np.isnan(rest_wave_data)
                if np.any(nan_mask):
                    self.restframe_wavelengths.data[nan_mask] = 0

        # Handle observed flux nu
        if self.observed_flux_nu is not None:
            flux_nu_data = self.observed_flux_nu.data
            if isinstance(flux_nu_data, astropy.units.Quantity):
                nan_mask = np.isnan(flux_nu_data.value)
                if np.any(nan_mask):
                    new_values = flux_nu_data.value.copy()
                    new_values[nan_mask] = 0
                    self.observed_flux_nu.data = new_values * flux_nu_data.unit
            else:
                nan_mask = np.isnan(flux_nu_data)
                if np.any(nan_mask):
                    self.observed_flux_nu.data[nan_mask] = 0

            # Handle uncertainty for flux_nu if present
            if self.observed_flux_nu.uncertainty is not None:
                uncertainty_array = self.observed_flux_nu.uncertainty.array
                if uncertainty_array is not None:
                    nan_mask = np.isnan(uncertainty_array)
                    if np.any(nan_mask):
                        uncertainty_array[nan_mask] = 0

        # Handle observed flux lambda
        if self.observed_flux_lambda is not None:
            flux_lambda_data = self.observed_flux_lambda.data
            if isinstance(flux_lambda_data, astropy.units.Quantity):
                nan_mask = np.isnan(flux_lambda_data.value)
                if np.any(nan_mask):
                    new_values = flux_lambda_data.value.copy()
                    new_values[nan_mask] = 0
                    self.observed_flux_lambda.data = new_values * flux_lambda_data.unit
            else:
                nan_mask = np.isnan(flux_lambda_data)
                if np.any(nan_mask):
                    self.observed_flux_lambda.data[nan_mask] = 0

            # Handle uncertainty for flux_lambda if present
            if self.observed_flux_lambda.uncertainty is not None:
                uncertainty_array = self.observed_flux_lambda.uncertainty.array
                if uncertainty_array is not None:
                    nan_mask = np.isnan(uncertainty_array)
                    if np.any(nan_mask):
                        uncertainty_array[nan_mask] = 0

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
                equivalencies=astropy.units.spectral_density(rest_wave_quantity)
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


    def set_boundarys(self, lower_boundary=None, upper_boundary=None):
        """
        Set the lower and upper boundaries for the spectrum.

        Parameters
        ----------
        lower_boundary : astropy.units.Quantity, optional
            The lower boundary of the spectrum. If None, no lower boundary is set.
        upper_boundary : astropy.units.Quantity, optional
            The upper boundary of the spectrum. If None, no upper boundary is set.
        """
        if lower_boundary is not None and upper_boundary is None:
            lower_boundary = lower_boundary.to(self.restframe_wavelengths.unit)
            indices= self.restframe_wavelengths.data >= lower_boundary.value
            self.processing_wavelengths = self.restframe_wavelengths[indices]
            self.processing_flux = self.observed_flux_lambda[indices] if self.observed_flux_lambda is not None else self.observed_flux_nu[indices]
        if upper_boundary is not None and lower_boundary is None:
            upper_boundary = upper_boundary.to(self.restframe_wavelengths.unit)
            indices= self.restframe_wavelengths.data <= upper_boundary.value
            self.processing_wavelengths = self.restframe_wavelengths[indices]
            self.processing_flux = self.observed_flux_lambda[indices] if self.observed_flux_lambda is not None else self.observed_flux_nu[indices]

        if lower_boundary is not None and upper_boundary is not None:
            lower_boundary = lower_boundary.to(self.restframe_wavelengths.unit)
            upper_boundary = upper_boundary.to(self.restframe_wavelengths.unit)
            indices = (self.restframe_wavelengths.data >= lower_boundary.value) & (self.restframe_wavelengths.data <= upper_boundary.value)
            self.processing_wavelengths = self.restframe_wavelengths[indices]
            self.processing_flux = self.observed_flux_lambda[indices] if self.observed_flux_lambda is not None else self.observed_flux_nu[indices]

        self.processing_wavelengths = self.processing_wavelengths.convert_unit_to(
            astropy.units.AA)


    def show(self):
        """Display the processed spectrum."""
        plt.figure(figsize=(20,10))
        if self.processing_wavelengths is not None and self.processing_flux is not None:
            if isinstance(self.processing_wavelengths, astropy.nddata.NDDataArray):
                wave_data = self.processing_wavelengths.data
                wave_unit = self.processing_wavelengths.unit
            else:
                wave_data = self.processing_wavelengths
                wave_unit = None

            if isinstance(self.processing_flux, astropy.nddata.NDDataArray):
                flux_data = self.processing_flux.data
                flux_unit = self.processing_flux.unit
            else:
                flux_data = self.processing_flux
                flux_unit = None

            plt.plot(wave_data, flux_data, label='Processed Spectrum')
            plt.xlabel(f'Wavelength ({wave_unit})' if wave_unit else 'Wavelength')
            plt.ylabel(f'Flux ({flux_unit})' if flux_unit else 'Flux')
            plt.title(f'Spectrum at Redshift {self.redshift.value:.3f}')
            plt.legend()
            plt.show()

    def __repr__(self):
        """String representation of the spectrum object."""
        wave_data_repr = "N/A"
        num_points_str = "N/A"

        # Check restframe wavelengths
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

        return f"Spectrum_1d(z={self.redshift.value:.3f}, λ_obs={wave_data_repr}, {num_points_str} points)"

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
            raise TypeError("line_restframe_wavelengths must be an astropy.units.Quantity or a list of astropy.units.Quantity objects.")

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

    def fit_single_gaussian(self,initial_guess=None):
        """
        Fits a single Gaussian to the spectrum data, the initial guess will be generated based on the observed fluxes.

        Returns
        -------
        dict
            A dictionary containing the fit results, including the fitted parameters and the covariance matrix.
        """

        try:
            # Extract observed wavelengths and fluxes
            obs_wavelengths = self.spectrum.processing_wavelengths.convert_unit_to(astropy.units.AA).data
            obs_flux_lambda = self.spectrum.processing_flux.data
            #NumPy array

            # Initial guess for the Gaussian parameters
            amplitude_guess = obs_flux_lambda.max()  # Initial guess for the amplitude
            mean_guess = obs_wavelengths[np.argmax(obs_flux_lambda)]  # Initial guess for the mean (wavelength of max flux)
            stddev_guess = 10 * astropy.units.AA  # Initial guess for the width, can be adjusted
            #int or float here

            if initial_guess is not None:
                amplitude_guess = initial_guess[0]
                mean_guess = initial_guess[1]
                stddev_guess = initial_guess[2] * astropy.units.AA
            elif not isinstance(stddev_guess, astropy.units.Quantity):
                raise TypeError("stddev_guess must be an astropy.units.Quantity object.")
            else:
                stddev_guess = stddev_guess.to(astropy.units.AA)
                initial_guess = [amplitude_guess, mean_guess, stddev_guess.value]

            #print(f"Initial guess for Gaussian parameters: {initial_guess}")

            # Fit the Gaussian using scipy.optimize.curve_fit

            popt, pcov = scipy.optimize.curve_fit(self.gaussian,
                                                    obs_wavelengths,
                                                    obs_flux_lambda,
                                                    p0=initial_guess,
                                                    maxfev=self.max_iterations)

            y_fit = self.gaussian(obs_wavelengths, *popt)

            integrated_flux, integration_error = scipy.integrate.quad(self.gaussian, obs_wavelengths.min(), obs_wavelengths.max(), args=tuple(popt),epsabs=0)* self.spectrum.processing_flux.unit* self.spectrum.processing_wavelengths.unit

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

    def check_line(self, line_restframe_wavelength, mean_fit, tolerance=10* astropy.units.AA):
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
            raise TypeError("line_restframe_wavelength must be an astropy.units.Quantity object.")
        if not isinstance(mean_fit, astropy.units.Quantity):
            raise TypeError("mean_fit must be an astropy.units.Quantity object.")
        if not isinstance(tolerance, astropy.units.Quantity):
            raise TypeError("tolerance must be an astropy.units.Quantity object.")

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
            raise TypeError("fit_result must be a dictionary containing the fit results.")

        if 'fitted_curve' not in fit_result or 'parameters' not in fit_result:
            raise ValueError("fit_result must contain 'fitted_curve' and 'parameters' keys.")

        obs_wavelengths = self.spectrum.processing_wavelengths.convert_unit_to(astropy.units.AA).data
        obs_flux_lambda = self.spectrum.processing_flux.data

        fig,ax= plt.subplots(figsize=(20, 10))

        if is_residual:
            ax.plot(obs_wavelengths, obs_flux_lambda, label=f"Residual Spectrum {component_index}", color='blue', alpha=0.5)
            title=f'Component {component_index} - Gaussian Fit Residuals'

        else:
            ax.plot(obs_wavelengths, obs_flux_lambda, label=f"Observed Spectrum {component_index}", color='blue', alpha=0.5)
            title=f'Component {component_index} - Gaussian Fit Result'

        if fit_result['success']:
            ax.plot(obs_wavelengths, fit_result['fitted_curve'].data, label=f"Fitted Curve {component_index}", color='red', alpha=0.7)

        ax.set_xlabel("Wavelength (Angstrom)", fontsize=14)
        ax.set_ylabel("Flux (erg/cm^2/s/Angstrom)", fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.legend()
        ax.set_xlim(obs_wavelengths.min(), obs_wavelengths.max())

        if not is_residual:
            flux_margin= 0.1 * np.nanmax(obs_flux_lambda)
            ax.set_ylim(np.nanmin(obs_flux_lambda) - flux_margin, np.nanmax(obs_flux_lambda) + flux_margin)

        plt.grid()
        plt.show()

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
            raise TypeError("fit_result must be a dictionary containing the fit results.")

        if 'parameters' not in fit_result or 'covariance' not in fit_result:
            raise ValueError("fit_result must contain 'parameters' and 'covariance' keys.")

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
            print(f"Component {component_index} Fit Failed: {fit_result['error']}")
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
        line_integrated_flux = 0 * self.spectrum.processing_flux.unit * self.spectrum.processing_wavelengths.unit

        for component_index in range(self.max_components):

            if not isinstance(line_restframe_wavelength, astropy.units.Quantity):
                raise TypeError("line_restframe_wavelength must be an astropy.units.Quantity object.")

            # print(f"\n{'='*60}")
            # print(f"Fitting Component {component_index + 1} for Line at {line_restframe_wavelength:.3f}")
            # print(f"Tolerance: {tolerance:.3f}")
            # print(f"{'='*60}")

            if component_index ==self.max_components - 1:
                indice= np.argmin(
                    (self.spectrum.processing_wavelengths.data - line_restframe_wavelength.value))
                initial_guess= [self.spectrum.processing_flux.data[indice],
                                line_restframe_wavelength.value,
                                5]
            else:
                initial_guess = None

            fit_result= self.fit_single_gaussian(initial_guess=initial_guess)
            if not fit_result['success']:
                print(f"Component {component_index + 1} fit failed: {fit_result['error']}")
                quit()

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
                is_residual=component_index > 0
                self.plot_fit_result(fit_result, component_index, is_residual=is_residual)

            self.spectrum.processing_flux = astropy.nddata.NDDataArray(
                data=self.spectrum.processing_flux.data - fit_result['fitted_curve'].data,
                unit=self.spectrum.processing_flux.unit
            )

            if plot_results and component_index< self.max_components - 1:
                plt.figure(figsize=(20, 10))
                plt.plot(self.spectrum.processing_wavelengths.data, self.spectrum.processing_flux.data, label=f"Residual Spectrum after Component {component_index + 1}", color='blue', alpha=0.5)
                plt.xlabel("Wavelength (Angstrom)", fontsize=14)
                plt.ylabel("Flux (erg/cm^2/s/Angstrom)", fontsize=14)
                plt.title(f"Residual Spectrum after Component {component_index + 1} Fitting", fontsize=16)
                plt.legend()
                plt.xlim(self.spectrum.processing_wavelengths.data.min(), self.spectrum.processing_wavelengths.data.max())
                flux_margin = 0.1 * np.nanmax(self.spectrum.processing_flux.data)
                plt.ylim(np.nanmin(self.spectrum.processing_flux.data) - flux_margin, np.nanmax(self.spectrum.processing_flux.data) + flux_margin)
                plt.grid()
                plt.show()

            if fit_result['is_within_tolerance']:
                #print(f"Component {component_index} is within tolerance for line at {line_restframe_wavelength:.3f}.")
                line_integrated_flux += fit_result['integrated_flux']
                break


        if len(self.fit_results) ==0:
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

    def plot_final_decomposition(self, line_restframe_wavelength,figure_name=None):
        """
        Plots the final decomposition of the spectral line with all fitted components.

        Parameters
        ----------
        line_restframe_wavelength : astropy.units.Quantity
            The rest-frame wavelength of the spectral line to plot.
        tolerance : astropy.units.Quantity, optional
            The tolerance range for checking the fit (default is 10 * astropy.units.AA).
        """

        self.spectrum.processing_wavelengths=self.spectrum.processing_wavelengths.convert_unit_to(astropy.units.AA)
        self.spectrum.processing_flux=self.spectrum.processing_flux.convert_unit_to(astropy.units.erg / (astropy.units.cm**2 * astropy.units.s * astropy.units.AA))

        subindices= np.where(
            (self.spectrum.restframe_wavelengths.convert_unit_to(astropy.units.AA).data>=self.spectrum.processing_wavelengths.data.min()) &
            (self.spectrum.restframe_wavelengths.convert_unit_to(astropy.units.AA).data<=self.spectrum.processing_wavelengths.data.max())
        )[0]

        plt.figure(figsize=(20, 10))

        plt.step(self.spectrum.processing_wavelengths.data, self.spectrum.observed_flux_lambda.data[subindices], label="Observed Spectrum", color='blue', alpha=0.5, linewidth=1.6)

        colors = ['red', 'blue', 'green', 'orange', 'purple']

        for i, fit_result in enumerate(self.fit_results):
            if fit_result['success']:

                params= [fit_result['parameters']['amplitude'].value,
                         fit_result['parameters']['mean'].value,
                            fit_result['parameters']['stddev'].value]

                component_flux= self.gaussian(
                    self.spectrum.processing_wavelengths.data, *params
                )

                label= f"Component {fit_result['component_index']}"

                if fit_result['is_within_tolerance']:
                    label+=f'(Line at {line_restframe_wavelength:.3f})'

                plt.plot(
                    self.spectrum.processing_wavelengths.data,
                    component_flux,
                    label=label,
                    color=colors[i % len(colors)],
                    alpha=0.7
                )

        plt.xlabel("Wavelength (Angstrom)", fontsize=14)
        plt.ylabel("Flux (erg/cm^2/s/Angstrom)", fontsize=14)
        plt.title(f"Final Decomposition for Line at {line_restframe_wavelength:.3f}", fontsize=16)
        plt.legend()
        plt.xlim(self.spectrum.processing_wavelengths.data.min(), self.spectrum.processing_wavelengths.data.max())
        flux_margin = 0.1 * np.nanmax(self.spectrum.observed_flux_lambda.data[subindices])
        plt.ylim(np.nanmin(self.spectrum.observed_flux_lambda.data[subindices]) - flux_margin, np.nanmax(self.spectrum.observed_flux_lambda.data[subindices]) + flux_margin)
        plt.grid()
        plt.savefig(f"./fig/Final_Decomposition_{figure_name if figure_name else 'spectrum'}.png")
        plt.close()



