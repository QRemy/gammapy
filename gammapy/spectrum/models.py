# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Spectral models for Gammapy.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from . import integrate_spectrum
from ..extern.bunch import Bunch


__all__ = [
    'SpectralModel',
    'PowerLaw',
    'ExponentialCutoffPowerLaw',
]


# Note: Consider to move stuff from _models_old.py here
class SpectralModel(object):
    """Spectral model base class.

    Derived classes should store their parameters as ``Bunch`` in an instance
    attribute called ``parameters``, see for example
    `~gammapy.spectrum.models.PowerLaw`.
    """
    def __call__(self, energy):
        """Call evaluate method of derived classes"""
        return self.evaluate(energy, **self.parameters)

    def __str__(self):
        """String representation"""
        ss = self.__class__.__name__
        for parname, parval in self.parameters.items():
            ss += '\n{parname} : {parval:.3g}'.format(**locals())
        return ss

    def integral(self, emin, emax, **kwargs):
        """
        Integrate spectral model numerically.

        .. math::

            F(E_{min}, E_{max}) = \int_{E_{min}}^{E_{max}}\phi(E)dE

        Parameters
        ----------
        emin : float, `~astropy.units.Quantity`
            Lower bound of integration range.
        emax : float, `~astropy.units.Quantity`
            Upper bound of integration range
        """
        return integrate_spectrum(self, emin, emax, **kwargs)

    def energy_flux(self, emin, emax, **kwargs):
        """
        Compute energy flux in given energy range.

        .. math::

            G(E_{min}, E_{max}) = \int_{E_{min}}^{E_{max}}E \phi(E)dE

        Parameters
        ----------
        emin : float, `~astropy.units.Quantity`
            Lower bound of integration range.
        emax : float, `~astropy.units.Quantity`
            Upper bound of integration range
        """
        def f(x): return x * self(x)
        return integrate_spectrum(f, emin, emax, **kwargs)

    def to_dict(self):
        """Serialize to dict"""
        retval = dict()

        retval['name'] = self.__class__.__name__
        retval['parameters'] = list()
        for parname, parval in self.parameters.items():
            retval['parameters'].append(dict(name=parname,
                                             val=parval.value,
                                             unit=str(parval.unit)))
        return retval

    @classmethod
    def from_dict(cls, val):
        """Serialize from dict"""
        kwargs = dict()
        for _ in val['parameters']:
            kwargs[_['name']] = _['val'] * u.Unit(_['unit'])
        return cls(**kwargs)

    def plot(self, energy_range, ax=None,
             energy_unit='TeV', flux_unit='cm-2 s-1 TeV-1',
             energy_power=0, n_points=100, **kwargs):
        """Plot `~gammapy.spectrum.SpectralModel`

        kwargs are forwarded to :func:`~matplotlib.pyplot.errorbar`

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        energy_range : `~astropy.units.Quantity`
            Plot range
        energy_unit : str, `~astropy.units.Unit`, optional
            Unit of the energy axis
        flux_unit : str, `~astropy.units.Unit`, optional
            Unit of the flux axis
        energy_power : int, optional
            Power of energy to multiply flux axis with
        n_points : int, optional
            Number of evaluation nodes

        Returns
        -------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        """

        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax

        x_min = np.log10(energy_range[0].to('TeV').value)
        x_max = np.log10(energy_range[1].to('TeV').value)
        xx = np.logspace(x_min, x_max, n_points) * u.Unit('TeV')
        yy = self(xx)
        x = xx.to(energy_unit).value
        y = yy.to(flux_unit).value
        y = y * np.power(x, energy_power)
        flux_unit = u.Unit(flux_unit) * np.power(u.Unit(energy_unit), energy_power)
        ax.plot(x, y, **kwargs)
        ax.set_xlabel('Energy [{}]'.format(energy_unit))
        ax.set_ylabel('Flux [{}]'.format(flux_unit))
        ax.set_xscale("log", nonposx='clip')
        ax.set_yscale("log", nonposy='clip')
        return ax


class PowerLaw(SpectralModel):
    r"""Spectral power-law model.

    .. math::

        \phi(E) = \phi_0 \cdot \left( \frac{E}{E_0} \right)^{-\Gamma}

    Parameters
    ----------
    index : float, `~astropy.units.Quantity`
        :math:`\Gamma`
    amplitude : float, `~astropy.units.Quantity`
        :math:`Phi_0`
    reference : float, `~astropy.units.Quantity`
        :math:`E_0`
    """
    def __init__(self, index, amplitude, reference):
        self.parameters = Bunch(index=index,
                                amplitude=amplitude,
                                reference=reference)

    @staticmethod
    def evaluate(energy, index, amplitude, reference):
        return amplitude * (energy / reference) ** (-index)

    def integral(self, emin, emax):
        r"""
        Integrate power law analytically.

        .. math::

            F(E_{min}, E_{max}) = \int_{E_{min}}^{E_{max}}\phi(E)dE = \left.
            \phi_0 \frac{E_0}{-\Gamma + 1} \left( \frac{E}{E_0} \right)^{-\Gamma + 1}
            \right \vert _{E_{min}}^{E_{max}}


        """
        pars = self.parameters

        val = -1 * pars.index + 1
        prefactor = pars.amplitude * pars.reference / val
        upper = (emax / pars.reference) ** val
        lower = (emin / pars.reference) ** val
        return prefactor * (upper - lower)

    def energy_flux(self, emin, emax):
        r"""
        Compute energy flux in given energy range analytically.

        .. math::

            G(E_{min}, E_{max}) = \int_{E_{min}}^{E_{max}}E \phi(E)dE = \left.
            \phi_0 \frac{E_0^2}{-\Gamma + 2} \left( \frac{E}{E_0} \right)^{-\Gamma + 2}
            \right \vert _{E_{min}}^{E_{max}}


        Parameters
        ----------
        emin : float, `~astropy.units.Quantity`
            Lower bound of integration range.
        emax : float, `~astropy.units.Quantity`
            Upper bound of integration range
        """
        pars = self.parameters
        val = -1 * pars.index + 2

        prefactor = pars.amplitude * pars.reference ** 2 / val
        upper = (emax / pars.reference) ** val
        lower = (emin / pars.reference) ** val
        return prefactor * (upper - lower)

    def to_sherpa(self, name='default'):
        """Return `~sherpa.models.PowLaw1d`

        Parameters
        ----------
        name : str, optional
            Name of the sherpa model instance
        """
        import sherpa.models as m
        model = m.PowLaw1D('powlaw1d.' + name)
        model.gamma = self.parameters.index.value
        model.ref = self.parameters.reference.to('keV').value
        model.ampl = self.parameters.amplitude.to('cm-2 s-1 keV-1').value
        return model


class ExponentialCutoffPowerLaw(SpectralModel):
    r"""Spectral exponential cutoff power-law model.

    .. math::

        \phi(E) = \phi_0 \cdot \left(\frac{E}{E_0}\right)^{-\Gamma} \exp(-\lambda E)

    Parameters
    ----------
    index : float, `~astropy.units.Quantity`
        :math:`\Gamma`
    amplitude : float, `~astropy.units.Quantity`
        :math:`\phi_0`
    reference : float, `~astropy.units.Quantity`
        :math:`E_0`
    lambda : float, `~astropy.units.Quantity`
        :math:`\lambda`
    """
    def __init__(self, index, amplitude, reference, lambda_):
        self.parameters = Bunch(index=index,
                                amplitude=amplitude,
                                reference=reference,
                                lambda_=lambda_)

    @staticmethod
    def evaluate(energy, index, amplitude, reference, lambda_):
        pwl = amplitude * (energy / reference) ** (-index)
        try:
            cutoff = np.exp(-energy * lambda_)
        except AttributeError:
            from uncertainties.unumpy import exp
            cutoff = exp(-energy * lambda_)
        return pwl * cutoff
