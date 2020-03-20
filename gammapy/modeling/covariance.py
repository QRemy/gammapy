# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Covariance class"""
import numpy as np
import scipy
from astropy.table import Table, hstack
from .parameter import Parameters

__all__ = ["Covariance"]


class Covariance:
    """Parameter covariance class

    Parameters
    ----------
    parameters : `~gammapy.modeling.Parameters`
        Parameter list
    data : `~numpy.ndarray`
        Covariance data array

    """

    def __init__(self, parameters, data=None, filename=None):
        self.parameters = parameters
        self.filename = filename
        if data is None:
            data = np.diag([p.error ** 2 for p in self.parameters])

        self._data = np.asanyarray(data, dtype=float)

    @property
    def shape(self):
        """Covariance shape"""
        npars = len(self.parameters)
        return npars, npars

    @property
    def data(self):
        """Covariance data (`~numpy.ndarray`)"""
        return self._data

    @data.setter
    def data(self, value):
        value = np.asanyarray(value)

        npars = len(self.parameters)
        shape = (npars, npars)
        if value.shape != shape:
            raise ValueError(
                f"Invalid covariance shape: {value.shape}, expected {shape}"
            )

        self._data = value

    @staticmethod
    def _expand_factor_matrix(matrix, parameters):
        """Expand covariance matrix with zeros for frozen parameters"""
        npars = len(parameters)
        matrix_expanded = np.zeros((npars, npars))
        mask = np.array([par.frozen for par in parameters])
        free_parameters = ~(mask | mask[:, np.newaxis])
        matrix_expanded[free_parameters] = matrix.ravel()
        return matrix_expanded

    @classmethod
    def from_factor_matrix(cls, parameters, matrix):
        """Set covariance from factor covariance matrix.

        Used in the optimizer interface.
        """
        npars = len(parameters)

        if not matrix.shape == (npars, npars):
            matrix = cls._expand_factor_matrix(matrix, parameters)

        scales = [par.scale for par in parameters]
        scale_matrix = np.outer(scales, scales)
        data = scale_matrix * matrix

        return cls(parameters, data=data)

    @classmethod
    def from_stack(cls, covar_list):
        """Stack sub-covariance matrices from list

        Parameters
        ----------
        covar_list : list of `Covariance`
            List of sub-covariances

        Returns
        -------
        covar : `Covariance`
            Stacked covariance
        """
        parameters = Parameters.from_stack([_.parameters for _ in covar_list])

        covar = cls(parameters)

        for subcovar in covar_list:
            covar.set_subcovariance(subcovar)

        return covar

    @classmethod
    def read(cls, models, filename):
        """Read covariance data from file

        Parameters
        ----------
        parameters : `~gammapy.modeling.Parameters`
            Parameter list

        filename : str
            Filename
        """
        t = Table.read(filename, format="ascii.fixed_width")
        t.remove_column("Parameters")
        arr = np.array(t)
        data = arr.view(np.float).reshape(arr.shape + (-1,))
        return cls(models.parameters, data=data, filename=filename)

    def write(self, models, filename, **kwargs):
        """Write covariance to file

        Parameters
        ----------
        filename : str
            Filename
        **kwargs : dict
            Keyword arguments passed to `~astropy.table.Table.write`

        """
        from gammapy.modeling.models import SkyModel

        self.filename = filename
        param_names = []
        for model in models:
            suffix = model.name
            if isinstance(model, SkyModel):
                submodels = [
                    model.spectral_model,
                    model.spatial_model,
                    model.temporal_model,
                ]
                for m in submodels:
                    if m is not None:
                        for p in m.parameters:
                            suffix += "_" + m.__class__.__name__
                            param_names.append(p.name + "@" + suffix)
            else:
                for p in model.parameters:
                    suffix += "_" + model.__class__.__name__
                    param_names.append(p.name + "@" + suffix)
        # TODO: maybe nicer/shorter names can be found
        if len(param_names) != 0:
            t1 = Table()
            t1["Parameters"] = param_names
            t2 = Table(self.data, names=param_names)
            t = hstack([t1, t2])
            t.write(filename, format="ascii.fixed_width", delimiter="|")

    def get_subcovariance(self, parameters):
        """Get sub-covariance matrix

        Parameters
        ----------
        parameters : `Parameters`
            Sub list of parameters.

        Returns
        -------
        covariance : `~numpy.ndarray`
            Sub-covariance.
        """
        idx = [self.parameters.index(par) for par in parameters]
        data = self._data[np.ix_(idx, idx)]
        return self.__class__(parameters=parameters, data=data)

    def set_subcovariance(self, covar):
        """Set sub-covariance matrix

        Parameters
        ----------
        parameters : `Parameters`
            Sub list of parameters.

        """

        idx = [self.parameters.index(par) for par in covar.parameters]

        if not np.allclose(self.data[np.ix_(idx, idx)], covar.data):
            self.data[idx, :] = 0
            self.data[:, idx] = 0

        self._data[np.ix_(idx, idx)] = covar.data

    def plot_correlation(self, ax=None, **kwargs):
        """Plot correlation matrix.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis to plot on.
        **kwargs : dict
            Keyword arguments passed to `~gammapy.visualisation.plot_heatmap`

        Returns
        -------
        ax : `~matplotlib.axes.Axes`, optional
            Axis

        """
        import matplotlib.pyplot as plt
        from gammapy.visualization import plot_heatmap, annotate_heatmap

        npars = len(self.parameters)
        figsize = (npars * 0.8, npars * 0.65)

        plt.figure(figsize=figsize)

        ax = plt.gca() if ax is None else ax

        kwargs.setdefault("cmap", "coolwarm")

        names = self.parameters.names
        im, cbar = plot_heatmap(
            data=self.correlation,
            col_labels=names,
            row_labels=names,
            ax=ax,
            vmin=-1,
            vmax=1,
            cbarlabel="Correlation",
            **kwargs,
        )
        annotate_heatmap(im=im)
        return ax

    @property
    def correlation(self):
        r"""Correlation matrix (`numpy.ndarray`).

        Correlation :math:`C` is related to covariance :math:`\Sigma` via:

        .. math::
            C_{ij} = \frac{ \Sigma_{ij} }{ \sqrt{\Sigma_{ii} \Sigma_{jj}} }
        """
        err = np.sqrt(np.diag(self.data))

        with np.errstate(invalid="ignore", divide="ignore"):
            correlation = self.data / np.outer(err, err)

        return np.nan_to_num(correlation)

    @property
    def scipy_mvn(self):
        # TODO: use this, as in https://github.com/cdeil/multinorm/blob/master/multinorm.py
        return scipy.stats.multivariate_normal(
            self.parameters.values, self.data, allow_singular=True
        )

    def __str__(self):
        return str(self.data)

    def __array__(self):
        return self.data