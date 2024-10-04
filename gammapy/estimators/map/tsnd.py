# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from gammapy.maps import Map, MapAxes, MapAxis
from gammapy.utils.parallel import run_multiprocessing
from itertools import repeat
from .ts import TSMapEstimator
from gammapy.stats.utils import ts_to_sigma

__all__ = ["TSMapGridEstimator"]


class TSMapGridEstimator:
    """Compute test statistic maps from MapDataset for a grid of parameters

    Parameters
    ----------
    model : `~gammapy.modeling.model.SkyModel`
        Source model kernel.
    parameters : list of `~gammapy.modeling.Parameter` or `~gammapy.modeling.Parameters`
        Parameters of the `model` that have to be scanned.
        The scan values must be defined on each parameter via the `scan_values` attribute.
    tsmap_kwargs : dict, optional
        Arguments passed to `~gammapy.estimators.TSMapEstimator`.
        Default is None and the `TSMapEstimator` defaults are used.
    n_jobs : int, optional
        Number of processes used in parallel for the computation. Default is one,
        unless `~gammapy.utils.parallel.N_JOBS_DEFAULT` was modified. The number
        of jobs limited to the number of physical CPUs.
    parallel_backend : {"multiprocessing", "ray"}, optional
        Which backend to use for multiprocessing. Defaults to `~gammapy.utils.parallel.BACKEND_DEFAULT`.

    See also
    --------
    `~gammapy.estimators.TSMapEstimator`

    """

    def __init__(
        self, model, parameters, tsmap_kwargs=None, n_jobs=None, parallel_backend=None
    ):
        self.model = model
        self.parameters = parameters

        if tsmap_kwargs:
            tsmap_kwargs = {
                key: item for key, item in tsmap_kwargs.items() if key not in ["model"]
            }
        else:
            tsmap_kwargs = {}
        self.estimator = TSMapEstimator(
            model=model,
            **tsmap_kwargs,
        )

        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

    @property
    def indexes(self):
        """Indexing used to scan the parameter grid"""
        return list(
            zip(
                *[
                    _.flatten()
                    for _ in np.meshgrid(
                        *[
                            range(len(self.model.parameters[p].scan_values))
                            for p in self.parameters
                        ],
                        indexing="ij",
                    )
                ]
            )
        )

    def run(self, datasets):
        """
        Run test statistic map estimation.

        Requires a MapDataset with counts, exposure and background_model
        properly set to run.

        Notes
        -----
        The progress bar can be displayed for this function.

        Parameters
        ----------
        dataset : `~gammapy.datasets.Datasets` or `~gammapy.datasets.MapDataset`
            Map dataset or Datasets (list of MapDataset with the same spatial geometry).

        Returns
        -------
        results : list of `~gammapy.estimators.FluxMaps`
            List of `FluxMaps` returned by the estimator for each model scanned.

        """

        models = []
        for inds, pars in zip(self.indexes, repeat(self.parameters)):
            # TODO: Check if models and pars are in sync for multiprocessing if this is moved in _run_tsmap
            for par, idx in zip(pars, inds):
                par.value = par.scan_values[idx]
            models.append(self.model.copy())

        results = run_multiprocessing(
            self._run_tsmap,
            zip(
                models,
                repeat(self.estimator),
                repeat(datasets),
            ),
            backend=self.parallel_backend,
            pool_kwargs=dict(processes=self.n_jobs),
            task_name="TS maps grid",
        )
        return results

    @staticmethod
    def _run_tsmap(model, estimator, datasets):
        estimator.model = model
        return estimator.run(datasets)

    def to_maps(self, results):
        """
        Create a ND TS maps and maps of the optimal parameters


        Parameters
        ----------
        results : list
            List of flux maps returned by the `TSMapGridEstimator.run` method.

        Returns
        -------
        results : dict of `~gammapy.maps.Map`
            The dictionnary entries includes:
                * ts_scan : `~gammapy.maps.WcsNDMap` with one extra dimension for each paraemter scanned.
                * significance : Optimal significance considering a degree of freedom equal to the number of parameters scanned.
                * the name of each parameters scanned. Gives a map of the optimal parameter value.
                * the name of each `FluxMap` entry returned by the `TSMapEstimator`. Gives a map  with the value for the optimal model in each pixel.
        """

        geom = results[0].ts.geom
        parameters_axes = MapAxes(
            [
                MapAxis.from_nodes(p.scan_values, name=p.name, interp=p.interp)
                for p in self.parameters
            ]
        )
        geom_nd = geom.to_image().to_cube(
            [geom.axes["energy"]] + list(parameters_axes)[::-1]
        )

        maps = dict()
        ts_scan = Map.from_geom(geom_nd)
        for inds, result in zip(self.indexes, results):
            ts_scan.data[*inds, :, :, :] = result["ts"].data
        maps["ts_scan"] = ts_scan

        inds_best = [
            _.T for _ in _argmax_lastNaxes(ts_scan.data.T, len(self.parameters))
        ][::-1]
        ij, il, ik = np.indices(inds_best[0].shape)

        for name in self.estimator.selection_all:
            map_ = Map.from_geom(geom_nd)
            if "norm" in name:
                new_name = "dnde" if name == "norm" else "dnde_" + name.split("_")[1]
                factor = result.dnde_ref.value
                unit = result.dnde_ref.unit
            else:
                new_name = name
                factor = 1
                unit = result[name].unit
            for inds, result in zip(self.indexes, results):
                map_.data[*inds, :, :, :] = result[name].data * factor
            maps[new_name] = Map.from_geom(
                geom, data=map_.data[*inds_best, ij, il, ik], unit=unit
            )

        for inds, p in zip(inds_best, self.parameters):
            maps[p.name] = Map.from_geom(geom, data=p.scan_values[inds], unit=p.unit)

        sig = np.sign(maps["npred_excess"]) * ts_to_sigma(
            maps["ts"], df=len(self.parameters) + 1
        )
        maps["significance"] = Map.from_geom(geom, data=sig)
        return maps


def _argmax_lastNaxes(A, N):
    # https://stackoverflow.com/questions/30589211/numpy-argmax-over-multiple-axes-without-loop
    s = A.shape
    new_shp = s[:-N] + (np.prod(s[-N:]),)
    max_idx = A.reshape(new_shp).argmax(-1)
    return np.unravel_index(max_idx, s[-N:])
