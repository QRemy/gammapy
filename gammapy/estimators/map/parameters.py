# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from gammapy.maps import Map, MapAxes, MapAxis
from gammapy.utils.parallel import run_multiprocessing
from itertools import repeat
from .ts import TSMapEstimator


class ParameterMapEstimator:
    def __init__(
        self, model, parameters, tsmap_kwargs=None, n_jobs=None, parallel_backend=None
    ):
        self.model = model
        self.parameters = parameters
        self.indexes = list(
            zip(
                *[
                    _.flatten()
                    for _ in np.meshgrid(
                        *[
                            range(len(model.parameters[p].scan_values))
                            for p in parameters
                        ],
                        indexing="ij",
                    )
                ]
            )
        )

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

    def run(self, datasets):
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
            task_name="Parameter maps",
        )
        return self._to_maps(results)

    @staticmethod
    def _run_tsmap(model, estimator, datasets):
        estimator.model = model
        return estimator.run(datasets)

    def _to_maps(self, results):
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
            _.T for _ in argmax_lastNaxes(ts_scan.data.T, len(self.parameters))
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
        return maps


def argmax_lastNaxes(A, N):
    # https://stackoverflow.com/questions/30589211/numpy-argmax-over-multiple-axes-without-loop
    s = A.shape
    new_shp = s[:-N] + (np.prod(s[-N:]),)
    max_idx = A.reshape(new_shp).argmax(-1)
    return np.unravel_index(max_idx, s[-N:])
