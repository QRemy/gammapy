# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import ray
import collections.abc
import copy
import numpy as np
import logging
import inspect
from astropy.table import vstack, Table
from astropy import units as u
from gammapy.modeling.models import Models, ProperModels, BackgroundModel
from gammapy.utils.scripts import make_name, make_path, read_yaml, write_yaml
from gammapy.utils.table import table_from_row_data
from gammapy.data import GTI


log = logging.getLogger(__name__)


__all__ = ["Dataset", "Datasets", "DatasetsActor"]


class Dataset(abc.ABC):
    """Dataset abstract base class.

    TODO: add tutorial how to create your own dataset types.

    For now, see existing examples in Gammapy how this works:

    - `gammapy.datasets.MapDataset`
    - `gammapy.datasets.SpectrumDataset`
    - `gammapy.datasets.FluxPointsDataset`
    """

    _residuals_labels = {
        "diff": "data - model",
        "diff/model": "(data - model) / model",
        "diff/sqrt(model)": "(data - model) / sqrt(model)",
    }

    @property
    @abc.abstractmethod
    def tag(self):
        pass

    @property
    def mask(self):
        """Combined fit and safe mask"""
        if self.mask_safe is not None and self.mask_fit is not None:
            return self.mask_safe & self.mask_fit
        elif self.mask_fit is not None:
            return self.mask_fit
        elif self.mask_safe is not None:
            return self.mask_safe

    def stat_sum(self):
        """Total statistic given the current model parameters."""
        stat = self.stat_array()

        if self.mask is not None:
            stat = stat[self.mask.data]

        return np.sum(stat, dtype=np.float64)

    @abc.abstractmethod
    def stat_array(self):
        """Statistic array, one value per data point."""

    def copy(self, name=None):
        """A deep copy."""
        new = copy.deepcopy(self)
        name = make_name(name)
        new._name = name
        # propagate new dataset name
        if new._models is not None:
            for m in new._models:
                if m.datasets_names is not None:
                    for k, d in enumerate(m.datasets_names):
                        if d == self.name:
                            m.datasets_names[k] = name
                    if hasattr(new, "background_model") and m == new.background_model:
                        m._name = name + "-bkg"
        return new

    @staticmethod
    def _compute_residuals(data, model, method="diff"):
        with np.errstate(invalid="ignore"):
            if method == "diff":
                residuals = data - model
            elif method == "diff/model":
                residuals = (data - model) / model
            elif method == "diff/sqrt(model)":
                residuals = (data - model) / np.sqrt(model)
            else:
                raise AttributeError(
                    f"Invalid method: {method!r}. Choose between 'diff',"
                    " 'diff/model' and 'diff/sqrt(model)'"
                )
        return residuals


class Datasets(collections.abc.MutableSequence):
    """Dataset collection.

    Parameters
    ----------
    datasets : `Dataset` or list of `Dataset`
        Datasets
    """

    def __init__(self, datasets=None):
        if datasets is None:
            datasets = []

        if isinstance(datasets, Datasets):
            datasets = datasets._datasets
        elif isinstance(datasets, Dataset):
            datasets = [datasets]
        elif not isinstance(datasets, list):
            raise TypeError(f"Invalid type: {datasets!r}")

        unique_names = []
        for dataset in datasets:
            if dataset.name in unique_names:
                raise (ValueError("Dataset names must be unique"))
            unique_names.append(dataset.name)

        self._datasets = datasets

    @property
    def parameters(self):
        """Unique parameters (`~gammapy.modeling.Parameters`).

        Duplicate parameter objects have been removed.
        The order of the unique parameters remains.
        """
        return self.models.parameters.unique_parameters

    @property
    def models(self):
        """Unique models (`~gammapy.modeling.Models`).

        Duplicate model objects have been removed.
        The order of the unique models remains.
        """
        return ProperModels(self)

    @property
    def names(self):
        return [d.name for d in self._datasets]

    @property
    def is_all_same_type(self):
        """Whether all contained datasets are of the same type"""
        return len(set(_.__class__ for _ in self)) == 1

    @property
    def is_all_same_shape(self):
        """Whether all contained datasets have the same data shape"""
        return len(set(_.data_shape for _ in self)) == 1

    @property
    def is_all_same_energy_shape(self):
        """Whether all contained datasets have the same data shape"""
        return len(set(_.data_shape[0] for _ in self)) == 1

    @property
    def energy_axes_are_aligned(self):
        """Whether all contained datasets have aligned energy axis"""
        axes = [d.counts.geom.axes["energy"] for d in self]
        return np.all([axes[0].is_aligned(ax) for ax in axes])

    def stat_sum(self):
        """Compute joint likelihood"""
        stat_sum = 0
        # TODO: add parallel evaluation of likelihoods
        for dataset in self:
            stat_sum += dataset.stat_sum()
        return stat_sum

    def select_time(self, t_min, t_max, atol="1e-6 s"):
        """Select datasets in a given time interval.

        Parameters
        ----------
        t_min, t_max : `~astropy.time.Time`
            Time interval
        atol : `~astropy.units.Quantity`
            Tolerance value for time comparison with different scale. Default 1e-6 sec.

        Returns
        -------
        datasets : `Datasets`
            Datasets in the given time interval.

        """
        atol = u.Quantity(atol)

        datasets = []

        for dataset in self:
            t_start = dataset.gti.time_start[0]
            t_stop = dataset.gti.time_stop[-1]

            if t_start >= (t_min - atol) and t_stop <= (t_max + atol):
                datasets.append(dataset)

        return self.__class__(datasets)

    def slice_energy(self, e_min, e_max):
        """Select and slice datasets in energy range

        Parameters
        ----------
        e_min, e_max : `~astropy.units.Quantity`
            Energy bounds to compute the flux point for.

        Returns
        -------
        datasets : Datasets
            Datasets

        """
        datasets = []

        for dataset in self:
            # TODO: implement slice_by_coord() and simplify?
            energy_axis = dataset.counts.geom.axes["energy"]
            try:
                group = energy_axis.group_table(edges=[e_min, e_max])
            except ValueError:
                log.info(
                    f"Dataset {dataset.name} does not contribute in the energy range"
                )
                continue

            is_normal = group["bin_type"] == "normal   "
            group = group[is_normal]

            slices = {
                "energy": slice(int(group["idx_min"][0]), int(group["idx_max"][0]) + 1)
            }

            name = f"{dataset.name}-{e_min:.3f}-{e_max:.3f}"
            dataset_sliced = dataset.slice_by_idx(slices, name=name)

            # TODO: Simplify model handling!!!!
            models = []

            for model in dataset.models:
                if isinstance(model, BackgroundModel):
                    models.append(dataset_sliced.background_model)
                else:
                    models.append(model)

            dataset_sliced.models = models
            datasets.append(dataset_sliced)

        return self.__class__(datasets=datasets)

    @property
    # TODO: make this a method to support different methods?
    def energy_ranges(self):
        """Get global energy range of datasets.

        The energy range is derived as the minimum / maximum of the energy
        ranges of all datasets.

        Returns
        -------
        e_min, e_max : `~astropy.units.Quantity`
            Energy range.
        """

        e_mins, e_maxs = [], []

        for dataset in self:
            energy_axis = dataset.counts.geom.axes["energy"]
            e_mins.append(energy_axis.edges[0])
            e_maxs.append(energy_axis.edges[-1])

        return u.Quantity(e_mins), u.Quantity(e_maxs)

    def __str__(self):
        str_ = self.__class__.__name__ + "\n"
        str_ += "--------\n\n"

        for idx, dataset in enumerate(self):
            str_ += f"Dataset {idx}: \n\n"
            str_ += f"\tType       : {dataset.tag}\n"
            str_ += f"\tName       : {dataset.name}\n"
            try:
                instrument = set(dataset.meta_table["TELESCOP"]).pop()
            except (KeyError, TypeError):
                instrument = ""
            str_ += f"\tInstrument : {instrument}\n\n"

        return str_.expandtabs(tabsize=2)

    def copy(self):
        """A deep copy."""
        return copy.deepcopy(self)

    @classmethod
    def read(
        cls,
        path,
        filedata="_datasets.yaml",
        filemodel="_models.yaml",
        lazy=True,
        cache=True,
    ):
        """De-serialize datasets from YAML and FITS files.

        Parameters
        ----------
        path : str, Path
            Base directory of the datasets files.
        filedata : str
            file path or name of yaml datasets file
        filemodel : str
            file path or name of yaml models file
        lazy : bool
            Whether to lazy load data into memory
        cache : bool
            Whether to cache the data after loading.


        Returns
        -------
        dataset : `gammapy.datasets.Datasets`
            Datasets
        """
        from . import DATASET_REGISTRY

        path = make_path(path)

        if (path / filedata).exists():
            filedata = path / filedata
        else:
            filedata = make_path(filedata)
        if (path / filemodel).exists():
            filemodel = path / filemodel
        else:
            filemodel = make_path(filemodel)

        models = Models.read(filemodel)
        data_list = read_yaml(filedata)

        datasets = []
        for data in data_list["datasets"]:
            if (path / data["filename"]).exists():
                data["filename"] = str(make_path(path / data["filename"]))

            dataset_cls = DATASET_REGISTRY.get_cls(data["type"])
            dataset = dataset_cls.from_dict(data, models, lazy=lazy, cache=cache)
            datasets.append(dataset)
        return cls(datasets)

    def write(self, path, prefix="", overwrite=False, write_covariance=True):
        """Serialize datasets to YAML and FITS files.

        Parameters
        ----------
        path : `pathlib.Path`
            path to write files
        prefix : str
            common prefix of file names
        overwrite : bool
            overwrite datasets FITS files
        write_covariance : bool
            save covariance or not
        """

        path = make_path(path).resolve()
        datasets_dictlist = []
        for dataset in self._datasets:
            filename = f"{prefix}_data_{dataset.name}.fits"
            dataset.write(path / filename, overwrite)
            datasets_dictlist.append(dataset.to_dict(filename=filename))
        datasets_dict = {"datasets": datasets_dictlist}

        write_yaml(datasets_dict, path / f"{prefix}_datasets.yaml", sort_keys=False)
        self.models.write(
            path / f"{prefix}_models.yaml",
            overwrite=overwrite,
            write_covariance=write_covariance,
        )

    def stack_reduce(self, name=None):
        """Reduce the Datasets to a unique Dataset by stacking them together.

        This works only if all Dataset are of the same type and if a proper
        in-place stack method exists for the Dataset type.

        Returns
        -------
        dataset : ~gammapy.utils.Dataset
            the stacked dataset
        """
        if not self.is_all_same_type:
            raise ValueError(
                "Stacking impossible: all Datasets contained are not of a unique type."
            )

        dataset = self[0].copy(name=name)
        for ds in self[1:]:
            dataset.stack(ds)
        return dataset

    def info_table(self, cumulative=False, region=None):
        """Get info table for datasets.

        Parameters
        ----------
        cumulative : bool
            Cumulate info across all observations

        Returns
        -------
        info_table : `~astropy.table.Table`
            Info table.
        """
        if not self.is_all_same_type:
            raise ValueError("Info table not supported for mixed dataset type.")

        stacked = self[0].copy(name=self[0].name)

        rows = [stacked.info_dict()]

        for dataset in self[1:]:
            if cumulative:
                stacked.stack(dataset)
                row = stacked.info_dict()
            else:
                row = dataset.info_dict()

            rows.append(row)

        return table_from_row_data(rows=rows)

    # TODO: merge with meta table?
    @property
    def gti(self):
        """GTI table"""
        time_intervals = []

        for dataset in self:
            if dataset.gti is not None:
                interval = (dataset.gti.time_start[0], dataset.gti.time_stop[-1])
                time_intervals.append(interval)

        return GTI.from_time_intervals(time_intervals)

    @property
    def meta_table(self):
        """Meta table"""
        tables = [d.meta_table for d in self]

        if np.all([table is None for table in tables]):
            meta_table = Table()
        else:
            meta_table = vstack(tables)

        meta_table.add_column([d.tag for d in self], index=0, name="TYPE")
        meta_table.add_column(self.names, index=0, name="NAME")
        return meta_table

    def __getitem__(self, key):
        return self._datasets[self.index(key)]

    def __delitem__(self, key):
        del self._datasets[self.index(key)]

    def __setitem__(self, key, dataset):
        if isinstance(dataset, Dataset):
            if dataset.name in self.names:
                raise (ValueError("Dataset names must be unique"))
            self._datasets[self.index(key)] = dataset
        else:
            raise TypeError(f"Invalid type: {type(dataset)!r}")

    def insert(self, idx, dataset):
        if isinstance(dataset, Dataset):
            if dataset.name in self.names:
                raise (ValueError("Dataset names must be unique"))
            self._datasets.insert(idx, dataset)
        else:
            raise TypeError(f"Invalid type: {type(dataset)!r}")

    def index(self, key):
        if isinstance(key, (int, slice)):
            return key
        elif isinstance(key, str):
            return self.names.index(key)
        elif isinstance(key, Dataset):
            return self._datasets.index(key)
        else:
            raise TypeError(f"Invalid type: {type(key)!r}")

    def __len__(self):
        return len(self._datasets)


class DatasetsActor(Datasets):
    """A modified Dataset collection for parallel evaluation using ray actors.
    Fore now only available if composed only of MapDataset.

    Parameters
    ----------
    datasets : `Datasets`
        Datasets
    """

    def __init__(self, datasets=None):
        from .map import MapDatasetActor

        if datasets is not None:
            self._datasets = datasets._datasets
            self._actors = [MapDatasetActor.remote(d) for d in self._datasets]
            self._copy_cavariance_data()

    @classmethod
    def from_actors(cls, actors):
        """create from previously defined actors
        This will work only if models are unique to their datasets,
        for example using actors obtain after data reduction with only background
        """
        from .map import MapDataset

        da = cls()
        da._actors = actors
        names = ray.get([a.get_attr.remote("name") for a in actors])
        # create ghost datasets linked to actors by name
        da._datasets = [MapDataset(name=name) for name in names]
        # copy model to create global model
        a_models = ray.get([a.get_models.remote() for a in actors])
        for d, models in zip(da._datasets, a_models):
            d.models = Models(models)
        da._copy_cavariance_data() 
        return da


    def __getattr__(self, attr):
        def wrapper(update_remote=False, **kwargs):
            if update_remote:
                self._update_remote_models()
            results = ray.get([a.get_attr.remote(attr) for a in self._actors])
            for res in results:
                if inspect.ismethod(res):
                    res = res(**kwargs) #no longer parallel but works with plots
            return results
        return wrapper

    def _copy_cavariance_data(self):
        # TODO: this avoid  ValueError: assignment destination is read-only in set_subcovariance
        # self._data[np.ix_(idx, idx)] = covar.data.copy()
        # better way/place to do that ?
        for m in self.models:
            m._covariance.data = m._covariance._data.copy()

    def _update_remote_models(self):
        args = [list(d.models) for d in self._datasets]
        ray.get([a.set_models.remote(arg) for a, arg in zip(self._actors, args)])

    def stat_sum(self):
        """Compute joint likelihood"""
        args = [d.models.parameters.get_parameter_values() for d in self._datasets]
        ray.get(
            [a.set_parameter_values.remote(arg) for a, arg in zip(self._actors, args)]
        )
        # blocked until set_parameters_factors on actors complete
        res = ray.get([a.stat_sum.remote() for a in self._actors])
        return np.sum(res)

