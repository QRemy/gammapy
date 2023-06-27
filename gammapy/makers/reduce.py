import logging
from multiprocessing import Pool
import numpy as np
from astropy.coordinates import Angle
from gammapy.maps import Map
from gammapy.datasets import Datasets, MapDataset, MapDatasetOnOff, SpectrumDataset
from gammapy.modeling.models import Models
from .core import Maker
from .safe import SafeMaskMaker
import os

log = logging.getLogger(__name__)


__all__ = [
    "DatasetsMaker",
]

def make_dataset(makers, dataset, observation):
    """Make single dataset.

    Parameters
    ----------
    dataset : `~gammapy.datasets.MapDataset`
        Reference dataset
    observation : `Observation`
        Observation
    """
    log.info(f"Computing dataset for observation {observation.obs_id}")
    for maker in makers:
        log.info(f"Running {maker.tag}")
        dataset = maker.run(dataset=dataset, observation=observation)
    return dataset

class DatasetsMaker(Maker):
    """Run makers in a chain

    Parameters
    ----------
    makers : list of `Maker` objects
        Makers
    stack_datasets : bool
        If True stack into the reference dataset (see `run` method arguments).
    n_jobs : int
        Number of processes to run in parallel
    cutout_mode : {'trim', 'partial', 'strict'}
        Used only to cutout the refrence MapDataset around each processed observation.
        Mode is an option for Cutout2D, for details see `~astropy.nddata.utils.Cutout2D`.
        Default is "trim".
    cutout_width : tuple of `~astropy.coordinates.Angle`
        Angular sizes of the region in (lon, lat) in that specific order.
        If only one value is passed, a square region is extracted.
        If None it returns an error, except if the list of makers includes a `SafeMaskMaker`
        with the offset-max method defined. In that case it is set to two times `offset_max`.
    outdir : str
        If provided the individual datasets will be stored in the given directory. 
    read_only : bool
        Only read exiting datasets in `outdir` if exits otherwise perform data-reduction loop.
        True by default.
    """

    tag = "DatasetsMaker"

    def __init__(
        self,
        makers,
        stack_datasets=True,
        n_jobs=None,
        cutout_mode="trim",
        cutout_width=None,
        outdir=None,
        read_only=True,
        skip_missing=False,
    ):
        self.log = logging.getLogger(__name__)
        self.makers = makers
        self.cutout_mode = cutout_mode
        if cutout_width is not None:
            cutout_width = Angle(cutout_width)
        self.cutout_width = cutout_width
        self._apply_cutout = True
        if self.cutout_width is None:
            if self.offset_max is None:
                self._apply_cutout = False
            else:
                self.cutout_width = 2 * self.offset_max
        self.n_jobs = n_jobs
        self.stack_datasets = stack_datasets
        self.outdir = outdir
        self.read_only = read_only
        self.skip_missing = skip_missing
        self._datasets = []
        self._error = False

    @property
    def offset_max(self):
        maker = self.safe_mask_maker
        if maker is not None and hasattr(maker, "offset_max"):
            return maker.offset_max

    @property
    def safe_mask_maker(self):
        for m in self.makers:
            if isinstance(m, SafeMaskMaker):
                return m

    def read_dataset(self, observation):
        if self.outdir is not None:
            name = f"run_{observation.obs_id}"
            filename = f"{self.outdir}/{name}_dataset.fits"
            if os.path.isfile(filename) :
                dataset_obs = MapDataset.read(filename, name=name)
            else :
                return None
            # TODO: write/read datasets yaml instead
            # otherwise read works only for one datasest type
                
            filename = f"{self.outdir}/run_{observation.obs_id}_models.yaml"
            if self._dataset.models is not None and not self.read_only:
                #TODO cutout templates
                models = self._dataset.models.copy()
                models.reassign(self._dataset.name, dataset_obs.name)
            elif os.path.isfile(filename) and self.read_only:
                models = Models.read(filename)
            else:
                models=Models([])
            dataset_obs.models = models
            #print(models.names)
            return dataset_obs

    def prepare_dataset(self, dataset, observation):
        """Cutout dataset for a given observation.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Reference dataset
        observation : `Observation`
            Observation
        """
        
        if self._apply_cutout:
            cutouts_kwargs = {
                "position": observation.pointing_radec.galactic,
                "width": self.cutout_width,
                "mode": self.cutout_mode,
                "name": f"run_{observation.obs_id}",
            }
            dataset_obs = dataset.cutout(
                **cutouts_kwargs,
            )
        else:
            dataset_obs = dataset.copy(name=f"run_{observation.obs_id}")

        if dataset.models is not None:
            #TODO cutout templates
            models = dataset.models.copy()
            models.reassign(dataset.name, dataset_obs.name)
            dataset_obs.models = models
            #print(models.names)
        return dataset_obs


    def callback(self, dataset):
        isvalid = np.any(dataset.mask_safe.data)
        if not isvalid :
            print(f"Discard {dataset.name}, empty mask")
        elif self.outdir is not None and not self.read_only:
            filename = f"{self.outdir}/{dataset.name}_models_full.yaml"
            dataset.models.write(filename, overwrite=True)

            filename = f"{self.outdir}/{dataset.name}_models.yaml"
            bkg = Models([dataset.background_model])
            bkg.write(filename, overwrite=True, write_covariance=False)

            filename = f"{self.outdir}/{dataset.name}_dataset.fits"
            dataset.write(filename, overwrite=True)

        if isvalid and self.stack_datasets:
            if isinstance(self._dataset, MapDataset) and isinstance(
                dataset, MapDatasetOnOff
            ):
                dataset = dataset.to_map_dataset(dataset)
            self._dataset.stack(dataset)

    def error_callback(self, dataset):
        # parallel run could cause a memory error with non-explicit message.
        self._error = True


    def run(self, dataset, observations):
        """Run data reduction

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Reference dataset (used only for stacking if datasets are provided)
        observations : `Observations`
            Observations

        Returns
        -------
        datasets : `~gammapy.datasets.Datasets`
            Datasets

        """
        n_obs = len(observations)
        if isinstance(dataset, MapDataset):
            # also valid for Spectrum as it inherits from MapDataset
            self._dataset = dataset
        else:
            raise TypeError("Invalid reference dataset.")

        if isinstance(dataset, SpectrumDataset):
            self._apply_cutout = False

        if self.n_jobs is not None and self.n_jobs > 1:
            n_jobs = min(self.n_jobs, n_obs)
            ct = 0
            ct_total = 0
            with Pool(processes=n_jobs) as pool:
                log.info("Using {} jobs.".format(n_jobs))
                results = []
                for obs in observations:
                    base = self.read_dataset(obs)
                    if base is not None and self.read_only:
                        self.callback(base)
                    else:
                        try:
                            obs.bkg # FileNotFoundError ?
                        except:
                            continue
                        ct += 1
                        ct_total += 1
                        if base is None :
                            base = self.prepare_dataset(dataset, obs)
                            makers = self.makers
                        elif not self.read_only :
                            base.mask_safe = Map.from_geom(base.mask_safe.geom,data=True)
                            makers = [m for m in self.makers if m.tag in ["FoVBackgroundMaker","SafeMaskMaker"]]                       
                        result = pool.apply_async(
                            make_dataset,
                            (   
                                makers,
                                base,
                                obs,
                            ),
                            callback=self.callback,
                            error_callback=self.error_callback,
                        )
                        results.append(result)
                        # chunk wait async run is done
                        # use apply_async instead of starmap_async
                        # because the callback has to applied at each iteration
                        if ct==n_jobs or ct_total==n_obs:
                            [result.wait() for result in results]
                            results = []
                            ct = 0
            if self._error:
                raise RuntimeError("Execution of a sub-process failed")
        else:
            for obs in observations:

                base = self.read_dataset(obs)
                if base is not None and self.read_only:
                    self.callback(base)
                else:
                    try:
                        obs.bkg # FileNotFoundError ?
                    except:
                        continue
                    if base is None:
                        if self.skip_missing :
                            continue
                        else:
                            base = self.prepare_dataset(dataset, obs)
                            makers = self.makers
                    elif not self.read_only :
                        base.mask_safe = Map.from_geom(base.mask_safe.geom,data=True)
                        makers = [m for m in self.makers if m.tag in ["FoVBackgroundMaker","SafeMaskMaker"]]                       
                    result = make_dataset(makers, base, obs)
                    self.callback(result)

        if self.stack_datasets:
            return Datasets([self._dataset])
        else:
            return None

