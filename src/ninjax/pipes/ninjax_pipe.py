import os
import json
import numpy as np
from astropy.time import Time
import inspect
import time

from jimgw.single_event.waveform import Waveform, RippleTaylorF2, RippleIMRPhenomD_NRTidalv2, RippleIMRPhenomD_NRTidalv2_no_taper, RippleIMRPhenomD
from jimgw.jim import Jim
from jimgw.single_event.detector import Detector, TriangularNetwork2G, H1, L1, V1, ET
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.single_event.overlapping_likelihood import HeterodynedDoubleTransientLikelihoodFD, DoubleTransientLikelihoodFD
from jimgw.prior import *
from jimgw.base import LikelihoodBase

import ninjax.pipes.pipe_utils as utils
from ninjax.pipes.pipe_utils import logger
from ninjax.pipes.gw_pipe import GWPipe
from ninjax.parser import ConfigParser
from ninjax.likelihood import LikelihoodWithTransforms
from ninjax import transforms

# TODO: can we make this more automated?
LIKELIHOODS_DICT = {"TransientLikelihoodFD": TransientLikelihoodFD, 
                    "HeterodynedTransientLikelihoodFD": HeterodynedTransientLikelihoodFD,
                    "DoubleTransientLikelihoodFD": DoubleTransientLikelihoodFD, 
                    "HeterodynedDoubleTransientLikelihoodFD": HeterodynedDoubleTransientLikelihoodFD,
                    }
GW_LIKELIHOODS = ["TransientLikelihoodFD", "HeterodynedTransientLikelihoodFD", "DoubleTransientLikelihoodFD", "HeterodynedDoubleTransientLikelihoodFD"]


class NinjaxPipe(object):
    
    def __init__(self, outdir: str):
        """Loads the config file and sets up the JimPipe object."""
        
        # Check if the output directory is valid
        logger.info("Checking and setting outdir")
        if not self.check_valid_outdir(outdir):
            raise ValueError(f"Outdir {outdir} must exist and must contain 'config.ini' and 'prior.prior'")
        self.outdir = outdir
        
        logger.info("Loading the given config")
        self.config = self.load_config()
        self.config["outdir"] = self.outdir
        self.dump_complete_config()
        
        # Setting some of the hyperparameters and the setup
        self.seed = self.get_seed()
        self.sampling_seed = self.get_sampling_seed()
        self.run_sampler = eval(self.config["run_sampler"])
        self.flowmc_hyperparameters = self.set_flowmc_hyperparameters()
        
        logger.info("Loading the priors")
        self.complete_prior = self.set_prior()
        self.naming = self.complete_prior.naming
        self.n_dim = len(self.naming)
        # FIXME: this breaks for some priors
        # self.complete_prior_bounds = self.set_prior_bounds(), but it is only used for now in the heterodyned likelihoods, so we can skip it
        self.complete_prior_bounds = []
        logger.info("Finished prior setup")
        
        # Set the transforms
        logger.info(f"Setting the transforms")
        self.transforms_str_list: str = self.set_transforms_str_list()
        self.transforms = self.set_transforms()
        
        # Finally, create the likelihood
        logger.info(f"Setting the likelihood")
        likelihood_str: str = self.config["likelihood"]
        self.check_valid_likelihood(likelihood_str)
        self.original_likelihood = self.set_original_likelihood(likelihood_str)
        
        self.likelihood = LikelihoodWithTransforms(self.original_likelihood, self.transforms)
        
        # TODO: check if the setup prior -> transform -> likelihood is OK
        logger.info(f"Required keys for the likelihood: {self.likelihood.required_keys}")
        self.check_prior_transforms_likelihood_setup()

        # TODO: make the default keys to plot empty/None and use prior naming in that case
        logger.info(f"Will plot these keys: {self.keys_to_plot}")
        self.labels_to_plot = []
        recognized_labels = list(utils.LABELS_TRANSLATION_DICT.keys())
        for key in self.keys_to_plot:
            if key in recognized_labels:
                self.labels_to_plot.append(utils.LABELS_TRANSLATION_DICT[key])
            else:
                logger.info(f"Plot key {key} does not have a known LaTeX translation")
                self.labels_to_plot.append(key)
        logger.info(f"Will plot the labels: {self.labels_to_plot}")
        logger.info(f"Ninjax setup complete.")
        
    
    @property
    def outdir(self):
        return self._outdir
    
    @property
    def keys_to_plot(self):
        keys = self.config["keys_to_plot"]
        keys = keys.split(",")
        keys = [k.strip() for k in keys]
        return keys
    
    def check_valid_outdir(self, outdir: str) -> bool:
        """Check if the outdir exists and contains required files."""
        if not os.path.isdir(outdir):
            return False
        self.config_filename = os.path.join(outdir, "config.ini")
        self.prior_filename = os.path.join(outdir, "prior.prior")
        return all([os.path.isfile(self.config_filename), os.path.isfile(self.prior_filename)])
    
    @outdir.setter
    def outdir(self, outdir: str):
        logger.info(f"The outdir is set to {outdir}")
        self._outdir = outdir
        
    @property
    def nf_model_kwargs(self) -> dict:
        kwargs = eval(self.config["nf_model_kwargs"])
        if kwargs is None:
            return {}
        logger.info(f"Setting the NF model kwargs to {kwargs}")
        return kwargs
        
    def load_config(self) -> dict:
        """Set the configuration by parsing the user and default config files."""

        parser = ConfigParser()
        config_filename = os.path.join(self.outdir, "config.ini")
        user_config: dict = parser.parse(config_filename)
        
        # Parse the default config for non-specified keys
        default_config_filename = os.path.join(os.path.dirname(__file__), "default_config.ini")
        config: dict = parser.parse(default_config_filename)
        
        recognized_keys = set(config.keys())
        unrecognized_keys = set(user_config.keys()) - recognized_keys
        if len(unrecognized_keys) > 0:
            logger.warn(f"Unrecognized keys given: {unrecognized_keys}. These will be ignored")
        
        # Drop the unrecognized keys
        for key in unrecognized_keys:
            user_config.pop(key)
        
        config.update(user_config)
        logger.info(f"Arguments loaded into the config: {config}")
        
        return config
        
    def dump_complete_config(self):
        """Dumps the complete config after merging the user and the default settings to a JSON file"""
        complete_ini_filename = os.path.join(self.outdir, "complete_config.json")
        json.dump(self.config, open(complete_ini_filename, "w"), indent=4, cls=utils.CustomJSONEncoder)
        logger.info(f"Complete config file written to {os.path.abspath(complete_ini_filename)}")

    def set_prior(self) -> Composite:
        prior_list = []
        with open(self.prior_filename, "r") as f:
            for line in f:
                stripped_line = line.strip()
                
                if stripped_line == "":
                    logger.info("Encountered empty line in prior file, continue")
                    continue
                
                # Skip lines that are commented out
                if stripped_line.startswith("#"):
                    continue
                
                logger.info(f"   {stripped_line}")
                exec(stripped_line)
                
                prior_name = stripped_line.split("=")[0].strip()
                prior_list.append(eval(prior_name))
        
        return Composite(prior_list)
    
    def set_prior_bounds(self):
        # TODO: generalize this: (i) only for GW relative binning, (ii) xmin xmax might fail for more advanced priors
        return jnp.array([[p.xmin, p.xmax] for p in self.complete_prior.priors])
    
    
    def set_flowmc_hyperparameters(self) -> dict:
        hyperparameters = {
            "n_loop_training": int(self.config["n_loop_training"]),
            "n_loop_production": int(self.config["n_loop_production"]),
            "n_local_steps": int(self.config["n_local_steps"]),
            "n_global_steps": int(self.config["n_global_steps"]),
            "n_epochs": int(self.config["n_epochs"]),
            "n_chains": int(self.config["n_chains"]),
            "learning_rate": float(self.config["learning_rate"]),
            "max_samples": int(self.config["max_samples"]),
            "momentum": float(self.config["momentum"]),
            "batch_size": int(self.config["batch_size"]),
            "use_global": eval(self.config["use_global"]),
            "keep_quantile": float(self.config["keep_quantile"]),
            "train_thinning": int(self.config["train_thinning"]),
            "output_thinning": int(self.config["output_thinning"]),
            "n_sample_max": int(self.config["n_sample_max"]),
            "num_layers": int(self.config["num_layers"]),
            "hidden_size": [int(x) for x in self.config["hidden_size"].split(",")],
            "num_bins": int(self.config["num_bins"]),
            "save_training_chains": eval(self.config["save_training_chains"]),
            "eps_mass_matrix": float(self.config["eps_mass_matrix"]),
            "use_scheduler": eval(self.config["use_scheduler"]),
            "stopping_criterion_global_acc": float(self.config["stopping_criterion_global_acc"]),
            "nf_model_kwargs": self.nf_model_kwargs,
        }
        return hyperparameters
    
    def set_transforms_str_list(self):
        
        transforms_str_list = self.config["transforms"]
        if transforms_str_list is None or transforms_str_list == "None" or len(transforms_str_list) == 0:
            logger.info("No transforms provided in the config.ini")
            transforms_str_list = None
        else:
            transforms_str_list.strip()
            logger.info(f"Raw transforms list is {transforms_str_list}")
            transforms_str_list = transforms_str_list.split(",")
            logger.info(f"transforms_str_list has {len(transforms_str_list)} elements")
            
        return transforms_str_list
    
    def set_transforms(self) -> list[Callable]:
        all_transforms = dict(inspect.getmembers(transforms, inspect.isfunction))
        logger.info(f"DEBUG: Checking that all_transforms is OK: the list is {list(all_transforms.keys())}")
        
        transforms_list = [lambda x: x]
        # Check if the transforms are recognized
        for tfo_str in self.transforms_str_list:
            if tfo_str not in list(all_transforms.keys()):
                raise ValueError(f"Unrecognized transform is provided: {tfo_str}")
            
        transforms_list += [all_transforms[tfo_str] for tfo_str in self.transforms_str_list]
        
        return transforms_list
            
    
    @staticmethod
    def check_valid_likelihood(likelihood_str) -> None:
        if likelihood_str not in LIKELIHOODS_DICT:
            raise ValueError(f"Likelihood {likelihood_str} not supported. Supported likelihoods are {list(LIKELIHOODS_DICT.keys())}.")

    def set_original_likelihood(self, likelihood_str: str) -> LikelihoodBase:
        """Create the likelihood object depending on the given likelihood string."""
        
        # Set up everything needed for GW likelihood
        if likelihood_str in GW_LIKELIHOODS:
            logger.info("GW likelihood provided, setting up the GW pipe")
            # TODO: this is becoming quite cumbersome... perhaps there is a better way to achieve this?
            self.config["gw_is_overlapping"] = likelihood_str in ["DoubleTransientLikelihoodFD", "HeterodynedDoubleTransientLikelihoodFD"]
            self.gw_pipe = GWPipe(self.config, self.outdir, self.complete_prior, self.complete_prior_bounds, self.seed, self.transforms)
            
        # Create the likelihood
        if likelihood_str == "HeterodynedTransientLikelihoodFD":
            logger.info("Using GW HeterodynedTransientLikelihoodFD. Initializing likelihood")
            if self.gw_pipe.relative_binning_ref_params_equal_true_params:
                ref_params = self.gw_pipe.gw_injection
                logger.info("Using the true parameters as reference parameters for the relative binning")
            else:
                ref_params = None
                logger.info("Will search for reference waveform for relative binning")
            
            logger.info(f"Using the following kwargs for the GW likelihood: {self.gw_pipe.kwargs}")
            
            init_heterodyned_start = time.time()
            likelihood = HeterodynedTransientLikelihoodFD(
                self.gw_pipe.ifos,
                prior=self.complete_prior,
                bounds=self.complete_prior_bounds, 
                n_bins = self.gw_pipe.relative_binning_binsize,
                waveform=self.gw_pipe.waveform,
                reference_waveform=self.gw_pipe.reference_waveform,
                trigger_time=self.gw_pipe.trigger_time,
                duration=self.gw_pipe.duration,
                post_trigger_duration=self.gw_pipe.post_trigger_duration,
                ref_params=ref_params,
                **self.gw_pipe.kwargs
                )
            init_heterodyned_end = time.time()
            
            logger.info(f"Initialization of HeterodynedTransientLikelihoodFD took {init_heterodyned_end - init_heterodyned_start} seconds = {(init_heterodyned_end - init_heterodyned_start) / 60} minutes")
        
            print(likelihood.required_keys)
        
        elif likelihood_str == "TransientLikelihoodFD":
            
            logger.info(f"Using the following kwargs for the GW likelihood: {self.gw_pipe.kwargs}")
            
            logger.info("Using GW TransientLikelihoodFD. Initializing likelihood")
            likelihood = TransientLikelihoodFD(
                self.gw_pipe.ifos,
                waveform=self.gw_pipe.waveform,
                trigger_time=self.gw_pipe.trigger_time,
                duration=self.gw_pipe.duration,
                post_trigger_duration=self.gw_pipe.post_trigger_duration,
                **self.gw_pipe.kwargs
                )
            print(likelihood.required_keys)
        
        elif likelihood_str == "DoubleTransientLikelihoodFD":
            
            logger.info(f"Using the following kwargs for the GW likelihood: {self.gw_pipe.kwargs}")
            
            logger.info("Using GW TransientLikelihoodFD. Initializing likelihood")
            likelihood = DoubleTransientLikelihoodFD(
                self.gw_pipe.ifos,
                waveform=self.gw_pipe.waveform,
                trigger_time=self.gw_pipe.trigger_time,
                duration=self.gw_pipe.duration,
                post_trigger_duration=self.gw_pipe.post_trigger_duration,
                **self.gw_pipe.kwargs
                )
            print(likelihood.required_keys)
        
        elif likelihood_str == "HeterodynedDoubleTransientLikelihoodFD":
            if self.gw_pipe.relative_binning_ref_params_equal_true_params:
                ref_params = self.gw_pipe.gw_injection
                logger.info("Using the true parameters as reference parameters for the relative binning")
            else:
                ref_params = None
                logger.info("Will search for reference waveform for relative binning")
            
            logger.info(f"Using the following kwargs for the GW likelihood: {self.gw_pipe.kwargs}")
            
            logger.info("Using GW TransientLikelihoodFD. Initializing likelihood")
            init_heterodyned_start = time.time()
            likelihood = HeterodynedDoubleTransientLikelihoodFD(
                self.gw_pipe.ifos,
                prior=self.complete_prior,
                bounds=self.complete_prior_bounds, 
                n_bins = self.gw_pipe.relative_binning_binsize,
                waveform=self.gw_pipe.waveform,
                reference_waveform=self.gw_pipe.reference_waveform,
                trigger_time=self.gw_pipe.trigger_time,
                duration=self.gw_pipe.duration,
                post_trigger_duration=self.gw_pipe.post_trigger_duration,
                ref_params=ref_params,
                **self.gw_pipe.kwargs
                )
            print(likelihood.required_keys)
            init_heterodyned_end = time.time()
            
            logger.info(f"Initialization of HeterodynedTransientLikelihoodFD took around {int((init_heterodyned_end - init_heterodyned_start) / 60)} minutes")
        
        return likelihood
    
    def check_prior_transforms_likelihood_setup(self):
        """Check if the setup between prior, transforms, and likelihood is correct by a small test."""
        logger.info("Checking the setup between prior, transforms, and likelihood")
        sample = self.complete_prior.sample(jax.random.PRNGKey(self.seed), 3)
        logger.info(f"sample: {sample}")
        # sample_transformed = jax.vmap(self.likelihood.transform)(sample)
        # logger.info(f"sample_transformed: {sample_transformed}")
        
        # TODO: what if we actually need to give data instead of nothing?
        log_prob = jax.vmap(self.likelihood.evaluate)(sample, {})
        if jnp.isnan(log_prob).any():
            raise ValueError("Log probability is NaN. Something is wrong with the setup!")
        logger.info(f"log_prob: {log_prob}")

    def get_seed(self):
        if isinstance(self.config["seed"], int):
            return self.config["seed"]
        seed = eval(self.config["seed"])
        if seed is None:
            seed = np.random.randint(0, 999999)
            logger.info(f"No seed specified. Generating a random seed: {seed}")
        self.config["seed"] = seed
        return seed

    def get_sampling_seed(self):
        if isinstance(self.config["sampling_seed"], int):
            return self.config["sampling_seed"]
        sampling_seed = eval(self.config["sampling_seed"])
        if sampling_seed is None:
            sampling_seed = np.random.randint(0, 999999)
            logger.info(f"No sampling_seed specified. Generating a random sampling_seed: {sampling_seed}")
        self.config["sampling_seed"] = sampling_seed
        return sampling_seed