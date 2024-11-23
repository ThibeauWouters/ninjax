import os
import json
import copy
import numpy as np
from astropy.time import Time
import inspect

from jimgw.single_event.waveform import Waveform, RippleTaylorF2, RippleIMRPhenomD_NRTidalv2, RippleIMRPhenomD_NRTidalv2_no_taper, RippleIMRPhenomD
from jimgw.jim import Jim
from jimgw.single_event.detector import Detector, H1, L1, V1, ET, TriangularNetwork2G
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.prior import *
from jimgw.base import LikelihoodBase

import ninjax.pipe_utils as utils
from ninjax.pipe_utils import logger
from ninjax.pipes.EOSPipe import EOSPipe, NEP_CONSTANTS_DICT, NEP_NAMES
from ninjax.parser import ConfigParser
from ninjax.likelihood import LikelihoodWithTransforms, GW_EOS_Likelihood
from ninjax import transforms

# TODO: can we make this more automated?
WAVEFORMS_DICT = {"TaylorF2": RippleTaylorF2, 
                  "IMRPhenomD_NRTidalv2": RippleIMRPhenomD_NRTidalv2,
                  "IMRPhenomD": RippleIMRPhenomD,
                  }
SUPPORTED_WAVEFORMS = list(WAVEFORMS_DICT.keys())
BNS_WAVEFORMS = ["IMRPhenomD_NRTidalv2", "TaylorF2"]

LIKELIHOODS_DICT = {"TransientLikelihoodFD": TransientLikelihoodFD, 
                    "HeterodynedTransientLikelihoodFD": HeterodynedTransientLikelihoodFD,
                    "GW_EOS_Likelihood": GW_EOS_Likelihood
                    }
GW_LIKELIHOODS = ["TransientLikelihoodFD", "HeterodynedTransientLikelihoodFD", "GW_EOS_Likelihood"]

class GWPipe:
    def __init__(self, 
                 config: dict, 
                 outdir: str, 
                 prior: Composite,
                 prior_bounds: np.array, 
                 seed: int,
                 transforms: list[Callable]):
        self.config = config
        self.outdir = outdir
        self.complete_prior = prior
        self.complete_prior_bounds = prior_bounds
        self.seed = seed
        self.transforms = transforms
        
        # Initialize other GW-specific attributes
        self.eos_file = self.set_eos_file()
        self.is_BNS_run = self.waveform_approximant in BNS_WAVEFORMS
        self.psds_dict = self.set_psds_dict()
        self.ifos = self.set_ifos()
        self.waveform = self.set_waveform()
        self.reference_waveform = self.set_reference_waveform()
        
        # TODO: data loading if preprocesse data is shared
        # Check if an injection and if has to be loaded, or if provided GW data must be loaded
        self.is_gw_injection = eval(self.config["gw_injection"])
        logger.info(f"GW run is an injection")
        if self.is_gw_injection:
            # TODO: should separate load existing injection from creating new one
            self.gw_injection = self.set_gw_injection()
            self.dump_gw_injection()
        else:
            self.set_gw_data_from_npz()
            # self.set_detector_info() # needed? Duration, epoch, gmst,...
            
    @property
    def fmin(self):
        return float(self.config["fmin"])

    @property
    def fmax(self):
        return float(self.config["fmax"])
    
    @property
    def fref(self):
        return float(self.config["fref"])
    
    @property
    def gw_load_existing_injection(self):
        return eval(self.config["gw_load_existing_injection"])

    @property
    def gw_SNR_threshold_low(self):
        return float(self.config["gw_SNR_threshold_low"])

    @property
    def gw_SNR_threshold_high(self):
        return float(self.config["gw_SNR_threshold_high"])
    
    @property
    def post_trigger_duration(self):
        return float(self.config["post_trigger_duration"])
    
    @property
    def trigger_time(self):
        return float(self.config["trigger_time"])

    @property
    def waveform_approximant(self):
        return self.config["waveform_approximant"]
    
    @property
    def psd_file_H1(self):
        return self.config["psd_file_H1"]

    @property
    def psd_file_L1(self):
        return self.config["psd_file_L1"]

    @property
    def psd_file_V1(self):
        return self.config["psd_file_V1"]

    @property
    def psd_file_ET1(self):
        return self.config["psd_file_ET1"]
    
    @property
    def psd_file_ET2(self):
        return self.config["psd_file_ET2"]
    
    @property
    def psd_file_ET3(self):
        return self.config["psd_file_ET3"]
    
    @property
    def relative_binning_binsize(self):
        return int(self.config["relative_binning_binsize"])
    
    @property
    def relative_binning_ref_params_equal_true_params(self):
        return eval(self.config["relative_binning_ref_params_equal_true_params"])
    
    def set_psds_dict(self) -> dict:
        psds_dict = {"H1": self.psd_file_H1,
                     "L1": self.psd_file_L1,
                     "V1": self.psd_file_V1,
                     "ET1": self.psd_file_ET1,
                     "ET2": self.psd_file_ET2,
                     "ET3": self.psd_file_ET3}
        return psds_dict
    
    def set_eos_file(self) -> str:
        """
        Check if an EOS file for the lambdas has been provided and if in correct format.
        Returns None if the provided file is not recognized.
        """
        
        # TODO: get full file path if needed
        eos_file = str(self.config["eos_file"])
        logger.info(f"eos_file is {eos_file}")
        if eos_file.lower() == "none" or len(eos_file) == 0:
            logger.info("No eos_file specified. Will sample lambdas uniformly.")
            return None
        else:
            self.check_valid_eos_file(eos_file)
            logger.info(f"Using eos_file {eos_file} for BNS injections")
            
        return eos_file
    
    def check_valid_eos_file(self, eos_file):
        """
        Check if the Lambdas EOS file has the right format, i.e. it should have "masses_EOS" and "Lambdas_EOS" keys.
        """
        if not os.path.exists(eos_file):
            raise ValueError(f"eos_file {eos_file} does not exist")
        if not eos_file.endswith(".npz"):
            raise ValueError("eos_file must be an npz file")
        data: dict = np.load(eos_file)
        keys = list(data.keys())
        if "masses_EOS" not in keys:
            raise ValueError("Key `masses_EOS` not found in eos_file")
        if "Lambdas_EOS" not in keys:
            raise ValueError("Key `Lambdas_EOS` not found in eos_file")
        
        return
        
    def set_gw_injection(self):
        """
        Function that creates a GW injection, taking into account the given priors and the SNR thresholds.
        If an existing injection.json exists, will load that one. 
        # TODO: do not hardcode injection.json, make more flexible

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        logger.info(f"Setting up GW injection . . . ")
        logger.info(f"The SNR thresholds are: {self.gw_SNR_threshold_low} - {self.gw_SNR_threshold_high}")
        pass_threshold = False
        config_duration = eval(self.config["duration"])
        
        sample_key = jax.random.PRNGKey(self.seed)
        while not pass_threshold:
            
            # Generate the parameters or load them from an existing file
            injection_path = os.path.join(self.outdir, "injection.json")
            if self.gw_load_existing_injection:
                logger.info(f"Loading existing injection, path: {injection_path}")
                injection = json.load(open(injection_path))
            else:
                logger.info(f"Generating new injection")
                sample_key, subkey = jax.random.split(sample_key)
                injection = utils.generate_injection(injection_path, self.complete_prior, subkey)
            
            # TODO: here is where we might have to transform from prior to ripple/jim parameters
            
            # If a BNS run, we can infer Lambdas from a given EOS if desired and override the parameters
            if self.is_BNS_run and self.eos_file is not None:
                logger.info(f"Computing lambdas from EOS file {self.eos_file} . . . ")
                injection = utils.inject_lambdas_from_eos(injection, self.eos_file)
            
            # Get duration based on Mc and fmin if not specified
            if config_duration is None:
                duration = utils.signal_duration(self.fmin, injection["M_c"])
                duration = 2 ** np.ceil(np.log2(duration))
                duration = float(duration)
                logger.info(f"Duration is not specified in the config. Computed chirp time: for fmin = {self.fmin} and M_c = {injection['M_c']} is {duration}")
            else:
                duration = config_duration
                logger.info(f"Duration is specified in the config: {duration}")
                
            self.duration = duration
                
            # Construct frequencies array
            self.frequencies = jnp.arange(
                self.fmin,
                self.fmax,
                1. / self.duration
            )
            
            # Make any necessary conversions
            # FIXME: hacky way for now
            try:
                injection = self.apply_transforms(injection)
            except Exception as e:
                logger.error(f"Error in applying transforms: {e}")
                # raise ValueError("Error in applying transforms")
            
            # Setup the timing setting for the injection
            self.epoch = self.duration - self.post_trigger_duration
            self.gmst = Time(self.trigger_time, format='gps').sidereal_time('apparent', 'greenwich').rad
            
            # Get the array of the injection parameters
            true_param = {key: float(injection[key]) for key in self.waveform.required_keys + ["t_c", "psi", "ra", "dec"]}
            
            logger.info(f"The trial injection parameters are {true_param}")
            
            self.detector_param = {
                'psi':    injection["psi"],
                't_c':    injection["t_c"],
                'ra':     injection["ra"],
                'dec':    injection["dec"],
                'epoch':  self.epoch,
                'gmst':   self.gmst,
                }
            
            # Generating the geocenter waveform
            logger.info("Injecting signals . . .")
            self.h_sky = self.waveform(self.frequencies, true_param)
            key = jax.random.PRNGKey(self.seed)
            logger.info("self.ifos")
            logger.info(self.ifos)
            for ifo in self.ifos:
                key, subkey = jax.random.split(key)
                ifo.inject_signal(
                    subkey,
                    self.frequencies,
                    self.h_sky,
                    self.detector_param,
                    psd_file=self.psds_dict[ifo.name]
                )
                
                # TODO: remove once tested
                logger.info(f"Signal injected in ifo {ifo.name}. Frequencies, data, and PSD:")
                logger.info(ifo.frequencies)
                logger.info(ifo.data)
                logger.info(ifo.psd)
            
            # Compute the SNRs, and save to a dict to be dumped later on
            snr_dict = {}
            for ifo in self.ifos:
                # if ifo.name == "ET":
                #     snr_dict["ET1_SNR"] = utils.compute_snr(ifo[0], self.h_sky, self.detector_param)
                #     snr_dict["ET2_SNR"] = utils.compute_snr(ifo[1], self.h_sky, self.detector_param)
                #     snr_dict["ET3_SNR"] = utils.compute_snr(ifo[2], self.h_sky, self.detector_param)
                # else:
                snr = utils.compute_snr(ifo, self.h_sky, self.detector_param)
                logger.info(f"SNR for ifo {ifo.name} is {snr}")
                snr_dict[f"{ifo.name}_SNR"] = snr
            
            snr_list = list(snr_dict.values())
            self.network_snr = float(jnp.sqrt(jnp.sum(jnp.array(snr_list) ** 2)))
            
            logger.info(f"The network SNR is {self.network_snr}")
            
            # If the SNR is too low, we need to generate new parameters
            pass_threshold = self.network_snr > self.gw_SNR_threshold_low and self.network_snr < self.gw_SNR_threshold_high
            if not pass_threshold:
                if self.gw_load_existing_injection:
                    raise ValueError("SNR does not pass threshold, but loading existing injection. This should not happen!")
                else:
                    logger.info("The network SNR does not pass the threshold, trying again")
                    
        logger.info(f"Network SNR passes threshold")
        injection.update(snr_dict)
        injection["network_SNR"] = self.network_snr
        
        # Also add detector etc info
        self.detector_param["duration"] = self.duration
        injection.update(self.detector_param)
        
        return injection
    
    def apply_transforms(self, params: dict):
        for transform in self.transforms:
            params = transform(params)
        # FIXME: this hard-coding is not so nice
        params["iota"] = params["iota"] % (2 * np.pi)
        params["dec"] = params["dec"] % (2 * np.pi)
        return params
    
    def dump_gw_injection(self):
        logger.info("Sanity checking the GW injection for ArrayImpl")
        for key, value in self.gw_injection.items():
            logger.info(f"   {key}: {value}")
        
        with open(os.path.join(self.outdir, "injection.json"), "w") as f:
            json.dump(self.gw_injection, f, indent=4, cls=utils.CustomJSONEncoder)
    
    def set_gw_data_from_npz(self):
        # FIXME: this has to be added in the future
        # Make sure the duration is set here as well
        raise NotImplementedError
    
    def dump_gw_data(self) -> None:
        # Dump the GW data
        for ifo in self.ifos:
            ifo_path = os.path.join(self.outdir, f"{ifo.name}.npz")
            np.savez(ifo_path, frequencies=ifo.frequencies, data=ifo.data, psd=ifo.psd)

    def set_waveform(self) -> Waveform:
        if self.waveform_approximant not in SUPPORTED_WAVEFORMS:
            raise ValueError(f"Waveform approximant {self.waveform_approximant} not supported. Supported waveforms are {SUPPORTED_WAVEFORMS}.")
        waveform_fn = WAVEFORMS_DICT[self.waveform_approximant]
        waveform = waveform_fn(f_ref = self.fref)
        return waveform

    def set_reference_waveform(self) -> Waveform:
        if self.waveform_approximant == "IMRPhenomD_NRTidalv2":
            logger.info("Using IMRPhenomD_NRTidalv2 waveform. Therefore, we will use no taper as the reference waveform for the likelihood if relative binning is used")
            reference_waveform = RippleIMRPhenomD_NRTidalv2_no_taper
        else:
            reference_waveform = WAVEFORMS_DICT[self.waveform_approximant]
        reference_waveform = reference_waveform(f_ref = self.fref)
        return reference_waveform
    
    def set_ifos(self) -> list[Detector]:
        # Go from string to list of ifos
        supported_ifos = ["H1", "L1", "V1", "ET"]
        self.ifos_str: list[str] = self.config["ifos"].split(",")
        self.ifos_str = [x.strip() for x in self.ifos_str]
        
        ifos: list[Detector] = []
        for single_ifo_str in self.ifos_str:
            if single_ifo_str not in supported_ifos:
                raise ValueError(f"IFO {single_ifo_str} not supported. Supported IFOs are {supported_ifos}.")
            new_ifo = eval(single_ifo_str)
            if isinstance(new_ifo, TriangularNetwork2G):
                ifos += new_ifo.ifos
            else:
                ifos.append(new_ifo)
        return ifos
    
    
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
        self.complete_prior_bounds = self.set_prior_bounds()
        logger.info("Finished prior setup")
        
        # If an EOS prior is specified, we need to set up an EOS prior
        if self.has_eos_priors():
            logger.info("The prior contains EOS parameters. Setting up EOS pipe now")
            # FIXME: pass fixed EOS params?
            self.eos_pipe = EOSPipe(self.config, self.complete_prior)
        
        # Set the transforms
        logger.info(f"Setting the transforms")
        self.transforms_str_list: str = self.set_transforms_str_list()
        self.transforms = self.set_transforms()
        
        # Finally, create the likelihood
        logger.info(f"Setting the likelihood")
        likelihood_str: str = self.config["likelihood"]
        self.check_valid_likelihood(likelihood_str)
        self.original_likelihood = self.set_original_likelihood(likelihood_str)
        logger.info(f"Original likelihood is set. Required keys are: {self.original_likelihood.required_keys}")
        
        self.likelihood = LikelihoodWithTransforms(self.original_likelihood, self.transforms)
        
        # TODO: check if the setup prior -> transform -> likelihood is OK
        logger.info(f"Required keys for the final likelihood: {self.likelihood.required_keys}")
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
                
                logger.info(f"   {stripped_line}")
                if stripped_line.startswith("#"):
                    continue
                exec(stripped_line)
                
                prior_name = stripped_line.split("=")[0].strip()
                prior_list.append(eval(prior_name))
        
        return Composite(prior_list)
    
    def set_prior_bounds(self):
        # TODO: generalize this: (i) only for GW relative binning, (ii) xmin xmax might fail for more advanced priors
        return jnp.array([[p.xmin, p.xmax] for p in self.complete_prior.priors])
    
    def has_eos_priors(self) -> bool:
        """
        Check if the prior containts prior over EOS parameters
        TODO: make this more robust?
        """
        
        # Check if there is an NEP in the naming
        naming = self.complete_prior.naming
        for name in naming:
            if name in NEP_NAMES or "n_CSE" in name or "cs2_CSE" in name:
                return True
        return False
    
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
                )
        
        elif likelihood_str == "GW_EOS_Likelihood":
            logger.info("Using GW with EOS likelihood. Initializing heterodyned GW likelihood")
            if self.gw_pipe.relative_binning_ref_params_equal_true_params:
                ref_params = self.gw_pipe.gw_injection
                logger.info("Using the true parameters as reference parameters for the relative binning")
            else:
                ref_params = None
                logger.info("Will search for reference waveform for relative binning")
            
            gw_likelihood = HeterodynedTransientLikelihoodFD(
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
                )
            
            likelihood = GW_EOS_Likelihood(gw_likelihood, self.eos_pipe)
        
        elif likelihood_str == "TransientLikelihoodFD":
            logger.info("Using GW TransientLikelihoodFD. Initializing likelihood")
            likelihood = TransientLikelihoodFD(
                self.gw_pipe.ifos,
                waveform=self.gw_pipe.waveform,
                trigger_time=self.gw_pipe.trigger_time,
                duration=self.gw_pipe.duration,
                post_trigger_duration=self.gw_pipe.post_trigger_duration,
                )
        
        return likelihood
    
    def check_prior_transforms_likelihood_setup(self):
        """Check if the setup between prior, transforms, and likelihood is correct by a small test."""
        logger.info("Checking the setup between prior, transforms, and likelihood")
        sample = self.complete_prior.sample(jax.random.PRNGKey(self.seed), 3)
        logger.info(f"sample: {sample}")
        sample_transformed = jax.vmap(self.likelihood.transform)(sample)
        logger.info(f"sample_transformed: {sample_transformed}")
        
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