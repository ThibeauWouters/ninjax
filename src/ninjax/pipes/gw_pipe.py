import os
import json
from typing import Callable
import numpy as np
from astropy.time import Time
import jax 
import jax.numpy as jnp

from jimgw.single_event.waveform import Waveform, RippleTaylorF2, RippleIMRPhenomD_NRTidalv2, RippleIMRPhenomD_NRTidalv2_no_taper, RippleIMRPhenomD
from jimgw.single_event.detector import Detector, TriangularNetwork2G, H1, L1, V1, ET
from jimgw.prior import Composite

import ninjax.pipes.pipe_utils as utils
from ninjax.pipes.pipe_utils import logger

WAVEFORMS_DICT = {"TaylorF2": RippleTaylorF2, 
                  "IMRPhenomD_NRTidalv2": RippleIMRPhenomD_NRTidalv2,
                  "IMRPhenomD": RippleIMRPhenomD,
                  }
SUPPORTED_WAVEFORMS = list(WAVEFORMS_DICT.keys())
BNS_WAVEFORMS = ["IMRPhenomD_NRTidalv2", "TaylorF2"]

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
    
    @property
    def kwargs(self) -> dict:
        _kwargs = eval(self.config["gw_kwargs"])
        if _kwargs is None:
            return {}
        return _kwargs
    
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
            # FIXME: hacky way for now --  if users specify iota in the injection, but sample over cos_iota and do the tfo, this breaks
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
    
    