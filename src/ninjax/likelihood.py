"""Gather all likelihoods that can be used by a ninjax program"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import Callable

from jimgw.base import LikelihoodBase
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD

from ninjax.pipes.EOSPipe import NEP_NAMES, EOSPipe
from ninjax.pipe_utils import logger
from ninjax.transforms import *

class ZeroLikelihood(LikelihoodBase):
    """Empty likelihood that constantly returns 0.0"""
    def __init__(self):
        
        # TODO: remove transform input?
        
        super().__init__()
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        return 0.0

class LikelihoodWithTransforms(LikelihoodBase):
    """Call an original likelihood but with some transforms applied to the parameters before evaluate"""
    def __init__(self, 
                 likelihood: LikelihoodBase, 
                 transforms: list[Callable]):
        self.likelihood = likelihood
        # TODO: if likelihood itself has a transform, we should add it here
        self.transforms = transforms
        if hasattr(likelihood, "transform"):
            logger.info("We are adding the likelihood's transform to the list of transforms")
            self.transforms.append(likelihood.transform)
        self.required_keys = likelihood.required_keys
        
    def transform(self, params: dict[str, Float]) -> dict[str, Float]:
        for transform in self.transforms:
            params = transform(params)
        return params
        
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        # We make a safe copy of the params to avoid modifying the original
        inner_params = {}
        inner_params.update(params)
        inner_params = self.transform(inner_params)
        return self.likelihood.evaluate(inner_params, data)
    
    
class GW_EOS_Likelihood:
    
    def __init__(self, 
                 gw_likelihood: LikelihoodBase,
                 eos_pipe: EOSPipe):
        self.gw_likelihood = gw_likelihood
        self.eos_pipe = eos_pipe
        self.required_keys = gw_likelihood.required_keys + eos_pipe.naming
        
        # Remove the lambdas, since these will be provided by the EOS pipe
        if "lambda_1" in self.required_keys and "lambda_2" in self.required_keys:
            self.required_keys.remove("lambda_1")
            self.required_keys.remove("lambda_2")
        
        logger.info(f"GW EOS likelihood requires the following keys: {self.required_keys}")
      
    def transform(self, params: dict[str, Float]) -> dict[str, Float]:
        # Solve the TOV:
        solved_eos = self.eos_pipe.transform_func(params)
        params.update(solved_eos)
        masses_EOS, Lambdas_EOS = params["masses_EOS"], params["Lambdas_EOS"]
        
        # TODO: check for MTOV here
        
        # Convert chirp mass and mass ratio to source masses, and get the lambdas
        params = detector_frame_M_c_q_to_source_frame_m_1_m_2(params)
        params = eos_masses_to_lambdas(params)
        
        return params
        
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        return self.gw_likelihood.evaluate(params, data)