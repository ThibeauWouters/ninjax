import os
import copy
import numpy as np
import jax.numpy as jnp
from jaxtyping import Array, Float

from jimgw.prior import Composite

# TODO: extend with other EOS models
from joseTOV.eos import MetaModel_EOS_model, MetaModel_with_CSE_EOS_model, construct_family

from ninjax.pipe_utils import logger

NEP_NAMES = ["E_sym", "L_sym", "K_sym", "Q_sym", "Z_sym", "K_sat", "Q_sat", "Z_sat", "nbreak"]

# FIXME: this needs to be generalized
NEP_CONSTANTS_DICT = {
    # This is a set of MM parameters that gives a decent initial guess for Hauke's Set A maximum likelihood EOS
    "E_sym": 33.431808,
    "L_sym": 77.178344,
    "K_sym": -129.761344,
    "Q_sym": 422.442807,
    "Z_sym": -1644.011429,
    
    "E_sat": -16.0,
    "K_sat": 285.527411,
    "Q_sat": 652.366343,
    "Z_sat": -1290.138303,
    
    "nbreak": 0.153406,
    
    "n_CSE_0": 3 * 0.16,
    "n_CSE_1": 4 * 0.16,
    "n_CSE_2": 5 * 0.16,
    "n_CSE_3": 6 * 0.16,
    "n_CSE_4": 7 * 0.16,
    "n_CSE_5": 8 * 0.16,
    "n_CSE_6": 9 * 0.16,
    "n_CSE_7": 10 * 0.16,
    
    "cs2_CSE_0": 0.5,
    "cs2_CSE_1": 0.7,
    "cs2_CSE_2": 0.5,
    "cs2_CSE_3": 0.4,
    "cs2_CSE_4": 0.8,
    "cs2_CSE_5": 0.6,
    "cs2_CSE_6": 0.9,
    "cs2_CSE_7": 0.8,
    
    # This is the final entry
    "cs2_CSE_8": 0.9,
}

class EOSPipe:
    
    def __init__(self, 
                 config: dict,
                 prior: Composite,
                 fixed_params: dict[str, float] = None):
        """Construct the EOS pipe with the given setup provided by the config and the EOS model informed by the prior"""
        self.config = config
        
        # TODO: check if specified priors are OK and making sense
        self.naming = []
        self.nb_CSE = 0
        for key in prior.naming:
            if key in NEP_NAMES or "n_CSE" in key or "cs2_CSE" in key:
                self.naming.append(key)
            if "n_CSE" in key:
                self.nb_CSE += 1
                
        logger.info(f"We have constructed an EOS pipe by fetching the following keys: {self.naming}")
        
        # Choose the correct transformation function based on the provided information on the EOS
        if self.nb_CSE > 0:
            eos = MetaModel_with_CSE_EOS_model(nmax_nsat=self.nmax_nsat,
                                               ndat_metamodel=self.ndat_metamodel,
                                               ndat_CSE=self.ndat_CSE,
                    )
            self.transform_func = self.transform_func_MM_CSE
        else:
            eos = MetaModel_EOS_model(nmax_nsat = self.nmax_nsat,
                                      ndat = self.ndat_metamodel)
        
            self.transform_func = self.transform_func_MM
        
        self.eos = eos
        
        # Remove those NEPs from the fixed values that we sample over
        if fixed_params is None:
            fixed_params = copy.deepcopy(NEP_CONSTANTS_DICT)
        
        self.fixed_params = fixed_params
        for name in self.naming:
            if name in list(self.fixed_params.keys()):
                self.fixed_params.pop(name)
        
        logger.info(f"We are given fixed parameters: {self.fixed_params}")
            
        # Construct a lambda function for solving the TOV equations, fix the given parameters
        self.construct_family_lambda = lambda x: construct_family(x, ndat = self.ndat_TOV, min_nsat = self.min_nsat_TOV)
            
    def transform_func_MM(self, params: dict[str, Float]) -> dict[str, Float]:
        
        params.update(self.fixed_params)
        NEP = {key: value for key, value in params.items() if "_sat" in key or "_sym" in key}
        
        # Create the EOS, ignore mu and cs2 (final 2 outputs)
        ns, ps, hs, es, dloge_dlogps, _, cs2 = self.eos.construct_eos(NEP)
        eos_tuple = (ns, ps, hs, es, dloge_dlogps)
        
        # Solve the TOV equations
        logpc_EOS, masses_EOS, radii_EOS, Lambdas_EOS = self.construct_family_lambda(eos_tuple)
    
        return_dict = {"logpc_EOS": logpc_EOS, "masses_EOS": masses_EOS, "radii_EOS": radii_EOS, "Lambdas_EOS": Lambdas_EOS,
                       "n": ns, "p": ps, "h": hs, "e": es, "dloge_dlogp": dloge_dlogps, "cs2": cs2}

        return return_dict

    def transform_func_MM_CSE(self, params: dict[str, Float]) -> dict[str, Float]:
        
        params.update(self.fixed_params)
        
        # Separate the MM and CSE parameters
        NEP = {key: value for key, value in params.items() if "_sat" in key or "_sym" in key}
        NEP["nbreak"] = params["nbreak"]
        
        ngrids = jnp.array([params[f"n_CSE_{i}"] for i in range(self.nb_CSE)])
        cs2grids = jnp.array([params[f"cs2_CSE_{i}"] for i in range(self.nb_CSE)])
        
        # Append the final cs2 value, which is fixed at nmax 
        ngrids = jnp.append(ngrids, jnp.array([self.nmax]))
        # Sort ngrids from lowest to highest
        ngrids = jnp.sort(ngrids)
        cs2grids = jnp.append(cs2grids, jnp.array([params[f"cs2_CSE_{self.nb_CSE}"]]))
        
        # Create the EOS, ignore mu and cs2 (final 2 outputs)
        ns, ps, hs, es, dloge_dlogps, _, cs2 = self.eos.construct_eos(NEP, ngrids, cs2grids)
        eos_tuple = (ns, ps, hs, es, dloge_dlogps)
        
        # Solve the TOV equations
        logpc_EOS, masses_EOS, radii_EOS, Lambdas_EOS = self.construct_family_lambda(eos_tuple)
    
        return_dict = {"logpc_EOS": logpc_EOS, "masses_EOS": masses_EOS, "radii_EOS": radii_EOS, "Lambdas_EOS": Lambdas_EOS,
                       "n": ns, "p": ps, "h": hs, "e": es, "dloge_dlogp": dloge_dlogps, "cs2": cs2}
        
        return return_dict
    
    @property
    def ndat_metamodel(self):
        return int(self.config["ndat_metamodel"])
    
    @property
    def nmax_nsat(self):
        return float(self.config["nmax_nsat"])
    
    @property
    def nmax(self):
        return self.nmax_nsat * 0.16
    
    @property
    def min_nsat_TOV(self):
        return float(self.config["min_nsat_TOV"])
    
    @property
    def ndat_TOV(self):
        return int(self.config["ndat_TOV"])
    
    @property
    def ndat_CSE(self):
        return int(self.config["ndat_CSE"])
    
    @property
    def nb_masses(self):
        return int(self.config["nb_masses"])
        