"""Transforms that can be used on a collection of parameters (gathered in a dict) and return the transformed params"""

import jax.numpy as jnp
from jaxtyping import Array, Float

def q_to_eta(params: dict) -> dict:
    params["eta"] = params["q"] / (1 + params["q"]) ** 2
    return params

def cos_iota_to_iota(params: dict):
    params["iota"] = jnp.arccos(params["cos_iota"])
    return params

def sin_dec_to_dec(params: dict) -> dict:
    params["dec"] = jnp.arcsin(params["sin_dec"])
    return params

def detector_frame_M_c_q_to_source_frame_m_1_m_2(params: dict) -> dict:
    M_c, q, d_L = params['M_c'], params['q'], params['d_L']
    H0 = params.get('H0', 67.4) # (km/s) / Mpc
    c = params.get('c', 299_792.4580) # km / s
    
    # Calculate source frame chirp mass
    z = d_L * H0 * 1e3 / c
    M_c_source = M_c / (1.0 + z)

    # Get source frame mass_1 and mass_2
    M_source = M_c_source * (1.0 + q) ** 1.2 / q**0.6
    m_1_source = M_source / (1.0 + q)
    m_2_source = M_source * q / (1.0 + q)
    
    params['m_1_source'] = m_1_source
    params['m_2_source'] = m_2_source

    return params

def eos_masses_to_lambdas(params: dict[str, Float]) -> dict[str, Float]:
    """Get the Lambdas from a given EOS (NS family) and source masses"""
    
    masses_EOS = params["masses_EOS"]
    Lambdas_EOS = params["Lambdas_EOS"]
    
    m_1 = params["m_1_source"]
    m_2 = params["m_2_source"]
    
    # Interpolate to get Lambdas
    lambda_1_interp = jnp.interp(m_1, masses_EOS, Lambdas_EOS, right = 1e12)
    lambda_2_interp = jnp.interp(m_2, masses_EOS, Lambdas_EOS, right = 1e12)
    
    params["lambda_1"] = lambda_1_interp
    params["lambda_2"] = lambda_2_interp
    
    return params