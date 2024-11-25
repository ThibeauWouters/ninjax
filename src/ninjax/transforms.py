"""Transforms that can be used on a collection of parameters (gathered in a dict) and return the transformed params"""

import jax.numpy as jnp

def q_to_eta(params: dict) -> dict:
    params["eta"] = params["q"] / (1 + params["q"]) ** 2
    return params

def cos_iota_to_iota(params: dict):
    params["iota"] = jnp.arccos(params["cos_iota"])
    return params

def sin_dec_to_dec(params: dict) -> dict:
    params["dec"] = jnp.arcsin(params["sin_dec"])
    return params


# For overlapping signals
def q1_to_eta1(params: dict) -> dict:
    params["eta_1"] = params["q_1"] / (1 + params["q_1"]) ** 2
    return params

def q2_to_eta2(params: dict) -> dict:
    params["eta_2"] = params["q_2"] / (1 + params["q_2"]) ** 2
    return params

def cos_iota1_to_iota1(params: dict):
    params["iota_1"] = jnp.arccos(params["cos_iota_1"])
    return params

def cos_iota2_to_iota2(params: dict):
    params["iota_2"] = jnp.arccos(params["cos_iota_2"])
    return params

def sin_dec1_to_dec1(params: dict) -> dict:
    params["dec_1"] = jnp.arcsin(params["sin_dec_1"])
    return params

def sin_dec2_to_dec2(params: dict) -> dict:
    params["dec_2"] = jnp.arcsin(params["sin_dec_2"])
    return params

def t_c_dt_to_tcs(params: dict) -> dict:
    params["t_c_1"] = params["t_c"]
    params["t_c_2"] = params["t_c"] + params["dt"]
    return params