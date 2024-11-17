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