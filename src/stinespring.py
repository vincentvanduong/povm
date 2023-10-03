import jax.numpy as jnp
import numpy as np

from jax import scipy, grad, jit, vmap
from jax import jacfwd, hessian
from jax import random

key = random.PRNGKey(0)


def eval_chi(matrix, theta):
    return jnp.einsum('iab, jcb, kca -> kij', theta, jnp.conjugate(theta), matrix)


def inner_product(unitary, chi):
   return jnp.einsum('ib, jb, kij -> kb', unitary, jnp.conjugate(unitary), chi)


def hamiltonian(params, basis):
    return jnp.einsum('a, aij -> ij', params, basis)


def A(theta, basis):
    return -1j * hamiltonian(theta, basis)


def loss(params, data, info):
    M = data
    basis, theta = info
    unitary = scipy.linalg.expm(A(params, basis))
    chi = eval_chi(M, theta)
    return jnp.einsum('ib, jb, aij -> ab', unitary, jnp.conjugate(unitary), chi)
