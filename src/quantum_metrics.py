import jax
import jax.numpy as jnp

def trace_squared(x, y) -> float:
    """The Trace Kernel"""
    # calculate the kernel
    return jnp.trace(jnp.conjugate(jnp.transpose(x)) @ y) ** 2


def trace_distance(x, y) -> float:
    """The Trace Distance, similar to total variation"""
    delta = x - y
    delta2 = jnp.conjugate(jnp.transpose(delta)) @ delta

    return 0.5 * jnp.trace(jax.scipy.linalg.sqrtm(delta2))


def trace_distance_squared(x, y) -> float:
    """The Trace Distance, similar to total variation"""
    delta = x - y
    delta2 = jnp.conjugate(jnp.transpose(delta)) @ delta

    return 0.5 * jnp.trace(jax.scipy.linalg.sqrtm(delta2)) ** 2


def fidelity_kernel(x, y) -> float:
    """The Trace Kernel"""
    # calculate the kernel
    z = x @ y
    return jnp.trace((jnp.conjugate(z.T) @ z))


def gram_matrix(kernel_func, x, y=None):
    if y is None:
        y = x
    mapx1 = jax.vmap(lambda a, b: kernel_func(a, b), in_axes=(0, None), out_axes=0)
    mapx2 = jax.vmap(lambda a, b: mapx1(a, b), in_axes=(None, 0), out_axes=1)
    return mapx2(x, y)


def eval(metric, povms1, povms2=None):
    if povms2 is None:
        povms2 = povms1
    matrix = jnp.real(gram_matrix(metric, x=povms1, y=povms2))
    matrix1, matrix2 = jnp.real(gram_matrix(metric, x=povms1)), jnp.real(gram_matrix(metric, x=povms2))
    norm1, norm2 = jnp.sqrt(matrix1.sum()), jnp.sqrt(matrix2.sum())
    return matrix.sum() / (norm1 * norm2)


def rank(povms, tol=1e-3):
    ranks = [jnp.linalg.matrix_rank(_, tol=tol) for _ in povms]
    return min(ranks), max(ranks)


def phi_(x, operator):
    return operator @ x @ jnp.conjugate(operator.T)


def phi(x, operators):
    mapo = jax.vmap(lambda a, b: phi_(a, b), in_axes=(None, 0), out_axes=0)
    return jnp.sum(mapo(x, operators), axis=0)


def choi_matrix(povms):
    map1 = jax.vmap(lambda x, y: phi(x, y), in_axes=(0, None), out_axes=0)
    map2 = jax.vmap(lambda x, y: map1(x, y), in_axes=(1, None), out_axes=1)
    map3 = jax.vmap(lambda x, y: jnp.kron(x, y), in_axes=(0, None), out_axes=0)
    map4 = jax.vmap(lambda x, y: map3(x, y), in_axes=(None, 1), out_axes=1)

    dim = povms[0].shape[0]
    basis = jnp.zeros((dim, dim, dim, dim))
    for i in range(basis.shape[0]):
        for j in range(basis.shape[1]):
            basis = basis.at[i, j, i, j].set(1)

    z = map2(basis, povms)

    return jnp.sum(jnp.kron(basis, z), axis=(0, 1))