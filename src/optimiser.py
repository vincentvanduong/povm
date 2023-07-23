import jax.numpy as jnp
import jax


def generate_lower_tri_basis(dim: int):
    basis = []

    for i in range(dim):
        for j in range(dim):
            if i >= j:
                lower_tri = jnp.zeros((dim, dim))
                lower_tri = lower_tri.at[i,j].set(1)
                basis.append(lower_tri)

    return jnp.asarray(basis)


def generate_povm(params, basis):
    theta_real, theta_imag = params[:, 0, :], params[:, 1, :]
    theta = theta_real + 1j * theta_imag
    L = jnp.einsum('ab,bij->aij', theta, basis)
    M = jnp.einsum('aij, akj -> aik', L, jnp.conjugate(L))
    return M


def loss_verbose(M, A):
    error = jnp.einsum('aij, aji', M, A)
    return jnp.linalg.norm(error)


def regularisation(M):
    return jnp.linalg.norm(jnp.sum(M, axis=0) - jnp.eye(M.shape[2]))


def loss(params, data, info):
    basis, shape, hyperparam = info
    params_ = jnp.reshape(params, shape)
    A = data
    M = generate_povm(params_, basis)
    return loss_verbose(M, A) + hyperparam * regularisation(M)


grad_loss = jax.grad(loss, argnums=0, holomorphic=False)


"""def optimise_povm(system, learning_rate, no_iters, no_epochs):
    samples1 = 
    params1 = jnp.ravel(samples1)
    for epoch in range(no_epochs):
        learning_rate_ = learning_rate * (10**-epoch)
        for _ in range(no_iters):
            derivative = grad_loss(params1, A, hyperparams)
            v = learning_rate_ * derivative / jnp.linalg.norm(derivative)
            params1 += -1*v
            delta2_ = loss(params1, A, hyperparams)
            M_ = generate_povm(jnp.reshape(params1, samples1.shape), lt_basis)
            error_ = jnp.max(jnp.abs(jnp.sum(M_, axis=0) - jnp.eye(M_.shape[1])))
            
        print(epoch, str(delta2_)[:5],error_)"""


def solve_params(system):
        weighted_density = jnp.einsum('nk, nij -> kij', system.params_samples, system.rho_out) / \
                           system.params_samples.shape[0]
        conditional_mean = jnp.einsum('mij, kij -> mk', system.povms, weighted_density)
        conditional_prob = jnp.einsum('mij, nij -> m', system.povms, system.rho_out) / system.params_samples.shape[0]
        params = jnp.transpose(jnp.real(jnp.transpose(conditional_mean) / jnp.real(conditional_prob)))
        return params