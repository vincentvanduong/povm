import jax.numpy as jnp

# Pauli Matrices Sigma
sx = jnp.asarray([[0, 1], [1, 0]])
sy = jnp.asarray([[0, -1j], [1j, 0]])
sz = jnp.asarray([[1, 0], [0, -1]])
sigma = jnp.asarray([sx, sy, sz])

# 2 Spins

interactions2 = jnp.kron(sigma[2], sigma[2])

def two_ising(params):
    """2 spin hamiltonian

    Args:
        params (array): parameters that define the hamiltoian couplings

    Returns:
        H (Hermitian matrices): hamiltonians genrated by the parameters
    """
    # H = Ising + B-field

    Jij, B = params[:, 0], params[:, 1:]

    # Angular Momentum J
    L = jnp.asarray([jnp.kron(s, jnp.eye(2)) + jnp.kron(jnp.eye(2), s) for s in sigma])

    H = jnp.einsum('n,ij->nij', Jij, interactions2) + jnp.einsum('nk,kij->nij', B, L)

    return H

# 3 Spins

# Ising Interaction: z-component of spins interact
ising12 = jnp.kron(jnp.eye(2), jnp.kron(sigma[2], sigma[2]))
ising23 = jnp.kron(sigma[2], jnp.kron(sigma[2], jnp.eye(2)))
ising13 = jnp.kron(sigma[2], jnp.kron(jnp.eye(2), sigma[2]))

ising = ising12 + ising23 + ising13

# Angular Momentum
L = []

for s in sigma:
    L_ = jnp.kron(jnp.eye(2), jnp.kron(jnp.eye(2), s))
    L_ += jnp.kron(jnp.eye(2), jnp.kron(s, jnp.eye(2)))
    L_ += jnp.kron(s, jnp.kron(jnp.eye(2), jnp.eye(2)))
    L.append(L_)
    
L = jnp.array(L)

def three_ising(params):
    """3 spin hamiltonian

    Args:
        params (array): parameters that define the hamiltoian couplings

    Returns:
        H (Hermitian matrices): hamiltonians genrated by the parameters
    """
    # H = Ising + B-field

    # Ising, B
    J, B = params[:, 0], params[:, 1:]
    H = jnp.einsum('n, ij->nij', J, ising) + jnp.einsum('nk, kij->nij', B, L)
    
    return H

# Four-body all to all ising interactions

four_body_ising = jnp.load('../data/hamiltonian_four-body_ising.npy')
four_body_L = jnp.load('../data/hamiltonian_four-body_L.npy')

def four_ising(params):
    """4 spin hamiltonian

    Args:
        params (array): parameters that define the hamiltoian couplings

    Returns:
        H (Hermitian matrices): hamiltonians genrated by the parameters
    """
    # H = Ising + B-field

    # Ising, B
    J, B = params[:, 0], params[:, 1:]
    H = jnp.einsum('n, ij->nij', J, four_body_ising) + jnp.einsum('nk, kij->nij', B, four_body_L)
    
    return H
