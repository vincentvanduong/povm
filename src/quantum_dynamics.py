import jax
import jax.numpy as jnp

class Prior:
    """
    Creates the probability distribution class for a prior.
    It can sample from the normal distribution by default.
    """

    def __init__(self):
        self.params = None
        self.key = None
        self.distribution = jax.random.multivariate_normal

    def sample(self, no_samples):
        return self.distribution(self.key, *self.params, shape=(no_samples,))


class QuantumSystem:
    """
    Quantum System class saves the Input State, the Hamiltonian, and the Hamiltonian's Prior
    The class enables unitary time evolution of the quantum density
    """

    def __init__(self, rho_in=None, hamiltonian=None, prior=None):
        self.hamiltonian = hamiltonian
        self.prior = prior
        self.rho_in = rho_in
        self.rho_in_svd = jnp.linalg.svd(rho_in, hermitian=True)


    def evolve(self, time, no_samples):
        self.params_samples = self.prior.sample(no_samples)
        self.hamiltonian_samples = self.hamiltonian(self.params_samples)
        self.unitary_samples = jax.vmap(jax.scipy.linalg.expm, in_axes=0)(-1j * time * self.hamiltonian_samples)
        u, s, _ = self.rho_in_svd
        sqrt_rho_in = jnp.einsum('ik, jk -> ij', u, jnp.diag(jnp.sqrt(s)))
        sqrt_rho_out = jnp.einsum('nik, kj-> nij', self.unitary_samples, sqrt_rho_in)
        self.rho_out = jnp.einsum('nik, njk -> nij', sqrt_rho_out, jnp.conjugate(sqrt_rho_out))


class QuantumOptimiser:
    def __init__(self, quantumsystem: QuantumSystem):
        self.quantumsystem=quantumsystem
        self.dim = self.quantumsystem.rho_out.shape[1]
        self.no_fields = self.quantumsystem.params_samples.shape[1]
        
        self.basis = self.generate_basis()
        self.dof = self.basis.shape[0]
        self.no_povms = None

        self.params = None
        self.params_field_ = None
        self.params_povms_ = None
        self.povms = None
        self.a_mu = None


    def generate_basis(self):
        basis = []
        dim = self.dim
        for i in range(dim):
            for j in range(dim):
                if i >= j:
                    lower_tri = jnp.zeros((dim, dim))
                    lower_tri = lower_tri.at[i,j].set(1)
                    basis.append(lower_tri)

        return jnp.asarray(basis)
    
    

    def wrap(self):
        params1 = jnp.reshape(self.params, (self.no_povms, self.no_fields + 2*self.dof))
        self.params_field_ = params1[:, :self.no_fields]
        self.params_povms_ = jnp.reshape(params1[:, self.no_fields:], (self.no_povms, 2, self.dof))
        
    def generate_povm(self):
        theta_real, theta_imag = self.params_povms_[:, 0, :], self.params_povms_[:, 1, :]
        theta = theta_real + 1j * theta_imag
        L = jnp.einsum('ab,bij->aij', theta, self.basis)
        M = jnp.einsum('aij, akj -> aik', L, jnp.conjugate(L))
        self.povms = M

    def mcmc1(self):
        samples = self.quantumsystem.params_samples
        no_samples = samples.shape[0]
        rho_out = self.quantumsystem.rho_out
    
        f1 = jnp.einsum('nij -> ij', rho_out) / no_samples
        f2 = jnp.einsum('na, nij -> aij', samples, rho_out) / no_samples
        f3 = jnp.einsum('n, nij', deviance(samples), rho_out) / no_samples

        self.f_ = (f1, f2, f3)
    
    def  mcmc2(self):
        f1, f2, f3 = self.f_
        a1 = jnp.einsum('a, ij -> aij', deviance(self.params_field_), f1)
        a2 = jnp.einsum('ab, bij -> aij', self.params_field_, f2)
        a3 = jnp.einsum('a, ij -> aij', jnp.ones(self.params_field_.shape[0]), f3)
        self.a_mu = a1 + -2*a2 + a3
    
    def cost_(self):
        return jnp.real(jnp.einsum('aij, aji', self.povms, self.a_mu))
    
    
def cost(params, hyper_param, measurement: QuantumOptimiser):
    measurement.params = params
    measurement.wrap()
    measurement.generate_povm()

    measurement.mcmc1()
    measurement.mcmc2()

    penalty = jnp.linalg.norm(jnp.sum(measurement.povms, axis=0) - jnp.eye(measurement.povms.shape[1]))**2

    return measurement.cost_() + hyper_param*penalty


grad_cost = jax.grad(cost, argnums=0)
hessian_cost = jax.jacfwd(jax.grad(cost, argnums=0), argnums=0)


        
def deviance_(x):
    return jnp.linalg.norm(x) ** 2

deviance = jax.vmap(deviance_, in_axes=(0))

