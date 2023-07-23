from src.quantum_objects import *
from src.quantum_dynamics import *
from src.optimiser import *

import time as ttime
import sys

start_time = ttime.time()
name = 'short_time'

# Setting-up the prior on the params = [J, Bx, By, Bz]
J = float(sys.argv[1]) / 100
mean = jnp.array([J, 0, 0, 0], dtype=jnp.float32)
covariance = jnp.diag(jnp.array([1e-16, 1e-1, 1e-1, 1e-1], dtype=jnp.float32))

prior = Prior()
prior.params = (mean, covariance)
prior.key = jax.random.PRNGKey(0)

# Setting-up the initial state
psi_in = jnp.array([0.5, 0.5j, 0.5, 0.5j], dtype=jnp.complex64)
rho_in = jnp.einsum('i,j->ij', psi_in, jnp.conjugate(psi_in))

# Setting-up the quantum system
sys = QuantumSystem(
    rho_in = rho_in,
    hamiltonian = two_ising,
    prior = prior
)
print('Initilisation complete')

# Set up system
time = 1
no_samples = int(1e6)
sys.evolve(time, no_samples)
print('Time evolution complete')

me = QuantumOptimiser(sys)
no_povms, no_fields, dof = 10, me.no_fields, me.dof
me.no_povms = no_povms

params = jax.random.normal(key=jax.random.PRNGKey(0), shape=(no_povms * (no_fields + 2*dof),))

me.params = params
me.wrap()
me.generate_povm()

cost_regularised_data = []
cost_data = []
params_data = []

print('Optimisation starting')

# Optimise
learning_rate = 1e-1
hyperparam = 0.1

for idx in range(5):
    grad = grad_cost(params, hyperparam, me)
    hessian  = hessian_cost(params, hyperparam, me)
    v_, _, _, _ = jnp.linalg.lstsq(hessian, grad)
    params -= v_
    cost_ = cost(params, hyperparam, me)
    cost_regularised_data.append(cost_)
    cost_data.append(cost(params, 0, me))
    print(learning_rate, hyperparam, idx, cost_, sep=' | ')
    params_data.append(params)

hyperparam = hyperparam * 10

for idx in range(5):
    grad = grad_cost(params, hyperparam, me)
    hessian  = hessian_cost(params, hyperparam, me)
    v_, _, _, _ = jnp.linalg.lstsq(hessian, grad)
    params -= v_
    cost_ = cost(params, hyperparam, me)
    cost_regularised_data.append(cost_)
    cost_data.append(cost(params, 0, me))
    print(learning_rate, hyperparam, idx, cost_, sep=' | ')
    params_data.append(params)

hyperparam = hyperparam * 10

for idx in range(5):
    grad = grad_cost(params, hyperparam, me)
    hessian  = hessian_cost(params, hyperparam, me)
    v_, _, _, _ = jnp.linalg.lstsq(hessian, grad)
    params -= v_
    cost_ = cost(params, hyperparam, me)
    cost_regularised_data.append(cost_)
    cost_data.append(cost(params, 0, me))
    print(learning_rate, hyperparam, idx, cost_, sep=' | ')
    params_data.append(params)

hyperparam = hyperparam * 100

for idx in range(5):
    grad = grad_cost(params, hyperparam, me)
    hessian  = hessian_cost(params, hyperparam, me)
    v_, _, _, _ = jnp.linalg.lstsq(hessian, grad)
    params -= v_
    cost_ = cost(params, hyperparam, me)
    cost_regularised_data.append(cost_)
    cost_data.append(cost(params, 0, me))
    print(learning_rate, hyperparam, idx, cost_, sep=' | ')
    params_data.append(params)

# Metrics

field_ = me.params_field_
povms_ = me.povms
rho_, _, _ = me.f_
prob_ = jnp.einsum('aij,ji->a',povms_,rho_)
variance_ = jnp.einsum('aij,aji->a', povms_, me.a_mu)
delta2_ = cost(params, 0, me)

print('-----','-------','-----')
print('-----','RESULTS','-----')
print('-----','-------','-----')

print('Cost:', cost(params, 0, me))
print('Penalty:', cost(params, 1, me)-cost(params, 0, me))
print('Amplification (dB):', -10 * jnp.log10(cost(params, 0, me)/jnp.trace(covariance)))

print('-------------------------------------------------------')
print('POVM','Probability','Parameters','Variance',sep='   |   ')
print('-------------------------------------------------------')

for mu in range(field_.shape[0]):
    prob = jnp.real(prob_[mu])
    fields = field_[mu]
    variance = jnp.around(jnp.real(variance_[mu]),5)
    print(mu, f'{prob:.5f}', fields, f'{variance:.8f}', sep='  ')

import matplotlib.pyplot as plt

plt.loglog(cost_data, label='cost function')
plt.legend()
plt.xlabel('iterations')
plt.title('convergence: '+name)
plt.savefig(name+'.png')

print("--- %s seconds ---" % (ttime.time() - start_time))


