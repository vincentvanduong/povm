import jax.numpy as jnp
import matplotlib.pyplot as plt

path = 'results/'

ising = jnp.load(path+'tomography1_ising.npy')
time = jnp.load(path+'tomography1_time.npy')
data = jnp.load(path+'tomography1_data.npy')

cost = jnp.array(data[:,0]).reshape(len(time), len(ising))
isings, times = jnp.meshgrid(ising, time)

cost_min = 0.5

# plt.pcolormesh(isings, times, jnp.real(cost), cmap='Reds')
# plt.clim(0,1)
# plt.title('Cost landscape: Fixed J, B_x~N(1,1)')
# plt.colorbar()
# plt.xlabel('J coupling')
# plt.ylabel('Time')
# plt.savefig('results/tomography1.pdf')

# plt.title('J slice: 0.2', fontsize=14)
# plt.plot(time, cost[:,5])
# plt.xlabel('Time')
# plt.ylabel('Cost')
# plt.savefig('results/tomography1_time.png')

fig, axes = plt.subplots(2, 1, sharey='row', figsize=(8,10), gridspec_kw = {'height_ratios':[1,1]})
fig.subplots_adjust(hspace=0.05, wspace=0.02)
axes[0].set_title('Cost landscape', fontsize=14)
c = axes[0].pcolor(isings, times, jnp.real(cost), cmap='Reds', vmin=cost_min, vmax=1)
fig.colorbar(c, ax=axes[0])

idx = 20

axes[1].set_title('J slice: J = {}'.format(ising[idx]), fontsize=14)
axes[1].plot(time, cost[:,idx])
axes[1].set_ylim(cost_min,1)

axes[0].set_ylabel('Time', fontsize=14)
axes[0].set_xlabel('J', fontsize=14)
axes[1].set_xlabel('Time', fontsize=14)
axes[1].set_ylabel('Cost', fontsize=14)
plt.tight_layout(w_pad=0.1, h_pad=0.1)
plt.savefig('results/tomography1.png')
plt.show()