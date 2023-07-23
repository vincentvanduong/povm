import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Importing
path = '../data/four-body3.npy'
data = jnp.load(path)


# Cleaning
ising = data[:,0]
time = data[:,1]
cost = data[:,2]

idx = (cost <= 0.35) & (0.195 <= cost)

time = time[idx]
ising = ising[idx]
cost = cost[idx]

x, y, z = ising, time, cost
triang = tri.Triangulation(x, y)


# Plotting
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
fig1, ax1 = plt.subplots(figsize=(4,3))

tpc = ax1.tripcolor(triang, z, shading='gouraud', vmax=0.3)

fig1.colorbar(tpc)
ax1.set_title('cost landscape')
ax1.set_xlabel('ising coupling')
ax1.set_ylabel('time')
plt.savefig('../results/four-body3.png', bbox_inches='tight', dpi=300)
