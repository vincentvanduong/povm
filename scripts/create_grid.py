import numpy as np

J_array = np.linspace(0, 2, 51)
t_array = np.linspace(0, 3, 51)

data = np.array(np.meshgrid(J_array, t_array)).T.reshape(-1,2)

np.save('data/array3.npy', data)

ddata = np.load('data/array3.npy')

print(ddata)