from src.quantum_objects import *
from src.quantum_dynamics import *
from src.optimiser import *
from src.quantum_metrics import *

import pickle

with open('results/example7.pkl', 'rb') as file:

    # Call load method to deserialze
    myvar = pickle.load(file)
  
# The solution
print('params opt:', myvar.params_field_, '', sep='\n')

print('povms opt:')
for _ in myvar.povms:
    print(_, '', sep='\n')

print('singular values:')
for _ in myvar.povms:
    u, s, vh = jnp.linalg.svd(_)
    print(jnp.around(s,4), '', sep='\n')
