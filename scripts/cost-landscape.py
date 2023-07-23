import sys
import numpy as onp
import jax.numpy as jnp

import os

directory = os.fsencode("../results/four-body3/")

data = []
    
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".npy"):
         data_ = onp.load("../results/four-body3/"+filename)
         data.append(data_)

data = onp.array(data)
onp.save('../data/four-body3.npy', data)
