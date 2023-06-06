import jax
from jax import numpy as jnp, pmap


# Assume we have 8 GPUs
print("Available devices:", jax.local_device_count())  # e.g., 8

# A simple function to be mapped
def f(x):
    return x ** 2

# Let's define an array that we'll distribute across devices
x = jnp.arange(8)

# Use pmap to apply f to x in parallel across multiple devices
pmap_f = pmap(f)

y = pmap_f(x)

import pdb; pdb.set_trace()

print(y)  # prints: [0, 1, 4, 9, ..., 49]
