import jax
import jax.numpy as jnp
import numpy as np

from jax.experimental.pjit import pjit
from jax.experimental.maps import Mesh
from jax.experimental import PartitionSpec

# Create device mesh
mesh = Mesh(np.array(jax.devices()).reshape((4, 2)), ("x", "y"))

# Create matrix and vector
M = jnp.eye(8)
v = jnp.arange(8).reshape((8, 1))

spec = PartitionSpec("x", "y")

f = pjit(jnp.dot,
         in_axis_resources=(spec, None),
         out_axis_resources=spec)

with mesh:
    output = f(M, M)

print(output)