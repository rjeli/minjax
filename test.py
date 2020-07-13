#!/usr/bin/env python3
import jax
import jax.numpy as np
from jax import grad, jit

def tanh(x):
    y = np.exp(-2.0*x)
    return (1.0 - y) / (1.0 + y)

gtanh = grad(tanh)
print(gtanh(1.))

c = jax.xla_computation(tanh, backend='cpu')(1.)
print(c.as_hlo_text())

s = c.as_serialized_hlo_module_proto()

import jaxopt_native as n

assert n.add(1, 2) == 3

n.take(s)

# from IPython import embed; embed()
