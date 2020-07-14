#!/usr/bin/env python3
import jax
import jax.numpy as np
from jax import grad, jit

def f2(x):
    x = x + .05
    x = x + .05
    x = x + .05
    x = x + .05
    x = x + .05
    x = x + .05
    x = x + .05
    x = x + .05
    return x

def tanh(x):
    a = x*2.3
    b = x*4.5 + f2(x) + f2(x)
    return a + b + x*(a+b)

print('python: tanh(12) == ', tanh(12))

c = jax.xla_computation(tanh, backend='cpu')(np.array([1.]))
print(c.as_hlo_text())

s = c.as_serialized_hlo_module_proto()

import jaxopt_native as n

assert n.add(1, 2) == 3

n.take(s)

# from IPython import embed; embed()
