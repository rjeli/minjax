import jax
import jax.numpy as np
import numpy as onp

from .c.minjax_c import add, Problem as _Problem

def fullname(f):
    return f.__module__ + '.' + f.__name__

class Problem:
    def __init__(self):
        self._problem = _Problem()

    def add_residual(self, costfn, x):
        assert x.dtype == onp.float32 and x.flags.c_contiguous and x.flags.writeable
        self.fn = jax.xla_computation(costfn, backend='cpu')(x)
        self.grad = jax.xla_computation(jax.grad(costfn), backend='cpu')(x)
        self.x = x

    def solve(self, verbose=False):
        fn_hlo = self.fn.as_serialized_hlo_module_proto()
        grad_hlo = self.grad.as_serialized_hlo_module_proto()
        print(self.fn.as_hlo_text())
        print(self.grad.as_hlo_text())
        self._problem.solve(fn_hlo, grad_hlo, self.x)
