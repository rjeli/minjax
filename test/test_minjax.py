import minjax
import numpy as onp
import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as np

key = jax.random.PRNGKey(0)

def test_add():
    assert minjax.add(1, 2) == 3

def simple_cost(x):
    return 2*x - 10

def l2_loss(f):
    def inner(x):
        y = f(x)
        return y**2
    return inner

def test_simple():
    p = minjax.Problem()
    x = onp.array(0., dtype=np.float32)
    p.add_residual(l2_loss(simple_cost), x)
    p.solve(verbose=True)
    print(x)
    assert np.isclose(x, 5., atol=1e-2)
