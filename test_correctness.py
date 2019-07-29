
import numpy as np
import os
import rk45
import chain_cython
import chain_oldpython
import chain_newpython

def chain_test(Chain, N, steps=10, friction=0):
    chain = Chain(N, friction=friction)
    chain.hanging_state(4, 0.1)
    stepper = rk45.RK45(chain, chain['state'])
    stepper['tmax'] = np.inf
    stepper['dt'] = 0.1 / N
    stepper['tol'] = 1e-8
    stepper['scheme'] = 'CK'
    stepper['adapative'] = False
    while stepper.iter <= steps:
        stepper()
    return chain['state']

def compare_codes(*args, **kw):
    old = chain_test(chain_oldpython.Chain, *args, **kw)
    new = chain_test(chain_newpython.Chain, *args, **kw)
    cython = chain_test(chain_cython.Chain, *args, **kw)
    print("  New matches old:", np.allclose(new, old))
    print("  Cython matches old:", np.allclose(cython, old))

print("No friction:")
compare_codes(N=100, friction=0)
print("Friction:")
compare_codes(N=100, friction=1.0)
