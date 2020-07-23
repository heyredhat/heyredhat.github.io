import qutip as qt
import numpy as np

def tm_squeezer(z, a):
    return (z.conjugate()*a[0]*a[1] - z*a[0].dag()*a[1].dag()).expm()

n = 5
a = [qt.tensor(qt.destroy(n), qt.identity(n)),\
     qt.tensor(qt.identity(n), qt.destroy(n))]


vac = qt.basis(n*n, 0)
vac.dims = [[n,n], [1,1]]

N = sum([a[i].dag()*a[i] for i in range(2)])

z = (np.random.randn(1) + np.random.randn(1)*1j)[0]
SQ = tm_squeezer(z, a)

sq_vac = SQ*vac

sq_a = [SQ*A*SQ.dag() for A in a]
sq_N = sum([sq_a[i].dag()*sq_a[i] for i in range(2)])

print("N on vac: %s" % qt.expect(N, vac))
print("N on sq_vac: %s" % qt.expect(N, sq_vac))
print("sq_N on vac: %s" % qt.expect(sq_N, vac))
print("sq_N on sq_vac: %s" % qt.expect(sq_N, sq_vac))
