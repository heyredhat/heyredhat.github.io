import jax
import jax.numpy as jp
import numpy as np
from magic import *

#######################################################################################

def make_views(pts):
    views = []
    for i in range(n):
        view = []
        for j in range(n):
            if i != j:
                to = pts[j]-pts[i]
                view.append(to/np.linalg.norm(to))
        views.append(view)
    return views

def views_constellations(views):
    return [XYZ_spin(view) for view in views]

def constellations_matrix(spins, orth=True, su2n=True):
    M = qt.Qobj(np.array([spin.full().T[0] for spin in spins]).T)
    if not orth:
        return M
    U, H = sc.linalg.polar(M)
    Q = qt.tensor(qt.Qobj(U), qt.identity(2)) if su2n else qt.Qobj(U)
    Q.dims = [[Q.shape[0]], [Q.shape[0]]]
    return Q

def matrix_views(W):
	return np.array([spin_XYZ(w) for w in W.full().T])

#######################################################################################

def npts(n):
	return np.random.randn(n, 3)

def normalize(v):
	return v/jp.linalg.norm(v)

def make_views(pts):
	n = len(pts)
	return jp.array([[normalize(pts[j]-pts[i])\
				for j in range(n) if i != j]\
					for i in range(n)])

def loss(pts, views):

	return jp.linalg.norm(make_views(pts) - views)

n = 3
learning_rate = 0.1
pts = npts(3)
views = make_views(pts)

W = constellations_matrix(views_constellations(views), orth=False, su2n=False)
U = (-1j*0.001*qt.jmat((n-1)/2, 'x')).expm()

W2 = U.dag()*W*U
views2 = matrix_views(W)

loss_grad = jax.jit(jax.grad(loss))
initial_pts = pts.copy()
ow = loss(initial_pts, views2)
while ow > 0.00001:
	grad = loss_grad(initial_pts, views2)
	initial_pts = initial_pts - learning_rate*grad
	ow = loss(initial_pts, views)
	print("loss: %.5f" % (ow))

