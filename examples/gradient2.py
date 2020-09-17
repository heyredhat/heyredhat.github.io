import jax
import jax.numpy as np
from magic import spin_XYZ
import numpy as onp
import itertools
import math
import qutip as qt

def xyz_c(xyz, pole="south"):
    x, y, z = xyz
    if (pole=="south"):
        if np.isclose(z,-1):
            return float("Inf")
        else:
            return x/(1+z) + 1j*y/(1+z)
    elif (pole=="north"):
        if np.isclose(z,1):
            return float("Inf")
        else:
            return x/(1-z) + 1j*y/(1-z)

def roots_coeffs(roots):
    n = len(roots)
    coeffs = np.array([((-1)**(-i))*sum([np.prod(term)\
        for term in itertools.combinations(roots, i)]) for i in range(0, len(roots)+1)])
    return coeffs/coeffs[0]

def roots_poly(roots):
    zeros = roots.count(0j)
    if zeros == len(roots):
        return [1j] + [0j]*len(roots)
    poles = roots.count(float("Inf"))
    roots = [root for root in roots if root != float('Inf')]
    if len(roots) == 0:
        return [0j]*poles + [1j]
    return np.concatenate([np.array([0j]*poles), roots_coeffs(roots)])

def poly_spin(poly):
    j = (len(poly)-1)/2.
    spin = []
    for m in np.arange(-j, j+1):
        i = int(m+j)
        spin.append(poly[i]/\
            (((-1)**(i))*np.sqrt(math.factorial(2*j)/\
                        (math.factorial(j-m)*math.factorial(j+m)))))
    aspin = np.array(spin)
    return aspin/np.linalg.norm(aspin)

def XYZ_spin(XYZ):
    return poly_spin(roots_poly([xyz_c(xyz) for xyz in XYZ]))

#######################################################################################

def views_constellations(views):
    return [XYZ_spin(view) for view in views]

def views_matrix(views):
    spins = views_constellations(views)
    return np.array([spin for spin in spins]).T

def matrix_views(W):
    return np.array([spin_XYZ(w) for w in W.T])

#######################################################################################

def npts(n):
    return onp.random.randn(n, 3)

def normalize(v):
    return v/np.linalg.norm(v)

def make_views(pts):
    n = len(pts)
    return np.array([[normalize(pts[j]-pts[i])\
                for j in range(n) if i != j]\
                    for i in range(n)])

def loss(pts, views):
    new_views = make_views(pts)
    C0 = views_matrix(views)
    C1 = views_matrix(new_views)
    return np.linalg.norm(make_views(pts) - views).real

n = 3
learning_rate = 1
pts = npts(3)
views = make_views(pts)

W = views_matrix(views)
U = (-1j*np.pi*qt.jmat((n-1)/2, 'x')).expm()

W2 = (U.dag()*qt.Qobj(W)*U).full()
views2 = matrix_views(W)

loss_grad = jax.grad(loss)
initial_pts = pts.copy()
ow = loss(initial_pts, views2)
while ow > 0.00001:
    grad = loss_grad(initial_pts, views2)
    initial_pts = initial_pts - learning_rate*grad
    ow = loss(initial_pts, views2)
    print("loss: %.5f" % (ow))

