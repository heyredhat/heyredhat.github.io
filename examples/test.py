import qutip as qt
import numpy as np
from magic import *
import scipy as sc
import vpython as vp

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

#######################################################################################

def gramian(vectors):
    return np.array([[np.dot(vectors[i], vectors[j]) for j in range(len(vectors))] for i in range(len(vectors))])

def recover_vecs_from_gramian(G):
    U, D, V = np.linalg.svd(G)
    M = np.dot(V[:3].T, np.sqrt(np.diag(D[:3])))
    return [m for m in M]

def rotation_between_pts(ptsA, ptsB):
    pts1, pts2 = ptsA[:], ptsB[:]
    C1 = sum(pts1)/len(pts1)
    C2 = sum(pts2)/len(pts2)
    for i in range(len(pts1)):
        pts1[i] = pts1[i] - C1
    for i in range(len(pts2)):
        pts2[i] = pts2[i] - C2
    pts1, pts2 = np.array(pts1), np.array(pts2)
    cov = np.dot(pts1.T, pts2)
    U, D, Vh = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(cov))
    R = np.dot(np.dot(Vh.T, np.diag([1,1,d])), U.T)
    return R, pts1, pts2, C1, C2

def test_rotation(R, pts1, pts2):
    print("pts1:")
    print(pts1)
    print("pts2:")
    print(pts2)
    print("R on pts1:")
    for pt in pts1:
        print(np.dot(R, pt))

#######################################################################################

n = 3
pts = [3*np.random.randn(3) for i in range(n)] 
#pts = [np.array([0, 0, 4*i - n/2]) for i in range(n)]

#G = gramian(pts)
#pts2 = recover_vecs_from_gramian(G)
#R, p1, p2, c1, c2 = rotation_between_pts(pts, pts2)
#test_rotation(R, p1, p2)

#######################################################################################

views = make_views(pts)
W = constellations_matrix(views_constellations(views), su2n=False, orth=True)

def show_xyzs(xyzs):
    for i in range(len(xyzs)):
        print(i)
        print(np.array(xyzs[i]))

xyzs = [spin_XYZ(qt.Qobj(w)) for w in W.full().T]
links = {}
used = []
print()
for i in range(n):
    for j in range(n-1):
        for k in range(n):
            for l in range(n-1):
                if i != k:
                    print(i, j, k, l)
                    print(xyzs[i][j], xyzs[k][l])
                    if np.isclose(xyzs[i][j], -1*xyzs[k][l]).all():
                        print("?")
                        if (i, j) not in used and (k, l) not in used:
                            links[(i, k)] = (j, l)
                            used.append((i, j))
                            used.append((k, l))
        print("--")
    print("-")

