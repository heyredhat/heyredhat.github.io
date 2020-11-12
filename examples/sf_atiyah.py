import qutip as qt
import numpy as np
import scipy as sc
import vpython as vp

from magic import *
from sf_gaussian_spin import *

import strawberryfields as sf
from strawberryfields.ops import *

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

def views_symplectic(views, inverted=True):
    P = constellations_matrix(views_constellations(views)).full()
    if inverted:
        P = P.conj().T
    H, h = make_Hh(P)
    S, s = make_Ss(H, h, expm=False)
    R, r = make_Rr(S, s)
    return R.real

def rand_xyz():
    xyz = np.random.randn(3) #np.array([0,0,-1])
    xyz = xyz/np.linalg.norm(xyz)
    return xyz

#######################################################################################

n = 3
inverted = True

#pts = [3*np.random.randn(3) for i in range(n)] 
pts = [np.array([2.5*i - n/2, 0, 0]) for i in range(n)]
views = make_views(pts)
initial_R = GaussianTransform(views_symplectic(views, inverted=not inverted))
initial_spin_axes = [rand_xyz() for i in range(n)]
#initial_spin_axes = [[1,0,0], [0,1,0],[0,0,1]]
initial_rots = [xyz_gaussianTransforms(xyz) for xyz in initial_spin_axes]

eng = sf.Engine("gaussian")
prog = sf.Program(2*n)
with prog.context as q:
    for i in range(0, 2*n, 2):
        Sgate(0.5) | q[i]
        initial_rots[int(i/2)][0] | (q[i], q[i+1])
        initial_rots[int(i/2)][1] | (q[i], q[i+1])
state = eng.run(prog).state

XYZ = {"X": qt.sigmax(), "Y": qt.sigmay(), "Z": qt.sigmaz()}
XYZs = [dict([(o, second_quantize(sc.linalg.block_diag(*\
                    [np.zeros((2,2)) if i !=j \
                        else XYZ[o].full() \
                for j in range(n)]), expm=False)[0]) \
                    for o in XYZ.keys()])
                        for i in range(n)]

def state_xyzs(state):
    global XYZs
    return np.array([[state.poly_quad_expectation(XYZ["X"])[0],\
                      state.poly_quad_expectation(XYZ["Y"])[0],\
                      state.poly_quad_expectation(XYZ["Z"])[0]]\
                            for i, XYZ in enumerate(XYZs)]).real
xyzs = state_xyzs(state)
initial_xyzs = xyzs[:]

#######################################################################################

vp.scene = vp.canvas(background=vp.color.white, width=800, height=600)
vcolors = [vp.vector(*np.random.rand(3)) for i in range(n)]
vspheres = [vp.sphere(color=vcolors[i],\
                      radius=1,\
                      opacity=0.1,\
                      pos=vp.vector(*pts[i]))\
                for i in range(n)]

vstars = [[vp.sphere(radius=0.15,\
                     emissive=True,\
                     pos=vspheres[i].pos+vp.vector(*views[i][j]))
            for j in range(n-1)]\
                for i in range(n)]
vspins = [vp.arrow(pos=vspheres[i].pos,\
                   axis=vp.vector(*xyzs[i]))\
            for i in range(n)]  
vorig = [[vp.sphere(pos=vspheres[i].pos+vspins[j].axis, \
                    color=vcolors[j], radius=0.15, opacity=0.6)\
                for j in range(n)]\
                    for i in range(n)]

#######################################################################################

def update_viz():
    global eng, pts, inverted, n
    global initial_rots, initial_R, initial_xyzs, t, state
    eng.reset()
    views = make_views(pts)
    R = GaussianTransform(views_symplectic(views, inverted=inverted))
    prog = sf.Program(2*n)
    with prog.context as q:
        for i in range(0, 2*n, 2):
            Sgate(0.5) | q[i]
            initial_rots[int(i/2)][0] | (q[i], q[i+1])
            initial_rots[int(i/2)][1] | (q[i], q[i+1])
        initial_R | q
        R | q
    state = eng.run(prog).state
    xyzs = state_xyzs(state)
    for i in range(n):
        vspheres[i].pos = vp.vector(*pts[i])
        for j in range(n-1):
            vstars[i][j].pos = vspheres[i].pos + \
                    vspheres[i].radius*vp.vector(*views[i][j])
        vspins[i].pos = vspheres[i].pos
        vspins[i].axis = vp.vector(*xyzs[i])
        for i in range(n):
            for j in range(n):
                vorig[i][j].pos = vspheres[i].pos+vp.vector(*initial_xyzs[j])
   
#######################################################################################

selected = -1
touched = False
def mousedown(e):
    global vspheres, selected
    picked = vp.scene.mouse.pick 
    if picked in vspheres:
        selected = vspheres.index(picked)
    else:
        selected = -1

def mouseup(e):
    global selected
    selected = -1

def mousemove(e):
    global pts, touched
    if selected != -1:
        pts[selected] = np.array(vp.scene.mouse.pos.value)
        touched = True

vp.scene.bind('mousedown', mousedown)
vp.scene.bind('mouseup', mouseup)
vp.scene.bind('mousemove', mousemove)

##################################################################

update_viz()
while True:
    if touched:
        update_viz()
        touched = False