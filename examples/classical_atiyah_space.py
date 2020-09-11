import qutip as qt
import numpy as np
import vpython as vp
from magic import *
import scipy as sc

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

def get_phase(q):
    c = sum(q.full().T[0][::-1])
    return np.exp(1j*np.angle(c))

def mat_coeffs(M, basis):
    return np.array([(o.dag()*M).tr() for i, o in enumerate(basis)])

def coeffs_mat(C, basis):
    return sum([C[i]*o for i, o in enumerate(basis)])

#######################################################################################

def su(n):
    annotations = []
    diagonals = [np.zeros((n,n), dtype=complex) for i in range(n)]
    for i in range(n):
        diagonals[i][i,i] = 1
        annotations.append(('Ez', i))
    xlike = [np.zeros((n, n), dtype=complex) for i in range(int(n*(n-1)/2))]
    r = 0
    for i in range(n):
        for j in range(n):
            if i < j:
                xlike[r][i,j] = 1/np.sqrt(2)
                xlike[r][j,i] = 1/np.sqrt(2)
                r +=1 
                annotations.append(('Ex', i, j))
    ylike = [np.zeros((n, n), dtype=complex) for i in range(int(n*(n-1)/2))]
    r = 0
    for i in range(n):
        for j in range(n):
            if i < j:
                ylike[r][i,j] = 1j/np.sqrt(2)
                ylike[r][j,i] = -1j/np.sqrt(2)
                r +=1 
                annotations.append(('Ey', i, j))
    return [qt.Qobj(o) for o in diagonals + xlike + ylike], annotations

def su2(n, half=True):
    XYZ = {"X": qt.sigmax(), "Y": qt.sigmay(), "Z": qt.sigmaz()}
    S = [dict([(o, (0.5 if half else 1)*qt.Qobj(\
          sc.linalg.block_diag(*\
            [np.zeros((2,2)) if i !=j else XYZ[o].full() \
                for j in range(n)]))) \
                    for o in XYZ.keys()])
                        for i in range(n)]
    sun, annotations = su(n)
    E = [(1/np.sqrt(2))*qt.tensor(o, qt.identity(2)) for o in sun]
    for e in E:
        e.dims = [[e.shape[0]], [e.shape[0]]]
    return S, E, annotations

def rand_su2n_state(n):
    return su2n_state([qt.rand_ket(2) for i in range(n)])

def su2n_state(spinors):
    return qt.Qobj(np.concatenate([q.full().T[0] for q in spinors]))

def split_su2n_state(state):
    v = state.full().T[0]
    return [qt.Qobj(np.array([v[i], v[i+1]])) for i in range(0, len(v), 2)]

def su2n_phases(state):
    return [get_phase(spinor) for spinor in split_su2n_state(state)]

def decompose_su2n(O, S, E):
    s = [dict([(o, (S[i][o]*O).tr()) for o in ["X", "Y", "Z"]]) for i in range(len(S))]
    e = [(E[i]*O).tr() for i in range(len(E))]
    return s, e

def reconstruct_su2n(s, e, S, E):
    terms = []
    for i in range(len(S)):
        for o in ["X", "Y", "Z"]:
            terms.append(s[i][o]*S[i][o]) if not np.isclose(s[i][o], 0) else None
    for i in range(len(E)):
        terms.append(e[i]*E[i]) if not np.isclose(e[i], 0) else None
    return sum(terms)

def display_su2n(s, e, annotations):
    r = 0
    for i in range(len(S)):
        print("s%d: %s" % (i, "".join(["%s: %s " % (o, s[i][o]) for o in ["X", "Y", "Z"]])))
    for i in range(len(E)):
        print("%s : %s" % (annotations[i], e[i]))

#######################################################################################

inverted = False
show_originals = True

n = 2
#pts = [3*np.random.randn(3) for i in range(n)] 
pts = [np.array([0, 0, 3*i - n/2]) for i in range(n)]

views = make_views(pts)
spinors = [qt.rand_ket(2) for i in range(n)]
state = su2n_state(spinors)
phases = su2n_phases(state)
S, E, annotations = su2(n)
P = constellations_matrix(views_constellations(views))

initial_state = P*state if inverted else P.dag()*state

##################################################################

vp.scene = vp.canvas(background=vp.color.white, width=1000, height=800)
vcolors = [vp.vector(*np.random.rand(3)) for i in range(n)]
pieces = split_su2n_state(state)

if show_originals:
    vspheres_ = [vp.sphere(color=vcolors[i], radius=pieces[i].norm(), opacity=0.05,\
                          pos=vp.vector(*pts[i]))\
                    for i in range(n)]
    vstars_ = [[vp.sphere(radius=0.15, color=vp.color.black, emissive=True,\
                         pos=vspheres_[i].pos+vp.vector(*views[i][j]))
                for j in range(n-1)]\
                    for i in range(n)]
    vphases_ = [vp.arrow(pos=vspheres_[i].pos,color=vp.color.yellow, opacity=0.6,\
                        axis=vp.vector(phases[i].real, phases[i].imag, 0))\
                    for i in range(n)]

vspheres = [vp.sphere(color=vcolors[i], radius=pieces[i].norm(), opacity=0.3,\
                      pos=vp.vector(*pts[i]))\
                for i in range(n)]

vstars = [[vp.sphere(radius=0.15, emissive=True,\
                     pos=vspheres[i].pos+vp.vector(*views[i][j]))
            for j in range(n-1)]\
                for i in range(n)]
vspins = [vp.arrow(pos=vspheres[i].pos,\
                   axis=2*vp.vector(qt.expect(S[i]["X"], state),\
                                    qt.expect(S[i]["Y"], state),\
                                    qt.expect(S[i]["Z"], state)))\
            for i in range(n)]  

vspins_ = [vp.arrow(pos=vspheres[i].pos, visible=False,\
                   axis=2*vp.vector(qt.expect(S[i]["X"], state),\
                                    qt.expect(S[i]["Y"], state),\
                                    qt.expect(S[i]["Z"], state)))\
            for i in range(n)]

vphases = [vp.arrow(pos=vspheres[i].pos,color=vp.color.magenta,\
                    axis=vp.vector(phases[i].real, phases[i].imag, 0),\
                    opacity=0.6)\
                for i in range(n)]

vorig = [[vp.sphere(pos=vspheres[i].pos+vspins_[j].axis, \
                    color=vcolors[j], radius=0.15, opacity=0.6)\
                for j in range(n)]\
                    for i in range(n)]

def update_viz(verbose=False):
    global n, pts, vspheres, vstars, vspins, initial_state
    global S, E, annotations, vspins_, inverted
    views = make_views(pts)
    P = constellations_matrix(views_constellations(views))
    s, e = decompose_su2n(P, S, E)
    spin_state = P.dag()*initial_state if inverted else P*initial_state
    pieces = split_su2n_state(spin_state)
    phases = su2n_phases(spin_state)
    #summing = [] if verbose else None
    for i in range(n):
        vspheres[i].pos = vp.vector(*pts[i])
        vspheres[i].radius = pieces[i].norm()
        for j in range(n-1):
            vstars[i][j].pos = vspheres[i].pos + vspheres[i].radius*vp.vector(*views[i][j])
        vspins[i].pos = vspheres[i].pos
        exp = 2*np.array([qt.expect(S[i]["X"], spin_state),\
                          qt.expect(S[i]["Y"], spin_state),\
                          qt.expect(S[i]["Z"], spin_state)])
        vspins[i].axis = vp.vector(*exp)
        vphases[i].pos = vspheres[i].pos
        vphases[i].axis = vp.vector(phases[i].real, phases[i].imag, 0)
        #summing.append(exp) if verbose else None
        for i in range(n):
            for j in range(n):
                vorig[i][j].pos = vspheres[i].pos+vspins_[j].axis
    if verbose:
        print("*****************************************")
        display_su2n(s, e, annotations)
        print("*****************************************")
        #print(sum(summing)) if verbose else None

##################################################################

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