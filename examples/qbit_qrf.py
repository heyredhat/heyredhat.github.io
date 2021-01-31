import numpy as np
import qutip as qt
import scipy as sc
from itertools import permutations, product

def construct_swap(a, b, dims):
    perm = list(range(len(dims)))
    perm[a], perm[b] = perm[b], perm[a]
    tensor_indices = list(product(*[[(i, j) for j in range(dims[i])] for i in range(len(perm))]))
    ptensor_indices = list(product(*[[(i, j) for j in range(dims[i])] for i in perm]))
    m = np.zeros((len(tensor_indices), len(tensor_indices)))
    for i, pind in enumerate(ptensor_indices):
        m[i, [tensor_indices.index(p) for p in permutations(pind) if p in tensor_indices][0]] = 1
    M = qt.Qobj(m)
    M.dims = [dims, dims]
    return M

def take_pov(state, from_pov, to_pov, G, basis=None, return_map=False):
    if from_pov == to_pov:
        return state
    n = len(state.dims[0])
    d = state.dims[0][0]
    basis = basis if type(basis) != type(None) else [qt.basis(d, i) for i in range(d)]
    g_inv = lambda i: G(i).dag()*basis[0]
    O = construct_swap(from_pov, to_pov, state.dims[0])*\
        sum([qt.tensor(*[qt.identity(d) if j == from_pov \
                        else g_inv(i)*basis[i].dag() if j == to_pov \
                        else G(i).dag() for j in range(n)]) for i in range(d)])
    return O if return_map else O*state    

######################################################

Xbasis = qt.sigmax().eigenstates()[1][::-1]
Ybasis = qt.sigmay().eigenstates()[1][::-1]
Zbasis = qt.sigmaz().eigenstates()[1][::-1]

######################################################

G = lambda i: qt.identity(2) if i == 0 else qt.sigmaz()
basis = Xbasis

#state = qt.tensor(Xbasis[0], qt.rand_ket(2), qt.rand_ket(2))
state = qt.tensor(Xbasis[0], Zbasis[1], Xbasis[0])
#state = qt.tensor(Xbasis[0], qt.bell_state("00"))

######################################################

#G = lambda i: qt.identity(2) if i == 0 else qt.sigmaz()
#basis = Ybasis

#state = qt.tensor(Ybasis[0], qt.rand_ket(2), qt.rand_ket(2))
#state = qt.tensor(Ybasis[0], Zbasis[1], Xbasis[0])
#state = qt.tensor(Ybasis[0], qt.bell_state("00"))

######################################################

#G = lambda i: qt.identity(2) if i == 0 else qt.sigmax()
#basis = Zbasis

#state = qt.tensor(Zbasis[0], qt.rand_ket(2), qt.rand_ket(2))
#state = qt.tensor(Zbasis[0], Zbasis[1], Xbasis[0])
#state = qt.tensor(Zbasis[0], qt.bell_state("11"))

######################################################

H0 = qt.tensor(qt.identity(2), qt.sigmay(), qt.identity(2))

######################################################

import vpython as vp

def create_viz():
    global state, G, basis
    scene = vp.canvas(background=vp.color.white)
    n = len(state.dims[0])
    colors = [vp.color.red, vp.color.green, vp.color.blue] if n == 3 else\
             [vp.vector(*np.random.random(3)) for i in range(n)]

    vpts = {}
    for i in range(n):
        pov_state = take_pov(state, 0, i, G, basis=basis)
        vp.label(text="pov: %d" % i,\
                 pos=vp.vector(3*i-n,-1.5,0),\
                 color=colors[i])
        vp.sphere(color=colors[i],\
                  opacity=0.1,\
                  pos=vp.vector(3*i-n, 0, 0))
        for k in range(n):
            partial = pov_state.ptrace(k)
            xyz = np.array([qt.expect(qt.sigmax(), partial),\
                            qt.expect(qt.sigmay(), partial),\
                            qt.expect(qt.sigmaz(), partial)])
            vpts[(i,k)] = vp.arrow(opacity=0.3,\
                                   pos=vp.vector(3*i-n, 0, 0)+0.01*vp.vector(*np.random.randn(3)),\
                                   color=colors[k],\
                                   axis=vp.vector(*xyz),\
                                   visible=not np.isclose(np.linalg.norm(xyz), 0))
    return vpts

vpts = create_viz()

def viz_update():
    global state, G, basis, vpts
    n = len(state.dims[0])
    for i in range(n):
        pov_state = take_pov(state, 0, i, G, basis=basis)
        for k in range(n):
            partial = pov_state.ptrace(k)
            xyz = np.array([qt.expect(qt.sigmax(), partial),\
                            qt.expect(qt.sigmay(), partial),\
                            qt.expect(qt.sigmaz(), partial)])
            vpts[(i,k)].axis = vp.vector(*xyz)
            vpts[(i,k)].visible = not np.isclose(np.linalg.norm(xyz), 0)

def evolve(H, dt=0.1, T=2*np.pi):
    global state
    U = (-1j*H*dt).expm()
    for t in range(int(T/dt)):
        state = U*state
        viz_update()
        
def measure(i, o='z'):
    global state
    n = len(state.dims[0])
    d = state.dims[0][0]
    dm = state.ptrace(i)
    O = qt.jmat(1/2, o)
    L, V = O.eigenstates()
    P = [v*v.dag() for v in V]
    probs = np.array([qt.expect(p, dm) for p in P])
    choice = np.random.choice(list(range(len(probs))), p=abs(probs/sum(probs)))
    proj = qt.tensor(*[qt.identity(d)]*i, P[choice], *[qt.identity(d)]*(n-i-1))
    print("%d collapses to %f" % (i, L[choice]))
    state = (proj*state).unit()
    viz_update()

######################################################


