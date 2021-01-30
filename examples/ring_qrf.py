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

def shift(d):
    return sum([qt.basis(d, 0)*qt.basis(d, i).dag() if i == d-1 else qt.basis(d,i+1)*qt.basis(d,i).dag() for i in range(d)])

def shift_generator(d):
    return 1j*qt.Qobj(sc.linalg.logm(shift(d)))

def cntrl_shift(d):
    T = shift(d)
    S = sum([qt.tensor(qt.basis(d,i)*qt.basis(d,i).dag(), T**i) for i in range(d)])
    CS = qt.Qobj(1j*sc.linalg.logm(S))
    CS.dims = [[d,d],[d,d]]
    return CS

######################################################

d = 8
X = shift(d)
G = lambda i: X**i
state = qt.tensor(qt.basis(d,0), qt.basis(d, 3), qt.basis(d, 5))
#state = qt.tensor(qt.basis(d, 0), qt.basis(d, 2), (qt.basis(d, 4)+qt.basis(d, 6).unit()))
#state = qt.tensor(qt.basis(d,0), sum([qt.tensor(qt.basis(d, i), qt.basis(d,i)) for i in range(d)]).unit())
#state = qt.tensor(qt.basis(d,0), (qt.tensor(qt.basis(d,2), qt.basis(d, 5)) + qt.tensor(qt.basis(d, 3), qt.basis(d,5))).unit())

H0 = qt.tensor([qt.identity(d), shift_generator(d), qt.identity(d)])
H1 = qt.tensor([qt.identity(d), cntrl_shift(d)])

######################################################

import vpython as vp

def create_viz():
    global state, G
    scene = vp.canvas(background=vp.color.white)
    n = len(state.dims[0])
    d = state.dims[0][0]
    coords = [vp.vector(root.real, root.imag, 0) for root in [np.exp(1j*2*np.pi*i/d)for i in range(d)]]
	colors = [vp.color.red, vp.color.green, vp.color.blue] if n == 3 else\
             [vp.vector(*np.random.random(3)) for i in range(n)]  

    vpts = {}
    for i in range(n):
        pov_state = take_pov(state, 0, i, G)
        vp.label(text="pov: %d" % i,\
                 pos=vp.vector(3*i-n,-1.5,0),\
                 color=colors[i])
        vp.ring(color=colors[i],\
                radius=1,\
                axis=vp.vector(0,0,1),\
                thickness=0.01,\
                pos=vp.vector(3*i-n, 0, 0))
        for k in range(n):
            partial = pov_state.ptrace(k).full()
            for j in range(d):
                vpts[(i, k, j)] = vp.sphere(opacity=0.5,\
                                            pos=vp.vector(3*i-n,np.random.randn()/25,0)+coords[j],\
                                            color=colors[k],\
                                            radius=partial[j,j].real/4)
    return vpts

vpts = create_viz()

def viz_update():
    global state, G, vpts
    n = len(state.dims[0])
    d = state.dims[0][0]
    for i in range(n):
        pov_state = take_pov(state, 0, i, G)
        for k in range(n):
            partial = pov_state.ptrace(k).full()
            for j in range(d):
                p = partial[j,j].real
                vpts[(i, k, j)].radius = partial[j,j].real/4

def evolve(H, dt=0.1, T=2*np.pi):
    global state
    U = (-1j*H*dt).expm()
    for t in range(int(T/dt)):
        state = U*state
        viz_update()
        
def measure(i):
    global state
    n = len(state.dims[0])
    d = state.dims[0][0]
    dm = state.ptrace(i).full()
    probs = np.array([dm[i,i].real for i in range(dm.shape[0])])
    choice = np.random.choice(list(range(len(probs))), p=abs(probs/sum(probs)))
    proj = qt.tensor(*[qt.identity(d)]*i, qt.basis(d, choice)*qt.basis(d, choice).dag(), *[qt.identity(d)]*(n-i-1))
    print("%d collapses to %d" % (i, choice))
    state = (proj*state).unit()
    viz_update()

######################################################


