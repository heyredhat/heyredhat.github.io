import qutip as qt
import numpy as np
from magic import *
from vhelper import *
scene = vp.canvas(background=vp.color.white, width=1000, height=800)

################################################################################################

def U_g(g):
    m = np.zeros((g.shape[0], g.shape[0]), dtype=complex)
    m[:,0] = g.full().T[0]
    return qt.Qobj(np.linalg.qr(m)[0])

def g_inv(g):
    d = g.shape[0]
    return U_g(g).dag()*qt.basis(d,0)

def pov(state, k):
    if k == 0:
        return state
    n = len(state.dims[0])
    d = state.dims[0][0]
    O = sum([qt.tensor(*[qt.identity(d) if j == 0 else \
                         g_inv(qt.basis(d,i))*qt.basis(d,i).dag() if j == k else \
                         U_g(qt.basis(d, i)) for j in range(n)])
                                       for i in range(d)])
    return qt.tensor_swap(O*state, (0, k))

################################################################################################

def to_xyz(dm):
    return np.array([qt.expect(qt.sigmax(), dm),\
                     qt.expect(qt.sigmay(), dm),\
                     qt.expect(qt.sigmaz(), dm)])
def disp_subsystems(state):
    for i in range(len(state.dims[0])):
        p = state.ptrace(i)
        print("%d: e: %.3f | %s\n%s" % (i, qt.entropy_vn(p), to_xyz(p), p.full()))
    print()

################################################################################################

state = qt.tensor(qt.basis(2,0), qt.bell_state("00"))

################################################################################################

n = len(state.dims[0])
d = state.dims[0][0]
j = (d-1)/2

dt = 0.01
H = qt.tensor(qt.identity(d), qt.rand_herm(d**(n-1)))
H.dims = [state.dims[0], state.dims[0]]
U = (-1j*H*dt).expm()

states = [pov(state, i) for i in range(n)]

################################################################################################

vdms = [[VisualDensityMatrix(states[i].ptrace(j), pos=[2*j, 3*n-3*i, 0])\
			for j in range(n)] for i in range(n)]
ventropies = [[vp.label(text="%.3f" % (qt.entropy_vn(states[i].ptrace(j))),\
				pos=vp.vector(2*j, 3*n-3*i-1.5, 0))\
			 for j in range(n)] for i in range(n)]

################################################################################################

input()
while True:
	state = U*state
	states = [pov(state, i) for i in range(n)]
	for i in range(n):
		for j in range(n):
			pt = states[i].ptrace(j)
			vdms[i][j].update(pt)
			ventropies[i][j].text = "%.3f" % (qt.entropy_vn(states[i].ptrace(j)))



