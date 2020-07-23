import qutip as qt
import numpy as np
from quantum_polyhedron import *
scene = vp.canvas(background=vp.color.white, width=1000, height=500)

a = QuantumPolyhedron([1/2,1/2,1/2,1/2])
b = QuantumPolyhedron([1/2,1/2,1/2,1/2])

state = sum([qt.tensor(qt.basis(a.d, i), qt.basis(a.d, i)) for i in range(a.d)]).unit()
#state = qt.rand_ket(a.d*b.d)
state.dims = [[a.d, b.d], [1,1]]
A = state.ptrace(0)
B = state.ptrace(1)

pA = QuantumPolyhedron(a.js, initial=A, pos=[-1,0,0], show_poly=True, show_spin=True)
pB = QuantumPolyhedron(b.js, initial=B, pos=[1,0,0], show_poly=True, show_spin=True)

def evolve(H=None, dt=0.01, T=100):
	global state
	if type(H) == type(None):
		H = qt.rand_herm(state.shape[0])
		H.dims = [state.dims[0], state.dims[0]]
	U = (-1j*dt*H).expm()
	for t in range(T):
		state = U*state
		A = state.ptrace(0)
		B = state.ptrace(1)
		pA.set(A)
		pB.set(B)

def measure(which, O):
	global state, a, b, pA, pB
	if which == "a":
		state = (qt.tensor(pA.measure(O, defer=True), qt.identity(b.d))*state).unit()
	else:
		state = (qt.tensor(qt.identity(a.d), pB.measure(O, defer=True))*state).unit()
	A = state.ptrace(0)
	B = state.ptrace(1)
	pA.set(A)
	pB.set(B)

#evolve()
#evolve(qt.tensor(qt.sigmax(), qt.identity(2)))
#measure("a", pA.INNER_PRODUCTS[0][1])
