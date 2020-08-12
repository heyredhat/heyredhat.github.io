import qutip as qt
import numpy as np
import vpython as vp
from magic import *

vp.scene.width = 1000
vp.scene.height = 800

CALC_ROOTS = True
CALC_SPHERE = True

dt = 0.01
n = 8
a = qt.destroy(n)
Q = qt.position(n)
QL, QV = Q.eigenstates()
N = qt.num(n)
NL, NV = N.eigenstates()
H = N + 1/2
U = (-1j*dt*H).expm()

def coherent(s):
	global n, a
	return np.exp(-s*np.conjugate(s)/2)*(s*a.dag()).expm()*(-np.conjugate(s)*a).expm()*qt.basis(n,0)

def oscillator_roots(q):
	n = q.shape[0]
	poly = [(c/np.sqrt(np.math.factorial(n-i))) for i, c in enumerate(q.full().T[0][::-1])]
	return [c.conjugate() for c in np.roots(poly)]

state = qt.basis(n,0)
amps = [state.overlap(v) for v in QV]
vamps = [vp.arrow(pos=vp.vector(QL[i], 0, 0),\
	      axis=vp.vector(amps[i].real, amps[i].imag, 0)) for i in range(n)]
vprobs = [vp.sphere(radius=0.1, color=vp.color.red,\
				    pos=vp.vector(QL[i], 1+3*(amps[i]*np.conjugate(amps[i])).real, 0))\
						for i in range(n)]
vexp = vp.sphere(color=vp.color.yellow, radius=0.3,\
					pos=vp.vector(qt.expect(Q, state), 0, 0))

grid_pts = 25
grid = np.linspace(-10, 10, grid_pts)
CS = [[coherent(x+1j*y) for y in grid] for x in grid]
cs = [[state.overlap(CS[i][j]) for j in range(grid_pts)] for i in range(grid_pts)]
vcs = [[vp.arrow(pos=vp.vector(x, y+12, 0), color=vp.color.green,\
				 axis=vp.vector(cs[i][j].real, cs[i][j].imag,0))
			for j, y in enumerate(grid)] for i, x in enumerate(grid)]

if CALC_ROOTS or CALC_SPHERE:
	C = oscillator_roots(state)

if CALC_ROOTS:
	vC = [vp.sphere(color=vp.color.magenta, radius=0.15,opacity=0.8,\
				   pos=vp.vector(r.real, r.imag+12, 0))\
						for r in C]

if CALC_SPHERE:
	vsphere = vp.sphere(color=vp.color.blue, opacity=0.3, pos=vp.vector(0,4,0))
	xyzs = [c_xyz(r) for r in C]
	vsC = [vp.sphere(radius=0.15,opacity=0.8,\
				    pos=vsphere.pos+vp.vector(*xyz))\
						for xyz in xyzs]

def keyboard(e):
	global state, n, NV, H, U
	key = e.key
	if key == "i":
		state = qt.rand_ket(n)
	elif key == "c":
		state = coherent(np.random.randint(1,5)*(np.random.randn() + np.random.randn()*1j))
	elif key == "h":
		H = qt.rand_herm(n)
		U = (-1j*dt*H).expm()
	elif key == "e":
		H = N + 1/2
		U = (-1j*dt*H).expm()		
	elif key == "n":
		H = H + qt.rand_herm(n)*0.1
		U = (-1j*dt*H).expm()		
	else:
		if key.isdigit():
			state = NV[int(key)]

vp.scene.bind('keydown', keyboard)
print("Loaded.")
T = 0
while True:
	state = U*state
	amps = [state.overlap(v) for v in QV]
	for i in range(n):
		vamps[i].axis = vp.vector(amps[i].real, amps[i].imag, 0)
		vprobs[i].pos = vp.vector(QL[i], 1+3*(amps[i]*np.conjugate(amps[i])).real, 0)
	vexp.pos = vp.vector(qt.expect(Q, state), 0, 0)

	cs = [[state.overlap(CS[i][j]) for j in range(grid_pts)] for i in range(grid_pts)]
	for i, x in enumerate(grid):
		for j, y in enumerate(grid):
			vcs[i][j].axis = vp.vector(cs[i][j].real, cs[i][j].imag,0)

	if CALC_ROOTS or CALC_SPHERE:
		if T % 5 == 0:
			C = oscillator_roots(state)

	if CALC_ROOTS:		
		if T % 5 == 0:	
			if len(vC) > len(C):
				for i in range(len(vC) - len(C)):
					vC[0].visible = False
					del vC[0]
			elif len(C) > len(vC):
				for i in range(len(C) - len(vC)):
					vC.append(vp.sphere(color=vp.color.magenta, radius=0.15, opacity=0.8))

			for i,r in enumerate(C):
				vC[i].pos = vp.vector(r.real, r.imag+12, 0)

	if CALC_SPHERE:
		if T % 5 == 0:
			if len(vsC) > len(C):
				for i in range(len(vsC) - len(C)):
					vsC[0].visible = False
					del vsC[0]
			elif len(C) > len(vsC):
				for i in range(len(C) - len(vsC)):
					vsC.append(vp.sphere(radius=0.15, opacity=0.8))
			xyzs = [c_xyz(r) for r in C]
			for i, xyz in enumerate(xyzs):
				vsC[i].pos = vsphere.pos+vp.vector(*xyz)
							
	T += 1
