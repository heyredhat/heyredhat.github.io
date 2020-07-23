import qutip as qt
import numpy as np
import vpython as vp
vp.scene.width = 1000
vp.scene.height = 800

dt = 0.01
n = 101
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

state = qt.basis(n,0)
amps = [state.overlap(v) for v in QV]
vamps = [vp.arrow(pos=vp.vector(QL[i], 0, 0),\
	      axis=vp.vector(amps[i].real, amps[i].imag, 0)) for i in range(n)]
vprobs = [vp.sphere(radius=0.1, color=vp.color.red,\
				    pos=vp.vector(QL[i], 1+3*(amps[i]*np.conjugate(amps[i])).real, 0))\
						for i in range(n)]
vexp = vp.sphere(color=vp.color.yellow, radius=0.3,\
					pos=vp.vector(qt.expect(Q, state), 0, 0))

grid_pts = 20
grid = np.linspace(-10, 10, 20)
CS = [[coherent(x+1j*y) for y in grid] for x in grid]
cs = [[state.overlap(CS[i][j]) for j in range(grid_pts)] for i in range(grid_pts)]
vcs = [[vp.arrow(pos=vp.vector(x, y+12, 0), color=vp.color.green,\
				 axis=vp.vector(cs[i][j].real, cs[i][j].imag,0))
			for j, y in enumerate(grid)] for i, x in enumerate(grid)]

def keyboard(e):
	global state, n, NV
	key = e.key
	if key == "i":
		state = qt.rand_ket(n)
	elif key == "c":
		state = coherent(np.random.randint(1,5)*(np.random.randn() + np.random.randn()*1j))
	else:
		state = NV[int(key)]

vp.scene.bind('keydown', keyboard)

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
