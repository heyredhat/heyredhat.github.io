import qutip as qt
import numpy as np
import vpython as vp
vp.scene.width = 1000
vp.scene.height = 800

CALC_ROOTS = True

dt = 0.001
n = 20
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

vpicker = vp.sphere(color=vp.color.green, radius=0.4, pos=vp.vector(0, 7, 0))
vplane = vp.box(pos=vp.vector(0, 7, 0), length=10, height=10, width=0.01)

if CALC_ROOTS:
	C = oscillator_roots(state)
	vC = [vp.sphere(color=vp.color.magenta, radius=0.15,opacity=0.8,\
				   pos=vp.vector(r.real, r.imag+7, 0))\
						for r in C]

selected = False
s = 0j
def mouse_click(e):
	global vpicker, selected
	if vp.scene.mouse.pick == vpicker:
		selected = True
	else:
		selected = False

def mouse_move(e):
	global vpicker, selected, state, s
	if selected:
		vpicker.pos = vp.scene.mouse.pos
		vpicker.pos.z = 0
		s = vpicker.pos.x + (vpicker.pos.y-7)*1j

vp.scene.bind('click', mouse_click)
vp.scene.bind('mousemove', mouse_move)

def display():
	amps = [state.overlap(v) for v in QV]
	for i in range(n):
		vamps[i].axis = vp.vector(amps[i].real, amps[i].imag, 0)
		vprobs[i].pos = vp.vector(QL[i], 1+3*(amps[i]*np.conjugate(amps[i])).real, 0)
	vexp.pos = vp.vector(qt.expect(Q, state), 0, 0)

T = 0
print("Loaded.")
while True:
	state = coherent(s)
	display() if T % 5 == 0 else None
	if CALC_ROOTS:
		if T % 5 == 0:
			C = oscillator_roots(state)
			if len(vC) > len(C):
				for i in range(len(vC) - len(C)):
					vC[0].visible = False
					del vC[0]
			elif len(C) > len(vC):
				for i in range(len(C) - len(vC)):
					vC.append(vp.sphere(color=vp.color.magenta, radius=0.15, opacity=0.8))

			for i,r in enumerate(C):
				vC[i].pos = vp.vector(r.real, r.imag+7, 0)
	T += 1