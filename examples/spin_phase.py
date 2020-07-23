import qutip as qt
import numpy as np
import vpython as vp
from magic import *

j = 5/2
n = int(2*j + 1)
dt = 0.1
state = qt.rand_ket(n)
H = qt.jmat(j, 'y')
U = (-1j*H*dt).expm()

vp.sphere(color=vp.color.blue, opacity=0.5)
initial_vstars = [vp.sphere(pos=vp.vector(*xyz),\
					       radius=0.2, emissive=True,\
					       color=vp.color.yellow) 
								for xyz in spin_XYZ(state)]

vstars = [vp.sphere(pos=vp.vector(*xyz),\
					radius=0.2, emissive=True) 
								for xyz in spin_XYZ(state)]

phase = get_phase(state)
initial_vphase = vp.arrow(pos=vp.vector(0,2,0),color=vp.color.yellow,\
				  axis=vp.vector(phase.real, phase.imag, 0))
vphase = vp.arrow(pos=vp.vector(0,2,0),\
				  axis=vp.vector(phase.real, phase.imag, 0))

touched = False
def keyboard(e):
	global state, U, touched
	k = e.key
	if k == "a":
		state = U.dag()*state
	elif k == "d":
		state = U*state
	touched = True

vp.scene.bind("keydown", keyboard)

while True:
	if touched:
		for i, xyz in enumerate(spin_XYZ(state)):
			vstars[i].pos = vp.vector(*xyz)
			phase = get_phase(state)
			vphase.axis = vp.vector(phase.real, phase.imag, 0)
		touched = False
