import scipy
import qutip as qt
import numpy as np
import vpython as vp

vp.scene.width = 900
vp.scene.height = 600

################################################################################

def qubit_xyz(spin):
	return [qt.expect(qt.sigmax(), spin),\
			qt.expect(qt.sigmay(), spin),\
			qt.expect(qt.sigmaz(), spin)]

################################################################################

spin_n = 2
clock_n = 7

################################################################################

spin_initial = qt.rand_ket(spin_n)
spin_H = qt.rand_herm(spin_n)
dt = (2*np.pi)/clock_n
spin_U = (-1j*spin_H*dt).expm()
psi_t = lambda t: (-1j*spin_H*t).expm()*spin_initial
spin_history = [psi_t(t*dt) for t in range(clock_n)]

################################################################################

H = sum([qt.tensor(spin_U, qt.basis(clock_n, t+1)*qt.basis(clock_n, t).dag())\
	    +qt.tensor(spin_U.dag(), qt.basis(clock_n, t)*qt.basis(clock_n, t+1).dag())\
		 	for t in range(clock_n-1)])
L, V = H.eigenstates()
H_projs = [v*v.dag() for v in V]

Z = qt.sigmaz()
ZL, ZV = Z.eigenstates()
Z_projs = [qt.tensor(v*v.dag(), qt.identity(clock_n)) for v in ZV]

TIME_projs = [qt.basis(clock_n,i)*qt.basis(clock_n,i).dag() for i in range(clock_n)]

SHIFT = np.array([[1 if j == i-1 or (i == 0 and j == clock_n-1) else 0 for j in range(clock_n)] for i in range(clock_n)])
SHIFT = qt.tensor(qt.identity(spin_n), qt.Qobj(SHIFT))

################################################################################

initial = qt.tensor(spin_initial, qt.basis(clock_n, 0))
state = initial.copy()

spin_part = state.ptrace(0)
clock_part = state.ptrace(1)

SL, SV = spin_part.eigenstates()
clock_probs = np.array([(clock_part*TIME_projs[i]).tr() for i in range(clock_n)])

################################################################################

colors = [vp.color.red, vp.color.orange, vp.color.yellow,\
		  vp.color.green, vp.color.white, vp.color.magenta]

vspin_sphere = vp.sphere(pos=vp.vector(-3,0,0),radius=2,\
						 color=vp.color.blue, opacity=0.4)
vspin_layers = [vp.sphere(pos=vspin_sphere.pos+vspin_sphere.radius*vp.vector(*qubit_xyz(SV[i])),\
			   color=colors[i], radius=0.5, opacity=SL[i]) for i in range(spin_n)]

################################################################################

vclock_ring = vp.ring(pos=vp.vector(2,0,0), 
					  axis=vp.vector(0,0,1),\
					  color=vp.color.red,\
					  radius=1, thickness=0.1, opacity=0.5)
roots_of_unity = [np.exp(2*np.pi*1j*i/clock_n) for i in range(clock_n)]
directions = [vp.vector(roots_of_unity[i].real,\
						roots_of_unity[i].imag,\
						0) for i in range(clock_n)]
vclock_arrows = [vp.arrow(pos=vclock_ring.pos,\
						  axis=directions[i],\
						  opacity=clock_probs[i]) for i in range(clock_n)]
vprojarrow = vp.arrow(pos=vclock_ring.pos, 
				      color=vp.color.white, 
				      opacity=0.7,\
					  visible=False)

################################################################################

for t in range(clock_n):
	moment_sphere = vp.sphere(pos=vclock_ring.pos+2*directions[t],\
					radius=0.5,\
					color=vp.color.blue,\
					opacity=0.5)
	moment_stars = vp.sphere(pos=moment_sphere.pos+0.5*vp.vector(*qubit_xyz(spin_history[t])),\
					 		 radius=0.1)

################################################################################

npdirections = [np.array([roots_of_unity[i].real,\
				  		  roots_of_unity[i].imag,\
				  		  0]) for i in range(clock_n)]
expected_time = sum([npdirections[i]*clock_probs[i] for i in range(clock_n)])
vexpected_time = vp.sphere(color=vp.color.red, radius=0.1,\
						   pos=vp.vector(*expected_time), opacity=0.5)

################################################################################

print(\
"""
   c: measure clock  
   v: measure spin-z
   b: measure total energy

   9/0: unshift/shift clock

   r: toggle clock proj
   e: tick back
   t: tick forward

   a/d: x-rot
   s/w: y-rot
   z/x: z-rot

   q: toggle evolution
   p: ground state
   o: random state
""")
def keyboard(e):
	global eta, clock_n, clock_probs, spin_n, Z_projs, ZV, H, running, ground, shift, display_proj, pointing_to, L
	key = e.key
	if key == "c":
		choice = np.random.choice(list(range(clock_n)), p=np.array(clock_probs)/sum(np.array(clock_probs)))
		eta = (qt.tensor(qt.identity(spin_n), clock_projs[choice])*eta).unit()
	elif key == "v":
		probs = np.array([(eta*eta.dag()*p).tr() for p in Z_projs]).real
		choice = np.random.choice(list(range(len(ZV))), p=probs/sum(probs))
		eta = (Z_projs[choice]*eta).unit()
	elif key == "b":
		probs = np.array([(eta*eta.dag()*p).tr() for p in HP]).real
		choice = np.random.choice(list(range(len(HP))), p=probs/sum(probs))
		eta = (HP[choice]*eta).unit()
		print("energy eigenvalue: %.4f" % (L[choice]))
	elif key == "9":
		eta = (shift*eta).unit()
	elif key == "0":
		eta = (shift.dag()*eta).unit()
	elif key == "r":
		if display_proj == False:
			display_proj = True
		else:
			display_proj = False
	elif key == "e":
		pointing_to -= 1
		if pointing_to < 0:
			pointing_to = clock_n-1
	elif key == "t":
		pointing_to += 1
		if pointing_to >= clock_n:
			pointing_to = 0
	elif key == "a":
		eta = qt.tensor((qt.jmat((spin_n-1)/2, 'x')*0.1*-1j).expm(), qt.identity(clock_n))*eta
	elif key == "d":
		eta = qt.tensor((qt.jmat((spin_n-1)/2, 'x')*0.1*1j).expm(), qt.identity(clock_n))*eta
	elif key == "s":
		eta = qt.tensor((qt.jmat((spin_n-1)/2, 'y')*0.1*-1j).expm(), qt.identity(clock_n))*eta
	elif key == "w":
		eta = qt.tensor((qt.jmat((spin_n-1)/2, 'y')*0.1*1j).expm(), qt.identity(clock_n))*eta
	elif key == "z":
		eta = qt.tensor((qt.jmat((spin_n-1)/2, 'z')*0.1*-1j).expm(), qt.identity(clock_n))*eta
	elif key == "x":
		eta = qt.tensor((qt.jmat((spin_n-1)/2, 'z')*0.1*1j).expm(), qt.identity(clock_n))*eta
	elif key == "q":
		if running == True:
			running = False
		else:
			running = True
	elif key == "p":
		eta = ground
	elif key == "o":
		eta_ = qt.rand_ket(eta.shape[0])
		eta_.dims = eta.dims
		eta = eta_
vp.scene.bind('keydown', keyboard)

################################################################################

evolving = True
display_proj = False
pointing_to = 0

DT = 0.005
U = (-1j*DT*H).expm()

while True:
	if evolving:
		state = U*state

	spin_part, clock_part = None, None
	if display_proj:
		vprojarrow.visible = True
		vprojarrow.axis = 1.5*vp.vector(directions[pointing_to])
		projected_state = qt.tensor(qt.identity(spin_n), TIME_projs[pointing_to])*state
		if projected_state.norm() != 0:
			projected_state = projected_state.unit()
			spin_part = projected_state.ptrace(0)
			clock_part = state.ptrace(1)
		else:
			continue
	else:
		vprojarrow.visible = False
		spin_part = state.ptrace(0)
		clock_part = state.ptrace(1)

	SL, SV = spin_part.eigenstates()

	clock_probs = [(clock_part*TIME_projs[i]).tr() for i in range(clock_n)]
	for i in range(clock_n):
		vclock_arrows[i].opacity = clock_probs[i]

	expected_time = sum([npdirections[i]*clock_probs[i] for i in range(clock_n)])
	vexpected_time.pos = vp.vector(*expected_time)+vclock_ring.pos

	for i in range(spin_n):
		vspin_layers[i].pos = vspin_sphere.pos + vspin_sphere.radius*vp.vector(*qubit_xyz(SV[i]))
		vspin_layers[i].opacity = SL[i]
