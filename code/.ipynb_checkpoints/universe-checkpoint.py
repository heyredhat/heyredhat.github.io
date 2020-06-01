import sys
import scipy
import qutip as qt
import numpy as np
from magic import *
import vpython as vp
from functools import reduce 

vp.scene.width = 1200
vp.scene.height = 800
vp.scene.background=vp.color.white
np.set_printoptions(threshold=sys.maxsize)

##############################################################################################

def commutator(a, b):
	return a*b - b*a

# Discrete Fourier Transform
def dft(N, basis=[]):
	w = np.exp(-2*np.pi*1j/N)
	d = qt.Qobj(np.array([[w**(i*j) for j in range(N)] for i in range(N)])/np.sqrt(N))
	if len(basis) > 0:
		d = d.transform(basis, inverse=True)
	return d

################################################
# FEYNMAN CLOCK STUFF

# Construct Feynman clock hamiltonian
def feynman_ham(unitaries):
	n = len(unitaries)
	return sum([qt.tensor(qt.basis(n, t+1)*qt.basis(n, t).dag(), unitaries[t])\
	   		   +qt.tensor(qt.basis(n, t)*qt.basis(n, t+1).dag(), unitaries[t].dag())\
		 			for t in range(n-1)])

# Construct Feynman hamiltonian with circular clock
def feynman_ham_circular(unitaries):
	n = len(unitaries)
	return sum([qt.tensor(qt.basis(n, t+1)*qt.basis(n, t).dag(), unitaries[t])\
	   		   +qt.tensor(qt.basis(n, t)*qt.basis(n, t+1).dag(), unitaries[t].dag())\
		 			if t < n-1 else\
		 		qt.tensor(qt.basis(n, 0)*qt.basis(n, t).dag(), unitaries[t])\
	   		   +qt.tensor(qt.basis(n, t)*qt.basis(n, 0).dag(), unitaries[t].dag())\
		 			for t in range(n)])

################################################
# FINITE DIMENSIONAL CLOCK STUFF

# Construct unitary "shift" operator
def shift(N):
	return qt.Qobj(np.array([[\
				1 if j == i-1  or i == 0 and j == N-1 else 0\
					 for j in range(N)] for i in range(N)]))

# Construct unitary "clock" operator
def clock(N):
	w = np.exp(-2*np.pi*1j/N)
	return qt.Qobj(np.array([[\
				w**i if i == j else 0\
					 for j in range(N)] for i in range(N)]))


# Construct finite dimensional / circular position and momentum operators
# and also checks them!
	# phi: generates clocks: like position
	# pi: generates shifts: like momentum
def make_clock(n, alpha=1, beta=1):
	if n % 2 != 1:
		print("N must be odd")
	else:
		l = int((n-1)/2)
		clck = clock(n)
		shft = shift(n)

		pi = qt.Qobj(np.array([[
			0 if i == j else\
			((1j*np.pi)/((2*l+1)*alpha))*\
				1/np.sin((2*np.pi*l*(i-j))/(2*l+1))
			 for j in range(-l, l+1)] 
			 	for i in range(-l, l+1)]))

		d = dft(n)
		phi = (-alpha/beta)*d*pi*d.dag()

		should_be_shft = (-1j*alpha*pi).expm()
		should_be_clck = (1j*beta*phi).expm()

		should_be_pi = (-beta/alpha)*d.dag()*phi*d
		should_be_phi = (-alpha/beta)*d*pi*d.dag()
		if np.isclose(shft.full(), should_be_shft.full()).all() and\
			np.isclose(clck.full(), should_be_clck.full()).all() and\
			np.isclose(pi.full(), should_be_pi.full()).all() and\
			np.isclose(phi.full(), should_be_phi.full()).all():
			return phi/(2*np.pi/n), pi/(2*np.pi/n)
		else:
			print("error!")

def clock_ham(n, alpha=1, beta=1):
	pi = make_clock(n, alpha=alpha, beta=beta)[1]
	return pi

################################################
# FINE DIMENSIONAL QUANTUM HARMONIC OSCILLATOR STUFF

def make_finite_oscillator(N, alpha=1, beta=1, osc_freq=1):
	pi, phi = make_clock(N)
	H_osc1 = 0.5*pi*pi + 0.5*osc_freq*osc_freq*phi*phi
	destroy = np.sqrt(osc_freq/2)*phi + (1j/np.sqrt(2*osc_freq))*pi
	create = np.sqrt(osc_freq/2)*phi - (1j/np.sqrt(2*osc_freq))*pi
	H_osc2 = osc_freq*(create*destroy + 0.5*commutator(destroy, create))
	if np.isclose(H_osc1.full(), H_osc2.full()).all():
		return H_osc1, phi, pi, create, destroy

def finite_oscillator_ham(n, alpha=1, beta=1, osc_freq=1):
	return make_finite_oscillator(n, alpha=alpha, beta=beta, osc_freq=osc_freq)[0]

##############################################################################################
# GRAPHICS

# For quantum spins of any j
class VSphere:
	def __init__(self, dm, pos):
		self.dm = dm
		self.pos = pos
		self.sL, self.sV = dm.eigenstates()
		self.vsphere = vp.sphere(color=vp.color.blue, 
									opacity=0.6,\
						  			pos=self.pos)
		self.vstars = [[vp.sphere(radius=0.1, 
									 pos=self.vsphere.pos+vp.vector(*xyz),\
									 opacity=self.sL[i].real,\
									 color=vp.color.hsv_to_rgb(vp.vector(i/len(self.sV),1,1)))
							for xyz in spin_XYZ(v)]\
								for i, v in enumerate(self.sV)]
		self.j = (dm.shape[0]-1)/2
		self.vspin_arrow = vp.arrow(color=vp.color.yellow,\
									pos=self.pos, shaftwidth=0.05,\
									axis=vp.vector(qt.expect(qt.jmat(self.j, 'x'), self.dm).real,\
												   qt.expect(qt.jmat(self.j, 'y'), self.dm).real,\
												   qt.expect(qt.jmat(self.j, 'z'), self.dm).real))

	def visible(self):
		self.vsphere.visible = True
		self.vspin_arrow.visible = True
		for i in range(len(self.vstars)):
			for j in range(len(self.vstars[i])):
				self.vstars[i][j].visible = True

	def invisible(self):
		if self.vsphere:
			self.vsphere.visible = False
		if self.vspin_arrow:
			self.vspin_arrow.visible = False
		if self.vstars:
			for i in range(len(self.vstars)):
				for j in range(len(self.vstars[i])):
					self.vstars[i][j].visible = False

	def destroy(self):
		self.invisible()
		self.vsphere = None
		self.vspin_arrow = None
		self.vstars = None

# Finite dimensional quantum circular clock
class VFiniteClock:
	def __init__(self, dm, LV, pos):
		self.dm = dm
		self.n = dm.shape[0]
		self.L, self.V = LV
		self.projectors = [v*v.dag() for v in self.V]
		self.pos = pos

		self.vring = vp.ring(pos=pos,\
							 axis=vp.vector(0,0,1),\
					  		 color=vp.color.red,\
					  		 radius=1, 
					  		 thickness=0.1, 
					  		 opacity=0.5)
		roots_of_unity = [np.exp(2*np.pi*1j*i/self.n) for i in range(self.n)]
		directions = [vp.vector(roots_of_unity[i].real,\
								roots_of_unity[i].imag,\
								0) for i in range(self.n)]
		clock_probs = np.array([(self.dm*self.projectors[i]).tr().real for i in range(self.n)])
		self.vclock_arrows = [vp.arrow(pos=self.vring.pos,\
								   axis=directions[i],\
								   opacity=clock_probs[i],\
								   color=vp.color.red) for i in range(self.n)]

		npdirections = [np.array([roots_of_unity[i].real,\
				  		  roots_of_unity[i].imag,\
				  		  0]) for i in range(self.n)]
		expected_time = sum([npdirections[i]*clock_probs[i] for i in range(self.n)])
		self.vexpected_time = vp.sphere(color=vp.color.red, radius=0.1,\
								   		pos=self.pos+vp.vector(*expected_time), opacity=0.5)


	def visible(self):
		self.vring.visible = True
		for arrow in self.vclock_arrows:
			arrow.visible = True

	def invisible(self):
		if self.vring:
			self.vring.visible = False
		if self.vclock_arrows:
			for arrow in self.vclock_arrows:
				arrow.visible = False
		if self.vexpected_time:
			self.vexpected_time.visible = False

	def destroy(self):
		self.invisible()
		self.vring = None
		self.vclock_arrows = None
		self.vexpected_time = None

# Truncated 1D quantum harmonic oscillator
class VHarmonicOscillator:
	def __init__(self, dm, pos):
		self.dm = dm
		self.pos = pos
		self.n = dm.shape[0]
		X = qt.position(self.n)
		L, V = X.eigenstates()
		self.projectors = [v*v.dag() for v in V]
		probs = [(self.dm*p).tr().real for p in self.projectors]
		spaced = L/max(L)
		self.vpts = [vp.sphere(pos=self.pos+vp.vector(spaced[i], 0, 0),\
							   radius=0.5*probs[i],#0.7/self.n,\
							   opacity=probs[i])\
						for i in range(self.n)]
		self.vexp = vp.sphere(pos=self.pos+vp.vector(qt.expect(X, dm).real/max(L),0,0),\
							  radius=0.1, color=vp.color.yellow, opacity=0.9)

	def visible(self):
		for pt in self.vpts:
			pt.visible = True
		self.vexp.visible = True

	def invisible(self):
		if self.vpts:
			for pt in self.vpts:
				pt.visible = False
		if self.vexp:
			self.vexp.visible = False

	def destroy(self):
		self.invisible()
		self.vpts = None
		self.vexp = None

# Truncated 2D quantum harmonic oscillator
class VDoubleHarmonicOscillator:
	def __init__(self, dm, pos):
		self.dm = dm.copy()
		self.pos = pos
		self.n = dm.shape[0]
		self.osc_n = int(np.sqrt(self.n))
		self.dm.dims = [[self.osc_n, self.osc_n], [self.osc_n,self.osc_n]]
		X = qt.position(self.osc_n)
		L, V = X.eigenstates()
		self.projectors = [[qt.tensor(v1*v1.dag(), v2*v2.dag()) for v2 in V] for v1 in V]
		probs = [[(self.dm*p).tr().real for p in row] for row in self.projectors]
		spaced = L/max(L)
		self.vpts = [[vp.sphere(pos=self.pos+vp.vector(spaced[i], spaced[j], 0),\
							    radius=0.5*probs[i][j],#0.7/self.n,\
							    opacity=probs[i][j])\
						for j in range(self.osc_n)]
							for i in range(self.osc_n)]
		self.vexp = vp.sphere(pos=self.pos+vp.vector(qt.expect(qt.tensor(X, qt.identity(self.osc_n)), self.dm).real/max(L),\
													 qt.expect(qt.tensor(qt.identity(self.osc_n), X), self.dm).real/max(L),\
																		0),\
							  radius=0.1, color=vp.color.yellow, opacity=0.9)

	def visible(self):
		for row in self.vpts:
			for pt in row:
				pt.visible = True
		self.vexp.visible = True

	def invisible(self):
		if self.vpts:
			for row in self.vpts:
				for pt in row:
					pt.visible = False
		if self.vexp:
			self.vexp.visible = False

	def destroy(self):
		self.invisible()
		self.vpts = None
		self.vexp = None

# A Feynman clock + system
class VFeynman:
	def __init__(self, dm, pos, structure):
		self.dm = dm.copy()
		self.pos = pos
		self.structure = structure[1:]

		types = []
		dims = []
		for kind in structure.split()[1:]:
			the_type = kind[:-1]
			types.append(the_type)
			its_dim = int(kind[-1])
			dims.append(its_dim)
		self.dm.dims = [dims, dims]
		self.n = len(types)

		clock_dm = self.dm.ptrace(0)
		self.vclock = VFiniteClock(clock_dm,\
					 qt.identity(clock_dm.shape[0]).eigenstates(),\
					 pos)

		self.vguys = []
		for i in range(1, self.n):
			if types[i] == "spin":
				self.vguys.append(VSphere(self.dm.ptrace(i),\
								  pos+vp.vector(0,2*i,0)))
			elif types[i] == "finite_clock":
				self.vguys.append(VFiniteClock(self.dm.ptrace(i),\
								  qt.identity(dims[i]).eigenstates(),\
								  pos+vp.vector(0,2*i,0)))
			elif types[i] == "harmonic_oscillator":
				self.vguys.append(VHarmonicOscillator(self.dm.ptrace(i),\
								  pos+vp.vector(0,2*i,0)))
			elif types[i] == "double_harmonic_oscillator":
				self.vguys.append(VDoubleHarmonicOscillator(self.dm.ptrace(i),\
								  pos+vp.vector(0,2*i,0)))

	def visible(self):
		self.vclock.visible()
		for vguy in self.vguys:
			vguy.visible()

	def invisible(self):
		self.vclock.invisible()
		for vguy in self.vguys:
			vguy.invisible()

	def destroy(self):
		self.vclock.destroy()
		for vguy in self.vguys:
			vguy.destroy()

##############################################################################################

class Universe:
	def __init__(self, solitary_hams, types, interaction_hams={}):
		self.n = len(solitary_hams)
		self.solitary_hams = solitary_hams
		self.types = types
		self.interaction_hams = interaction_hams
		self.dims = [h.shape[0] for h in solitary_hams]
		H = sum([qt.tensor(*[
				h1 if i == j else qt.identity(self.dims[j])\
					for j, h2 in enumerate(solitary_hams)])\
						for i, h1 in enumerate(solitary_hams)])
		self.upgraded_interaction_hams = [] 
		for systems, inter_ham in interaction_hams.items():
			others = list(filter(lambda s: s not in systems, list(range(self.n))))
			partone = qt.tensor(*[qt.identity(self.dims[o]) for o in others])
			partone = qt.tensor(partone, inter_ham)
			indices = others+list(systems)
			final = partone.permute([indices.index(i) for i in range(self.n)])
			self.upgraded_interaction_hams.append(final)
		H += sum(self.upgraded_interaction_hams)
		HL, HV = H.eigenstates()
		zero_subspace = []
		for i in range(len(HL)):
			if np.isclose(HL[i], 0):
				zero_subspace.append(HV[i])
		if len(zero_subspace) == 0:
			print("no zero energy eigenspace!")
			print(HL)
			self.H = H
			self.HL, self.HV = HL, HV
			return
		self.H = H
		self.HL, self.HV = HL, HV
		self.zero_subspace = zero_subspace
		self.state = sum(zero_subspace).unit()
		self.rho = self.state*self.state.dag()
		self.dfts = [dft(d, basis=self.solitary_hams[i].eigenstates()[1]) for i, d in enumerate(self.dims)]
		self.time_op = [self.dfts[i]*h*self.dfts[i].dag()\
							for i, h in enumerate(solitary_hams)]
		self.time_op_LV = [op.eigenstates() for op in self.time_op]
		self.time_op_projs = [[qt.tensor(*[v*v.dag()\
								if i == j else qt.identity(self.dims[j])\
							 	for j in range(len(solitary_hams))])\
									for k, v in enumerate(LV[1])]
									for i, LV in enumerate(self.time_op_LV)]
		self.setup_visuals()

	def rest_at(self, t=0, from_perspective=0, extra=None):
		dims = [h.shape[0] for h in self.solitary_hams]
		projected = self.time_op_projs[from_perspective][t % self.dims[from_perspective]]*self.rho
		if extra:
			projected = extra*projected
		if projected.norm() == 0:
			print("help!")
			return projected.ptrace(\
				tuple(filter(lambda el: el != from_perspective, list(range(len(self.solitary_hams))))))
		return projected.ptrace(\
				tuple(filter(lambda el: el != from_perspective, list(range(len(self.solitary_hams))))))\
					/projected.tr()

	def setup_visuals(self):
		left = -self.n/2-1.5
		self.vsystems = []
		for i in range(self.n):
			substate = self.rho.ptrace(i)
			if self.types[i] == 'spin':
				vsphere = VSphere(substate, vp.vector(left+2.5*i, 0, 0))
				vtimearrow = vp.arrow(pos=vsphere.pos, opacity=0.7,\
									  visible=False)
				self.vsystems.append([vsphere, vtimearrow])
			elif self.types[i] == 'finite_clock':
				vclock = VFiniteClock(substate,\
					self.time_op_LV[i], vp.vector(left+2.5*i, 0, 0))
				vtimearrow = vp.arrow(pos=vclock.pos, opacity=0.7,\
									  visible=False)
				self.vsystems.append([vclock, vtimearrow])
			elif self.types[i] == 'harmonic_oscillator':
				vosc = VHarmonicOscillator(substate, vp.vector(left+2.5*i, 0, 0))
				vtimearrow = vp.arrow(pos=vosc.pos, opacity=0.7,\
									  visible=False)
				self.vsystems.append([vosc, vtimearrow])
			elif self.types[i] == 'double_harmonic_oscillator':
				vosc = VDoubleHarmonicOscillator(substate, vp.vector(left+2.5*i, 0, 0))
				vtimearrow = vp.arrow(pos=vosc.pos, opacity=0.7,\
									  visible=False)
				self.vsystems.append([vosc, vtimearrow])
			elif self.types[i].startswith("feynman"):
				vfeyn = VFeynman(substate, vp.vector(left+2.5*i, 0, 0), self.types[i])
				vtimearrow = vp.arrow(pos=vfeyn.pos, opacity=0.7,\
									  visible=False)
				self.vsystems.append([vfeyn, vtimearrow])
		vp.scene.bind('keydown', self.keyboard)
		vp.scene.bind('click', self.mouse)
		self.time_toggles = [False]*self.n
		self.times = [0]*self.n
		self.temp = []
		self.gauge_evolving = False
		self.gauge_op = sum([r*self.HV[i]*self.HV[i].dag() for i, r in enumerate(np.random.randn(self.n))])
		self.grouped = False
		self.group = None
		self.extra_projectors_toggles = [False]*self.n
		self.extra_projectors_times = [0]*self.n
		self.extra_projectors_vtoggles = [vp.box(pos=self.vsystems[i][0].pos+\
												vp.vector(0,1.5,0),\
												size=vp.vector(0.5,0.5,0.5)) for i in range(self.n)]
		self.extra_projectors_varrows = [vp.arrow(pos=self.vsystems[i][0].pos+\
												vp.vector(0,2.5,0), visible=False) for i in range(self.n)]

	def mouse(self, event):
		selected = vp.scene.mouse.pick
		if selected in self.extra_projectors_vtoggles:
			i = self.extra_projectors_vtoggles.index(selected)
			self.extra_projectors_toggles[i] = False if self.extra_projectors_toggles[i] == True else True
		elif selected in self.extra_projectors_varrows:
			i = self.extra_projectors_varrows.index(selected)
			if self.extra_projectors_times[i] == self.dims[i]-1:
				self.extra_projectors_times[i] = 0
			else:
				self.extra_projectors_times[i] += 1
		self.vupdate()

	def keyboard(self, event):
		key = event.key
		dont_update = False
		if key.isdigit():
			dont_update = True
			if self.grouped:
				self.ungroup()
			i = int(key)
			if i < self.n:
				if self.time_toggles[i] == False:
					self.time_toggles = [False]*self.n
					self.time_toggles[i] = True
				else:
					self.time_toggles = [False]*self.n
		elif key == "[":
			dont_update = True
			if self.grouped:
				t = self.group["time"]
				if t == 0:
					self.group["time"] = self.group["dims"]-1
				else:
					self.group["time"] = t-1
			else:
				if True in self.time_toggles:
					which = self.time_toggles.index(True)
					if self.times[which] == 0:
						self.times[which] = self.dims[which]-1
					else:
						self.times[which] -= 1
		elif key == "]":
			dont_update = True
			if self.grouped:
				t = self.group["time"]
				if t == self.group["dims"]-1:
					self.group["time"] = 0
				else:
					self.group["time"] = t+1
			else:
				if True in self.time_toggles:
					which = self.time_toggles.index(True)
					if self.times[which] == self.dims[which]-1:
						self.times[which] = 0
					else:
						self.times[which] += 1
		elif key == "g":
			self.gauge_evolving = True if self.gauge_evolving == False else False
			print("evolving : %s" % self.gauge_evolving)
		elif key == "h":
			self.gauge_op = sum([r*self.HV[i]*self.HV[i].dag() for i, r in enumerate(np.random.randn(self.n))])
		elif key == "q":
			dont_update = True
			self.ungroup()
		if dont_update == False:
			if self.gauge_evolving:
				u = (-1j*self.gauge_op*0.1).expm()
				self.rho = u*self.rho*u.dag()
		self.vupdate() 

	def evolve(self, T=100):
		u = (-1j*self.gauge_op*0.1).expm()
		for i in range(T):
			self.rho = u*self.rho*u.dag()
			self.state = u*self.state
		self.vupdate()

	def vupdate(self):
		for t in self.temp:
			t.destroy()
		
		extra_projs = [qt.identity(self.H.shape[0])]
		extra_projs[0].dims = self.H.dims
		for i, toggled in enumerate(self.extra_projectors_toggles):
			if toggled == True:
				c = np.exp(2*np.pi*1j*self.extra_projectors_times[i]/self.dims[i])
				self.extra_projectors_varrows[i].axis = vp.vector(c.real, c.imag, 0)
				self.extra_projectors_varrows[i].visible = True
				extra_projs.append(self.time_op_projs[i][self.extra_projectors_times[i]])
			else:
				self.extra_projectors_varrows[i].visible = False
		extra_proj = reduce(lambda x, y: x*y, extra_projs)

		if self.grouped:
			for vs in self.vsystems:
				vs[1].visible = False

			c = np.exp(2*np.pi*1j*self.group["time"]/self.group["dims"])
			self.group["vtimearrow"].axis = vp.vector(c.real, c.imag, 0)
			for j in range(self.n):
				self.vsystems[j][0].invisible()
			local_time = self.group["time"]
			projected = extra_proj*self.group["groupTprojs"][local_time % self.group["dims"]]*self.rho

			self.temp = []
			for i in range(self.n):
				rhop = projected.ptrace(i)/projected.tr()
				v = None
				if self.types[i] == 'spin':
					v = VSphere(rhop, self.vsystems[i][0].pos)
				elif self.types[i] == 'finite_clock':
					v = VFiniteClock(rhop, self.time_op_LV[i], self.vsystems[i][0].pos)
				elif self.types[i] == 'harmonic_oscillator':
					v = VHarmonicOscillator(rhop, self.vsystems[i][0].pos)
				elif self.types[i] == 'double_harmonic_oscillator':
					v = VDoubleHarmonicOscillator(rhop, self.vsystems[i][0].pos)
				elif self.types[i].startswith("feynman"):
					v = VFeynman(substate, self.vsystems[i][0].pos, self.types[i])
				self.temp.append(v)
		else:
			found = False
			for i, marked in enumerate(self.time_toggles):
				if marked == True:
					c = np.exp(2*np.pi*1j*self.times[i]/self.dims[i])
					self.vsystems[i][-1].axis = vp.vector(c.real, c.imag, 0)
					self.vsystems[i][-1].visible = True

					for j in range(self.n):
						self.vsystems[j][0].invisible()

					local_time = self.times[i]
					local_state = self.time_op_LV[i][1][local_time]
					rest = self.rest_at(t=local_time, from_perspective=i, extra=extra_proj)
					local_v = None
					if self.types[i] == 'spin':
						local_v = VSphere(local_state*local_state.dag(), self.vsystems[i][0].pos)
					elif self.types[i] == 'finite_clock':
						local_v =  VFiniteClock(local_state*local_state.dag(),\
										self.time_op_LV[i], self.vsystems[i][0].pos)
					elif self.types[i] == 'harmonic_oscillator':
						local_v = VHarmonicOscillator(local_state*local_state.dag(), self.vsystems[i][0].pos)
					elif self.types[i] == 'double_harmonic_oscillator':
						local_v = VDoubleHarmonicOscillator(local_state*local_state.dag(), self.vsystems[i][0].pos)
					elif self.types[i].startswith("feynman"):
						local_v = VFeynman(local_state*local_state.dag(), self.vsystems[i][0].pos, self.types[i])

					everyone = list(range(self.n))
					del everyone[i]
					other_vs = []
					for j in range(self.n-1):
						if self.types[everyone[j]] == 'spin':
							other_vs.append(\
								VSphere(rest.ptrace(j) if len(rest.dims[0]) != 1 else rest,\
										self.vsystems[everyone[j]][0].pos))
						elif self.types[everyone[j]] == 'finite_clock':
							other_vs.append(\
								VFiniteClock(rest.ptrace(j) if len(rest.dims[0]) != 1 else rest,\
										self.time_op_LV[everyone[j]], 
										self.vsystems[everyone[j]][0].pos))
						elif self.types[everyone[j]] == 'harmonic_oscillator':
							other_vs.append(\
								VHarmonicOscillator(rest.ptrace(j) if len(rest.dims[0]) != 1 else rest,\
										self.vsystems[everyone[j]][0].pos))
						elif self.types[everyone[j]] == 'double_harmonic_oscillator':
							other_vs.append(\
								VDoubleHarmonicOscillator(rest.ptrace(j) if len(rest.dims[0]) != 1 else rest,\
										self.vsystems[everyone[j]][0].pos))
						elif self.types[everyone[j]].startswith('feynman'):
							other_vs.append(\
								VFeynman(rest.ptrace(j) if len(rest.dims[0]) != 1 else rest,\
										self.vsystems[everyone[j]][0].pos, self.types[everyone[j]]))

					self.temp = [local_v]+other_vs
					found = True
				else:
					self.vsystems[i][-1].visible = False
			if not found:
				for j in range(self.n):
					self.vsystems[j][0].visible()

	def gather(self, subsystems):
		if self.grouped:
			self.ungroup()
		groupH = sum([qt.tensor(*[self.solitary_hams[i] \
							if i == j else qt.identity(self.dims[j])\
								for j in subsystems]) for i in subsystems])
		for who, interaction in self.interaction_hams.items():
			flag = False
			for w in who:
				if w not in subsystems:
					flag = True
					break
			if not flag: # all interactors in subsystems
				others = list(filter(lambda s: s not in who, subsystems))
				if len(others) > 0:
					partone = qt.tensor(*[qt.identity(self.dims[o]) for o in others])
					partone = qt.tensor(partone, interaction)
					indices = others+list(who)
					final = partone.permute([indices.index(i) for i in subsystems])
					groupH += final
				else:
					groupH += interaction
		d = dft(groupH.shape[0], basis=groupH.eigenstates()[1])
		d.dims = groupH.dims
		groupT = d*groupH*d.dag()
		groupTL, groupTV = groupT.eigenstates()
		projs = [v*v.dag() for v in groupTV]
		others = list(filter(lambda el: el not in subsystems, list(range(self.n))))
		partone = qt.tensor(*[qt.identity(self.dims[o]) for o in others])
		partones = [qt.tensor(partone, proj) for proj in projs]
		indiceses = [others+list(subsystems) for i in range(len(partones))]
		groupTprojs = [part.permute([indiceses[k].index(i) for i in range(self.n)])\
						 for k, part in enumerate(partones)]	
		vtimearrow = vp.arrow(pos=vp.vector(0,-2,0), opacity=0.7,\
									  visible=True)
		vmarkers = [vp.pyramid(pos=self.vsystems[i][0].pos+\
							   vp.vector(0,-1,0),\
							   size=vp.vector(0.5,0.5,0.5)) for i in subsystems]
		self.grouped = True
		self.group = {"subsystems": subsystems,\
					  "others": others,
					  "groupH": groupH,\
					  "groupT": groupT,\
					  "groupTL": groupTL,\
					  "groupTV": groupTV,\
					  "groupTprojs": groupTprojs,\
					  "vtimearrow": vtimearrow,\
					  "vmarkers": vmarkers,\
					  "time": 0,\
					  "dims": groupH.shape[0]
					  }
		self.vupdate()

	def ungroup(self):
		self.grouped = False
		if self.group:
			self.group["vtimearrow"].visible = False
			for m in self.group["vmarkers"]:
				m.visible = False
			self.group = {}
		self.vupdate()

	def subs_now(self, subsystems):
		extra_projs = [qt.identity(self.H.shape[0])]
		extra_projs[0].dims = self.H.dims
		for i, toggled in enumerate(self.extra_projectors_toggles):
			if toggled == True:
				extra_projs.append(self.time_op_projs[i][self.extra_projectors_times[i]])
		extra_proj = reduce(lambda x, y: x*y, extra_projs)

		projected = self.rho
		if self.grouped:
			projected = extra_proj*self.group["groupTprojs"][self.group["time"] % self.group["dims"]]*projected
		else:
			for i, marked in enumerate(self.time_toggles):
				if marked == True:
					projected = extra_proj*self.time_op_projs[i][self.times[i] % self.dims[i]]*projected
		return projected.ptrace(subsystems)/projected.tr()

	def expect_now(self, op, subsystems):
		ra = self.subs_now(subsystems)
		return qt.expect(op, ra)

##############################################################################################

print("0..9 select subsystem")
print("[ ] shift time projection")
print("g gauge evolution")
print("h random gauge")
print("cmd: u.gather([subsystems])")
print("q ungroup")

##############################################################################################

u = Universe([clock_ham(5), qt.jmat(1.5, 'z'), qt.jmat(1.5, 'z'), qt.num(10)],\
			 ['finite_clock', 'spin', 'spin', 'harmonic_oscillator'])

##############################################################################################

#u = Universe([clock_ham(3), clock_ham(5), clock_ham(3), clock_ham(5)],\
#			 ['finite_clock', 'finite_clock', 'finite_clock', 'finite_clock'])

##############################################################################################

#n = 3
#N = qt.tensor(qt.num(n)+0.5, qt.identity(n)) + qt.tensor(qt.identity(n), qt.num(n)+0.5) 
#N.dims = [[n*n],[n*n]]
#u = Universe([qt.jmat(0.5, 'x'), qt.jmat(0.5, 'y'),qt.jmat(0.5, 'z'), qt.jmat(2.5, 'z'),N],\
#			["spin", "spin", "spin", "spin", "harmonic_oscillator"])

##############################################################################################
# NOTE THAT OSCILLATOR TIME STATES SEEM OUT OF ORDER...

#n = 5
#N = qt.tensor(qt.num(n)+0.5, qt.identity(n)) + qt.tensor(qt.identity(n), qt.num(n)+0.5) 
#N.dims = [[n*n],[n*n]]
#u = Universe([qt.jmat(4, 'z'), N, qt.num(5)],\
#			 ['spin', 'double_harmonic_oscillator', 'harmonic_oscillator'])

##############################################################################################

#u = Universe([qt.jmat(0.5, 'z'), qt.jmat(0.5, 'z'), qt.num(3)],\
#			 ['spin', 'spin', 'double_harmonic_oscillator'])

##############################################################################################

#u = Universe([qt.sigmaz(),\
#			  qt.sigmaz(),\
#			  qt.sigmaz(),\
#			  qt.sigmaz(),\
#			  qt.sigmaz(),\
#			  qt.sigmaz(),\
#			  qt.sigmaz(),\
#			  qt.sigmaz()],\
#			  ["spin", "spin", "spin", "spin", \
#			   "spin", "spin", "spin", "spin"])

##############################################################################################

#u = Universe([qt.jmat(4, "z"), qt.jmat(4, 'x')], ["spin", "spin"])

##############################################################################################

#u = Universe([qt.jmat(4, 'z'),\
#			  qt.num(5),\
#			  clock_ham(5)],\
#			  ["spin", "harmonic_oscillator", "finite_clock"])

##############################################################################################

#u = Universe([qt.jmat(4, 'z'),\
#			  qt.num(6),\
#			  qt.jmat(4, 'z')],\
#			  ["spin", "harmonic_oscillator", "spin"])

##############################################################################################

#u = Universe([qt.jmat(3, 'x'),\
#			   qt.num(9)],\
#			  ["spin", "harmonic_oscillator"])

##############################################################################################

#interaction = qt.jmat(1.5, 'z')
#interaction.dims = [[2,2], [2,2]]
#u = Universe([qt.jmat(0.5, 'z'), qt.jmat(0.5, 'z'), qt.num(6), qt.jmat(4.5,'z')],\
#			 ['spin', 'spin', 'harmonic_oscillator', 'spin'],
#			 interaction_hams={(0,1): interaction})

##############################################################################################

#THIS IS QUESTIONABLE
#unitaries = [(-1j*qt.sigmaz()*0.1).expm(),\
#			 (-1j*qt.sigmaz()*0.1).expm(),\
#			 (-1j*qt.sigmaz()*0.1).expm(),\
#			 (-1j*qt.sigmaz()*0.1).expm(),\
#			 (-1j*qt.sigmaz()*0.1).expm(),\
#			 (-1j*qt.sigmaz()*0.1).expm()]
#fh = feynman_ham(unitaries)
#fh.dims = [[fh.shape[0]], [fh.shape[0]]]
#u = Universe([fh, -fh],\
#			 ["feynman clock6 spin2", "feynman clock6 spin2"])

##############################################################################################

#unitaries = [(-1j*qt.sigmay()).expm(),\
#			 (-1j*qt.sigmay()).expm(),\
#			 (-1j*qt.sigmay()).expm(),\
#			 (-1j*qt.sigmay()).expm(),\
#			 (-1j*qt.sigmay()).expm(),\
#			 (-1j*qt.sigmay()).expm()]
#fh = feynman_ham(unitaries)
#fh = fh - qt.tensor(qt.identity(6), qt.identity(2)) -\
#		  qt.tensor(qt.identity(6), qt.identity(2))
#u = Universe([qt.identity(6), qt.identity(2), qt.jmat(3, 'z')+(1-0.19806226)],\
#			 ["finite_clock", "spin", "spin"],\
#			 interaction_hams={(0,1): fh})

##############################################################################################

#unitaries = [qt.tensor(qt.identity(2), qt.identity(2)),\
#			qt.cnot(),\
#			qt.tensor(qt.identity(2), qt.identity(2))]
#fh = feynman_ham(unitaries)
#fh = fh - qt.tensor(clock_ham(3), qt.identity(2), qt.identity(2)) -\
#		  qt.tensor(qt.identity(3), qt.sigmaz(), qt.identity(2)) - \
#		  qt.tensor(qt.identity(3), qt.identity(2), qt.sigmaz())
#u = Universe([clock_ham(3), qt.sigmaz(), qt.sigmaz(), qt.jmat(2,'z'), qt.jmat(2,'x')],\
#			 ["finite_clock", "spin", "spin", "spin", "spin"],\
#			 interaction_hams={(0,1,2): fh})

##############################################################################

# 1+1 DIRAC EQUATION. Interestingly, seems to need 2 clocks.
#m = 1
#n = 10
#interact = qt.tensor(qt.sigmax(), qt.identity(n))*qt.tensor(qt.identity(2), qt.momentum(n))+\
#			m*qt.tensor(qt.sigmaz(), qt.identity(n))

#interact = interact - qt.tensor(qt.sigmaz(), qt.identity(n)) -\
#					  qt.tensor(qt.identity(2), qt.momentum(n))

#u = Universe([qt.sigmaz(), qt.momentum(n), qt.jmat(2, "z"), qt.jmat(2, "z")-0.02137019],\
#			 ["spin", "harmonic_oscillator", "spin", "spin"],\
#			 interaction_hams={(0,1): interact})

