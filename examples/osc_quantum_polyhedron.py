import numpy as np
import qutip as qt
import polyhedrec
import itertools
import vpython as vp
from magic import *
from vhelper import *
from quantum_polyhedron import *

class OscQuantumPolyhedron:
	def __init__(self, nfaces=4, cutoff=4, pos=[0,0,0], show_poly=False, show_spin=False, initial=None):
		self.pos = np.array(pos)
		self.show_poly = show_poly
		self.show_spin = show_spin
		self.nfaces = nfaces
		self.cutoff = cutoff

		self.a = [[qt.tensor(qt.destroy(self.cutoff), qt.identity(self.cutoff)),\
				   qt.tensor(qt.identity(self.cutoff), qt.destroy(self.cutoff))] for i in range(self.nfaces)]
		for i in range(self.nfaces):
			self.a[i] = [qt.tensor(*[op if i == j else qt.tensor(qt.identity(self.cutoff), qt.identity(self.cutoff))\
							for j in range(self.nfaces)]) for op in self.a[i]]
		self.X = [osc_op_upgrade(qt.sigmax(), self.a[i]) for i in range(self.nfaces)]
		self.Y = [osc_op_upgrade(qt.sigmay(), self.a[i]) for i in range(self.nfaces)]
		self.Z = [osc_op_upgrade(qt.sigmaz(), self.a[i]) for i in range(self.nfaces)]

		X_total, Y_total, Z_total = sum(self.X), sum(self.Y), sum(self.Z)
		self.J2 = X_total*X_total + Y_total*Y_total + Z_total*Z_total
		self.JL, self.JV = self.J2.eigenstates()

		self.zeros = []
		for i, l in enumerate(self.JL):
			if np.isclose(l, 0):
				self.zeros.append(self.JV[i])
		self.zero_map = qt.Qobj(np.array([v.full().T[0] for v in self.zeros])).dag()
		self.zero_map.dims[0] = self.X[0].dims[0]
		self.d = len(self.zeros)

		if type(initial) == type(None):
			self.spin = qt.rand_ket(self.d)
		elif len(initial.dims[0]) == 1:
			self.spin = initial
		else:
			if initial.type == "ket":
				self.spin = repair(self.zero_map.dag()*initial)
			else:
				self.spin = self.zero_map.dag()*initial*self.zero_map

		self.INNER_PRODUCTS = [[self.X[i]*self.X[j] + self.Y[i]*self.Y[j] + self.Z[i]*self.Z[j]\
						 for j in range(self.nfaces)]\
						 	for i in range(self.nfaces)]
		self.INNER_PRODUCTS_ = [[self.tiny(self.INNER_PRODUCTS[i][j])\
				for j in range(self.nfaces)] for i in range(self.nfaces)]

		if self.show_spin:
			if self.spin.type == "ket":
				self.vspin = VisualSpin(self.spin, self.pos+np.array([2,0,0]))
			else:
				self.vspin = VisualDensityMatrix(self.spin, self.pos+np.array([2,0,0]))
		if self.show_poly and self.show_spin:
			p = self.poly()
			if type(p) == list:
				self.vpoly = VisualVectors(p, self.pos+np.array([-2,0,0]))
			else:
				self.vpoly = VisualPolyhedron(p, self.pos+np.array([-2,0,0]))

		if self.show_poly and not self.show_spin:
			p = self.poly()
			if type(p) == list:
				self.vpoly = VisualVectors(p, self.pos)
			else:
				self.vpoly = VisualPolyhedron(p, self.pos)

	def set(self, state):
		if len(state.dims[0]) == 1:
			self.spin = state
		else:
			self.spin = self.tiny(O=state)
		self.update_viz()

	def tiny(self, O=None):
		if type(O) == type(None):
			return self.spin
		elif O.type == "oper":
			return self.zero_map.dag()*O*self.zero_map
		else:
			return repair(self.zero_map.dag()*O)

	def big(self, O=None):
		if type(O) == type(None):
			if self.spin.type == "ket":
				return repair(self.zero_map*self.spin)
			else:
				return self.zero_map*self.spin*self.zero_map.dag()
		elif O.type == "oper":
			return self.zero_map*O*self.zero_map.dag()
		else:
			return repair(self.zero_map*O)

	def inner_products(self):
		return np.array([[qt.expect(self.INNER_PRODUCTS_[i][j], self.spin)\
					for j in range(self.nfaces)]\
							for i in range(self.nfaces)])
	def vectors(self):
		G = self.inner_products()
		U, D, V = np.linalg.svd(G)
		M = np.dot(V[:3].T, np.sqrt(np.diag(D[:3])))
		return [m for m in M]

	def areas(self):
		v = self.vectors()
		return [np.linalg.norm(v_) for v_ in v]

	def normals(self):
		return [v_/np.linalg.norm(v_) for v_ in self.vectors()]

	def closure(self):
		return np.isclose(sum(self.vectors()), np.array([0,0,0])).all()

	def poly(self):
		try:
			v = self.vectors()
			areas = [np.linalg.norm(v_) for v_ in v]
			normals = [v_/areas[i] for i, v_ in enumerate(v)]
			return polyhedrec.reconstruct(normals, areas) 
		except:
			print(self.closure())
			return self.vectors()

	def update_viz(self):
		if self.show_poly:
			p = self.poly()
			if type(p) == list:
				if type(self.vpoly) == VisualVectors:
					self.vpoly.update(p)
				else:
					self.vpoly.destroy()
					self.vpoly = VisualVectors(p, self.pos+np.array([-2,0,0]))
			else:
				if type(self.vpoly) == VisualPolyhedron:
					self.vpoly.update(p)
				else:
					self.vpoly.destroy()
					self.vpoly = VisualPolyhedron(p, self.pos+np.array([-2, 0, 0]))
		if self.show_spin:
			self.vspin.update(self.spin)

	def evolve(self, H=None, T=1000, dt=0.005):
		if type(H) == type(None):
			H = qt.rand_herm(self.d)
		U = (-1j*H*dt).expm()
		for i in range(T):
			if self.spin.type == "ket":
				self.spin = U*self.spin
			else:
				self.spin = U*self.spin*U.dag() 
			self.update_viz()

	def measure(self, O, tiny=False, defer=False):
		O_ = self.tiny(O) if tiny == False else O
		OL, OV = O_.eigenstates()
		OP = [ov*ov.dag() for ov in OV]
		probs = np.array([qt.expect(OP[i], self.spin) for i in range(len(OP))])
		choice = np.random.choice(list(range(len(OP))), p=abs(probs/sum(probs)))
		print("collapsed to %f" % OL[choice])
		if not defer:
			self.spin = (OP[choice]*self.spin).unit()
			self.update_viz()
		else:
			return OP[choice]

	#### Fancier stuff:
	def construct_operators(self, tiny=False):
		a = self.a
		if not tiny:
			self.E = [[a[i][0].dag()*a[j][0] + a[i][1].dag()*a[j][1]
							for j in range(self.nfaces)] \
							for i in range(self.nfaces)]
			self.F = [[a[i][0]*a[j][1] - a[i][1]*a[j][0]
							for j in range(self.nfaces)] \
							for i in range(self.nfaces)]
		else:
			self.E = [[self.tiny(a[i][0].dag()*a[j][0] + a[i][1].dag()*a[j][1])
							for j in range(self.nfaces)] \
							for i in range(self.nfaces)]
			self.F = [[self.tiny(a[i][0]*a[j][1] - a[i][1]*a[j][0])
							for j in range(self.nfaces)] \
							for i in range(self.nfaces)]

	def apply(self, O):
		if O.shape[0] == self.d:
			self.spin = O*self.spin
		else:
			self.spin = self.tiny(O*self.big())
		self.update_viz()

	# H is nfaces x nfaces
	def upgrade_operator(self, H): 
		H = H.full()
		terms = []
		for i in range(self.nfaces):
			for j in range(self.nfaces):
				terms.append(H[i][j]*self.E[i][j])
		return sum(terms)

	def quantize(self, classical_poly, J=1):
		spinors = classical_poly.spinors
		terms = []
		for i in range(self.nfaces):
			for j in range(self.nfaces):
				terms.append(\
				dual_bra(spinors[i])*spinors[j]*\
				(self.a[i][0].dag()*self.a[j][1].dag() -\
				 self.a[i][1].dag()*self.a[j][0].dag()))
		O = (1/np.sqrt(np.math.factorial(J)*np.math.factorial(J+1)))*\
			((0.5*sum(terms))**J)
		n = self.a[0][0].shape[0]
		vac = qt.basis(n, 0)
		vac.dims[0] = self.a[0][0].dims[0]
		state = O*repair(vac)
		self.spin = self.tiny(state)
		self.update_viz()

############################################################################

if __name__ == "__main__":
	scene = vp.canvas(background=vp.color.white, width=1000, height=500)
	p = OscQuantumPolyhedron(nfaces=4, cutoff=2, show_poly=True, show_spin=False)

	def evolve(H=None, dt=0.01, T=100):
		global state, A, B, pA, pB
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
		global state, A, B, pA, pB
		if which == "a":
			state = (qt.tensor(pA.measure(O, defer=True), qt.identity(p.d))*state).unit()
		else:
			state = (qt.tensor(qt.identity(p.d), pB.measure(O, defer=True))*state).unit()
		A = state.ptrace(0)
		B = state.ptrace(1)
		pA.set(A)
		pB.set(B)