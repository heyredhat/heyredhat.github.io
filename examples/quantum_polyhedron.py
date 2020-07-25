import numpy as np
import qutip as qt
import polyhedrec
import itertools
import vpython as vp
from magic import *
from vhelper import *

#####################################################################

def dual_ket(spinor):
	a, b = spinor.conj().full().T[0]
	return qt.Qobj(np.array([-b, a]))

def dual_bra(spinor):
	a, b = spinor.full().T[0]
	q = qt.Qobj(np.array([-b, a])).dag().conj()
	return q

def close_spinors(spinors):
	X = sum([spinor*spinor.dag() for spinor in spinors])
	g, d, gi = np.linalg.svd(X.full())
	d = np.diag(d)
	rho = np.linalg.det(d)
	L = np.dot(g, np.sqrt(d)*rho**(-1./4.))
	Li = qt.Qobj(np.linalg.inv(L))
	return [Li*spinor for spinor in spinors]

def unitary_spinors(U):
    col0, col1 = U[:,0], U[:, 1]
    return [qt.Qobj(np.array([col0[i], col1[i]])) for i in range(n)]

def spinor_xyz(spinor):
    return np.array([qt.expect(qt.sigmax(), spinor), qt.expect(qt.sigmay(), spinor), qt.expect(qt.sigmaz(), spinor)])

def recover_vectors(inner_products):
    U, D, V = np.linalg.svd(inner_products)
    M = np.dot(V[:3].T, np.sqrt(np.diag(D[:3])))
    return [m for m in M]

def inner_products(vectors):
	return np.array([[np.dot(vectors[i], vectors[j])\
				for j in range(len(vectors))]\
					for i in range(len(vectors))])

class ClassicalPolyhedron:
	def __init__(self, nfaces, show_poly=True, pos=[0,0,0], spinors=None):
		self.nfaces = nfaces
		self.show_poly = show_poly
		self.pos = pos
		self.spinors = spinors if type(spinors) != type(None) else close_spinors([qt.rand_ket(2) for i in range(self.nfaces)])
		self.poly = polyhedrec.reconstruct(self.normals(), self.areas())
		if self.show_poly:
			self.vpoly = VisualPolyhedron(self.poly, self.pos)

	def evolve(self, H=None, dt=0.01, T=100):
		if type(H) == type(None):
			H = qt.rand_herm(self.nfaces)
		U = (-1j*H*dt).expm().full()
		for t in range(T):
			self.spinors = [sum([U[i][j]*self.spinors[j]\
								for j in range(self.nfaces)])\
									for i in range(self.nfaces)]
			self.poly = polyhedrec.reconstruct(self.normals(), self.areas())
			self.vpoly.update(self.poly) if self.show_poly else None

	def vectors(self):
		return [spinor_xyz(s) for s in self.spinors]

	def areas(self):
		return [(spinor.dag()*spinor).full()[0][0].real for spinor in self.spinors]

	def normals(self):
		areas = self.areas()
		vectors = self.vectors()
		return [v/areas[i] for i, v in enumerate(vectors)]

	def inner_products(self):
		vectors = self.vectors()
		return np.array([[np.dot(vectors[i], vectors[j])\
					for j in range(self.nfaces)]\
						for i in range(self.nfaces)])

	def closure(self):
		return np.isclose(sum(self.vectors()), np.array([0,0,0])).all()

	def cross_spinors(self):
		cross_spinors = []
		for i in range(self.nfaces):
			row = []
			zi = self.spinors[i].full().T[0]
			for j in range(self.nfaces):
				zj = self.spinors[j].full().T[0]
				row.append((zj[0]*zi[1] - zi[0]*zj[1]))
			cross_spinors.append(row)
		return np.array(cross_spinors)

##################################################################

def repair(q):
	q.dims[1] = [1]*len(q.dims[0])
	return q

def make_total_spin(js):
	ns = [int(2*j+1) for j in js]

	X = [qt.tensor(*[qt.jmat(j, 'x') if k == i else qt.identity(ns[k])\
				for k in range(len(js))])\
					 for i, j in enumerate(js)]
	Y = [qt.tensor(*[qt.jmat(j, 'y') if k == i else qt.identity(ns[k])\
				for k in range(len(js))])\
					 for i, j in enumerate(js)]
	Z = [qt.tensor(*[qt.jmat(j, 'z') if k == i else qt.identity(ns[k])\
				for k in range(len(js))])\
					 for i, j in enumerate(js)]

	X_total = sum(X)
	Y_total = sum(Y)
	Z_total = sum(Z)
	J2 = X_total*X_total + Y_total*Y_total + Z_total*Z_total
	return X, Y, Z, J2

def epsilonj(j):
    n = int(2*j)
    epsilons = qt.tensor(*[np.sqrt(2)*qt.bell_state("11")]*(n))
    epsilons = epsilons.permute(list(range(0, 2*n, 2))+list(range(1, 2*n, 2)))
    S = sym_spin(n)
    return repair(qt.tensor(S, S)*epsilons)

def raise_indices(spins):
    input_js = [(d-1)/2 for d in spins.dims[0]]
    n = len(input_js)
    epsilons = qt.tensor(*[epsilonj(input_j)\
                               for input_j in input_js])
    state = repair(qt.tensor(epsilons, spins))  
    pairs = [(2*i+1, 2*n+i) for i in range(n)]
    return repair(qt.tensor_contract(state, *pairs)).conj()

def contract(poly_spins, input_spins, indices, verbose=False, restore_norm=False):
    js = [(d-1)/2 for d in poly_spins.dims[0]]
    ns = poly_spins.dims[0][:]
    n = len(js)
    input_js = [js[index] for index in indices]
    input_ns = [ns[index] for index in indices]
    if len(indices) == n:
        epsilons = qt.tensor(*[epsilonj(input_j) for input_j in input_js])
        epsilons = epsilons.permute(list(range(0, 2*n, 2))+list(range(1, 2*n, 2)))
        output = epsilons.dag()*qt.tensor(input_spins, poly_spins)
        output.dims = [[1],[1]]
        return output
    else: 
        epsilons = qt.tensor(*[epsilonj(input_j)*(np.sqrt(2*input_j+1) if restore_norm else 1) for input_j in input_js]) 
        offset = 3*len(indices)
        state = repair(qt.tensor(epsilons, input_spins, poly_spins))   
        pairs = [(2*i, 2*len(indices)+i) for i in range(len(indices))]+\
                [(2*i+1, index+offset) for i, index in enumerate(indices)]
        output_spins = repair(qt.tensor_contract(state, *pairs))
        if verbose:
            X0, Y0, Z0, J20 = make_total_spin(input_js)
            removed_indices = np.delete(js, indices)
            X1, Y1, Z1, J21 = make_total_spin(removed_indices)

            input_xyz = np.array([qt.expect(sum(X0), input_spins),\
                                  qt.expect(sum(Y0), input_spins),\
                                  qt.expect(sum(Z0), input_spins)])
            output_xyz = np.array([qt.expect(sum(X1), output_spins),\
                                   qt.expect(sum(Y1), output_spins),\
                                   qt.expect(sum(Z1), output_spins)])
            uinput_xyz = input_xyz/np.linalg.norm(input_xyz)
            uoutput_xyz = output_xyz/np.linalg.norm(output_xyz)
            print("input ang mo direction: %s " % uinput_xyz)
            print("output ang mo direction: %s " % uoutput_xyz)
            print("difference: %s" % (uinput_xyz-uoutput_xyz))
            print("norm of output: %s" % output_spins.norm())
            print("actual difference: %s" % (input_xyz-output_xyz))
    return output_spins

class QuantumPolyhedron:
	def __init__(self, js, pos=[0,0,0], show_poly=False, show_spin=False, initial=None):
		self.js = js
		self.pos = np.array(pos)
		self.show_poly = show_poly
		self.show_spin = show_spin

		self.nfaces = len(self.js)
		self.ns = [int(2*j+1) for j in self.js]
		self.X, self.Y, self.Z, self.J2 = make_total_spin(self.js)
		self.JL, self.JV = self.J2.eigenstates()

		self.zeros = []
		for i, l in enumerate(self.JL):
			if np.isclose(l, 0):
				self.zeros.append(self.JV[i])
		self.zero_map = qt.Qobj(np.array([v.full().T[0] for v in self.zeros])).dag()
		self.zero_map.dims[0] = self.ns[:]
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

############################################################################

if __name__ == "__main__":
	scene = vp.canvas(background=vp.color.white, width=1000, height=500)
	p = QuantumPolyhedron([1/2, 1/2, 1/2, 1/2], show_poly=True, show_spin=True)
	#p = QuantumPolyhedron([1/2, 1/2, 1/2, 1/2, 1/2, 1/2], show_poly=True, show_spin=True)
	#p = QuantumPolyhedron([1, 1, 1/2, 1/2], show_poly=True, show_spin=True)
	#print("total area: %s" % sum(p.areas()))
	#p.evolve(H=qt.rand_herm(p.d), dt=0.01, T=1000)
	#print("total area: %s" % sum(p.areas()))
	#p.measure(p.INNER_PRODUCTS[0][1])
	#print("total area: %s" % sum(p.areas()))
