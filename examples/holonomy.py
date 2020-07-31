import numpy as np
import qutip as qt
import itertools
import vpython as vp
from magic import *
from vhelper import *
from quantum_polyhedron import *
from osc_quantum_polyhedron import *
import scipy as sp
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize, precision=3)


def construct_poly_operators_(qp, tiny=False):
	a = qp.a
	if not tiny:
		qp.E = [[(a[i][0].dag()*a[j][0] + a[j][0].dag()*a[i][0] + \
			     a[i][1].dag()*a[j][1] + a[j][1].dag()*a[i][1])/2
						for j in range(qp.nfaces)] \
						for i in range(qp.nfaces)]
		qp.F = [[(a[i][0].dag()*a[j][0] - a[j][0].dag()*a[i][0] +\
				  a[i][1].dag()*a[j][1] - a[j][1].dag()*a[i][1])*(i/2)
						for j in range(qp.nfaces)] \
						for i in range(qp.nfaces)]
	else:
		qp.E = [[qp.tiny((a[i][0].dag()*a[j][0] + a[j][0].dag()*a[i][0] + \
			     a[i][1].dag()*a[j][1] + a[j][1].dag()*a[i][1])/2)
						for j in range(qp.nfaces)] \
						for i in range(qp.nfaces)]
		qp.F = [[qp.tiny((a[i][0].dag()*a[j][0] - a[j][0].dag()*a[i][0] +\
				  a[i][1].dag()*a[j][1] - a[j][1].dag()*a[i][1])*(i/2))
						for j in range(qp.nfaces)] \
						for i in range(qp.nfaces)]
	return qp

def construct_poly_operators(qp, tiny=False):
	a = qp.a
	if not tiny:
		qp.E = [[a[i][0].dag()*a[j][0] + a[i][1].dag()*a[j][1] + (1 if i==j else 0)*qp.ID
						for j in range(qp.nfaces)] \
						for i in range(qp.nfaces)]
		qp.F = [[a[i][0]*a[j][1] - a[i][1]*a[j][0]
						for j in range(qp.nfaces)] \
						for i in range(qp.nfaces)]
	else:
		qp.E = [[qp.tiny(a[i][0].dag()*a[j][0] + a[i][1].dag()*a[j][1])
						for j in range(qp.nfaces)] \
						for i in range(qp.nfaces)]
		qp.F = [[qp.tiny(a[i][0]*a[j][1] - a[i][1]*a[j][0])
						for j in range(qp.nfaces)] \
						for i in range(qp.nfaces)]
	return qp

# H is nfaces x nfaces
def upgrade_to_poly_operator(qp, H): 
	H = H.full()
	terms = []
	for i in range(qp.nfaces):
		for j in range(qp.nfaces):
			terms.append(H[i][j]*qp.E[i][j])
	return sum(terms)

def downgrade_from_poly_operator(qp, O):
	return qt.Qobj(np.array([[(qp.E[i][j].dag()*O).tr() for j in range(qp.nfaces)] for i in range(qp.nfaces)]))

def quantize(qp, classical_poly, J=1):
	spinors = classical_poly.spinors
	terms = []
	for i in range(qp.nfaces):
		for j in range(qp.nfaces):
			terms.append(\
			dual_bra(spinors[i])*spinors[j]*\
			(qp.a[i][0].dag()*qp.a[j][1].dag() -\
			 qp.a[i][1].dag()*qp.a[j][0].dag()))
	O = (1/np.sqrt(np.math.factorial(J)*np.math.factorial(J+1)))*\
		((0.5*sum(terms))**J)
	n = qp.a[0][0].shape[0]
	vac = qt.basis(n, 0)
	vac.dims[0] = qp.a[0][0].dims[0]
	state = O*repair(vac)
	qp.spin = qp.tiny(state)
	qp.update_viz()
	return qp

def construct_holonomy_operators(a, b, E):
	Ei = qt.Qobj(np.linalg.inv((sp.linalg.sqrtm(E.full() + np.eye(E.shape[0])))))
	Ei.dims = a[0].dims
	M = [[(a[1].dag()*b[0].dag() - a[0]*b[1]), (a[0]*b[0] + a[1].dag()*b[1].dag())],\
		   [(-a[1]*b[1] - a[0].dag()*b[0].dag()), (a[1]*b[0] - a[0].dag()*b[1].dag())]]
	for i in range(2):
		for j in range(2):
			M[i][j] = Ei*M[i][j]*Ei
	return M

def expect_holonomy(state, h):
	return np.array([[qt.expect(h[i][j], state) for j in range(2)] for i in range(2)])

def quantum_transport(a, H):
	return [sum([a[j]*H[i][j] for j in range(2)]) for i in range(2)]

def classical_transport(a, b):
	return (a*dual_bra(b) - dual_ket(a)*b.dag())/np.sqrt((a.dag()*a*b.dag()*b).full()[0][0])

def xyz_spinor(xyz):
	uxyz = xyz/np.linalg.norm(xyz)
	return np.sqrt(np.linalg.norm(xyz))*qt.Qobj(c_spinor(xyz_c(uxyz)))

############################################################################

def checkE(E):
	for i in range(len(E)):
		for j in range(len(E)):
			for k in range(len(E)):
				for l in range(len(E)):
					a = qt.commutator(E[i][j], E[k][l])
					b = (1 if j == k else 0)*E[i][l] - (1 if i == l else 0)*E[k][j]
					print(np.isclose(a.full(), b.full()).all())
if True:
	p = OscQuantumPolyhedron(nfaces=4, cutoff=2, show_poly=False, show_spin=False)
	construct_poly_operators(p, tiny=True)
	s = OscQuantumPolyhedron(nfaces=1, cutoff=2, show_poly=False, show_spin=False)
	construct_poly_operators(s, tiny=True)

	ha = qt.rand_herm(4)
	ha2 = downgrade_from_poly_operator(p, upgrade_to_poly_operator(p, ha))

	construct_poly_operators(p, tiny=False)
	ha3 = downgrade_from_poly_operator(p, upgrade_to_poly_operator(p, ha))


	#pa0 = [qt.tensor(s.ID, a_) for a_ in p.a[0]]
	#pa1 = [qt.tensor(s.ID, a_) for a_ in p.a[1]]
	#sa = [qt.tensor(a_, p.ID) for a_ in s.a[0]]

	#state = qt.tensor(s.big(), p.big())

	#H = construct_holonomy_operators(pa0, pa1, qt.tensor(s.ID, p.E[0][0]))
	#h = expect_holonomy(state, H)
	#trh = np.trace(h)

	#sa2 = quantum_transport(sa, H)

	#Ei = qt.Qobj(np.linalg.inv((sp.linalg.sqrtm(p.E[0][0].full() + np.eye(p.E[0][0].shape[0])))))
	#Ei.dims = p.F[0][1].dims
	#W = Ei*(p.F[0][1].dag() + p.F[0][1])*Ei
	#wa = qt.expect(W, p.big())

	#H_ = construct_holonomy_operators(p.a[0], p.a[1], p.E[0][0])
	#W2 = -1*(H_[0][0]+H_[1][1])


	#pv = p.vectors()
	#spinors = [xyz_spinor(va) for va in pv]

	#G = np.array([[classical_transport(spinors[i], spinors[j]).tr() for i in range(4)] for j in range(4)])
			
