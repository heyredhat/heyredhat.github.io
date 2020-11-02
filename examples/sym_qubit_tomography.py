import numpy as np
import qutip as qt

from copy import deepcopy
from math import factorial
from itertools import product

from magic import *

from qiskit import QuantumCircuit, execute, ClassicalRegister
from qiskit import Aer, IBMQ, transpile
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.quantum_info.operators import Operator
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter

############################################################################

# constructs linear map between spin-j states and the states of
# 2j symmetrized qubits
def spin_sym(j):
    n = int(2*j)
    if n == 0:
        return qt.Qobj([1])
    N = {}
    for p in product([0,1], repeat=n):
        if p.count(1) in N:
            N[p.count(1)] += qt.tensor(*[qt.basis(2, i) for i in p])
        else:
            N[p.count(1)] = qt.tensor(*[qt.basis(2, i) for i in p])
    Q = qt.Qobj(np.array([N[i].unit().full().T[0].tolist() for i in range(n+1)]))
    Q.dims[1] = [2]*n
    return Q.dag()

############################################################################

# convert back and forth from cartesian to spherical coordinates
# phi: [0, 2pi]
# theta: [0, pi]
def xyz_sph(x, y, z):
    return np.array([np.arctan2(y, x),\
                     np.arccos(z/np.sqrt(x**2 + y**2 + z**2))])

def sph_xyz(phi, theta):
    return np.array([np.sin(theta)*np.cos(phi),\
                     np.sin(theta)*np.sin(phi),\
                     np.cos(theta)])

############################################################################

# operators for preparing the control qubits
def Rk(k):
    return Operator((1/np.sqrt(k+1))*\
                    np.array([[1, -np.sqrt(k)],\
                              [np.sqrt(k), 1]]))

def Tkj(k, j):
    return Operator((1/np.sqrt(k-j+1))*\
                    np.array([[np.sqrt(k-j+1), 0, 0, 0],\
                              [0, 1, np.sqrt(k-j), 0],\
                              [0, -np.sqrt(k-j), 1, 0],\
                              [0, 0, 0, np.sqrt(k-j+1)]]))

############################################################################

j = 2
backend_name = "qasm_simulator"
shots = 10000

#backend_name = "ibmq_qasm_simulator"
#backend_name = "ibmq_16_melbourne"
#backend_name = "ibmq_athens"
shots = 8192

############################################################################

d = int(2*j+1)
n = int(2*j)
p = int(n*(n-1)/2)

# construct a spin state, get the XYZ locations of the stars
# and convert to spherical coordinates
spin_state = qt.rand_ket(d)
stars = spin_XYZ(spin_state)
angles = [xyz_sph(*star) for star in stars]

# the first p qubits are the control qubits
# the next n are the qubits to be symmetrized
circ = QuantumCircuit(p+n)

# rotate the n qubits so they're pointing in the
# directions of the stars
for i in range(n):
    circ.ry(angles[i][1], p+i)
    circ.rz(angles[i][0], p+i)

# apply the symmetrization algorithm:
# iterate over k, working with k control qubits
# in each round: apply the preparation to the controls
# (the U's and T's), then the fredkins, then the inverse
# of the control preparation.
offset = p
for k in range(1, n):
    offset = offset-k
    circ.append(Rk(k), [offset])
    for i in range(k-1):
        circ.append(Tkj(k, i+1), [offset+i+1, offset+i])
    for i in range(k-1, -1, -1):
        circ.fredkin(offset+i, p+k, p+i)
    for i in range(k-2, -1, -1):
        circ.append(Tkj(k, i+1).adjoint(), [offset+i+1, offset+i])    
    circ.append(Rk(k).adjoint(), [offset])

# let's look at it!
print(circ.draw())

# create a collection of circuits to do tomography 
# on just the qubits to be symmetrized
tomog_circs = state_tomography_circuits(circ, list(range(p, p+n)))

# create a copy for later
tomog_circs_sans_aux = deepcopy(tomog_circs)

# add in the measurements of the controls to 
# the tomography circuits
ca = ClassicalRegister(p)
for tomog_circ in tomog_circs:
    tomog_circ.add_register(ca)
    for i in range(p):
        tomog_circ.measure(i,ca[i])

# run on the backend
if backend_name == "qasm_simulator":
    backend = Aer.get_backend("qasm_simulator")
    job = execute(tomog_circs, backend, shots=shots)
    raw_results = job.result()
else:
    provider = IBMQ.load_account()
    job_manager = IBMQJobManager()
    backend = provider.get_backend(backend_name)
    job = job_manager.run(transpile(tomog_circs, backend=backend),\
                    backend=backend, name="spin_sym", shots=shots)
    raw_results = job.results().combine_results()

# have to hack the results to implement post-selection 
# on the controls all being 0's:
new_result = deepcopy(raw_results)
for resultidx, _ in enumerate(raw_results.results):
    old_counts = raw_results.get_counts(resultidx)
    new_counts = {}

    # remove the internal info associated to the control registers
    new_result.results[resultidx].header.creg_sizes = [new_result.results[resultidx].header.creg_sizes[0]]
    new_result.results[resultidx].header.clbit_labels = new_result.results[resultidx].header.clbit_labels[0:-p]
    new_result.results[resultidx].header.memory_slots = n

    # go through the results, and only keep the counts associated
    # to the desired post-selection of all 0's on the controls
    for reg_key in old_counts:
        reg_bits = reg_key.split(" ")
        if reg_bits[0] == "0"*p:
            new_counts[reg_bits[1]] = old_counts[reg_key]
        new_result.results[resultidx].data.counts = new_counts

# fit the results (note we use the copy of the tomography circuits
# w/o the measurements of the controls) and obtain a density matrix
tomog_fit = StateTomographyFitter(new_result, tomog_circs_sans_aux)
rho = tomog_fit.fit()

# downsize the density matrix for the 2j symmerized qubits
# to the density matrix of a spin-j system, and compare
# to the density matrix of the spin we started out with via the trace
S = spin_sym(j)
correct_jrho = spin_state*spin_state.dag()
our_jrho = S.dag()*qt.Qobj(rho, dims=[[2]*n, [2]*n])*S
print("overlap between actual and expected: %.4f" % (correct_jrho*our_jrho).tr().real)