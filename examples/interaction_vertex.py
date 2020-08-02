import qutip as qt
import numpy as np
from . import magic
from . import vhelper
from . import quantum_polyhedron
from examples.quantum_polyhedron import *
from examples.vhelper import *
from examples.magic import *

scene = vp.canvas(background=vp.color.white, width=1000, height=500)

p = QuantumPolyhedron([1/2,1/2,1/2,1/2], show_poly=True)
#p = QuantumPolyhedron([1,1,1/2,1/2], show_poly=True)

inputs = qt.rand_ket(2)
outputs = contract(p.big(), inputs, [0])

#inputs = qt.rand_ket(9)
#inputs.dims = [[3,3],[1,1]]
#outputs = contract(p.big(), inputs, [0,1])

#inputs = qt.rand_ket(2**4)
#inputs = raise_indices(p.big())
#inputs.dims = [[2]*4, [1]*4]
#outputs = contract(p.big(), inputs, [0,1,2,3])

vinputs = [VisualDensityMatrix(inputs.ptrace(i), [3*i-len(inputs.dims[0])+0.5, -2, 0])\
				for i in range(len(inputs.dims[0]))]
voutputs = [VisualDensityMatrix(outputs.ptrace(i), [3*i-len(outputs.dims[0])+0.5, 2, 0])\
				for i in range(len(outputs.dims[0]))] if outputs.dims != [[1],[1]] else \
					[VisualDensityMatrix(outputs, [0,2,0])]

def evolve(what, H, dt=0.01, T=200):
	global p, inputs, outputs, vinputs, voutputs
	if what == "polyhedron":
		for t in range(T):
			p.evolve(H, dt=dt, T=1)
			outputs = contract(p.big(), inputs, list(range(len(inputs.dims[0]))), restore_norm=True)
			if outputs.dims == [[1],[1]]:
				voutputs[0].update(outputs)
			else:
				for i, voutput in enumerate(voutputs):
					voutput.update(outputs.ptrace(i))
	elif what == "input":
		U = (-1j*dt*H).expm()
		for t in range(T):
			inputs = U*inputs
			outputs = contract(p.big(), inputs, list(range(len(inputs.dims[0]))), restore_norm=True)
			if outputs.dims == [[1],[1]]:
				voutputs[0].update(outputs)
			else:
				for i, voutput in enumerate(voutputs):
					voutput.update(outputs.ptrace(i))
			for i, vinput in enumerate(vinputs):
				vinput.update(inputs.ptrace(i))

#evolve("polyhedron", qt.rand_herm(p.d))
#evolve("input", qt.tensor(qt.sigmax(), qt.identity(2)))
#evolve("input", qt.tensor(qt.identity(2),qt.sigmax()))
#evolve("input", qt.jmat(1/2, 'x'))
#evolve("input", qt.tensor(qt.jmat(1, 'x'), qt.identity(3)))

#evolve("input", qt.tensor(qt.jmat(1/2, 'x'),qt.identity(2), qt.identity(2), qt.identity(2)))
#R = qt.rand_herm(2**4)
#R.dims = [[2]*4, [2]*4]
#evolve("input",R)

