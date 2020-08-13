import qutip as qt
import numpy as np
import vpython as vp

n_pos = 5
n_max = 3

n_osc = n_pos*2
a = [qt.tensor([qt.destroy(n_max) if i == j else qt.identity(n_max)\
		for j in range(n_osc)])\
		 for i in range(n_osc)]

Q = qt.position(n_pos)
QL, QV = Q.eigenstates()
P = qt.momentum(n_pos)
PL, PV = Q.eigenstates()

