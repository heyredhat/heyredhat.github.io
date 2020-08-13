import qutip as qt
import numpy as np
import vpython as vp
from magic import *

def osc_op_upgrade(O, a):
    n = len(a)
    O = O.full()
    terms = []
    for i in range(n):
        for j in range(n):
            terms.append(a[i].dag()*O[i][j]*a[j])
    return sum(terms)

def osc_state_upgrade(s, a):
    n = len(a)
    s = s.full().T[0] if(type(s) == qt.Qobj) else s
    terms = []
    for i in range(n):
        terms.append(a[i].dag()*s[i])
    return sum(terms)

##############################################################################

n_pos = 6
n_max = 2
d = (n_max**n_pos)**2
n_osc = 2*n_pos

##############################################################################

a = [qt.tensor([qt.destroy(n_max) if i == j else qt.identity(n_max)\
        for j in range(n_osc)])\
         for i in range(n_osc)]

Q = qt.position(n_pos)
QL, QV = Q.eigenstates()
P = qt.momentum(n_pos)
PL, PV = P.eigenstates()

Z = qt.sigmaz()
ZL, ZV = Z.eigenstates()

##############################################################################

def a_pos(x, m):
    global a, Q, QL, QV, P, PL, PV
    B = qt.tensor(QV[list(QL).index(x)], ZV[list(ZL).index(m)]).full().T[0]
    return sum([c*a[i] for i, c in enumerate(B)])

def a_mo(p, m):
    global a, Q, QL, QV, P, PL, PV
    B = qt.tensor(PV[list(PL).index(p)], ZV[list(ZL).index(m)]).full().T[0]
    return sum([c*a[i] for i, c in enumerate(B)])

a_qm = [[a_pos(x, 1.), a_pos(x, -1.)] for x in QL]
n_qm = [[aq[1].dag()*aq[1], aq[0].dag()*aq[0]]  for aq in a_qm]

a_pm = [[a_mo(p, 1.), a_mo(p, -1.)] for p in PL]
n_pm = [[ap[1].dag()*ap[1], ap[0].dag()*ap[0]]  for ap in a_pm]

a_q = [sum([a_pos(x, m) for m in ZL])/np.sqrt(len(ZL)) for x in QL]
a_p = [sum([a_mo(p, m) for m in ZL])/np.sqrt(len(ZL)) for p in PL]
a_m = [sum([a_pos(x, m) for x in QL])/np.sqrt(len(QL)) for m in ZL]

Xsq = [osc_op_upgrade(qt.sigmax(), a_qm[i]) for i in range(n_pos)]
Ysq = [osc_op_upgrade(qt.sigmay(), a_qm[i]) for i in range(n_pos)]
Zsq = [osc_op_upgrade(qt.sigmaz(), a_qm[i]) for i in range(n_pos)]

Xsp = [osc_op_upgrade(qt.sigmax(), a_pm[i]) for i in range(n_pos)]
Ysp = [osc_op_upgrade(qt.sigmay(), a_pm[i]) for i in range(n_pos)]
Zsp = [osc_op_upgrade(qt.sigmaz(), a_pm[i]) for i in range(n_pos)]

def create_star(x, spinor):
    global a_qm, QL
    return osc_state_upgrade(spinor, a_qm[list(QL).index(x)])

##############################################################################

def expectations(state, pos=True):
    global nq_m, np_m, Xsq, Ysq, Zsq, QL, PL, Xsp, Ysp, Zsp
    if pos:
        return [[2*QL[i], qt.expect(nq[0], state), qt.expect(nq[1], state), \
                    (qt.expect(Xsq[i], state), qt.expect(Ysq[i], state), qt.expect(Zsq[i], state))]\
                    for i, nq in enumerate(n_qm)]
    else:
        return [[2*PL[i], qt.expect(nq[0], state), qt.expect(nq[1], state), \
                    (qt.expect(Xsp[i], state), qt.expect(Ysp[i], state), qt.expect(Zsp[i], state))]\
                    for i, nq in enumerate(n_pm)]

def prnt(state, pos=True):
    E = expectations(state, pos=pos)
    for i, e in enumerate(E):
        print("@%.3f: %.3f" % (e[0], e[1]+e[2]))
        print("@%.3f, %.3f: %.3f" % (e[0], -1., e[1]))
        print("@%.3f, %.3f: %.3f" % (e[0], 1., e[2]))
        print("\tX: %.3f, Y: %.2f, Z: %.3f" % e[3])
        print('-')

##############################################################################

def coherent(s, n):
    a = qt.destroy(n)
    return np.exp(-s*np.conjugate(s)/2)*(s*a.dag()).expm()*(-np.conjugate(s)*a).expm()*qt.basis(n,0)

vacuum = qt.basis(d, 0)
vacuum.dims = [[n_max]*n_pos*2, [1]*n_pos*2]

z = (np.random.randn(1)[0] + np.random.randn(1)[0]*1j)/2
coherent_state = coherent(z, n_pos)
spinor = qt.rand_ket(2)
xyz = spin_XYZ(spinor)[0]
#print(xyz)

state = (osc_state_upgrade(qt.tensor(coherent_state, spinor), a)*vacuum).unit()
state.dims = [[n_max]*n_pos*2, [1]*n_pos*2]

#state = qt.rand_ket(d)
#state.dims = [[n_max]*n_pos*2, [1]*n_pos*2]

#state = qt.basis(d, 0)
#state.dims = [[n_max]*n_pos*2, [1]*n_pos*2]
#state = (create_star(QL[1], spinor)*create_star(QL[1], spinor)*state).unit()

dt = 0.001
#H = sum([a_.dag()*a_ + 0.5 for a_ in a_p])
H = osc_op_upgrade(qt.tensor(qt.create(n_pos)*qt.destroy(n_pos) + 0.5, qt.identity(2)), a)
H.dims = [[n_max]*n_pos*2, [n_max]*n_pos*2]
U = (-1j*H*dt).expm()

##############################################################################

E = expectations(state)
vnums = [vp.cylinder(pos=vp.vector(E[i][0], 0, 0), \
            color=vp.color.red, axis=vp.vector(0, E[i][1]+E[i][2],0))\
                for i in range(n_pos)]
vspheres = [vp.sphere(pos=vp.vector(E[i][0], -2, 0), color=vp.color.blue, opacity=0.3)\
        for i in range(n_pos)]
vxyzs = [vp.arrow(pos=vspheres[i].pos, axis=vp.vector(*E[i][3])) for i in range(n_pos)]
vdownlabels = [vp.label(pos=vp.vector(E[i][0], -5, 0), height=10, \
                    text="-1: %.3f" % (E[i][1])) for i in range(n_pos)]
vuplabels = [vp.label(pos=vp.vector(E[i][0], -4, 0), height=10, \
                    text="1: %.3f" % (E[i][2])) for i in range(n_pos)]

##############################################################################

while True:
    state = U*state
    E = expectations(state, pos=True)
    for i in range(n_pos):
        vnums[i].axis = vp.vector(0, E[i][1]+E[i][2],0)
        vxyzs[i].axis = vp.vector(*E[i][3])
        vdownlabels[i].text = "-1: %.3f" % (E[i][1])
        vuplabels[i].text = "1: %.3f" % (E[i][2])

