import qutip as qt
import numpy as np

def anticommutator(a, b):
    return a*b + b*a

#########################################################################################

def fermion_operators(n):
    return [qt.tensor(*[qt.destroy(2) if i == j\
                else (qt.sigmaz() if j < i\
                    else qt.identity(2))\
                        for j in range(n)])\
                            for i in range(n)]

def test_fermion_operators(f):
    for i in range(len(f)):
        for j in range(len(f)):
            d = f[i].shape[0]
            test1 = anticommutator(f[i], f[j]).full()
            test2 = anticommutator(f[i], f[j].dag()).full()
            if not \
                (np.isclose(test1, np.zeros((d,d))).all()\
                    and \
                ((np.isclose(test2, np.zeros((d,d))).all() and i != j)\
                        or (np.isclose(test2, np.eye(d)).all() and i == j))):
                return False
    return True

#########################################################################################

n = 6
IDn = qt.identity(2**n)
IDn.dims = [[2]*n, [2]*n]

f = fermion_operators(n)
N = sum([a.dag()*a for a in f]) # number operator
I = qt.basis(2**n, 0) # vacuum state
I.dims = [[2]*n, [1]*n]

#########################################################################################

def majorana_operators(f):
    L, R = [], []
    for i in range(len(f)):
        L.append((1/np.sqrt(2))*(f[i] + f[i].dag()))
        R.append((1j/(np.sqrt(2)))*(f[i] - f[i].dag()))
    return L, R

def test_majorana_operators(m):
    for i in range(len(m)):
        for j in range(len(m)):
            d = m[i].shape[0]
            test = anticommutator(m[i], m[j]).full()
            if not ((i == j and np.isclose(test, np.eye(d)).all()) or\
                (i != j and np.isclose(test, np.zeros((d,d))).all())):
                return False
    return True

def majoranas_to_fermions(m):
    return [(m[i] + 1j*m[i+1])/np.sqrt(2) for i in range(0, len(m)-1, 2)],\
            [(m[i] - 1j*m[i+1])/np.sqrt(2) for i in range(0, len(m)-1, 2)]

#########################################################################################

mL, mR = majorana_operators(f)

Lf, Lfdag = majoranas_to_fermions(mL)
NLf = sum([Lfdag[i]*Lf[i] for i in range(len(Lf))]) # L number operator
Rf, Rfdag = majoranas_to_fermions(mR)
NRf = sum([Rfdag[i]*Rf[i] for i in range(len(Rf))]) # R number operator

NLR = NLf+NRf
NLRl, NLRv = NLR.eigenstates()
CB = np.array([v.full().T[0] for v in NLRv]).T
ILR = I.transform(CB)

#########################################################################################

from itertools import combinations
from functools import reduce

def to_majorana_basis(op, m):
    N = len(m)
    terms = []
    for n in range(N+1):
        if n == 0:
            terms.append(op.tr()/m[0].shape[0])
        else:
            for pr in combinations(list(range(N)), n):
                s = reduce(lambda x,y: x*y, [m[p] for p in pr])
                terms.append((s.dag()*op).tr()*2**(n-2))
    return qt.Qobj(np.array(terms))

def from_majorana_basis(op, m):
    op = op.full().T[0]
    N = len(m)
    c = 0
    terms = []
    for n in range(N+1):
        if n == 0:
            terms.append(op[c]*qt.identity(m[0].shape[0]))
            terms[-1].dims = m[0].dims
            c += 1
        else:
            for pr in combinations(list(range(N)), n):
                s = reduce(lambda x,y: x*y, [m[p] for p in pr])
                terms.append(op[c]*s)
                terms[-1].dims = m[0].dims
                c += 1
    return sum(terms)

#########################################################################################

n_ = int(n/2)
f_ = fermion_operators(n_)
m_ = reduce(lambda x, y: x+y, majorana_operators(f_))

O = qt.rand_unitary(2**n_)
O.dims = [[2]*n_, [2]*n_]

OL = from_majorana_basis(to_majorana_basis(O, m_), mL)
OR = from_majorana_basis(to_majorana_basis(O, m_), [1j*r for r in mR])

def size(O, i=None):
    global f, mL, m_, I, N
    majorana_state = (from_majorana_basis(to_majorana_basis(O, m_), mL)*I).unit()
    if type(i) != type(None):
        return qt.expect(f[i].dag()*f[i], majorana_state)
    else:
        return qt.expect(N, majorana_state)

#########################################################################################

import math

def random_syk_couplings(m):
    J = {}
    for i in range(m):
        for j in range(m):
            for k in range(m):
                for l in range(m):
                    if i < j and j < k and k < l:
                        J[(i, j, k, l)] = np.random.normal()
    return J

def syk_ham(couplings, m):
    Jterms = []
    for ijkl, c in couplings.items():
        i, j, k, l = ijkl
        Jterms.append(c*m[i]*m[j]*m[k]*m[l])
    return (-1/(math.factorial(4)))*sum(Jterms)

J = random_syk_couplings(n)
E = syk_ham(J, m_)
EL = syk_ham(J, mL)
ER = syk_ham(J, [1j*r for r in mR]) 

OLt = lambda t: (1j*EL*t).expm()*OL*(-1j*EL*t).expm()
ORt = lambda t: (1j*ER*t).expm()*OR*(-1j*ER*t).expm()

#########################################################################################

def construct_thermal_dm(H, beta=0):
    return (-beta*H*(1/2)).expm()/np.sqrt((-beta*H*(1/2)).expm().tr())

rho = construct_thermal_dm(E, beta=0)
rhoR = from_majorana_basis(to_majorana_basis(rho, m_), [1j*r for r in mR])
TFD = (rhoR*I).unit()

#########################################################################################

def cold_size(O, i=None):
    global f, mL, m_, TFD, N
    majorana_state = (from_majorana_basis(to_majorana_basis(O, m_), mL)*TFD).unit()
    if type(i) != type(None):
        return qt.expect(f[i].dag()*f[i], majorana_state)
    else:
        return qt.expect(N, majorana_state)
    
#########################################################################################

A = OLt(-10)*TFD
B = ORt(10)*TFD
A_ = A.full().T[0]
B_ = B.full().T[0]
D_ = np.array([B_[i]/A_[i] for i in range(len(A_))])
D = qt.Qobj(np.diag(D_)) # our e^{igV}
D.dims = [A.dims[0], A.dims[0]]

#########################################################################################

def commutator(a, b):
    return a*b - b*a

X = -1j*m_[0]*m_[2]
Y = -1j*m_[2]*m_[1]
Z = -1j*m_[1]*m_[0]

#########################################################################################

msg = qt.basis(2,0)
big_I = qt.tensor(msg, I)
big_TFD = qt.tensor(msg, TFD)

#########################################################################################

msgXYZ = {"I": qt.tensor(qt.identity(2), IDn),\
          "X": qt.tensor(qt.sigmax(), IDn),\
          "Y": qt.tensor(qt.sigmay(), IDn),\
          "Z":qt.tensor(qt.sigmaz(), IDn)}

def Ostar(state):
    global msgXYZ
    return [qt.expect(msgXYZ[o], state) for o in ["I", "X", "Y", "Z"]]

#########################################################################################

LXYZ = {"I": qt.tensor(qt.identity(2), IDn),\
        "X": qt.tensor(qt.identity(2), from_majorana_basis(to_majorana_basis(X, m_), mL)),\
        "Y": qt.tensor(qt.identity(2), from_majorana_basis(to_majorana_basis(Y, m_), mL)),\
        "Z": qt.tensor(qt.identity(2), from_majorana_basis(to_majorana_basis(Z, m_), mL))}

RXYZ = {"I": qt.tensor(qt.identity(2), IDn),\
        "X": qt.tensor(qt.identity(2), from_majorana_basis(to_majorana_basis(X, m_), [1j*r for r in mR])),\
        "Y": qt.tensor(qt.identity(2), from_majorana_basis(to_majorana_basis(Y, m_), [1j*r for r in mR])),\
        "Z": qt.tensor(qt.identity(2), from_majorana_basis(to_majorana_basis(Z, m_), [1j*r for r in mR]))}

def Lstar(state):
    global LXYZ
    return [qt.expect(LXYZ[o],state) for o in ["I", "X", "Y", "Z"]]

def Rstar(state):
    global RXYZ
    return [qt.expect(RXYZ[o],state) for o in ["I", "X", "Y", "Z"]]

#########################################################################################

from itertools import product
from qutip.qip.operations.gates import swap

def pauli_basis(n):
    IXYZ = {"I": qt.identity(2), "X": qt.sigmax(), "Y": qt.sigmay(), "Z": qt.sigmaz()}
    names, ops = [], []
    for P in product(IXYZ, repeat=n):
        names.append("".join(P))
        ops.append(qt.tensor(*[IXYZ[p] for p in P]))
    return names, ops

def to_pauli(op, Pops):
    return np.array([(o.dag()*op).tr() for o in Pops])/np.sqrt(len(Pops))

############################################

SWAP = swap(N=2, targets=[0,1])
Pnames, Pops = pauli_basis(2)
SWAPp = to_pauli(SWAP, Pops)

INSERT = sum([SWAPp[i]*msgXYZ[name[0]]*LXYZ[name[1]] for i, name in enumerate(Pnames)])

#########################################################################################

tiny_N = sum([a.dag()*a for a in f[1:]]) # number operator

state = big_TFD.copy()
big_EL = qt.tensor(qt.identity(2), EL)
big_ER = qt.tensor(qt.identity(2), ER)
big_N = qt.tensor(qt.identity(2), tiny_N)#N)

def teleportation(state, g, t=10):
    global big_EL, big_ER, big_N, INSERT
    return (-1j*big_ER*t).expm()*(1j*g*big_N).expm()*(-1j*big_EL*t).expm()*INSERT*(1j*big_EL*t).expm()*state

G = np.linspace(-10, 10, 300)
Zs = [qt.expect(RXYZ["Z"], teleportation(state, g)) for g in G]
g = G[np.argmin(Zs)]

state2 = teleportation(state, g)
exiting_star = Rstar(state2)

#########################################################################################

BOOST = qt.tensor(qt.identity(2), ER - EL)
ETA = EL + ER - g*tiny_N#N
TE = qt.tensor(qt.identity(2), ETA - qt.expect(ETA, TFD))
PR = qt.tensor(qt.identity(2), -ER - g*N/2)
PL = qt.tensor(qt.identity(2), -EL - g*N/2)
P = -1j*commutator(BOOST, TE)

#########################################################################################

LfermionNs =  [qt.tensor(qt.identity(2), Lfdag[i]*Lf[i]) for i in range(len(Lf))]
RfermionNs =  [qt.tensor(qt.identity(2), Rfdag[i]*Rf[i]) for i in range(len(Rf))]
CfermionNs = [qt.tensor(qt.identity(2), a.dag()*a) for a in f]

import matplotlib.pyplot as plt
import vpython as vp

class SYKGraphics:
    def __init__(self, show_LRfermions=True,\
                            show_Cfermions=True,\
                            show_XYZexpectations=True):
        self.show_LRfermions = show_LRfermions
        self.show_Cfermions = show_Cfermions
        self.show_XYZexpectations = show_XYZexpectations
        self.make_graphics()

    def make_graphics(self):
        global n, n_
        self.pts = 0
        if self.show_LRfermions:
            self.pts += 2
        if self.show_Cfermions:
            self.pts += 1

        self.fig, self.axes = plt.subplots(1, self.pts, figsize=(9,4))
        if self.pts == 1:
            self.axes = [self.axes]

        self.axis_info = {}
        running = 0
        if self.show_LRfermions:
            self.num_fermL = self.axes[running].bar(np.arange(n_), [0]*(n_))
            self.axes[running].set_xticks(np.arange(n_))
            self.axes[running].set_xticklabels(["L%d"%i for i in range(n_)])
            self.axis_info["num_fermL"] = self.axes[running]
            running += 1
            self.num_fermR = self.axes[running].bar(np.arange(n_), [0]*(n_))
            self.axes[running].set_xticks(np.arange(n_))
            self.axes[running].set_xticklabels(["R%d"% i for i in range(n_)])
            self.axis_info["num_fermR"] = self.axes[running]
            running += 1
        if self.show_Cfermions:
            self.num_cferm = self.axes[running].bar(np.arange(n), [0]*(n))
            self.axes[running].set_xticks(np.arange(n))
            self.axes[running].set_xticklabels(["C%d"% (i) for i in range(n)])
            self.axis_info["num_cferm"] = self.axes[running]
            running += 1
        if self.show_XYZexpectations:
            self.Lvsphere = vp.sphere(pos=vp.vector(-2,0,0),\
                                     color=vp.color.red,\
                                     opacity=0.5)
            self.Rvsphere = vp.sphere(pos=vp.vector(2,0,0),\
                                     color=vp.color.blue,\
                                     opacity=0.5)
            self.Lvstar = vp.sphere(pos=self.Lvsphere.pos, radius=0.3, emissive=True)
            self.Rvstar = vp.sphere(pos=self.Rvsphere.pos, radius=0.3, emissive=True)
        self.fig.canvas.draw()

    def view(self):
        global Lf, Lfdag, n, Rfdag, f, state

        lfn = [qt.expect(LfermionNs[i], state) for i in range(len(Lf))]
        rfn = [qt.expect(RfermionNs[i], state) for i in range(len(Rf))]
        cn = [qt.expect(CfermionNs[i], state) for i in range(len(f))]

        print("LEFT:")
        for i, F in enumerate(lfn):
            print(" fermion %d: %f" % (i, lfn[i])) if self.show_LRfermions else None
        print("RIGHT:")
        for i, F in enumerate(rfn):
            print(" fermion %d: %f" % (i, rfn[i])) if self.show_LRfermions else None
        if self.show_Cfermions:
            print("COMPLEX:")
            for i, c in enumerate(cn):
                print("  cfermion %d: %f" % (i, c))

        if self.show_LRfermions:
            [b.set_height(v) for b, v in zip(self.num_fermL, lfn)]
            self.axis_info["num_fermL"].set_ylim([0,1])
            [b.set_height(v) for b, v in zip(self.num_fermR, rfn)]
            self.axis_info["num_fermR"].set_ylim([0,1])

        if self.show_Cfermions:
            [b.set_height(v) for b, v in zip(self.num_cferm, cn)]
            self.axis_info["num_cferm"].set_ylim([0,1])

        if self.show_XYZexpectations:
            self.Lvstar.pos = self.Lvsphere.pos + vp.vector(*Lstar(state)[1:])
            self.Rvstar.pos = self.Rvsphere.pos + vp.vector(*Rstar(state)[1:])
        self.fig.canvas.draw()
        plt.pause(0.00001)
        print()

syk = SYKGraphics()
syk.view()

def evolve(op=None, dt=0.1, T=100, sign=-1j):
    global syk, state, TE
    if type(op) == type(None):
        op = TE
    U = (sign*op*dt).expm()
    for t in range(T):
        state = (U*state).unit()
        syk.view()

def insert(t=0):
    global state, syk, big_EL, INSERT
    #print("evolving back in time...")
    #dt = 0.1 if t < 0 else -0.1
    #evolve(op=big_EL, dt=dt, T=abs(int(t/dt)), sign=1j)
    print("inserting...")
    state = INSERT*state
    state = state.unit()
    syk.view()