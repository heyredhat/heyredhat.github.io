import math
import qutip as qt
import numpy as np
from itertools import combinations
from functools import reduce 
from itertools import product
from qutip.qip.operations.gates import swap
import matplotlib.pyplot as plt
import vpython as vp

def fermion_operators(n):
    return [qt.tensor(*[qt.destroy(2) if i == j\
                else (qt.sigmaz() if j < i\
                    else qt.identity(2))\
                        for j in range(n)])\
                            for i in range(n)]

def split_fermion(f):
    return (f + f.dag()), 1j*(f - f.dag())

def join_majoranas(m, n):
    return (1/2)*(m + 1j*n)#, (1/2)*(m - 1j*n)

#########################################################################################

def anticommutator(a, b):
    return a*b + b*a

def test_fermions(f):
    d = f[0].shape[0]
    for i in range(len(f)):
        for j in range(len(f)):
            test1 = anticommutator(f[i], f[j]).full()
            test2 = anticommutator(f[i], f[j].dag()).full()
            if not \
                (np.isclose(test1, np.zeros((d,d))).all()\
                    and \
                ((np.isclose(test2, np.zeros((d,d))).all() and i != j)\
                        or (np.isclose(test2, np.eye(d)).all() and i == j))):
                return False
    return True

def test_majoranas(m):
    d = m[0].shape[0]
    for i in range(len(m)):
        for j in range(len(m)):
            test = anticommutator(m[i], m[j]).full()
            if not ((i == j and np.isclose(test, 2*np.eye(d)).all()) or\
                    (i != j and np.isclose(test, np.zeros((d,d))).all())):
                return False
    return True

#########################################################################################

def majorana_basis(m):
    n = len(m)
    prefac = 2**(-n/4)
    M = {}
    for i in range(n+1):
        if i == 0:
            M["I"] = prefac*qt.identity(m[0].shape[0])
            M["I"].dims = m[0].dims
        else:
            for string in combinations(list(range(n)), i):
                M[string] = prefac*reduce(lambda x, y: x*y, [m[j] for j in string])
    return M

def op_coeffs(O, M):
    return dict([(string, (string_op.dag()*O).tr()) for string, string_op in M.items()])

def coeffs_op(coeffs, M):
    return sum([coeffs[string]*string_op for string, string_op in M.items()])

def prettyc(coeffs):
    for string, coeff in coeffs.items():
        if not np.isclose(coeff, 0):
            print("%s : %s" % (string, coeff))

#########################################################################################

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

def construct_thermal_dm(H, beta=0):
    return (-beta*H).expm()/(-beta*H).expm().tr()

def construct_tfd(H, beta=0):
    rho_ = (-beta*H*(1/2)).expm()/np.sqrt((-beta*H*(1/2)).expm().tr())
    rhoL = coeffs_op(op_coeffs(rho_, M), ML)
    return (rhoL*c_vac).unit()
    
#########################################################################################

def majoranas_qubit(m):
    I = qt.identity(m[0].shape[0])
    I.dims = m[0].dims
    return {"I": I,
            "X": -1j*m[0]*m[2],\
            "Y": -1j*m[2]*m[1],\
            "Z": -1j*m[1]*m[0]}

def test_XYZ(XYZ):
    return [qt.commutator(XYZ["X"], XYZ["Y"]) == 2j*XYZ["Z"],\
            qt.commutator(XYZ["Y"], XYZ["Z"]) == 2j*XYZ["X"],\
            qt.commutator(XYZ["Z"], XYZ["X"]) == 2j*XYZ["Y"],\
            qt.commutator(XYZ["X"], XYZ["X"]) == qt.commutator(XYZ["Y"], XYZ["Y"])\
                                              == qt.commutator(XYZ["Z"], XYZ["Z"]),\
            qt.commutator(XYZ["X"], XYZ["X"]).norm() == 0]

def qubit_xyz(state, XYZ):
    return np.array([qt.expect(XYZ["X"], state),\
                     qt.expect(XYZ["Y"], state),\
                     qt.expect(XYZ["Z"], state)])

def pauli_basis(n):
    IXYZ = {"I": qt.identity(2), "X": qt.sigmax(), "Y": qt.sigmay(), "Z": qt.sigmaz()}
    return dict([("".join(P), qt.tensor(*[IXYZ[p]/2 for p in P])) for P in product(IXYZ, repeat=n)])

#########################################################################################

n = 4
beta = 1

f = fermion_operators(n)
fvac = qt.basis(2**n)
fvac.dims = [[2]*n, [1]*n]

m = []
for i in range(n):
    m.extend(split_fermion(f[i]))
M = majorana_basis(m)

N = 2*n
lr = fermion_operators(N)

mL = []
for i in range(n): 
    mL.extend(split_fermion(lr[i]))

mR = []
for i in range(n, N): 
    mR.extend(split_fermion(lr[i]))

cf = [join_majoranas(mL[i], mR[i]) for i in range(N)]

N_lr = sum([lr[i].dag()*lr[i] for i in range(N)])
N_c = sum([cf[i].dag()*cf[i] for i in range(N)])

lr_vac = qt.basis(2**N, 0)
lr_vac.dims = [[2]*N, [1]*N]

N_cL, N_cV = N_c.eigenstates()
c_vac = N_cV[0]

ML = majorana_basis(mL)
MR = majorana_basis([-1j*m_ for m_ in mR])

SYK = random_syk_couplings(N)
H = syk_ham(SYK, m)
HL = syk_ham(SYK, mL)
HR = syk_ham(SYK, [-1j*m_ for m_ in mR]) 

rho = construct_thermal_dm(H, beta=beta)
TFD = construct_tfd(H, beta=beta)

######################################################################################### 

def size(O, i=None):
    global M, ML, N_c, cf, c_vac
    majorana_state = coeffs_op(op_coeffs(O, M), ML)*c_vac
    if type(i) == type(None):
        return qt.expect(N_c, majorana_state)
    else:
        return qt.expect(cf[i].dag()*cf[i], majorana_state)

def cold_size(O, i=None):
    global M, ML, N_c, cf, TFD
    majorana_state = coeffs_op(op_coeffs(O, M), ML)*TFD
    if type(i) == type(None):
        return qt.expect(N_c, majorana_state)
    else:
        return qt.expect(cf[i].dag()*cf[i], majorana_state)

######################################################################################### 

XYZ_L = majoranas_qubit(mL[:3])
XYZ_R = majoranas_qubit(mR[:3])

IDrest = qt.identity(2**N)
IDrest.dims = [[2]*N, [2]*N]

msg = (qt.basis(2,0)+qt.basis(2,1)).unit()
state = qt.tensor(msg, TFD)

XYZ_msg = {"I": qt.tensor(qt.identity(2), IDrest),\
           "X": qt.tensor(qt.sigmax(), IDrest),\
           "Y": qt.tensor(qt.sigmay(), IDrest),\
           "Z": qt.tensor(qt.sigmaz(), IDrest)}

XYZ_L_ = dict([(op_name, qt.tensor(qt.identity(2), op)) for op_name, op in XYZ_L.items()])
XYZ_R_ = dict([(op_name, qt.tensor(qt.identity(2), op)) for op_name, op in XYZ_R.items()])

P = pauli_basis(2)
SWAP = swap(N=2, targets=[0,1])
SWAPc = op_coeffs(SWAP, P)
INSERT = sum([coeff*XYZ_msg[op_name[0]]*XYZ_L_[op_name[1]] for op_name, coeff in SWAPc.items()])

HL_ = qt.tensor(qt.identity(2), HL)
HR_ = qt.tensor(qt.identity(2), HR)
SIZE = qt.tensor(qt.identity(2), sum([cf_.dag()*cf_ for cf_ in cf[2:]]))

def wormhole(state, g, t=10):
    global HL, HR, SIZE
    return (-1j*HR_*t).expm()*(1j*g*SIZE).expm()*(-1j*HL_*t).expm()*INSERT*(1j*HL_*t).expm()*state

print("optimizing...")

G = np.linspace(-5, 5, 50)
X = [qt.expect(XYZ_R_["X"], wormhole(state, g)) for g in G]
g = G[np.argmax(X)] # since we started with the msg being [1,0,0]

final_state = wormhole(state, g)
final_qubit = qubit_xyz(final_state, XYZ_R_)

print("final R qubit: %s" % final_qubit)

#plt.plot(G, X, linewidth=2.0)
#plt.xlabel("g")
#plt.ylabel("<X>")
#plt.show()

#########################################################################################

N_lefts = [qt.tensor(qt.identity(2), lr[i].dag()*lr[i]) for i in range(n)]
N_rights = [qt.tensor(qt.identity(2), lr[i].dag()*lr[i]) for i in range(n, N)]
N_cmplxs = [qt.tensor(qt.identity(2), cf[i].dag()*cf[i]) for i in range(N)]

vp.scene.background = vp.color.white

class SYKGraphics:
    def __init__(self, show_LRfermions=True,\
                       show_Cfermions=True,\
                       show_XYZexpectations=True):
        self.show_LRfermions = show_LRfermions
        self.show_Cfermions = show_Cfermions
        self.show_XYZexpectations = show_XYZexpectations
        self.make_graphics()

    def make_graphics(self):
        global N, n

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
            self.num_fermL = self.axes[running].bar(np.arange(n), [0]*(n))
            self.axes[running].set_xticks(np.arange(n))
            self.axes[running].set_xticklabels(["L%d"%i for i in range(n)])
            self.axis_info["num_fermL"] = self.axes[running]
            running += 1
            self.num_fermR = self.axes[running].bar(np.arange(n), [0]*(n))
            self.axes[running].set_xticks(np.arange(n))
            self.axes[running].set_xticklabels(["R%d"% i for i in range(n)])
            self.axis_info["num_fermR"] = self.axes[running]
            running += 1
        if self.show_Cfermions:
            self.num_cferm = self.axes[running].bar(np.arange(N), [0]*(N))
            self.axes[running].set_xticks(np.arange(N))
            self.axes[running].set_xticklabels(["C%d"% (i) for i in range(N)])
            self.axis_info["num_cferm"] = self.axes[running]
            running += 1
        if self.show_XYZexpectations:
            self.Lvsphere = vp.sphere(pos=vp.vector(-1.5,0,0),\
                                      color=vp.color.red,\
                                      opacity=0.2)
            self.Rvsphere = vp.sphere(pos=vp.vector(1.5,0,0),\
                                      color=vp.color.blue,\
                                      opacity=0.2)
            self.Lvstar = vp.arrow(pos=self.Lvsphere.pos, axis=vp.vector(0,0,0))
            self.Rvstar = vp.arrow(pos=self.Rvsphere.pos, axis=vp.vector(0,0,0))
        self.fig.canvas.draw()

    def view(self):
        global N_lefts, N_rights, N_cmplxs, XYZ_L_, XYZ_R_

        lfn = [qt.expect(o, state) for o in N_lefts]
        rfn = [qt.expect(o, state) for o in N_rights]
        cn = [qt.expect(o, state) for o in N_cmplxs]
        xyzL = qubit_xyz(state, XYZ_L_)
        xyzR = qubit_xyz(state, XYZ_R_)

        print("LEFT:")
        for i, F in enumerate(lfn):
            print(" fermion %d: %f" % (i, lfn[i])) if self.show_LRfermions else None
        print("RIGHT:")
        for i, F in enumerate(rfn):
            print(" fermion %d: %f" % (i, rfn[i])) if self.show_LRfermions else None
        print("COMPLEX:")
        for i, c in enumerate(cn):
            print("  cfermion %d: %f" % (i, c))
        print("LQUBIT: %s" % xyzL)
        print("RQUBIT: %s" % xyzR)

        if self.show_LRfermions:
            [b.set_height(v) for b, v in zip(self.num_fermL, lfn)]
            self.axis_info["num_fermL"].set_ylim([0,1])
            [b.set_height(v) for b, v in zip(self.num_fermR, rfn)]
            self.axis_info["num_fermR"].set_ylim([0,1])

        if self.show_Cfermions:
            [b.set_height(v) for b, v in zip(self.num_cferm, cn)]
            self.axis_info["num_cferm"].set_ylim([0,1])

        if self.show_XYZexpectations:
            self.Lvstar.axis = vp.vector(*xyzL)
            self.Rvstar.axis = vp.vector(*xyzR)

        self.fig.canvas.draw()
        plt.pause(0.00001)
        print()

syk_graphics = SYKGraphics()
syk_graphics.view()

#########################################################################################

BOOST = HR_ - HL_
ETA = HR_ + HL_ - g*SIZE
E = ETA - qt.expect(ETA, state)
PR = -HR_ - g*SIZE/2
PL =  -HL_ - g*SIZE/2
P = -1j*qt.commutator(BOOST, E)

def evolve(op=None, dt=0.1, T=100, sign=-1j):
    global syk_graphics, state, E
    if type(op) == type(None):
        op = E
    U = (sign*op*dt).expm()
    for t in range(T):
        state = U*state
        syk_graphics.view()

def insert(t=0, dt=0.1):
    global syk_graphics, state, INSERT, HL_
    evolve(op=HL_, dt=dt, T=abs(int(t/dt)), sign=1j)
    state = INSERT*state
    evolve(op=HL_, dt=dt, T=abs(int(t/dt)), sign=-1j)
    syk_graphics.view()
