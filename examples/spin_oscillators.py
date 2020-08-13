import qutip as qt
import numpy as np
import vpython as vp
from magic import *
from itertools import product
vp.scene.background = vp.color.white

####################################################################################

def upgrade(O, a):
    O = O.full()
    terms = []
    for i in range(2):
        for j in range(2):
            terms.append(a[i].dag()*O[i][j]*a[j])
    return sum(terms)

def upgrade_state(spin, a):
    n = spin.shape[0]
    z, w = [a_.dag() for a_ in a]
    return sum([c*((z**(n-i-1))*(w**(i)))/\
                    np.sqrt(np.math.factorial(i)*np.math.factorial(n-i-1))\
                         for i, c in enumerate(spin.full().T[0])])

def upgrade_state2(spin, a):
    return reduce(lambda x, y: x*y, [sum([c*a[i].dag()\
        for i, c in enumerate(xyz_spinor(xyz))]) for xyz in spin_XYZ(spin)])

####################################################################################

dt = 0.005
n = 5

a = [qt.tensor(qt.destroy(n), qt.identity(n)),\
     qt.tensor(qt.identity(n), qt.destroy(n))]

XYZ = {"X": upgrade(qt.sigmax(), a),\
       "Y": upgrade(qt.sigmay(), a),\
       "Z": upgrade(qt.sigmaz(), a)}

X, Y, Z = XYZ["X"], XYZ["Y"], XYZ["Z"]     
XL, XV = X.eigenstates()
XP = [v*v.dag() for v in XV]
YL, YV = Y.eigenstates()
YP = [v*v.dag() for v in YV]
ZL, ZV = Z.eigenstates()
ZP = [v*v.dag() for v in ZV]

def check(X, Y, Z):
    print(qt.commutator(X, Y) - 2*1j*Z)
    print(qt.commutator(Y, Z) - 2*1j*X)
    print(qt.commutator(Z, X) - 2*1j*Y)

####################################################################################

N = sum([a[i].dag()*a[i]for i in range(2)])
NL, NV = N.eigenstates()
NP = [v*v.dag() for v in NV]

tensor_basis_labels = list(product(list(range(n)), repeat=2))
D = np.array([sum(l) for l in tensor_basis_labels])
E = sorted(list(set(D)))

def extract(state):
    global NL, NV, E, D
    return [qt.Qobj(state[np.where(D == e)[0]][:(e+1)][::-1]) for e in E]

####################################################################################

state = qt.basis(n*n, 0)#qt.rand_ket(n*n)
state.dims = [[n, n], [1,1]]

spin = qt.rand_ket(4)
state = upgrade_state(spin, a)*state
#state = (upgrade_state2(spin, a)*state).unit()

H = N + 1#X#qt.rand_herm(n*n)
H.dims = [[n, n], [n, n]]
U = (-1j*H*dt).expm()

####################################################################################

Q = [qt.tensor(qt.position(n), qt.identity(n)),\
     qt.tensor(qt.identity(n), qt.position(n))]
Q_ = qt.position(n)
QL, QV = Q_.eigenstates()
Qstates = [[qt.tensor(QV[i], QV[j]) for j in range(n)] for i in range(n)]
Qprojs = [[Qstates[i][j]*Qstates[i][j].dag() for j in range(n)] for i in range(n)]

####################################################################################

vorig_sph = vp.sphere(pos=vp.vector(0, 5, 0), color=vp.color.red, opacity=0.5)
voring_stars = [vp.sphere(pos=vorig_sph.pos + vp.vector(*xyz), radius=0.2, emissive=True)\
                    for xyz in spin_XYZ(spin)]

####################################################################################

jstates = extract(state)
js = [(js.shape[0]-1)/2 for js in jstates]
L = len(jstates)
vspheres = [vp.sphere(pos=vp.vector(-L + 2.5*i, -5, 0),\
                      color=vp.color.blue, opacity=jstates[i].norm())\
                for i in range(L)]
vstars = []
for i in range(L):
    vsrow = []
    if jstates[i].norm() != 0:
        for xyz in spin_XYZ(jstates[i]):
            vsrow.append(vp.sphere(pos=vspheres[i].pos + vp.vector(*xyz),\
                             radius=0.2, emissive=True))
    else:
        for i in range(int(2*js[i])):
            vsrow.append(vp.sphere(radius=0.2, emissive=True, visible=False))
    vstars.append(vsrow)

varrows = []
for i in range(L):
    s = jstates[i].unit() if jstates[i].norm() != 0 else jstates[i]
    if js[i] == 0:
        varrows.append(vp.arrow(pos=vspheres[i].pos, \
            axis=vp.vector(s[0][0][0].real, s[0][0][0].imag, 0)))
    else:
        varrows.append(vp.arrow(pos=vspheres[i].pos, \
            axis=vp.vector(*[qt.expect(qt.jmat(js[i], O), jstates[i]) for O in ["x", "y", "z"]])))

####################################################################################

pos_amps = [[state.overlap(Qstates[i][j])
                for j in range(n)]
                    for i in range(n)]
vpos = [[vp.arrow(pos=vp.vector(QL[i], QL[j], 0),\
                  color=vp.color.black,\
                  axis=vp.vector(pos_amps[i][j].real, pos_amps[i][j].imag, 0))\
            for j in range(n)] for i in range(n)]

vpos_exp = vp.sphere(pos=vp.vector(qt.expect(Q[0], state).real, qt.expect(Q[1], state).real, 0),\
                     color=vp.color.yellow, radius=0.4)

####################################################################################

def keyboard(e):
    global n, state, XP, XL, YP, YL, ZP, ZL, NP, NL, QP, QL, H, U, X, Y, Z
    k = e.key
    if k == "x":
        probs = np.array([qt.expect(XP[i], state) for i in range(n*n)])
        print(probs)
        choice = np.random.choice(list(range(n*n)), p=abs(probs/sum(probs)))
        state = (XP[choice]*state).unit()
        print(XL[choice])
    elif k == "y":
        probs = np.array([qt.expect(YP[i], state) for i in range(n*n)])
        print(probs)
        choice = np.random.choice(list(range(n*n)), p=abs(probs/sum(probs)))
        state = (YP[choice]*state).unit()
        print(YL[choice])
    elif k == "z":
        probs = np.array([qt.expect(ZP[i], state) for i in range(n*n)])
        print(probs)
        choice = np.random.choice(list(range(n*n)), p=abs(probs/sum(probs)))
        state = (ZP[choice]*state).unit()
        print(ZL[choice])
    elif k == "n":
        probs = np.array([qt.expect(NP[i], state) for i in range(n*n)])
        print(probs)
        choice = np.random.choice(list(range(n*n)), p=abs(probs/sum(probs)))
        state = (NP[choice]*state).unit()
        print(NL[choice])
    elif k == "i":
        state = qt.rand_ket(n*n)
        state.dims = [[n, n], [1,1]]
    elif k == "q":
        probs = []
        indices = []
        for i in range(n):
            for j in range(n):
                probs.append(qt.expect(Qprojs[i][j], state))
                indices.append((i, j))
        probs = np.array(probs)
        print(probs)
        choice = np.random.choice(list(range(n*n)), p=abs(probs/sum(probs)))
        i, j = indices[choice]
        state = (Qprojs[i][j]*state).unit()
        print(QL[i], QL[j])
    elif k == "e":
        H = N + 1#X#qt.rand_herm(n*n)
        H.dims = [[n, n], [n, n]]
        U = (-1j*H*dt).expm()
    elif k == "h":
        H = qt.rand_herm(n*n)
        H.dims = [[n, n], [n, n]]
        U = (-1j*H*dt).expm()
    elif k == "s":
        H = [X, Y, Z][np.random.randint(3)]
        H.dims = [[n, n], [n, n]]
        U = (-1j*H*dt).expm()

vp.scene.bind('keydown', keyboard)

####################################################################################

while True:
    state = U*state
    jstates = extract(state)

    for i in range(L):
        vspheres[i].opacity = jstates[i].norm()
        if jstates[i].norm() != 0:
            for j, xyz in enumerate(spin_XYZ(jstates[i])):
                vstars[i][j].pos = vspheres[i].pos + vp.vector(*xyz)
                vstars[i][j].visible = True
        else:
            if js[i] != 0:
                for j in range(int(2*js[i])):
                    vstars[i][j].visible = False
        s = jstates[i].unit() if jstates[i].norm() != 0 else jstates[i]
        if js[i] == 0:
            varrows[i].axis=vp.vector(s[0][0][0].real, s[0][0][0].imag, 0)
        else:
            varrows[i].axis=vp.vector(*[qt.expect(qt.jmat(js[i], O), s) for O in ["x", "y", "z"]])

    for i in range(n):
        for j in range(n):
            amp = state.overlap(Qstates[i][j])
            vpos[i][j].axis = vp.vector(amp.real, amp.imag, 0)

    vpos_exp.pos = vp.vector(qt.expect(Q[0], state).real, qt.expect(Q[1], state).real, 0)
            