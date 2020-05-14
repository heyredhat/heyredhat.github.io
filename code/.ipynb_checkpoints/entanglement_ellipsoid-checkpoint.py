import qutip as qt
import numpy as np
import vpython as vp
from itertools import product
scene = vp.canvas(background=vp.color.white)

#######################################

IXYZ = {"I": qt.identity(2),\
        "X": qt.sigmax(),\
        "Y": qt.sigmay(),\
        "Z": qt.sigmaz()}

def state_to_four(state):
    global IXYZ
    return qt.Qobj(np.array([qt.expect(IXYZ[o], state)\
                            for o in ["I", "X", "Y", "Z"]]))

def four_to_herm(four):
    global IXYZ
    four = four.full().T[0]
    return sum([four[i]*IXYZ[o]/2 for i, o in enumerate(["I", "X", "Y", "Z"])])

def dm_to_four(dm):
    global IXYZ
    return  qt.Qobj(np.array([[(dm*qt.tensor(IXYZ[x], IXYZ[y])).tr()\
                for y in ["I", "X", "Y", "Z"]]\
                    for x in ["I", "X", "Y", "Z"]]))/2

def four_to_dm(four):
    global IXYZ
    four = four.full()
    terms = []
    for i, y in enumerate(["I", "X", "Y", "Z"]):
        for j, x in enumerate(["I", "X", "Y", "Z"]):
            terms.append(four[i][j]*qt.tensor(IXYZ[x], IXYZ[y]))
    return sum(terms)/2

#######################################

show_both = False
state = qt.rand_ket(4)
#state = qt.tensor(qt.rand_ket(2), qt.rand_ket(2))
#state = qt.bell_state("00")
state.dims = [[2,2],[1,1]]
R = dm_to_four(state*state.dag())

N = 4
TXYZs = []
for x in np.linspace(-1,1,N):
    for y in np.linspace(-1,1,N):
        for z in np.linspace(-1,1,N):
            xyz = np.array([x,y,z])
            if(np.linalg.norm(xyz) > 0):
                xyz = xyz/np.linalg.norm(xyz)
                TXYZs.append(qt.Qobj(np.concatenate(([1], xyz))))

vAsphere = vp.sphere(pos=vp.vector(-1.5,0,0),\
                     opacity=0.2, color=vp.color.blue)
vAarrow = vp.arrow(pos=vAsphere.pos)
vBsphere = vp.sphere(pos=vp.vector(1.5,0,0),\
                     opacity=0.2, color=vp.color.red)
vBarrow = vp.arrow(pos=vBsphere.pos)
vApts = [vp.sphere(radius=0.07,\
                    pos=vAsphere.pos+vp.vector(*txyz.full().T[0][1:].real),\
                    color=vp.vector(*txyz.full().T[0][1:].real))\
                        for i, txyz in enumerate(TXYZs)]
vBpts = [vp.sphere(radius=0.07,\
                   color=vp.vector(*txyz.full().T[0][1:].real))\
                        for i, txyz in enumerate(TXYZs)]

if show_both:
    vAsphere_ = vp.sphere(pos=vp.vector(-1.5,-2,0),\
                         opacity=0.2, color=vp.color.blue)
    vAarrow_ = vp.arrow(pos=vAsphere_.pos)
    vBsphere_ = vp.sphere(pos=vp.vector(1.5,-2,0),\
                         opacity=0.2, color=vp.color.red)
    vBarrow_ = vp.arrow(pos=vBsphere_.pos)
    vBpts_ = [vp.sphere(radius=0.07,\
                        pos=vBsphere_.pos+vp.vector(*txyz.full().T[0][1:].real),\
                        color=vp.vector(*txyz.full().T[0][1:].real))\
                            for i, txyz in enumerate(TXYZs)]
    vApts_ = [vp.sphere(radius=0.07,\
                        color=vp.vector(*txyz.full().T[0][1:].real))\
                            for i, txyz in enumerate(TXYZs)]

#######################################

def review(normalize=True):
    global LXYZ, RXYZ, TXYZs, vBts, vAarrow, vBarrow, state, R, show_both

    for i, txyz in enumerate(TXYZs):
        btxyz = txyz.dag()*R
        if normalize:
            vBpts[i].pos = vBsphere.pos+vp.vector(*(btxyz.full()[0][1:]/btxyz.full()[0][0]).real)
        else:
            vBpts[i].pos = vBsphere.pos+vp.vector(*btxyz.full()[0][1:].real)

        if show_both:
            atxyz = R*txyz
            if normalize:
                vApts_[i].pos = vAsphere_.pos + vp.vector(*(atxyz.full().T[0][1:]/atxyz.full().T[0][0]).real)
            else:
                vApts_[i].pos = vAsphere_.pos + vp.vector(*atxyz.full().T[0][1:].real)
    

    vAarrow.axis = vp.vector(*R.full()[:,0][1:].real)
    vBarrow.axis = vp.vector(*R.full()[0][1:].real)

    if show_both:
        vAarrow_.axis = vAarrow.axis
        vBarrow_.axis = vBarrow.axis

review()

#######################################

def rand_op():
    op = qt.rand_herm(4)
    op.dims = [[2,2], [2,2]]
    return op

def evolve(op=None, i=None, dt=0.08, T=20):
    global state, R
    if op == None:
        op = rand_op()
    U = (-1j*op*dt).expm()
    if i == 0:
        U = qt.tensor(U, qt.identity(2))
    elif i == 1:
        U = qt.tensor(qt.identity(2), U)
    for t in range(T):
        state = U*state
        R = dm_to_four(state*state.dag())
        review()

#######################################
#evolve(op=qt.sigmaz(), i=0)
#evolve(op=qt.sigmaz(), i=1)
#evolve()
