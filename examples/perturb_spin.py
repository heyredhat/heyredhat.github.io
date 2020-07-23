import numpy as np
import qutip as qt
import vpython as vp

vp.scene.background = vp.color.white
vp.scene.width = 1000
vp.scene.height = 800

##########################################################################################

# from the south pole
def c_xyz(c):
        if c == float("Inf"):
            return np.array([0,0,-1])
        else:
            x, y = c.real, c.imag
            return np.array([2*x/(1 + x**2 + y**2),\
                             2*y/(1 + x**2 + y**2),\
                   (1-x**2-y**2)/(1 + x**2 + y**2)])

# np.roots takes: p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]
def poly_roots(poly):
    head_zeros = 0
    for c in poly:
        if c == 0:
            head_zeros += 1 
        else:
            break
    return [float("Inf")]*head_zeros + [complex(root) for root in np.roots(poly)]

def spin_poly(spin):
    j = (spin.shape[0]-1)/2.
    v = spin
    poly = []
    for m in np.arange(-j, j+1, 1):
        i = int(m+j)
        poly.append(v[i]*\
            (((-1)**(i))*np.sqrt(np.math.factorial(2*j)/\
                        (np.math.factorial(j-m)*np.math.factorial(j+m)))))
    return poly

def spin_XYZ(spin):
    return [c_xyz(root) for root in poly_roots(spin_poly(spin))]

def get_phase(v):
    c = None
    if isinstance(v, qt.Qobj):
        v = v.full().T[0]
    i = (v!=0).argmax(axis=0)
    c = v[i]
    return np.exp(1j*np.angle(c))

##########################################################################################

def display(spin, where, radius=1):
    j = (spin.shape[0]-1)/2
    vsphere = vp.sphere(color=vp.color.blue,\
                        opacity=0.5,\
                        radius=radius,
                        pos=where)
    vstars = [vp.sphere(emissive=True,\
                        make_trail=True,\
                        radius=radius*0.3,\
                        color=vp.vector(*np.random.rand(3)),\
                        pos=vsphere.pos+vsphere.radius*vp.vector(*xyz))\
                            for i, xyz in enumerate(spin_XYZ(spin.full().T[0]))]
    varrow = vp.arrow(pos=vsphere.pos,\
                      axis=vsphere.radius*vp.vector(qt.expect(qt.jmat(j, 'x'), spin),\
                                                    qt.expect(qt.jmat(j, 'y'), spin),\
                                                    qt.expect(qt.jmat(j, 'z'), spin)))
    phase = get_phase(spin)
    vphase = vp.arrow(pos=vp.vector(0,3,0),\
                      color=vp.color.yellow,\
                      axis=vp.vector(phase.real, phase.imag, 0))
    return vsphere, vstars, varrow, vphase

def fix_stars(old_stars, new_stars):
    ordering = [None]*len(old_stars)
    for i, old_star in enumerate(old_stars):
        dists = np.array([np.linalg.norm(new_star-old_star)\
                            for new_star in new_stars])
        minim = np.argmin(dists)
        if np.count_nonzero(dists == dists[minim]) == 1:
            ordering[i] = new_stars[minim]
        else:
            print ("WH")
            return new_stars
    return ordering

dont_use_last_stars = False
def update(spin, vsphere, vstars, varrow, vphase):
    global dont_use_last_stars
    j = (spin.shape[0]-1)/2
    good_stars = None
    new_stars = spin_XYZ(spin.full().T[0])
    if not dont_use_last_stars:
        old_stars = [np.array([vstar.pos.x, vstar.pos.y, vstar.pos.z]) for vstar in vstars]
        good_stars = fix_stars(old_stars, new_stars)
    else:
        good_stars = new_stars
        dont_use_last_stars = False
    for i, xyz in enumerate(good_stars):
        vstars[i].pos = vsphere.pos+vsphere.radius*vp.vector(*xyz)
    varrow.axis = vsphere.radius*vp.vector(qt.expect(qt.jmat(j, 'x'), spin),\
                                           qt.expect(qt.jmat(j, 'y'), spin),\
                                           qt.expect(qt.jmat(j, 'z'), spin))
    phase = get_phase(spin)
    vphase.axis = vp.vector(phase.real, phase.imag, 0)
    return vsphere, vstars, varrow

##########################################################################################

j = 3/2
n = int(2*j+1)
dt = 0.002
XYZ = {"X": qt.jmat(j, 'x'),\
       "Y": qt.jmat(j, 'y'),\
       "Z": qt.jmat(j, 'z')}
state = qt.rand_ket(n)#qt.basis(n, 0)#
H = qt.rand_herm(n)#qt.jmat(j, 'x')#qt.rand_herm(n)#
HL, HV = H.eigenstates()
HP = [v*v.dag() for v in HV]
U = (-1j*H*dt).expm()

vsphere, vstars, varrow, vphase = display(state, vp.vector(0,0,0), radius=2)

evolving = True
def keyboard(e):
    global state, XYZ, dt, n, H, U, HP, HL, evolving, vstars, dont_use_last_stars
    key = e.key
    DT = 0.1
    if key == "a":
        state = (DT*1j*XYZ["X"]).expm()*state
    elif key == "d":
        state = (-DT*1j*XYZ["X"]).expm()*state
    elif key == "s":
        state = (DT*1j*XYZ["Y"]).expm()*state
    elif key == "w":
        state = (-DT*1j*XYZ["Y"]).expm()*state
    elif key == "z":
        state = (DT*1j*XYZ["Z"]).expm()*state
    elif key == "x":
        state = (-DT*1j*XYZ["Z"]).expm()*state
    elif key == "i":
        state = qt.rand_ket(n)
        dont_use_last_stars = True
    elif key == "o":
        H = qt.rand_herm(n)
        U = (-1j*H*dt).expm()
        HL, HV = H.eigenstates()
        HP = [v*v.dag() for v in HV]
    elif key == "e":
        probs = np.array([qt.expect(HP[i], state) for i in range(n)])
        choice = np.random.choice(list(range(n)), p=abs(probs/sum(probs)))
        state = (HP[choice]*state).unit()
        print(HL[choice])
        dont_use_last_stars = True
    elif key == "p":
        evolving = False if evolving else True
    elif key == "t":
        for vstar in vstars:
            vstar.make_trail = False if vstar.make_trail else True
    elif key == "c":
        for vstar in vstars:
            vstar.clear_trail()

vp.scene.bind('keydown', keyboard)

while True:
    if evolving:
        state = U*state
    update(state, vsphere, vstars, varrow, vphase)
    vp.rate(1000)
   