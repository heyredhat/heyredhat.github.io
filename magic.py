import math
#import sympy
import numpy as np
import functools
import qutip as qt
import itertools
import scipy
from functools import *
from itertools import *
import scipy.linalg

####################################################################

def d_j(d):
    return (d-1)/2

def j_d(j):
    return 2*j + 1

####################################################################

def normalize(v):
    n = np.linalg.norm(v)
    if n != 0:
        return v/n
    else:
        return v

def get_phase(v):
    c = None
    if isinstance(v, qt.Qobj):
        v = v.full().T[0]
    i = (v!=0).argmax(axis=0)
    c = v[i]
    return np.exp(1j*np.angle(c))

def normalize_phase(v):
    return v/get_phase(v)

####################################################################

def random_complex_unit_vector(d):
	v = np.random.randn(d) + 1j*np.random.randn(d)
	return v/np.linalg.norm(v)

def random_hermitian_matrix(d):
	H = np.random.randn(d,d) + 1j*np.random.randn(d, d)
	return H + H.T.conj()

####################################################################

def c_xyz(c, pole="south"):
    if(pole == "south"):
        if c == float("Inf"):
            return np.array([0,0,-1])
        else:
            x, y = c.real, c.imag
            return np.array([2*x/(1 + x**2 + y**2),\
                             2*y/(1 + x**2 + y**2),\
                   (1-x**2-y**2)/(1 + x**2 + y**2)])
    elif (pole == "north"):
        if c == float("Inf"):
            return np.array([0,0,1])
        else:
            x, y = c.real, c.imag
            return np.array([2*x/(1 + x**2 + y**2),\
                             2*y/(1 + x**2 + y**2),\
                   (-1+x**2+y**2)/(1 + x**2 + y**2)])

def xyz_c(xyz, pole="south"):
    x, y, z = xyz
    if (pole=="south"):
        if np.isclose(z,-1):
            return float("Inf")
        else:
            return x/(1+z) + 1j*y/(1+z)
    elif (pole=="north"):
        if np.isclose(z,1):
            return float("Inf")
        else:
            return x/(1-z) + 1j*y/(1-z)

####################################################################

def spinor_xyz(spinor):
    if isinstance(spinor, np.ndarray):
        spinor = qt.Qobj(spinor)
    return np.array([qt.expect(qt.sigmax(), spinor),\
                     qt.expect(qt.sigmay(), spinor),\
                     qt.expect(qt.sigmaz(), spinor)])

def xyz_spinor(xyz):
    return c_spinor(xyz_c(xyz))

def spin_xyz(spin):
    if isinstance(spinor, np.ndarray):
        spinor = qt.Qobj(spinor)
    j = d_j(spin.shape[0])
    return np.array([qt.expect(qt.jmat(j, 'x'), spinor),\
                     qt.expect(qt.jmat(j, 'y'), spinor),\
                     qt.expect(qt.jmat(j, 'z'), spinor)])

####################################################################

def spinor_c(spinor):
    a, b = None, None
    if isinstance(spinor, qt.Qobj):
        a, b = spinor.full().T[0]
    else:
        a, b = spinor
    if np.isclose(a,0):
        return float('Inf')
    else:
        return b/a

def c_spinor(c):
    if c == float('Inf'):
        return np.array([0,1])
    else:
        return normalize(np.array([1, c]))

####################################################################

def spin_poly(spin):
    if isinstance(spin, qt.Qobj):
        spin = spin.full().T[0]
    j = (spin.shape[0]-1)/2.
    v = spin
    poly = []
    for m in np.arange(-j, j+1, 1):
        i = int(m+j)
        poly.append(v[i]*\
            (((-1)**(i))*math.sqrt(math.factorial(2*j)/\
                        (math.factorial(j-m)*math.factorial(j+m)))))
    return poly

def poly_spin(poly):
    j = (len(poly)-1)/2.
    spin = []
    for m in np.arange(-j, j+1):
        i = int(m+j)
        spin.append(poly[i]/\
            (((-1)**(i))*math.sqrt(math.factorial(2*j)/\
                        (math.factorial(j-m)*math.factorial(j+m)))))
    aspin = np.array(spin)
    return aspin/np.linalg.norm(aspin)

####################################################################

def poly_roots(poly):
    head_zeros = 0
    for c in poly:
        if c == 0:
            head_zeros += 1 
        else:
            break
    return [float("Inf")]*head_zeros + [complex(root) for root in np.roots(poly)]

def roots_coeffs(roots):
    n = len(roots)
    coeffs = np.array([((-1)**(-i))*sum([np.prod(term) for term in itertools.combinations(roots, i)]) for i in range(0, len(roots)+1)])
    return coeffs/coeffs[0]
        
def roots_poly(roots):
    zeros = roots.count(0j)
    if zeros == len(roots):
        return [1j] + [0j]*len(roots)
    poles = roots.count(float("Inf"))
    roots = [root for root in roots if root != float('Inf')]
    if len(roots) == 0:
        return [0j]*poles + [1j]
    #Z = sympy.symbols("Z")
    #Poly = sympy.Poly(functools.reduce(lambda a, b: a*b, [Z-root for root in roots]), domain="CC")
    #return [0j]*poles + [complex(c) for c in Poly.all_coeffs()] + [0j]*zeros
    return [0j]*poles + roots_coeffs(roots).tolist() #+ [0j]*(zeros-1)

####################################################################

def spin_roots(spin):
    if isinstance(spin, qt.Qobj):
        spin = spin.full().T[0]
    return poly_roots(spin_poly(spin))

def roots_spin(roots):
    return poly_spin(roots_poly(roots))

####################################################################

def spin_XYZ(spin):
    if isinstance(spin, qt.Qobj):
        spin = spin.full().T[0]
    return [c_xyz(root) for root in poly_roots(spin_poly(spin))]

def XYZ_spin(XYZ):
    #XYZ = [np.array(xyz) for xyz in XYZ]
    #XYZ = [xyz/np.linalg.norm(xyz) for xyz in XYZ]
    return qt.Qobj(poly_spin(roots_poly([xyz_c(xyz) for xyz in XYZ])))

####################################################################

def differentiate_spin(spin, times=1):
    def _differentiate_spin_(spin):
        n = len(spin)-1
        for i in range(len(spin)):
            spin[i] *= n
            n -= 1;
        return normalize(spin[:-1])
    spin = spin.copy()
    for t in range(times):
        spin = _differentiate_spin_(spin)
    return spin

def integrate_spin(spin, times=1):
    def _integrate_spin_(spin):
        return np.array(spin.tolist()+[0j])
    spin = spin.copy()
    for t in range(times):
        spin = _integrate_spin_(spin)
    return spin    

####################################################################

def sph_xyz(sph):
    r, theta, phi = sph;
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return np.array([x,y,z])

def xyz_sph(xyz):
    x, y, z = xyz
    r =  np.sqrt(x*x + y*y + z*z)
    theta =  np.arccos(z/r)
    phi =  np.arctan2(y,x)
    return np.array([r,theta,phi])

####################################################################

def eval_spin_at_c(spin, at):
    if at == float("Inf"):
        return eval_spin_at_c(XYZ_spin([c_xyz(xyz_c(xyz, pole="north")) for xyz in spin_XYZ(spin)]), 0)
    if isinstance(spin, qt.Qobj):
        spin = spin.full().T[0]
    normalized = spin/spin[0]
    poly = spin_poly(normalized)
    n = len(poly)
    return sum([poly[i]*(np.power(at,(n-i-1))) for i in range(n)])

def eval_spin_at(spin, at):
    return eval_spin_at_c(spin, xyz_c(at))

def eval_spin(spin, n=50, a=0.6):
    n=20
    theta, phi = np.linspace(0, 2*np.pi, 2*n), np.linspace(0, np.pi, n/2)
    points = []
    for t in theta:
        for p in phi:
            xyz = sph_xyz(np.array([1,t,p]))
            c = eval_spin_at(spin, xyz)
            points.append([xyz.tolist(), [c.real, c.imag], np.abs(c), np.angle(c)/(2*np.pi), c_xyz(c).tolist(), [((np.angle(c)/(np.pi))+1)/2, (1-np.power(a, np.abs(c))), 0.5]])
    return points

####################################################################

def husimi(spin, at):
    if isinstance(spin, qt.Qobj):
        spin = spin.full().T[0]
    n_stars = int(2*d_j(len(spin)))
    coherent = XYZ_spin([at]*n_stars)
    return np.vdot(spin, coherent)

def coherent(at, n):
    return XYZ_spin([at]*n)

def husimi_snapshot(spin, n=30):
    theta, phi = np.linspace(0, 2*np.pi, 16), np.linspace(0, np.pi, 16)
    points = []
    for t in theta:
        points_row = []
        for p in phi:
            #points.append([xyz.tolist(), [c.real, c.imag], np.abs(c), np.angle(c)/(2*np.pi), c_xyz(c).tolist(), [((np.angle(c)/(np.pi))+1)/2, (1-np.power(a, np.abs(c))), 0.5]])
            xyz = sph_xyz(np.array([1,t,p]))           
            h = husimi(spin, xyz)
            a = np.angle(h)
            points_row.append([xyz.tolist(), [np.abs(h), [np.cos(a), np.sin(a)]]])
        points.append(points_row)
    return points

############################################################

def sym_spin(n):
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
    return Q

def symmeterize(pieces):
    return sum(qt.tensor(*perm)\
        for perm in permutations(pieces, len(pieces))).unit()

def symmeterize_indices(tensor):
    n = tensor.norm()
    pieces = [tensor.permute(p) for p in permutations(list(range(len(tensor.dims[0]))))]
    v = sum(pieces)
    return v if v.norm() == 0 else n*v.unit()

############################################################

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

############################################################

def possible_j3s(j1, j2):
    J3 = [j1-m2 for m2 in np.arange(-j2, j2+1)]\
            if j1 > j2 else\
                [j2-m1 for m1 in np.arange(-j1, j1+1)]
    #print(sum([2*j+1 for j in J3]) == (2*j1+1)*(2*j2+1))
    return J3[::-1]

def tensor_clebsch(j1, j2):
    J3 = possible_j3s(j1, j2)
    states = []
    labels = []
    for j3 in J3:
        substates = []
        sublabels = []
        for m3 in np.arange(-j3, j3+1):
            terms = []
            for m1 in np.arange(-j1, j1+1):
                for m2 in np.arange(-j2, j2+1):
                    terms.append(\
                        qt.clebsch(j1, j2, j3, m1, m2, m3)*\
                        qt.tensor(qt.spin_state(j1, m1),\
                                    qt.spin_state(j2, m2)))
            substates.append(sum(terms))
            sublabels.append((j3, m3))
        states.extend(substates[::-1])
        labels.append(sublabels[::-1])
    return qt.Qobj(np.array([state.full().T[0] for state in states])), labels

def clebsch_split(state, sectors):
    v = state.full().T[0]
    dims = [int(2*sector[0][0] + 1) for sector in sectors]
    running = 0
    clebsch_states = []
    for d in dims:
        clebsch_states.append(qt.Qobj(v[running:running+d]))
        running += d
    return clebsch_states

def clebsch_double_osc(sectors):
    N = int(2*sectors[-1][0][0])+1
    clebsch_basis = []
    for i, sector in enumerate(sectors):
        n = int(2*sector[0][0])
        for x in range(n, -1, -1):
            clebsch_basis.append((x,n-x))
    L = len(clebsch_basis)
    O = sum([qt.tensor(\
                qt.tensor(qt.basis(N, c[0]), qt.basis(N, c[1])),\
                qt.basis(L, i).dag())\
                    for i, c in enumerate(clebsch_basis)])
    O.dims = [[O.shape[0]], [O.shape[1]]]
    return O

def double_osc_clebsch(n):
    osc_basis = []
    for i in range(n):
        for j in range(n):
            osc_basis.append((i, j))
    clebsch_basis = []
    cb = []
    for m in range(2*n):
        sector = []
        for i in range(m+1):
            t = (m-i, i)
            if t[0] <= n-1 and t[1] <= n-1:
                clebsch_basis.append(t)
                sector.append(t)
        if len(sector) > 0:
            cb.append(sector)
    #print(osc_basis)
    #print(clebsch_basis)
    O = sum([qt.tensor(\
                qt.basis(len(clebsch_basis), clebsch_basis.index(osc_basis[i])),\
                qt.tensor(qt.basis(n, osc_basis[i][0]), qt.basis(n, osc_basis[i][1])).dag())\
                    for i in range(len(osc_basis))])
    O.dims = [[O.shape[0]], [O.shape[1]]]
    return O, cb
    
##############################################################################

