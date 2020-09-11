from sympy import *
#from symengine import *
import qutip as qt
import numpy as np
import scipy as sc
from functools import reduce
from magic import *

#######################################################################################

class State:
    def __init__(self, symbols, expr=0, qtype='ket'):
        self.vars = symbols
        self.expr = expr
        self.qtype = qtype
        self.symbolic = False

    def __call__(self, *args):
        return self.expr.copy().subs(list(zip(self.vars, args)))

    def __lshift__(self, expr):
        self.expr = expr

    def __add__(self, other):
        if type(other) == State:
            return State(self.vars, \
                         expr=self.expr.copy()+other.expr.copy(), \
                         qtype=self.qtype)
        else:
            return State(self.vars, \
                         expr=self.expr.copy()+other, \
                         qtype=self.qtype)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if type(other) == State:
            return State(self.vars, \
                         expr=self.expr.copy()-other.expr.copy(), \
                         qtype=self.qtype)
        else:
            return State(self.vars, \
                         expr=self.expr.copy()-other, \
                         qtype=self.qtype)


    def __rsub__(self, other):
        if type(other) == State:
            return State(self.vars, \
                         expr=other.expr.copy()-self.expr.copy(), \
                         qtype=self.qtype)
        else:
            return State(self.vars, \
                         expr=other-self.expr.copy(),\
                         qtype=self.qtype)

    def __mul__(self, other):
        if type(other) == State and self.qtype == 'bra' and other.qtype =='ket':
            if self.symbolic:
                limits = [(v, -oo, oo) for v in self.vars]
                current = conjugate(self.expr.copy())*other.expr.copy()
                for limit in limits:
                    current = integrate(current, limit)
                return current
            else:
                expr = conjugate(self.expr.copy())*other.expr.copy()
                lamb = lambdify(self.vars, expr)
                return sc.integrate.nquad(lambda *args: sc.real(lamb(*args)),\
                                          [[-sc.inf, sc.inf] for v in self.vars],\
                                          opts={"epsabs": 1.49 * 10**(-4),\
                                                "epsrel": 1.49 * 10**(-4),\
                                                "maxp1": 4})[0]\
                        + 1j*sc.integrate.nquad(lambda *args: sc.imag(lamb(*args)),\
                                          [[-sc.inf, sc.inf] for v in self.vars],\
                                          opts={"epsabs": 1.49 * 10**(-4),\
                                                "epsrel": 1.49 * 10**(-4),\
                                                "maxp1": 4})[0]
        elif type(other) == Operator and self.qtype == 'bra':
                return (other*self.dag()).dag()
        else:
            return State(self.vars,\
                         expr=other*self.expr.copy(),\
                         qtype=self.qtype)

    def __rmul__(self, other):
        return State(self.vars,\
                     expr=other*self.expr.copy(),\
                     qtype=self.qtype)

    def dag(self):
        return State(self.vars, \
                     expr=self.expr.copy(), \
                     qtype=('bra' if self.qtype == 'ket' else 'ket'))

    def copy(self):
        return State(self.vars, \
                     expr=self.expr.copy(),\
                     qtype=self.qtype)

    def __repr__(self):
        return "%s: %s" % (self.qtype, self.expr if\
                                       self.qtype == 'ket' else \
                                       conjugate(self.expr))

#######################################################################################

class Operator:
    def __init__(self, func=lambda state: 1):
        self.func = func

    def __call__(self, state):
        return self.func(state)

    def __add__(self, other):
        if type(other) == Operator:
            return Operator(func=lambda state: self.func(state) + other.func(state))
        else:
            return Operator(func=lambda state: self.func(state) + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if type(other) == Operator:
            return Operator(func=lambda state: self.func(state) - other.func(state))
        else:
            return Operator(func=lambda state: self.func(state) - other)

    def __rsub__(self, other):
        if type(other) == Operator:
            return Operator(func=lambda state: other.func(state) - self.func(state))
        else:
            return Operator(func=lambda state: other - self.func(state))

    def __mul__(self, other):
        if type(other) == State and other.qtype == 'ket':
            return self.func(other.copy())
        elif type(other) == Operator:
            return Operator(func=lambda state: self.func(other.func(state)))
        else:            
            return Operator(func=lambda state: other*self.func(state))

    def __rmul__(self, other):
        return Operator(func=lambda state: other*self.func(state))    

    def copy(self):
        return Operator(func=self.func.copy())    

#######################################################################################

class Field:
    def __init__(self, n):
        self.n = n
        self.vars = [symbols("[%d]" % i) for i in range(n)]
        self.vac = State(self.vars,\
                          expr=prod([pi**(-1/4)*exp(-(self.vars[i]**2)/2) for i in range(n)]))
        self._ = self.vac.copy()

        self.Q = [Operator(func=\
                            lambda state, i=i: \
                                self.vars[i]*state) \
                  for i in range(self.n)]
        self.P = [Operator(func=\
                            lambda state, i=i: \
                                State(self.vars, \
                                      expr=\
                                      -I*diff(state.expr.copy(), self.vars[i])))\
                        for i in range(self.n)]  
        self.A = [Operator(func=\
                            lambda state, i=i: \
                                   (1/sqrt(2)*(self.Q[i] + I*self.P[i])*state))\
                           for i in range(self.n)]
        self.A_ = [Operator(func=\
                            lambda state, i=i: \
                                   (1/sqrt(2)*(self.Q[i] - I*self.P[i])*state))\
                           for i in range(self.n)]
        self.N = [Operator(func=\
                            lambda state, i=i: \
                                   self.A_[i]*self.A[i]*state)\
                           for i in range(self.n)]

    def apply(self, O):
        self._ = O*self._

    def expect(self, O):
        return self._.dag()*O*self._

    def create(self, n, osc=0):
        if n == 0:
            return Operator(func=lambda state: state)
        return (1/sqrt(factorial(n)))*\
                reduce(lambda x, y: x*y, [self.A_[osc] for i in range(n)])

    def nth(self, n, osc=0, sticky=False):
        result = self.create(n, osc)*self.vac
        self._ = result if sticky else self._
        return result

    def quantize_operator(self, O, A=None, A_=None):
        A = A if type(A) != type(None) else self.A
        A_ = A_ if type(A_) != type(None) else self.A_
        o = O.full()
        terms = []
        for i in range(len(A)):
            for j in range(len(A)):
                terms.append(A_[i]*complex(o[i][j])*A[j])
        return sum(terms)

    def quantize_state(self, state, A=None, A_=None, lower=False):
        A = A if type(A) != type(None) else self.A
        A_ = A_ if type(A_) != type(None) else self.A_
        v = state.full().T[0]
        return sum([complex(v[i])*A_[i] for i in range(len(A_))])*sqrt(2/len(A_)) if not lower else \
                sum([complex(v[i])*A[i] for i in range(len(A))])*sqrt(2/len(A_)) 

    def __repr__(self):
        s = "%d oscillator(s):\n  %s\n" % (self.n, self._)
        for i in range(self.n):
            s += "   <N%d>: %s\n" % (i, str(self._.dag()*self.N[i]*self._))
        return s[:-1]

#######################################################################################

def get_phase(q):
    c = sum(q.full().T[0][::-1])
    return np.exp(1j*np.angle(c))

def mat_coeffs(M, basis):
    return np.array([(o.dag()*M).tr() for i, o in enumerate(basis)])

def coeffs_mat(C, basis):
    return sum([C[i]*o for i, o in enumerate(basis)])

#######################################################################################

def su(n):
    diagonals = [np.zeros((n,n), dtype=complex) for i in range(n)]
    for i in range(n):
        diagonals[i][i,i] = 1
    xlike = [np.zeros((n, n), dtype=complex) for i in range(int(n*(n-1)/2))]
    r = 0
    for i in range(n):
        for j in range(n):
            if i < j:
                xlike[r][i,j] = 1/np.sqrt(2)
                xlike[r][j,i] = 1/np.sqrt(2)
                r +=1 
    ylike = [np.zeros((n, n), dtype=complex) for i in range(int(n*(n-1)/2))]
    r = 0
    for i in range(n):
        for j in range(n):
            if i < j:
                ylike[r][i,j] = 1j/np.sqrt(2)
                ylike[r][j,i] = -1j/np.sqrt(2)
                r +=1 
    return [qt.Qobj(o) for o in diagonals + xlike + ylike]

def su2(n, half=True):
    XYZ = {"X": qt.sigmax(), "Y": qt.sigmay(), "Z": qt.sigmaz()}
    S = [dict([(o, (0.5 if half else 1)*qt.Qobj(\
          sc.linalg.block_diag(*\
            [np.zeros((2,2)) if i !=j else XYZ[o].full() \
                for j in range(n)]))) \
                    for o in XYZ.keys()])
                        for i in range(n)]
    E = [qt.tensor(o, qt.identity(2)) for o in su(n)]
    for e in E:
        e.dims = [[e.shape[0]], [e.shape[0]]]
    return S, E

def rand_su2n_state(n):
    return su2n_state([qt.rand_ket(2) for i in range(n)])

def su2n_state(spinors):
    return qt.Qobj(np.concatenate([q.full().T[0] for q in spinors]))

def split_su2n_state(state):
    v = state.full().T[0]
    return [qt.Qobj(np.array([v[i], v[i+1]])) for i in range(0, len(v), 2)]

def su2n_phases(state):
    return [get_phase(spinor) for spinor in split_su2n_state(state)]

##################################################################

f = Field(2)

a = qt.rand_ket(2)
XYZ = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
xyz = [qt.expect(o, a) for o in XYZ]

XYZf = [f.quantize_operator(o) for o in XYZ]
f.apply(f.quantize_state(a))
#xyzf = [f.expect(o) for o in XYZf]

##################################################################

#n = 2
#cS, cE = su2(n)
#cstate = rand_su2n_state(n)
#cdms = [spinor*spinor.dag() for spinor in split_su2n_state(cstate)]
#cO = cE[2]

#f = Field(2*n)
#qS = [dict([(o, f.quantize_operator(O))\
#              for o, O in xyz.items()])\
#                for xyz in cS]
#qE = [f.quantize_operator(o) for o in cE]
#qcreate = f.quantize_state(cstate)
#f.apply(qcreate)
#qO = qE[2]
#transformed = qO*f._

#cstate2 = cO*cstate
#cdsm2 = [spinor*spinor.dag() for spinor in split_su2n_state(cstate2)]

##################################################################

class Spinors:
    def __init__(self, n, spinors=[]):
        self.n = n
        self.S, self.E = su2(n)
        self._ = su2n_state(spinors)\
                      if len(spinors) > 0 else rand_su2n_state(n)

    def apply(self, O):
        self._ = O*self._

    def dms(self):
        return [spinor*spinor.dag() for spinor in split_su2n_state(self._)]

    def xyz(self, i=None):
        if type(i) != type(None):
            return np.array([qt.expect(self.S[i][o], self._)\
                                for o in ["X", "Y", "Z"]])
        return np.array([[qt.expect(self.S[i][o], self._)\
                                for o in ["X", "Y", "Z"]]\
                                    for i in range(self.n)])

    def phases(self):
        return su2n_phases(self._)

    def __repr__(self):
        s = "%d spinors:\n" % self.n
        s += str(self.xyz())
        return s

##################################################################

class Spins:
    def __init__(self, n, spinors=[]):
        self.n = n
        self.c = Spinors(n, spinors=spinors)

        self.f = Field(2*n)
        self.S = [dict([(o, 2*self.f.quantize_operator(O))\
                      for o, O in xyz.items()])\
                        for xyz in self.c.S]
        self.E = [self.f.quantize_operator(o) for o in self.c.E]

        self.init_lift = self.f.quantize_state(self.c._)
        self.f.apply(self.init_lift)
        self.last = self.f._.copy()
        self.phase = 1

    def apply(self, O, parallel=True, update_phase=False):
        if parallel:
            self.c.apply(O)
        if update_phase:
            self.last = self.f._.copy()
        P = self.f.quantize_operator(O) if type(O) == qt.Qobj else O
        self.f.apply(P)
        if update_phase:
            self.phase = self.last.dag()*self.f._

    def xyz(self, i=None):
        if type(i) != type(None):
            return np.array([self.f.expect(self.S[i][o])\
                                for o in ["X", "Y", "Z"]])
        return np.array([[self.f.expect(self.S[i][o]) \
                                for o in ["X", "Y", "Z"]]\
                                    for i in range(self.n)])

    def __repr__(self):
        s = "%d spins:\n" % (self.n)
        s += "(classical) %s\n" % str(self.c)
        s += "(quantum):\n%s" % self.xyz()
        return s

##################################################################

def make_views(pts):
    views = []
    for i in range(n):
        view = []
        for j in range(n):
            if i != j:
                to = pts[j]-pts[i]
                view.append(to/np.linalg.norm(to))
        views.append(view)
    return views

def views_constellations(views):
    return [XYZ_spin(view) for view in views]

def constellations_matrix(spins, orth=True):
    M = qt.Qobj(np.array([spin.full().T[0] for spin in spins]).T)
    if not orth:
        return M
    U, H = sc.linalg.polar(M)
    Q = qt.tensor(qt.Qobj(U), qt.identity(2))
    Q.dims = [[Q.shape[0]], [Q.shape[0]]]
    return Q

def constellations_matrix_(spins, orth=True):
    M = qt.Qobj(np.array([spin.full().T[0] for spin in spins]).T)
    if not orth:
        return M
    U, H = sc.linalg.polar(M)
    Q = qt.tensor(qt.Qobj(U))
    return Q

##################################################################

n = 2
s = Spins(n, spinors=[qt.basis(2,0), qt.basis(2,1)])

#pts = [np.array([0, 0, i-1/2]) for i in range(n)]
pts = [np.random.randn(3) for i in range(n)]
views = make_views(pts)
SUN = constellations_matrix(views_constellations(views))

s.apply(SUN)

print(s)
print(s.f._.dag()*s.f._)
# s.f._.dag()*s.f._ == 1
