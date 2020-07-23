from __future__ import division, print_function
import numpy as np
from scipy.optimize import root as __root
import warnings

"""
To keep compatibility with python2 while retaining functionality, rename xrange
to range if xrange is defined (i.e., if python2 is used).
"""
try:
    range = xrange
except NameError:
    pass
"""
If python2 is being used, import map, filter and zip from future_builtins so
that they behave like in python3.
"""
try:
    from future_builtins import map, filter, zip
except ImportError:
    pass


class ReconstructError(Exception):
    """
    Python-exception-derived object raised by the reconstruct() function. The
    exception is raised when the algorithm to reconstruct the polyhedron fails.
    """
    pass


def __removeDuplicates(unormals,areas,rtol,atol):
    """
    Unit normals should all be distinct. This function "merges" duplicate
    vectors into one by summing the associated areas.
    """
    n = len(unormals)
    I = list(range(n))
    eq_classes = [] # Equivalence classes of distinct vectors

    while I:
        spam = [I[0]]
        for i in reversed(range(1,len(I))):
            if np.isclose(unormals[I[0]],unormals[I[i]],rtol,atol).all():
                spam.append(I[i])
                del I[i]
        del I[0]
        eq_classes.append(spam)

    U = [unormals[e[0]] for e in eq_classes]
    A = [sum(areas[i] for i in e) for e in eq_classes]

    if len(eq_classes) < n:
        warnings.warn('\
            There are duplicate unit normals. They have been merged into one, \
            and the associated areas have been summed.')

    return U,A


class Polyhedron():
    def __init__(self,vertices,vertex_adjacency_matrix,faces,
                 face_adjacency_matrix,unormals,areas,H):
        self.vertices = tuple(vertices)
        self.v_adjacency_matrix = vertex_adjacency_matrix
        self.f_adjacency_matrix = face_adjacency_matrix
        self.faces = tuple(Face(verticesIDs,unormal,area) for
                      verticesIDs,unormal,area in zip(faces,unormals,areas))
        self.inequalities = tuple(np.array([h] + list(-u))
                                  for h,u in zip(H,unormals))
        self.volume = sum(a*h for a,h in zip(areas,H))/3
    
    def print_inequality(self,index):
        ineq = self.inequalities[index]
        spam = ' + '.join([str(a) + '*' + x
                           for a,x in zip(-ineq[1:],['x','y','z'])])
        spam += ' <= ' + str(ineq[0])
        print(spam)
        

class Face():
    def __init__(self,verticesIDs,unormal,area):
        self.vertices = tuple(verticesIDs)
        self.unormal = unormal.copy()
        self.area = area


def reconstruct(unormals,areas,D=None,options={}):
    """
    Reconstruct (up to translation) a polyhedron from its outward unit normals
    and face areas.

    The unit normals must span 3D space, the areas must be positive and the
    closure constraint ``sum(areas[i]*unormals[i] for i in range(n)) == 0``
    must be satisfied, otherwise a ValueError exception is raised.
    
    Inputs
    ------
    unormals : list
        The list of unit normals (each normals should be a numpy array or a
        list/tuple).
    areas : list
        The list of face areas.
    D : float, optional
        Parameter affecting the initial guess for the root finding algorithm:
        each face is at distance D from a point inside the polyhedron. If D is
        not specified the function computes the appropriate distance; an
        explicit value should only be used if the algorithm with the built-in
        distance fails.
        The value entered for D has to be positive.
    options : dict, optional
        A dictionary of additional options. The default is:
            {'rtol': 1e-05,'atol': 1e-08,'ftol': 1.5e-08,'xtol': 1.5e-08}
        'rtol' : float
            Relative tolerance parameter used when comparing numbers with
            numpy.isclose()
        'atol' : float
            Absolute tolerance parameter used when comparing numbers with
            numpy.isclose(). Used when comparing something with 0.
        'ftol': 1.5e-08
            Relative error desired when comparing the function to 0 in the
            root finding algorithm.
        'ftol': 1.5e-08
            Relative error desired in theapproximate solution in the root
            finding algorithm.
    
    Returns
    -------
    P : Polyhedron
        Reconstructed polyhedron as an object of the Polyhedron class.

    Raises
    ------
    ReconstructError
        If the root algorithm is unable to reconstruct the polyhedron.
        In this case, specifying an explicit value for D may solve the problem. 
    
    Notes
    -----
    The unit normals should be distinct; if they are not, duplicate vectors
    are going to be merged into one by summing the associated areas. As a
    consequence, the returned polyhedron may have less faces than expected.

    Examples
    --------
    >>> import polyhedrec as pr
    
    >>> unormals = [numpy.array([-0.17447189, -0.43229383, -0.88469294]),
                    numpy.array([ 0.60815785,  0.73855608,  0.29099648]),
                    numpy.array([-0.53342276, -0.44253185, -0.72085069]),
                    numpy.array([-0.29876108,  0.45137263,  0.84083563]),
                    numpy.array([-0.41552563, -0.69427644,  0.58763822])]
    
    >>> areas = [7.049773119515445,
                 8.521720027005253,
                 1.9790800361257508,
                 2.5983018666756377,
                 5.1034298342172928]
    
    >>> P = pr.reconstruct(unormals,areas)
    
    >>> P.vertices
    (array([-0.,  0., -0.]),
     array([ 6.02548243, -5.56480979,  1.53087654]),
     array([ 0.60717762, -2.63092577,  1.16582546]),
     array([-0.44385591,  0.31140014,  0.13727997]),
     array([ 1.70842807, -2.31842303,  2.31374446]),
     array([-0.2875289 , -1.90977608,  1.3851845 ]))
    
    >>>P.f_adjacency_matrix 
    array([[0, 1, 1, 0, 1],
           [1, 0, 1, 1, 1],
           [1, 1, 0, 1, 1],
           [0, 1, 1, 0, 1],
           [1, 1, 1, 1, 0]])
    
    >>> P.v_adjacency_matrix
    array([[0, 1, 1, 1, 0, 0],
           [1, 0, 1, 0, 1, 0],
           [1, 1, 0, 0, 0, 1],
           [1, 0, 0, 0, 1, 1],
           [0, 1, 0, 1, 0, 1],
           [0, 0, 1, 1, 1, 0]])
    
    >>> P.inequalities
    (array([ 0.        ,  0.17447189,  0.43229383,  0.88469294]),
     array([ 0.        , -0.60815785, -0.73855608, -0.29099648]),
     array([ 0.        ,  0.53342276,  0.44253185,  0.72085069]),
     array([ 0.38859427,  0.29876108, -0.45137263, -0.84083563]),
     array([ 2.25937552,  0.41552563,  0.69427644, -0.58763822]))
    
    >>> P.print_inequality(0)
    -0.174471886106*x + -0.432293832351*y + -0.884692943043*z <= 0.0
    
    >>> P.faces[0].vertices
    (0, 1, 2)
    
    >>> P.faces[0].area
    (7.049773119515445)
    
    >>> P.faces[0].unormal
    array([-0.17447189, -0.43229383, -0.88469294])
    """
    
    options.setdefault('rtol', 1e-05)
    options.setdefault('atol', 1e-08)
    options.setdefault('ftol', 1.5e-08)
    options.setdefault('xtol', 1.5e-08)
    
    rtol = options['rtol']
    atol = options['atol']
    ftol = options['rtol']
    xtol = options['atol']
    
    unormals = [np.array(u) for u in unormals]

    # Check if the input satisfies the requirements. 
    if D is not None and D < 0:
        raise ValueError('D should be positive.')

    if (np.array(areas) <= 0 ).any():
        raise ValueError('The areas should be positive.')
    
    if not np.isclose(
            sum(a*u for a,u in zip(areas,unormals)),0,rtol,atol).all():
        raise ValueError('The normals do not sum to zero.')

    if not np.isclose([np.dot(u,u) for u in unormals],1).all():
        raise ValueError('The normals are not unit vectors.')
    
    # Check if there are any repeated unormals and merge them if necessary  
    unormals, areas = __removeDuplicates(unormals,areas,rtol,atol)
    
    n = len(unormals)
    
    dots = np.array(
        [[0]*i + [0.5] + [np.dot(unormals[i],unormals[j])
         for j in range(i+1,n)] for i in range(n)])
    dots = (dots + dots.T)
    
    a = []
    for k in range(n):
        spam = np.array(
            [[0]*(i+1) + [np.dot(unormals[k],np.cross(unormals[i],unormals[j]))
             if j!=k else 0 for j in range(i+1,n)] if i!=k
            else [0]*n for i in range(n)])
        spam = (spam - spam.T)
        a.append(spam)
    del spam
    
    # Find three independent normals or raise an Exception if unable to.
    gen = [0]
    for i in range(1,n):
        if not np.isclose(abs(dots[0,i]),1,rtol,atol):
            gen.append(i)
            break
    for k in filter(lambda x: len(gen)==2 and x not in gen, range(n)):
        if not np.isclose(a[k][gen[0],gen[1]],0,rtol,atol):
            gen.append(k)
            break
    if len(gen) < 3:
        raise ValueError('The normals do not span 3D space.')
    
    norms = np.array([[0]*(i+1) + [1 - dots[i,j]**2 for j in range(i+1,n)]
                      for i in range(n)])
    norms = (norms + norms.T)
    N = [np.array([[0 if j==i
                    else -0.5*dots[i,k] if np.isclose(norms[i,j],0,rtol,atol)
                    else (dots[i,j]*dots[j,k] - dots[i,k])/norms[i,j] if j!=i
                    else 0 for j in range(n)] if i!=k
                   else [-1]*n for i in range(n)]) for k in range(n)]
    
    global L,r,Lmax,Lmin
    
    Lmax = np.empty([n,n])
    Lmin = np.empty([n,n])
    
    def area(h,jac=True):
        global L,r,Lmax,Lmin
        
        H = np.insert(h,[gen[0],gen[1]-1,gen[2]-2],0)
        r = np.array([[H[j] - dots[i,j]*H[i] if j!=i else 0 for j in range(n)]
                      for i in range(n)])
        b = []
        for k in range(n):
            spam = np.array([[0]*(i+1) + [r[i,k] + r[i,j]*N[k][j,i] if j!=k
                             else 0 for j in range(i+1,n)] if i!=k else [0]*n
                             for i in range(n)])
            spam = (spam + spam.T)
            b.append(spam)
        del spam
        
        L = np.zeros([n,n])
        if jac:
            c = np.zeros([n,n])
            y = [np.zeros([n,n]) for k in range(n)]
        for i in range(n):
            for j in range(i+1,n):
                if np.isclose(norms[i,j],0,rtol,atol):
                    L[i,j] = 0
                    L[j,i] = 0
                else:
                    Lmin[i,j],Lmax[i,j] = -np.inf, np.inf
                    kmax,kmin = None, None
                    for k in filter(lambda x: x!=i and x!=j, range(n)):
                        if np.isclose(a[k][i,j],0,rtol,atol):
                            if b[k][i,j] < 0:
                                Lmax[i,j] = 0
                                Lmin[i,j] = 0
                                break
                        elif a[k][i,j] > 0:
                            spam = b[k][i,j]/a[k][i,j]
                            if spam < Lmax[i,j]:
                                Lmax[i,j] = spam
                                kmax = k
                        else:
                            spam = b[k][i,j]/a[k][i,j]
                            if spam > Lmin[i,j]:
                                Lmin[i,j] = spam
                                kmin = k
                    L[i,j] = max(0,Lmax[i,j]-Lmin[i,j])
                    L[j,i] = L[i,j]
                
                if jac and L[i,j] != 0:
                    if kmax is not None:
                        c[i,j] += N[kmax][i,j]/a[kmax][i,j]
                        c[j,i] += N[kmax][j,i]/a[kmax][i,j]
                        spam = 1/a[kmax][i,j]
                        y[kmax][i,j] += spam
                        y[kmax][j,i] += spam
                    if kmin is not None:
                        c[i,j] -= N[kmin][i,j]/a[kmin][i,j]
                        c[j,i] -= N[kmin][j,i]/a[kmin][i,j]
                        spam = 1/a[kmin][i,j]
                        y[kmin][i,j] -= spam
                        y[kmin][j,i] -= spam
                     
        #compute area
        A = np.array([0.5*sum(L[i,j]*r[i,j] if j!= i else 0 for j in range(n))
                      - areas[i] for i in range(n)])

        if jac:
            #compute Jacobian
            J = []
            for k in filter(lambda x: x not in gen, range(n)):
                J.append([])
                for i in range(n):
                    if i == k:
                        spam = sum(c[i,j]*r[i,j] - L[i,j]*dots[i,j] if j!=i
                                   else 0 for j in range(n))
                        J[-1].append(0.5 * spam)
                    else:
                        spam = sum(y[k][i,j]*r[i,j] if j!=i
                                   and j!=k else 0 for j in range(n))
                        if L[i,k] != 0:
                            spam += c[k,i]*r[i,k] + L[i,k]
                        J[-1].append(0.5 * spam)
            J = np.matrix(J)
            return A,J
        else:
            return A
            
    
    coeff = np.linalg.solve([[dots[i,j] for j in gen] for i in gen],[-1]*3)
    c = sum(coeff[i] * unormals[gen[i]] for i in range(3))
    h0 = np.array([1 + np.dot(unormals[i],c)
                   for i in filter(lambda x: x not in gen, range(n))])
    if D is not None:
        d = D
    else:
        d = np.sqrt(np.average(areas/(area(h0,False) + areas)))
    h0 = d*h0  
    
    sol = __root(area, h0, method='lm', jac=True, options={'col_deriv': 1,
                                                         'ftol': ftol,
                                                         'xtol': xtol})
    
    if not np.isclose(sol.fun,0,atol,rtol).all():
        if D is None:
            raise ReconstructError('\
                The algorithm was not able to reconstruct the polyhedron; try \
                to pass a custom value for D as input. For reference, the \
                value computed by the algorithm was {0}.'.format(d))
        else:
            raise ReconstructError('\
                The algorithm was not able to reconstruct the polyhedron; try \
                to pass a different value for D as input.')

    H = np.insert(sol.x,[gen[0],gen[1]-1,gen[2]-2],0)
    
    face_ad_matrix = (L>0).astype(int)

    num_edges = int(sum(sum(face_ad_matrix))/2)
    num_vertices = num_edges - n + 2
    vert_ad_matrix = np.zeros([num_vertices,num_vertices],dtype=int)

    faces = [[] for i in range(n)]
    
    vertices = []

    endpoints = np.full([n,n],None)
    
    for i in range(n):
        intersections = [j for j in range(n) if L[i,j]>0]
        '''
        Sort intersections in counterclockwise order from first one. The order
        is chosen by computing the angles between the components of the
        unormals[j] orthogonal to unormals[i].
        '''
        first = intersections[0]
        intersections = [intersections[k] for k in np.argsort(
            [np.arctan2(a[j][i,first],dots[first,j] - dots[i,j]*dots[first,i])
             for j in intersections])]
        for k in range(len(intersections)):
            j = intersections[k]
            nextj = intersections[(k+1) % len(intersections)]
            '''
            For each edge e_ij we add the end vertex, obtained by traversing
            the edge in the direction of unormals[i] x unormals[j].
            If the edge e_ij has been considered already (i.e. j > i) or the
            associated vertex has already been added, so we do nothing.
            Likewise, if the edge e_ij has not been considered but the next one
            in the list has, the end vertex has already been added from a
            different face, so we do nothing.
            ''' 
            if j > i:
                if nextj > i:
                    o_ij = (r[j,i]*unormals[i] + r[i,j]*unormals[j])/norms[i,j]
                    vertex = o_ij + Lmax[i,j]*np.cross(unormals[i],unormals[j])
                    vertices.append(vertex)
                    idx = len(vertices) - 1                   
                else:
                    idx = endpoints[nextj,i]
            else:
                idx = endpoints[i,j]
            endpoints[i,j] = idx
            endpoints[nextj,i] = idx
            faces[i].append(idx)
        # Update vertex adjacency matrix with information from face i
        for j in intersections:
            vert_ad_matrix[endpoints[i,j],endpoints[j,i]] = 1
    
    del L,Lmin,Lmax,r # Clean up global variables.
    return Polyhedron(vertices,vert_ad_matrix,faces,
                      face_ad_matrix,unormals,areas,H)
