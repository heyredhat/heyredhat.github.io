import numpy as np
import qutip as qt
import vpython as vp
from magic import *
scene = vp.canvas(background=vp.color.white)

def coherent_states(j, N=25):
    theta = np.linspace(0, math.pi, N)
    phi = np.linspace(0, 2*math.pi, N)
    THETA, PHI = np.meshgrid(theta, phi)
    return ([[qt.spin_coherent(j, THETA[i][k], PHI[i][k])\
                    for k in range(N)] for i in range(N)], THETA, PHI)

def husimi(state, CS):
    cs, THETA, PHI = CS
    N = len(THETA)
    Q = np.zeros_like(THETA, dtype=complex)
    for i in range(N):
        for j in range(N):
            amplitude = cs[i][j].overlap(state)
            Q[i][j] = amplitude#*np.conjugate(amplitude)
    pts = []
    for i, j, k in zip(Q, THETA, PHI):
        for q, t, p in zip(i, j, k):
            pts.append([q, sph_xyz([1, t, p])])
    return pts

def tangent_rotations(CS):
    cs, THETA, PHI = CS
    rots = []
    for j, k in zip(THETA, PHI):
        for t, p in zip(j, k):
            normal = sph_xyz([1, t, p])
            tan = sph_xyz([1, t+np.pi/2, p])
            vv = np.cross(tan, normal)
            vv = vv/np.linalg.norm(vv)
            trans = np.array([tan, vv, normal])
            itrans = np.linalg.inv(trans)
            rots.append(itrans)
    return rots

dt = 0.01
j = 3/2
n = int(2*j+1)

CS = coherent_states(j, N=20)
rots = tangent_rotations(CS)

state = qt.rand_ket(n)
H = qt.rand_herm(n)
U = (-1j*H*dt).expm()

vsphere = vp.sphere(color=vp.color.blue,\
                    opacity=0.4)
vstars = [vp.sphere(radius=0.2, emissive=True,\
                    pos=vp.vector(*xyz)) for xyz in spin_XYZ(state)]

pts = husimi(state, CS)
#vpts = [vp.sphere(pos=vp.vector(*pt[1]), radius=0.5*pt[0]) for pt in pts]
vpts = []
for i, pt in enumerate(pts):
    amp, normal = pt
    amp_vec = np.array([amp.real, amp.imag, 0])
    amp_vec = np.dot(rots[i], amp_vec)
    vpts.append(vp.arrow(pos=vp.vector(*normal), axis=0.5*vp.vector(*amp_vec)))

while True:
    state = U*state
    for i, xyz in enumerate(spin_XYZ(state)):
        vstars[i].pos = vp.vector(*xyz)
    pts = husimi(state, CS)
    #for i, pt in enumerate(pts):
    #    vpts[i].radius = 0.5*pt[0]
    for i, pt in enumerate(pts):
        amp, normal = pt
        amp_vec = np.array([amp.real, amp.imag, 0])
        amp_vec = np.dot(rots[i], amp_vec)
        vpts[i].axis = 0.5*vp.vector(*amp_vec)
    vp.rate(2000)