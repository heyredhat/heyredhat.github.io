import numpy as np
import qutip as qt
import scipy as sc

import strawberryfields as sf
from strawberryfields.ops import *

import matplotlib.pyplot as plt
import matplotlib.animation as animation

########################################################################

def make_Hh(A, B=None, h=None):
    B = np.zeros(A.shape) if type(B) == type(None) else B
    h = np.zeros(A.shape[0]) if type(h) == type(None) else h
    return np.block([[A, B],\
                     [B.conj(), A.conj()]]),\
           np.concatenate([h, h.conj()])

def rand_Hh(n):
    A = qt.rand_herm(n).full()
    B = np.random.randn(n, n) + 1j*np.random.randn(n, n)
    B = B @ B.T + B.T @ B
    h = np.random.randn(n) + 1j*np.random.randn(n)
    return make_Hh(A, B, h)

########################################################################

def omega_c(n):
    return sc.linalg.block_diag(np.eye(n), -np.eye(n))

def make_Ss(H, h, expm=False, theta=1):
    n = int(len(h)/2)
    omega = omega_c(n)
    S = sc.linalg.expm(-1j*(theta/2)*omega@H) if expm else H
    try:
        s = ((S - np.eye(2*n)) @ np.linalg.inv(H)) @ h
    except:
        s = ((S - np.eye(2*n)) @ np.linalg.pinv(H)) @ h
    return S, s

def test_c(S):
    n = int(len(S)/2)
    WC = omega_c(n)
    return np.allclose(S @ WC @ S.conj().T, WC)

########################################################################

def omega_r(n):
    return np.block([[np.zeros((n,n)), np.eye(n)],\
                     [-np.eye(n), np.zeros((n,n))]])
    
def make_Rr(S, s):
    n = int(len(s)/2)
    L = (1/np.sqrt(2))*np.block([[np.eye(n), np.eye(n)],\
                             [-1j*np.eye(n), 1j*np.eye(n)]])
    return (L @ S @ L.conj().T).real, (L @ s).real

def make_Rr2(S, s):
    n = int(len(s)/2)
    E, F = S[0:n, 0:n], S[0:n, n:]
    return np.block([[(E+F).real, -(E-F).imag],\
                     [(E+F).imag, (E-F).real]]),\
           np.sqrt(2)*np.concatenate([s[0:n].real,\
                                      s[0:n].imag])
def test_r(R):
    n = int(len(R)/2)
    WR = omega_r(n)
    return np.allclose(R @ WR @ R.T, WR)

########################################################################

def second_quantize(O, expm=True, theta=1):
    n = O.shape[0]
    Op = O.full() if type(O) == qt.Qobj else O
    H, h = make_Hh(Op, np.zeros((n,n)), np.zeros(n))
    S, s = make_Ss(H, h, expm=expm, theta=theta)
    R, r = make_Rr(S, s)
    return R, r

########################################################################

def make_XYZ():
    return {"X": second_quantize(qt.sigmax(), expm=False)[0],\
            "Y": second_quantize(qt.sigmay(), expm=False)[0],\
            "Z": second_quantize(qt.sigmaz(), expm=False)[0]}

def state_xyz(state, XYZ=None):
    if type(XYZ) == type(None):
        XYZ = make_XYZ()
    a = np.array([state.poly_quad_expectation(XYZ["X"])[0],\
                  state.poly_quad_expectation(XYZ["Y"])[0],\
                  state.poly_quad_expectation(XYZ["Z"])[0]]).real
    return a/np.linalg.norm(a) if np.linalg.norm(a) != 0 else a

########################################################################

import math

def xyz_sph(x,y,z):
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = math.atan2(y, x)
    theta = math.acos(z/r)
    return r, phi, theta

def xyz_gaussianTransforms(xyz):
    r, phi, theta = xyz_sph(*xyz)
    IR1 = GaussianTransform(second_quantize(qt.sigmax(), expm=True, theta=-theta)[0])
    IR2 = GaussianTransform(second_quantize(qt.sigmaz(), expm=True, theta=phi-np.pi/2)[0])
    return IR1, IR2

########################################################################

if __name__ == "__main__":
    n = 2 # number of modes
    N = 50 # mesh size
    fps = 24 # frames per sec
    frn = 60 # frame number
    E = qt.sigmay() # which rotation
    initial_state = None
    #initial_state = np.random.randn(3) #np.array([1,0,0])
    #initial_state = initial_state/np.linalg.norm(initial_state)

    eng = sf.Engine('gaussian')

    thetas = np.linspace(0, 2*np.pi, frn)
    Rs = [GaussianTransform(\
            second_quantize(E, expm=True, theta=theta)[0])\
                for theta in thetas] # Generate a Gaussian transformation for each step in the movie

    Q = np.linspace(-5, 5, N+1)
    P = np.linspace(-5, 5, N+1)
    Q_, P_ = np.meshgrid(Q, P)

    # These will hold the Wigner data for each step in the movie, for each oscillator
    zarray0 = np.zeros((N+1, N+1, frn))
    zarray1 = np.zeros((N+1, N+1, frn))

    # This will hold the XYZ data for the qubit, for each step in the movie
    xyzs = np.zeros((3, frn))
    XYZ = make_XYZ()

    if initial_state != None:
        IR1, IR2 = xyz_gaussianTransforms(initial_state)

    for i, theta in enumerate(thetas): # for each frame
        prog = sf.Program(2)
        with prog.context as q:
            Vac | q[0] # start off in the vacuum
            Vac | q[1]
            Sgate(1) | q[0] # squeeze the first qubit to get Z+
            if type(initial_state) != type(None):
                IR1 | (q[0], q[1]) # apply the GaussianTransforms
                IR2 | (q[0], q[1]) # to prepare the initial state
            Rs[i] | (q[0], q[1]) # perform the rotation for this frame
        state = eng.run(prog).state # get the state

        Z0 = state.wigner(0, Q, P)
        Z1 = state.wigner(1, Q, P)

        zarray0[:,:,i] = Z0 # save the wigner data
        zarray1[:,:,i] = Z1
        xyzs[:,i] = state_xyz(state, XYZ=XYZ) # save the xyz data

        eng.reset()

    def sphere(): # makes a mesh of a sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)
        return x, y, z

    def update_plot(frame_number, zarray0, zarray1, xyzs, plot):
        plot[0].remove()
        plot[0] = ax0.plot_surface(Q_, P_, zarray0[:,:,frame_number],\
                                cmap="RdYlGn", lw=0.5, rstride=1, cstride=1)
        plot[1].remove()
        plot[1] = ax1.plot_surface(Q_, P_, zarray1[:,:,frame_number],\
                                cmap="RdYlGn", lw=0.5, rstride=1, cstride=1)
        plot[2].remove()
        plot[2] = ax2.plot_surface(*sphere(), color="g", alpha=0.1)
        plot[3].remove()
        plot[3] = ax2.quiver(0,0,0,*xyzs[:,frame_number], arrow_length_ratio=0.3)

    fig = plt.figure(figsize=(6,10))
    ax0 = fig.add_subplot(311, projection="3d")
    ax0.set_zlim(0, 0.18)
    ax1 = fig.add_subplot(312, projection="3d")
    ax1.set_zlim(0, 0.18)
    ax2 = fig.add_subplot(313, projection="3d")

    plot = [ax0.plot_surface(Q_, P_, zarray0[:,:,0],\
                                cmap="RdYlGn", lw=0.5, rstride=1, cstride=1),\
            ax1.plot_surface(Q_, P_, zarray1[:,:,0],\
                                cmap="RdYlGn", lw=0.5, rstride=1, cstride=1),\
            ax2.plot_surface(*sphere(), color="g", alpha=0.1),\
            ax2.quiver(0,0,0,*xyzs[:,0], arrow_length_ratio=0.3)]

    fig.tight_layout()
    ani = animation.FuncAnimation(fig, update_plot, frn,\
                fargs=(zarray0, zarray1, xyzs, plot), interval=1000/fps)
    
    ani.save('movie.mp4',writer='ffmpeg',fps=fps)
    #ani.save('movie.gif',writer='imagemagick',fps=fps)
    #plt.show()