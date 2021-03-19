import numpy as np
import vpython as vp
import scipy as sc
import scipy.linalg

def xy_vxyz(xy):
    return vp.vector(xy[0], xy[1], 0)

def flip(b):
    return False if b else True

class FrameViz2D:
    def __init__(self, R):
        self.d, self.n = R.shape
        self.R = R
        self.D = np.linalg.pinv(R)
        self.g = R.T @ R
        self.phi = np.linalg.pinv(self.g)
        self.xy = np.zeros(2)

        ######################################################

        self.selected = False
        self.scene = vp.canvas(width=1200, height=800)
        self.vxy = vp.sphere(radius=0.1,\
                             color=vp.color.yellow,\
                             emissive=True)

        ######################################################

        self.vcartesian_axes = [vp.arrow(axis=vp.vector(*axis),\
                                         color=vp.color.blue,\
                                         shaftwidth=0.01)\
                                    for axis in [[1,0,0],
                                                 [-1,0,0],\
                                                 [0,1,0],\
                                                 [0,-1,0]]]
        self.vcartesian_projs = [vp.cylinder(radius=0.01,\
                                             color=vp.color.purple,\
                                             axis=vp.vector(0,0,0))\
                                    for i in range(self.d)]

        ######################################################

        self.vframe_axes = [vp.arrow(axis=xy_vxyz(v), 
                                      shaftwidth=0.01,\
                                        color=vp.color.red,\
                                        visible=False)\
                                    for v in R.T]
        self.vframe_axes.extend([vp.arrow(axis=-xy_vxyz(v), 
                                      shaftwidth=0.01,\
                                        color=vp.color.red,\
                                        visible=False)\
                                    for v in R.T])
        self.vframe_projs = [vp.cylinder(radius=0.01,\
                                         color=vp.color.orange,\
                                             visible=False,\
                                         axis=vp.vector(0,0,0))\
                                    for i in range(self.n)]

        ######################################################

        self.vdualframe_axes = [vp.arrow(axis=xy_vxyz(v), 
                                           shaftwidth=0.01,\
                                             color=vp.color.green,\
                                             visible=False)\
                                    for v in self.D]
        self.vdualframe_axes.extend([vp.arrow(axis=-xy_vxyz(v), 
                                      shaftwidth=0.01,\
                                        color=vp.color.green,\
                                        visible=False)\
                                    for v in self.D])
        self.vdualframe_projs = [vp.cylinder(radius=0.01,\
                                             color=vp.vector(115, 231, 126)/256,\
                                                 visible=False,\
                                             axis=vp.vector(0,0,0))\
                                    for i in range(self.n)]

        ######################################################

        if self.n == 3:
            self.vdual_axes = [vp.arrow(axis=vp.vector(*d),\
                                        color=vp.vector(156, 204, 1)/256,\
                                        shaftwidth=0.01,\
                                           visible=False) for d in self.D.T]
            self.vdual_axes.extend([vp.arrow(axis=-vp.vector(*d),\
                                        color=vp.vector(156, 204, 1)/256,\
                                        shaftwidth=0.01,\
                                           visible=False) for d in self.D.T])
            self.vdual_xyz = vp.sphere(radius=0.05,\
                                         color=vp.color.green,\
                                          visible=False)
            self.vdual_projs = [vp.cylinder(radius=0.01,\
                                            color=vp.color.cyan,\
                                                visible=False,\
                                            axis=vp.vector(0,0,0))\
                                    for i in range(self.d)]

        ######################################################

        self.vq_arrows = [vp.arrow(shaftwidth=0.02,\
                                   color=vp.color.magenta,\
                                   axis=vp.vector(0,0,0),\
                                      visible=False)\
                                        for i in range(self.n)]
        self.vq_pts = [vp.sphere(radius=0.02,\
                                 color=vp.color.magenta,\
                                    visible=False)\
                            for i in range(self.n)]

        ######################################################

        self.vp_arrows = [vp.arrow(shaftwidth=0.02,\
                                   color=vp.color.yellow,\
                                   axis=vp.vector(0,0,0),\
                                      visible=False)\
                                        for i in range(self.n)]
        self.vp_pts = [vp.sphere(radius=0.02,\
                                 color=vp.color.yellow,\
                                    visible=False)\
                            for i in range(self.n)]

        ######################################################

        self.scene.bind("mousedown", self.mousedown)
        self.scene.bind("mouseup", self.mouseup)
        self.scene.bind("mousemove", self.mousemove)
        self.scene.bind("keydown", self.keydown)

    def mousedown(self):
        if self.scene.mouse.pick == self.vxy:
            self.selected = True
        else:
            self.selected = False

    def mouseup(self):
        self.selected = False

    def mousemove(self):
        if self.selected:
            self.vxy.pos = self.scene.mouse.pos
            self.vxy.pos.z = 0
            self.xy = np.array([self.vxy.pos.x, self.vxy.pos.y])

            self.vcartesian_projs[0].pos = vp.vector(self.xy[0], 0, 0)
            self.vcartesian_projs[1].pos = vp.vector(0, self.xy[1], 0)
            self.vcartesian_projs[0].axis = self.vxy.pos - self.vcartesian_projs[0].pos
            self.vcartesian_projs[1].axis = self.vxy.pos - self.vcartesian_projs[1].pos

            xyz = self.R.T @ self.xy
            for i in range(self.n):
                proj = xyz[i]*self.R.T[i]/np.linalg.norm(self.R.T[i])**2
                self.vframe_projs[i].pos = vp.vector(proj[0], proj[1], 0)
                self.vframe_projs[i].axis = self.vxy.pos - self.vframe_projs[i].pos

            dxyz = self.D @ self.xy
            for i in range(self.n):
                proj = dxyz[i]*self.D[i]/np.linalg.norm(self.D[i])**2
                self.vdualframe_projs[i].pos = vp.vector(proj[0], proj[1], 0)
                self.vdualframe_projs[i].axis = self.vxy.pos - self.vdualframe_projs[i].pos

            if self.n == 3:
                self.vdual_xyz.pos = vp.vector(*xyz)
                for i in range(self.d):
                    proj = self.xy[i]*self.D.T[i]/np.linalg.norm(self.D.T[i])**2
                    self.vdual_projs[i].pos = vp.vector(*proj)
                    self.vdual_projs[i].axis = self.vdual_xyz.pos - self.vdual_projs[i].pos 

            qxyz = self.phi @ xyz
            qpts = [qxyz[i]*self.R.T[i] for i in range(self.n)]
            vqpts = [xy_vxyz(qpt) for qpt in qpts]
            running = vp.vector(0,0,0)
            for i in range(self.n):
                self.vq_pts[i].pos = vqpts[i]
                self.vq_arrows[i].pos = running
                self.vq_arrows[i].axis = vqpts[i]
                running += vqpts[i]

            ppts = [xyz[i]*self.D[i] for i in range(self.n)]
            vppts = [xy_vxyz(ppt) for ppt in ppts]
            running = vp.vector(0,0,0)
            for i in range(self.n):
                self.vp_pts[i].pos = vppts[i]
                self.vp_arrows[i].pos = running
                self.vp_arrows[i].axis = vppts[i]
                running += vppts[i]

    def keydown(self, e):
        k = e.key
        if k == "c":
            for v in self.vcartesian_axes:
                v.visible = flip(v.visible)
            for v in self.vcartesian_projs:
                v.visible = flip(v.visible)
        elif k == "f":
            for v in self.vframe_axes:
                v.visible = flip(v.visible)
            for v in self.vframe_projs:
                v.visible = flip(v.visible)
        elif k == "d":
            for v in self.vdualframe_axes:
                v.visible = flip(v.visible)
            for v in self.vdualframe_projs:
                v.visible = flip(v.visible)
        elif k == "3" and self.n == 3:
            for v in self.vdual_axes:
                v.visible = flip(v.visible)
            self.vdual_xyz.visible = flip(self.vdual_xyz.visible)
            for v in self.vdual_projs:
                v.visible = flip(v.visible)
        elif k == "q":
            for v in self.vq_arrows:
                v.visible = flip(v.visible)
            for v in self.vq_pts:
                v.visible = flip(v.visible)    
        elif k == "p":
            for v in self.vp_arrows:
                v.visible = flip(v.visible)
            for v in self.vp_pts:
                v.visible = flip(v.visible)    

def equiangular_frame2D():
    roots = [np.exp(2j*np.pi*i/3) for i in range(3)]
    return np.array([[r.real, r.imag] for r in roots]).T

def random_unbiased_tight_frame(d, n, rtol=1e-15, atol=1e-15):
    n = int(d*(d+1)/2) if n == None else n
    R = np.random.randn(d, n)
    done = False
    while not (np.allclose(R @ R.T, (n/d)*np.eye(d), rtol=rtol, atol=atol) and\
               np.allclose(np.linalg.norm(R, axis=0), np.ones(n), rtol=rtol, atol=atol)):
        R = sc.linalg.polar(R)[0]
        R = np.array([state/np.linalg.norm(state) for state in R.T]).T
    return R

def random_biased_tight_frame(d, n=None):
    n = int(d*(d+1)/2) if n == None else n
    R = np.random.randn(d, n)
    return np.sqrt(n/d)*sc.linalg.polar(R)[0]

def random_frame(d, n=None):
    n = int(d*(d+1)/2) if n == None else n
    R = np.random.randn(d, n)
    return R

vframe = FrameViz2D(random_frame(2, 3))

print("c: cartesian")
print("f: frame")
print("d: dual frame")
print("3: 3D dual frame")
print("q: frame reconstruction")
print("p: dual reconstruction")


