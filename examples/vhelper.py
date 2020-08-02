import numpy as np
import qutip as qt
import itertools
import vpython as vp
from . import magic
from examples.magic import *

############################################################################

class VisualSpin:
	def __init__(self, spin, center=[0,0,0]):
		self.spin = spin
		self.j = (self.spin.shape[0]-1)/2
		self.vcenter = vp.vector(*center)
		self.vsphere = vp.sphere(pos=self.vcenter, opacity=0.5, color=vp.color.blue)
		self.vstars = [vp.sphere(pos=self.vsphere.pos+vp.vector(*xyz), radius=0.2)\
							for xyz in spin_XYZ(self.spin)]
		self.X, self.Y, self.Z = qt.jmat(self.j, 'x'), qt.jmat(self.j, 'y'), qt.jmat(self.j, 'z')
		self.varrow = vp.arrow(pos=self.vcenter, axis=vp.vector(qt.expect(self.X, spin),\
																qt.expect(self.Y, spin),\
																qt.expect(self.Z, spin)))

	def update(self, spin):
		self.spin = spin
		for i, xyz in enumerate(spin_XYZ(self.spin)):
			self.vstars[i].pos = self.vsphere.pos+vp.vector(*xyz)
		self.varrow.axis = vp.vector(qt.expect(self.X, spin),\
									 qt.expect(self.Y, spin),\
									 qt.expect(self.Z, spin))

	def destroy():
		for vstar in vstars:
			vstar.visible = False
		self.varrow.visible = False
		self.vstars = []
		self.varrow = None

############################################################################

class VisualDensityMatrix:
	def __init__(self, dm, pos=[0,0,0]):
		self.pos = vp.vector(*pos)
		if dm.dims == [[1],[1]]:
			self.dm = dm
			z = self.dm.full()[0][0]
			self.vsphere = vp.sphere(color=vp.color.blue, 
									 opacity=dm.norm()/2,\
						  			 pos=self.pos)
			self.vspin_arrow =  vp.arrow(pos=self.pos, shaftwidth=0.1, color=vp.color.magenta,\
									axis=vp.vector(z.real, z.imag, 0))
		else:
			self.dm = dm.unit()
			self.sL, self.sV = dm.eigenstates()
			self.vsphere = vp.sphere(color=vp.color.blue, 
										opacity=dm.norm()/2,\
							  			pos=self.pos)
			self.colors = [vp.vector(*np.random.rand(3)) for i in range(len(self.sV))]
			self.vstars = [[vp.sphere(radius=0.2, 
										 pos=self.vsphere.pos+vp.vector(*xyz),\
										 opacity=self.sL[i].real,\
										 color=self.colors[i])
								for xyz in spin_XYZ(v)]\
									for i, v in enumerate(self.sV)]
			self.j = (dm.shape[0]-1)/2
			self.vspin_arrow = vp.arrow(pos=self.pos, shaftwidth=0.1, color=vp.color.magenta,\
										axis=vp.vector(qt.expect(qt.jmat(self.j, 'x'), self.dm).real,\
													   qt.expect(qt.jmat(self.j, 'y'), self.dm).real,\
													   qt.expect(qt.jmat(self.j, 'z'), self.dm).real))

	def update(self, dm):
		if dm.dims == [[1],[1]]:
			self.dm = dm
			z = self.dm.full()[0][0]
			self.vsphere.opacity = dm.norm()/2
			self.vspin_arrow.axis = vp.vector(z.real, z.imag, 0)
		else:
			self.vsphere.opacity = dm.norm()/2
			self.dm = dm.unit()
			self.sL, self.sV = dm.eigenstates()
			for i, v in enumerate(self.sV):
				for j, xyz in enumerate(spin_XYZ(v)):
					self.vstars[i][j].pos = self.vsphere.pos+vp.vector(*xyz)
					self.vstars[i][j].opacity = self.sL[i].real
			self.vspin_arrow.axis = vp.vector(qt.expect(qt.jmat(self.j, 'x'), self.dm).real,\
											  qt.expect(qt.jmat(self.j, 'y'), self.dm).real,\
											  qt.expect(qt.jmat(self.j, 'z'), self.dm).real)


	def visible(self):
		self.vsphere.visible = True
		self.vspin_arrow.visible = True
		for i in range(len(self.vstars)):
			for j in range(len(self.vstars[i])):
				self.vstars[i][j].visible = True

	def invisible(self):
		if self.vsphere:
			self.vsphere.visible = False
		if self.vspin_arrow:
			self.vspin_arrow.visible = False
		if self.vstars:
			for i in range(len(self.vstars)):
				for j in range(len(self.vstars[i])):
					self.vstars[i][j].visible = False

	def destroy(self):
		self.invisible()
		self.vsphere = None
		self.vspin_arrow = None
		self.vstars = None

############################################################################

class VisualVectors:
	def __init__(self, vectors, center=[0,0,0]):
		self.vectors = vectors
		self.varrows = [vp.arrow(pos=vp.vector(*center),\
								 axis=vp.vector(*v)) for v in self.vectors]

	def update(self, vectors):
		self.vectors = vectors
		for i, v in enumerate(self.vectors):
			self.varrows[i].axis = vp.vector(*v)

	def destroy(self):
		for varrow in self.varrows:
			varrow.visible = False
		self.varrows = []

############################################################################

class VisualPolyhedron:
	def __init__(self, poly, center=[0,0,0]):			
		self.poly = poly
		self.nfaces = len(self.poly.faces)
		colors = [vp.vector(*np.random.rand(3)) for i in range(self.nfaces)]

		self.vcenter = vp.vector(*center)
		self.vcolors = colors

		center = sum([v for v in self.poly.vertices])/len(self.poly.vertices)
		vertices = [vertex-center for vertex in self.poly.vertices]

		vpoints = [vp.sphere(pos=self.vcenter+vp.vector(*v),\
							 radius=0.02, emissive=True, opacity=0.6)\
						for v in vertices]
		vfaces = []
		vnormals = []
		vedges = []
		for f, face in enumerate(self.poly.faces):
			vtriangles = []
			for tri in itertools.combinations(face.vertices, 3):
				triangle = vp.triangle(vs=[vp.vertex(pos=self.vcenter+vp.vector(*vertices[v]),\
													 color=self.vcolors[f],\
													 opacity=0.5,\
													 emissive=True,\
													 shininess=0.5) for v in tri])

				vtriangles.append(triangle)
			vfaces.append(vtriangles)
			face_center = sum([vertices[vertex] for vertex in face.vertices])/len(face.vertices)
			vnormals.append(vp.arrow(pos=self.vcenter+vp.vector(*face_center),
									 color=self.vcolors[f],\
									 opacity=0.5,\
									 axis=vp.vector(*face.unormal)/3))
		self.vpoints, self.vfaces, self.vnormals = vpoints, vfaces, vnormals


	def update(self, poly):
		self.poly = poly
		center = sum([v for v in self.poly.vertices])/len(self.poly.vertices)
		vertices = [vertex-center for vertex in self.poly.vertices]

		for i, v in enumerate(vertices):
			self.vpoints[i].pos = self.vcenter+vp.vector(*v)

		for vface in self.vfaces:
			for vtri in vface:
				vtri.visible = False
		self.vfaces = []
		for f, face in enumerate(self.poly.faces):
			vtriangles = []
			for tri in itertools.combinations(face.vertices, 3):
				triangle = vp.triangle(vs=[vp.vertex(pos=self.vcenter+vp.vector(*vertices[v]),\
													 color=self.vcolors[f],\
													 opacity=0.5,\
													 emissive=True,\
													 shininess=0.5) for v in tri])

				vtriangles.append(triangle)
			self.vfaces.append(vtriangles)
			face_center = sum([vertices[vertex] for vertex in face.vertices])/len(face.vertices)
			self.vnormals[f].pos = self.vcenter+vp.vector(*face_center)
			self.vnormals[f].axis = vp.vector(*face.unormal)/3
		

	def destroy(self):
		for vpoint in self.vpoints:
			vpoint.visible = False
		for vface in self.vfaces:
			for vtri in vface:
				vtri.visible = False
		for vnormal in self.vnormals:
			vnormal.visible = False
		self.vpoints, self.vfaces, self.vnormals = [], [], []
