import itertools


import matplotlib.pylab as plt
import meshio
import netgen.occ as occ
from netgen.occ import gp_Trsf
import ngsolve as ng
import dolfin as df
import numpy as np
import pyacvd
import pyvista as pv

from dolfin import near
from mpi4py import MPI
from netgen.meshing import IdentificationType
from ngsolve import Mesh
from ngsolve.webgui import Draw
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from tqdm import trange


# The indices of the full stiffness matrix of (orthorhombic) interest
Voigt_notation = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
print("test!!")

def full_3x3_to_Voigt_6_index(i, j):
    if i == j:
        return i
    return 6-i-j


def Voigt_6x6_to_full_3x3x3x3(C):
    """
    Convert from the Voigt representation of the stiffness matrix to the full
    3x3x3x3 representation.

    Parameters
    ----------
    C : array_like
        6x6 stiffness matrix (Voigt notation).

    Returns
    -------
    C : array_like
        3x3x3x3 stiffness matrix.
    """

    C = np.asarray(C)
    C_out = np.zeros((3, 3, 3, 3), dtype=float)
    for i, j, k, l in itertools.product(range(3), range(3), range(3), range(3)):
        Voigt_i = full_3x3_to_Voigt_6_index(i, j)
        Voigt_j = full_3x3_to_Voigt_6_index(k, l)
        C_out[i, j, k, l] = C[Voigt_i, Voigt_j]
    return C_out


def full_3x3x3x3_to_Voigt_6x6(C, tol=1e-3, check_symmetry=True):
    """
    Convert from the full 3x3x3x3 representation of the stiffness matrix
    to the representation in Voigt notation. Checks symmetry in that process.
    """

    C = np.asarray(C)
    Voigt = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            k, l = Voigt_notation[i]
            m, n = Voigt_notation[j]
            Voigt[i, j] = C[k, l, m, n]
            """
            print('---')
            print("k,l,m,n", C[k,l,m,n])
            print("m,n,k,l", C[m,n,k,l])
            print("l,k,m,n", C[l,k,m,n])
            print("k,l,n,m", C[k,l,n,m])
            print("m,n,l,k", C[m,n,l,k])
            print("n,m,k,l", C[n,m,k,l])
            print("l,k,n,m", C[l,k,n,m])
            print("n,m,l,k", C[n,m,l,k])
            print('---')
            """
            if check_symmetry:
                assert abs(Voigt[i, j]-C[m, n, k, l]) < tol, \
                    '1 Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
                    .format(i, j, Voigt[i, j], m, n, k, l, C[m, n, k, l])
                assert abs(Voigt[i, j]-C[l, k, m, n]) < tol, \
                    '2 Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
                    .format(i, j, Voigt[i, j], l, k, m, n, C[l, k, m, n])
                assert abs(Voigt[i, j]-C[k, l, n, m]) < tol, \
                    '3 Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
                    .format(i, j, Voigt[i, j], k, l, n, m, C[k, l, n, m])
                assert abs(Voigt[i, j]-C[m, n, l, k]) < tol, \
                    '4 Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
                    .format(i, j, Voigt[i, j], m, n, l, k, C[m, n, l, k])
                assert abs(Voigt[i, j]-C[n, m, k, l]) < tol, \
                    '5 Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
                    .format(i, j, Voigt[i, j], n, m, k, l, C[n, m, k, l])
                assert abs(Voigt[i, j]-C[l, k, n, m]) < tol, \
                    '6 Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
                    .format(i, j, Voigt[i, j], l, k, n, m, C[l, k, n, m])
                assert abs(Voigt[i, j]-C[n, m, l, k]) < tol, \
                    '7 Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
                    .format(i, j, Voigt[i, j], n, m, l, k, C[n, m, l, k])

    return Voigt


def rotate_elastic_constants(C, A, tol=1e-6):
    """
    Return rotated elastic moduli for a general crystal given the elastic
    constant in Voigt notation.

    Parameters
    ----------
    C : array_like
        6x6 matrix of elastic constants (Voigt notation).
    A : array_like
        3x3 rotation matrix.

    Returns
    -------
    C : array
        6x6 matrix of rotated elastic constants (Voigt notation).
    """

    A = np.asarray(A)

    # Is this a rotation matrix?
    if np.sometrue(np.abs(np.dot(np.array(A), np.transpose(np.array(A))) -
                          np.eye(3, dtype=float)) > tol):
        raise RuntimeError('Matrix *A* does not describe a rotation.')

    # Rotate
    return full_3x3x3x3_to_Voigt_6x6(np.einsum('ia,jb,kc,ld,abcd->ijkl',
                                               A, A, A, A,
                                               Voigt_6x6_to_full_3x3x3x3(C)))


def init_logger(log_file_name='solver.log'):
    import logging
    logger = logging.getLogger('logger')
    logger.propagate = False
    logger.setLevel(logging.INFO)

    # our first handler is a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler_format = '[%(levelname)s]: %(message)s'
    console_handler.setFormatter(logging.Formatter(console_handler_format))
    logger.addHandler(console_handler)

    # the second handler is a file handler
    file_handler = logging.FileHandler('solver.log')
    file_handler.setLevel(logging.INFO)
    file_handler_format = '%(asctime)s | [%(levelname)s] | %(lineno)d: %(message)s'
    file_handler.setFormatter(logging.Formatter(file_handler_format))
    logger.addHandler(file_handler)
    return logger


class ExternalCell2Dolfin(object):
    def __init__(self, mesh):
        self.mesh = mesh
        # self.mesh = df.UnitCubeMesh(10, 10, 10)
        self.centre = (self.mesh.coordinates().max(
            0) + self.mesh.coordinates().min(0)) / 2
        # vycentrovani
        self.mesh.translate(df.Point(-self.centre))
        self.xmin, self.ymin, self.zmin = self.mesh.coordinates().min(0)
        self.xmax, self.ymax, self.zmax = self.mesh.coordinates().max(0)

        self.lx = self.xmax - self.xmin
        self.ly = self.ymax - self.ymin
        self.lz = self.zmax - self.zmin

        self.EPS = 1e-5
        self.get_markers()

    def get_left_corner(self):
        EPS = self.EPS
        xmin, ymin, zmin = self.xmin, self.ymin, self.zmin

        class get_corner(df.SubDomain):
            def inside(self, x, on_boundary):
                return df.near(x[0], xmin, EPS) \
                    and df.near(x[1], ymin, EPS) \
                    and df.near(x[2], zmin, EPS) and on_boundary
        return get_corner()

    def get_markers(self):
        EPS = self.EPS
        xmin, ymin, zmin = self.xmin, self.ymin, self.zmin
        xmax, ymax, zmax = self.xmax, self.ymax, self.zmax

        class neg_x(df.SubDomain):
            def inside(self, x, on_boundary):
                return df.near(x[0], xmin, EPS) and on_boundary

        class neg_y(df.SubDomain):
            def inside(self, x, on_boundary):
                return df.near(x[1], ymin, EPS) and on_boundary

        class pos_x(df.SubDomain):
            def inside(self, x, on_boundary):
                return df.near(x[0], xmax, EPS) and on_boundary

        class pos_y(df.SubDomain):
            def inside(self, x, on_boundary):
                return df.near(x[1], ymax, EPS) and on_boundary

        class neg_z(df.SubDomain):
            def inside(self, x, on_boundary):
                return df.near(x[2], zmin, EPS) and on_boundary

        class pos_z(df.SubDomain):
            def inside(self, x, on_boundary):
                return df.near(x[2], zmax, EPS) and on_boundary

        self.boundaries = {'neg_x': neg_x(), 'pos_x': pos_x(),
                           'neg_y': neg_y(), 'pos_y': pos_y(),
                           'neg_z': neg_z(), 'pos_z': pos_z()}

    def get_surface_measures(self):
        self.subdomains = df.MeshFunction("size_t", self.mesh,
                                          self.mesh.topology().dim() - 1)
        self.subdomains.set_all(0)
        for i, key in enumerate(self.boundaries.keys()):
            self.boundaries[key].mark(self.subdomains, i + 1)

        return df.Measure('ds', domain=self.mesh, subdomain_data=self.subdomains)

    def get_surface_measure(self):
        self.subdomains = df.MeshFunction("size_t", self.mesh,
                                          self.mesh.topology().dim() - 1)

        self.subdomains.set_all(0)
        for i, key in enumerate(self.boundaries.keys()):
            self.boundaries[key].mark(self.subdomains, 1)

        return df.Measure('ds', domain=self.mesh, subdomain_data=self.subdomains)

    def save_markers(self, fname='markers.pvd'):
        file = df.File(fname)
        file << self.subdomains

    def save_mesh(self, fname="reentrant_cell.pvd"):
        file = df.File(fname)
        file << self.mesh

    def check_periodic(self):
        points = self.mesh.coordinates()
        pcs = [[self.xmin, self.xmax],
               [self.ymin, self.ymax],
               [self.zmin, self.zmax]]
        nms = ['x', 'y', 'z']
        for i in range(len(pcs)):
            pc_min, pc_max = pcs[i]
            print(f'Checking {nms[i]}-periodicity...', end=" ", flush=True)
            points_min = points[(points[:, i] >= (pc_min-self.EPS))
                                & (points[:, i] <= (pc_min + self.EPS))]
            points_max = points[(points[:, i] >= (pc_max-self.EPS))
                                & (points[:, i] <= (pc_max + self.EPS))]
            if len(points_min) == len(points_max):
                print('PASS')
            else:
                print('FAILED')


def line(point1, point2, r, cup=False):
    lp1 = occ.Sphere(occ.Pnt(point1), r=r)
    lp2 = occ.Sphere(occ.Pnt(point2), r=r)
    h = np.linalg.norm(point2-point1)
    direction = (point2-point1)/h
    line = occ.Cylinder(occ.Pnt(point1),
                        occ.gp_Vec(tuple(direction)),
                        r=r, h=h)
    if cup:
        return line + lp1 + lp2
    else:
        return line

def Cell50Block(nX, nY, nZ, L, r, mesh_order=2, minh=0.2, maxh=0.4):
    for i in range(nX):
        for j in range(nY):
            for k in range(nZ):
                lp1x = np.array([-L/2. + L*i, L*j, L*k])
                lp2x = np.array([ L/2. + L*i, L*j, L*k])
                linex = line(lp1x, lp2x, r)

                lp1y = np.array([L*i, -L/2 + L*j, L*k])
                lp2y = np.array([L*i,  L/2 + L*j, L*k])
                liney = line(lp1y, lp2y, r)

                lp1z = np.array([L*i, L*j, -L/2 + L*k])
                lp2z = np.array([L*i, L*j,  L/2 + L*k])
                linez = line(lp1z, lp2z, r)

                if i==0 and j==0 and k==0:
                    RVE = linex + liney + linez
                else:
                    RVE += linex + liney + linez

    RVE = RVE.Move((L/2-nX*L/2,L/2-nY*L/2,L/2-nZ*L/2))

    minX = -L*nX/2 + 1e-3
    maxX = L*nX/2 - 1e-3    
    trf = occ.gp_Trsf.Translation(nX*L * occ.X)
    RVE.faces[occ.X < minX].Identify(RVE.faces[occ.X > maxX], "periodicX",
                                  IdentificationType.PERIODIC, trf)
    minY = -L*nY/2 + 1e-3
    maxY = L*nY/2 - 1e-3  
    trf = occ.gp_Trsf.Translation(nY*L * occ.Y)
    RVE.faces[occ.Y < minY].Identify(RVE.faces[occ.Y > minY], "periodicY",
                                  IdentificationType.PERIODIC, trf)    
    minZ = -L*nZ/2 + 1e-3
    maxZ = L*nZ/2 - 1e-3  
    trf = occ.gp_Trsf.Translation(nZ*L * occ.Z)
    RVE.faces[occ.Z < minZ].Identify(RVE.faces[occ.Z > minZ], "periodicZ",
                                  IdentificationType.PERIODIC, trf)

    posX = occ.X >= L/2. + L*(nX-1)
    negX = occ.X <= -L/2.

    posY = occ.Y >= L/2. + L*(nY-1)
    negY = occ.Y <= -L/2.

    posZ = occ.Z >= L/2. + L*(nZ-1)
    negZ = occ.Z <= -L/2.

    RVE.faces[posX].name = 'pos_x'
    RVE.faces[posX].col = (0, 0, 1)
    RVE.faces[negX].name = 'neg_x'
    RVE.faces[negX].col = (0, 0, 1)

    RVE.faces[posY].name = 'pos_y'
    RVE.faces[posY].col = (0, 0, 1)
    RVE.faces[negY].name = 'neg_y'
    RVE.faces[negY].col = (0, 0, 1)

    RVE.faces[posZ].name = 'pos_z'
    RVE.faces[posZ].col = (0, 0, 1)
    RVE.faces[negZ].name = 'neg_z'
    RVE.faces[negZ].col = (0, 0, 1)


    geo = occ.OCCGeometry(RVE)
    # geo.save_STEP("geometry.stp")
    mesh = Mesh(geo.GenerateMesh(minh=minh, maxh=maxh))
    mesh.Curve(mesh_order)
    return mesh

def Cell50(L, r, mesh_order=2, minh=0.2, maxh=0.4):
    lp1x = np.array([- L/2., 0, 0])
    lp2x = np.array([+ L/2., 0, 0])
    linex = line(lp1x, lp2x, r)

    lp1y = np.array([0, -L/2, 0])
    lp2y = np.array([0,  L/2, 0])
    liney = line(lp1y, lp2y, r)

    lp1z = np.array([0, 0, -L/2])
    lp2z = np.array([0, 0, L/2])
    linez = line(lp1z, lp2z, r)

    RVE = linex + liney + linez
    minX = -L/2 + 1e-3
    maxX = L/2 - 1e-3   
    trf = occ.gp_Trsf.Translation(L * occ.X)
    RVE.faces[occ.X < minX].Identify(RVE.faces[occ.X > maxX], "periodicX",
                                  IdentificationType.PERIODIC, trf)
    
    minY = -L/2 + 1e-3
    maxY = L/2 - 1e-3  
    trf = occ.gp_Trsf.Translation(L * occ.Y)
    RVE.faces[occ.Y < minY].Identify(RVE.faces[occ.Y > maxY], "periodicY",
                                  IdentificationType.PERIODIC, trf)
    
    minZ = -L/2 + 1e-3
    maxZ = L/2 - 1e-3  
    trf = occ.gp_Trsf.Translation(L * occ.Z)
    RVE.faces[occ.Z < minZ].Identify(RVE.faces[occ.Z > maxZ], "periodicZ",
                                  IdentificationType.PERIODIC, trf)

    posX = occ.X >= L/2.
    negX = occ.X <= -L/2.

    posY = occ.Y >= L/2.
    negY = occ.Y <= -L/2.

    posZ = occ.Z >= L/2.
    negZ = occ.Z <= -L/2.

    RVE.faces[posX].name = 'pos_x'
    RVE.faces[posX].col = (0, 0, 1)
    RVE.faces[negX].name = 'neg_x'
    RVE.faces[negX].col = (0, 0, 1)

    RVE.faces[posY].name = 'pos_y'
    RVE.faces[posY].col = (0, 0, 1)
    RVE.faces[negY].name = 'neg_y'
    RVE.faces[negY].col = (0, 0, 1)

    RVE.faces[posZ].name = 'pos_z'
    RVE.faces[posZ].col = (0, 0, 1)
    RVE.faces[negZ].name = 'neg_z'
    RVE.faces[negZ].col = (0, 0, 1)

    geo = occ.OCCGeometry(RVE)
    mesh = Mesh(geo.GenerateMesh(minh=minh, maxh=maxh))
    mesh.Curve(mesh_order)
    return mesh

def Cell21X(L, r, mesh_order=2, minh=0.2, maxh=0.4):
    boundN = np.array([+ L/2. + 100*r, + L/2. + 100*r, - L/2. - 100*r])
    boundP = np.array([- L/2. - 100*r, - L/2. - 100*r, + L/2. + 100*r])
    boundn = np.array([+ L/2., + L/2., - L/2.])
    boundp = np.array([- L/2., - L/2., + L/2.])

    lp1 = np.array([- L/2., - L/2., - L/2.])
    lp2 = np.array([- L/2., + L/2., - L/2.])
    lp3 = np.array([+ L/2., + L/2., - L/2.])
    lp4 = np.array([+ L/2., - L/2., - L/2.])
    lp5 = np.array([- L/2., - L/2., + L/2.])
    lp6 = np.array([- L/2., + L/2., + L/2.])
    lp7 = np.array([+ L/2., + L/2., + L/2.])
    lp8 = np.array([+ L/2., - L/2., + L/2.])
    line1 = line(lp1, lp7, r)
    line2 = line(lp2, lp8, r)
    line3 = line(lp3, lp5, r)
    line4 = line(lp4, lp6, r)
    nline1 = line(lp7, lp1, r)
    nline2 = line(lp8, lp2, r)
    nline3 = line(lp5, lp3, r)
    nline4 = line(lp6, lp4, r)
    

    RVE = line1 + nline2 + line3 + nline4
    lp_outer = occ.Box(occ.Pnt(boundN), occ.Pnt(boundP))
    lp_inner = occ.Box(occ.Pnt(boundn), occ.Pnt(boundp))
    GG = lp_outer-lp_inner
    RVE -= GG
    

    minX = RVE.faces.Min(occ.X).center[0] + 1e-4
    maxX = RVE.faces.Max(occ.X).center[0] - 1e-4
    trf = occ.gp_Trsf.Translation(L * occ.X)
    RVE.faces[occ.X < minX].Identify(RVE.faces[occ.X > maxX], "periodicX",
                                  IdentificationType.PERIODIC, trf)
    # xNfaces = RVE.faces[occ.X < minX]
    # xPfaces = RVE.faces[occ.X > maxX]
    # print(xNfaces[2].center)
    # print(xPfaces[1].center)
    # xNfaces[0].Identify(xPfaces[0], "periodicX",
    #                               IdentificationType.PERIODIC, trf)
    # xNfaces[1].Identify(xPfaces[2], "periodicX",
    #                               IdentificationType.PERIODIC, trf)
    # xNfaces[2].Identify(xPfaces[1], "periodicX",
    #                               IdentificationType.PERIODIC, trf)
    # xNfaces[3].Identify(xPfaces[3], "periodicX",
    #                               IdentificationType.PERIODIC, trf)

        

    minY = RVE.faces.Min(occ.Y).center[0] + 1e-3
    maxY = RVE.faces.Max(occ.Y).center[0] - 1e-3  
    trf = occ.gp_Trsf.Translation(L * occ.Y)
    RVE.faces[occ.Y < minY].Identify(RVE.faces[occ.Y > maxY], "periodicY",
                                  IdentificationType.PERIODIC, trf)   
  
    
    minZ = RVE.faces.Min(occ.Z).center[0] + 1e-3
    maxZ = RVE.faces.Max(occ.Z).center[0] - 1e-3  
    trf = occ.gp_Trsf.Translation(L * occ.Z)
    RVE.faces[occ.Z < minZ].Identify(RVE.faces[occ.Z > maxZ], "periodicZ",
                                  IdentificationType.PERIODIC, trf)

    geo = occ.OCCGeometry(RVE)
    mesh = Mesh(geo.GenerateMesh(minh=minh, maxh=maxh))
    mesh.Curve(mesh_order)
    return mesh


def Cell22(L, r, mesh_order=2, minh=0.2, maxh=0.4):
    boundN = np.array([+ L/2. + 10*r, + L/2. + 10*r, - L/2. - 10*r])
    boundP = np.array([- L/2. - 10*r, - L/2. - 10*r, + L/2. + 10*r])
    boundn = np.array([+ L/2., + L/2., - L/2.])
    boundp = np.array([- L/2., - L/2., + L/2.])

    lp1xn = np.array([- 1.5*L/2., - L/4., - L/4.])
    lp2xn = np.array([- 1.5*L/2., - L/4., + L/4.])
    lp3xn = np.array([- L/2., + L/4., + L/4.])
    lp4xn = np.array([- L/2., + L/4., - L/4.])
    lp1xp = np.array([+ 1.5*L/2., - L/4., - L/4.])
    lp2xp = np.array([+ 1.5*L/2., - L/4., + L/4.])
    lp3xp = np.array([+ L/2., + L/4., + L/4.])
    lp4xp = np.array([+ L/2., + L/4., - L/4.])

    lp1yn = np.array([- L/4., - L/2., - L/4.])
    lp2yn = np.array([- L/4., - L/2., + L/4.])
    lp3yn = np.array([+ L/4., - L/2., + L/4.])
    lp4yn = np.array([+ L/4., - L/2., - L/4.])
    lp1yp = np.array([- L/4., + L/2., - L/4.])
    lp2yp = np.array([- L/4., + L/2., + L/4.])
    lp3yp = np.array([+ L/4., + L/2., + L/4.])
    lp4yp = np.array([+ L/4., + L/2., - L/4.])

    lp1zn = np.array([- L/4., - L/4., - 1.5*L/2.])
    lp2zn = np.array([- L/4., + L/4., - 1.5*L/2.])
    lp3zn = np.array([+ L/4., + L/4., - L/2.])
    lp4zn = np.array([+ L/4., - L/4., - L/2.])
    lp1zp = np.array([- L/4., - L/4., + 1.5*L/2.])
    lp2zp = np.array([- L/4., + L/4., + 1.5*L/2.])
    lp3zp = np.array([+ L/4., + L/4., + L/2.])
    lp4zp = np.array([+ L/4., - L/4., + L/2.])

    line1x = line(lp1xn, lp2xp, r) #
    line2x = line(lp2xn, lp1xp, r) #
    line3x = line(lp3xn, lp3xp, r)
    line4x = line(lp4xn, lp4xp, r)
    line1y = line(lp1yn, lp1yp, r)
    line2y = line(lp2yn, lp2yp, r)
    line3y = line(lp3yn, lp3yp, r)
    line4y = line(lp4yn, lp4yp, r)
    line1z = line(lp1zn, lp2zp, r) #
    line2z = line(lp2zn, lp1zp, r) #
    line3z = line(lp3zn, lp3zp, r)
    line4z = line(lp4zn, lp4zp, r)
    RVE = line1x + line2x + line3x + line4x + line1y + line2y + line3y + line4y + line1z + line2z + line3z + line4z
    lp_outer = occ.Box(occ.Pnt(boundN), occ.Pnt(boundP))
    lp_inner = occ.Box(occ.Pnt(boundn), occ.Pnt(boundp))
    GG = lp_outer-lp_inner
    RVE -= GG
    
    # minX = -L/2 + 1e-3
    # maxX = L/2 - 1e-3  
    # trf = occ.gp_Trsf.Translation(L * occ.X)
    # RVE.faces[occ.X < minX].Identify(RVE.faces[occ.X > maxX], "periodicX",
    #                               IdentificationType.PERIODIC, trf)
  
    # minY = -L/2 + 1e-3
    # maxY = L/2 - 1e-3  
    # trf = occ.gp_Trsf.Translation(L * occ.Y)
    # RVE.faces[occ.Y < minY].Identify(RVE.faces[occ.Y > maxY], "periodicY",
    #                               IdentificationType.PERIODIC, trf)   
  
    
    # minZ = -L/2 + 1e-3
    # maxZ = L/2 - 1e-3  
    # trf = occ.gp_Trsf.Translation(L * occ.Z)
    # RVE.faces[occ.Z < minZ].Identify(RVE.faces[occ.Z > maxZ], "periodicZ",
    #                               IdentificationType.PERIODIC, trf)

    posX = occ.X >= L/2.
    negX = occ.X <= -L/2.

    posY = occ.Y >= L/2.
    negY = occ.Y <= -L/2.

    posZ = occ.Z >= L/2.
    negZ = occ.Z <= -L/2.

    RVE.faces[posX].name = 'pos_x'
    RVE.faces[posX].col = (0, 0, 1)
    RVE.faces[negX].name = 'neg_x'
    RVE.faces[negX].col = (0, 0, 1)

    RVE.faces[posY].name = 'pos_y'
    RVE.faces[posY].col = (0, 0, 1)
    RVE.faces[negY].name = 'neg_y'
    RVE.faces[negY].col = (0, 0, 1)

    RVE.faces[posZ].name = 'pos_z'
    RVE.faces[posZ].col = (0, 0, 1)
    RVE.faces[negZ].name = 'neg_z'
    RVE.faces[negZ].col = (0, 0, 1)

    geo = occ.OCCGeometry(RVE)
    mesh = Mesh(geo.GenerateMesh(minh=minh, maxh=maxh))
    mesh.Curve(mesh_order)
    return mesh


def Cell23(L, r, K, mesh_order=2, minh=0.2, maxh=0.4):

    boundN = np.array([+ L/2. + 100*r, + L/2. + 10*r, - L/2. - 100*r])
    boundP = np.array([- L/2. - 100*r, - L/2. - 10*r, + L/2. + 100*r])
    boundn = np.array([+ L/2., + L/2., - L/2.])
    boundp = np.array([- L/2., - L/2., + L/2.])

    lp1xn = np.array([- K*L/2., - L/4., - L/4.])
    lp2xn = np.array([- K*L/2., - L/4., + L/4.])
    lp3xn = np.array([- K*L/2., + L/4., + L/4.])
    lp4xn = np.array([- K*L/2., + L/4., - L/4.])
    lp1xp = np.array([+ K*L/2., - L/4., - L/4.])
    lp2xp = np.array([+ K*L/2., - L/4., + L/4.])
    lp3xp = np.array([+ K*L/2., + L/4., + L/4.])
    lp4xp = np.array([+ K*L/2., + L/4., - L/4.])

    lp1yn = np.array([- L/4., - K*L/2., - L/4.])
    lp2yn = np.array([- L/4., - K*L/2., + L/4.])
    lp3yn = np.array([+ L/4., - K*L/2., + L/4.])
    lp4yn = np.array([+ L/4., - K*L/2., - L/4.])
    lp1yp = np.array([- L/4., + K*L/2., - L/4.])
    lp2yp = np.array([- L/4., + K*L/2., + L/4.])
    lp3yp = np.array([+ L/4., + K*L/2., + L/4.])
    lp4yp = np.array([+ L/4., + K*L/2., - L/4.])

    lp1zn = np.array([- L/4., - L/4., - K*L/2.])
    lp2zn = np.array([- L/4., + L/4., - K*L/2.])
    lp3zn = np.array([+ L/4., + L/4., - K*L/2.])
    lp4zn = np.array([+ L/4., - L/4., - K*L/2.])
    lp1zp = np.array([- L/4., - L/4., + K*L/2.])
    lp2zp = np.array([- L/4., + L/4., + K*L/2.])
    lp3zp = np.array([+ L/4., + L/4., + K*L/2.])
    lp4zp = np.array([+ L/4., - L/4., + K*L/2.])

    lp1x = np.array([- L/2., 0, 0])
    lp2x = np.array([+ L/2., 0, 0]) 
    lp1y = np.array([0, -L/2, 0])
    lp2y = np.array([0,  L/2, 0])
    lp1z = np.array([0, 0, -L/2])
    lp2z = np.array([0, 0, L/2])
    


    line1x = line(lp1xn, lp2xp, r) #
    line2x = line(lp2xn, lp1xp, r) #
    line3x = line(lp3xn, lp4xp, r)
    line4x = line(lp4xn, lp3xp, r)
    line1y = line(lp1yn, lp2yp, r)
    line2y = line(lp2yn, lp1yp, r)
    line3y = line(lp3yn, lp4yp, r)
    line4y = line(lp4yn, lp3yp, r)
    line1z = line(lp1zn, lp2zp, r) #
    line2z = line(lp2zn, lp1zp, r) #
    line3z = line(lp3zn, lp4zp, r)
    line4z = line(lp4zn, lp3zp, r)

    linex = line(lp1x, lp2x, r)
    liney = line(lp1y, lp2y, r)
    linez = line(lp1z, lp2z, r)

    RVE = line1x + line2x + line3x + line4x + line1y + line2y + line3y + line4y + line1z + line2z + line3z + line4z + linex + liney + linez
    lp_outer = occ.Box(occ.Pnt(boundN), occ.Pnt(boundP))
    lp_inner = occ.Box(occ.Pnt(boundn), occ.Pnt(boundp))
    GG = lp_outer-lp_inner
    RVE -= GG
    
    minX = -L/2 + 1e-3
    maxX = L/2 - 1e-3  
    trf = occ.gp_Trsf.Translation(L * occ.X)
    RVE.faces[occ.X < minX].Identify(RVE.faces[occ.X > maxX], "periodicX",
                                  IdentificationType.PERIODIC, trf)
  
    minY = -L/2 + 1e-3
    maxY = L/2 - 1e-3  
    trf = occ.gp_Trsf.Translation(L * occ.Y)
    RVE.faces[occ.Y < minY].Identify(RVE.faces[occ.Y > maxY], "periodicY",
                                  IdentificationType.PERIODIC, trf)   
  
    
    minZ = -L/2 + 1e-3
    maxZ = L/2 - 1e-3  
    trf = occ.gp_Trsf.Translation(L * occ.Z)
    RVE.faces[occ.Z < minZ].Identify(RVE.faces[occ.Z > maxZ], "periodicZ",
                                  IdentificationType.PERIODIC, trf)

    posX = occ.X >= L/2.
    negX = occ.X <= -L/2.

    posY = occ.Y >= L/2.
    negY = occ.Y <= -L/2.

    posZ = occ.Z >= L/2.
    negZ = occ.Z <= -L/2.

    RVE.faces[posX].name = 'pos_x'
    RVE.faces[posX].col = (0, 0, 1)
    RVE.faces[negX].name = 'neg_x'
    RVE.faces[negX].col = (0, 0, 1)

    RVE.faces[posY].name = 'pos_y'
    RVE.faces[posY].col = (0, 0, 1)
    RVE.faces[negY].name = 'neg_y'
    RVE.faces[negY].col = (0, 0, 1)

    RVE.faces[posZ].name = 'pos_z'
    RVE.faces[posZ].col = (0, 0, 1)
    RVE.faces[negZ].name = 'neg_z'
    RVE.faces[negZ].col = (0, 0, 1)

    geo = occ.OCCGeometry(RVE)
    mesh = Mesh(geo.GenerateMesh(minh=minh, maxh=maxh))
    mesh.Curve(mesh_order)
    return mesh


def Cell23X(L, r, K, mesh_order=2, minh=0.2, maxh=0.4):

    boundN = np.array([+ L/2. + 100*r, + L/2. + 10*r, - L/2. - 100*r])
    boundP = np.array([- L/2. - 100*r, - L/2. - 10*r, + L/2. + 100*r])
    boundn = np.array([+ L/2., + L/2., - L/2.])
    boundp = np.array([- L/2., - L/2., + L/2.])

    lp1xn = np.array([- K*L/2., - L/4., - L/4.])
    lp2xn = np.array([- K*L/2., - L/4., + L/4.])
    lp3xn = np.array([- K*L/2., + L/4., + L/4.])
    lp4xn = np.array([- K*L/2., + L/4., - L/4.])
    lp1xp = np.array([+ K*L/2., - L/4., - L/4.])
    lp2xp = np.array([+ K*L/2., - L/4., + L/4.])
    lp3xp = np.array([+ K*L/2., + L/4., + L/4.])
    lp4xp = np.array([+ K*L/2., + L/4., - L/4.])

    lp1yn = np.array([- L/4., - K*L/2., - L/4.])
    lp2yn = np.array([- L/4., - K*L/2., + L/4.])
    lp3yn = np.array([+ L/4., - K*L/2., + L/4.])
    lp4yn = np.array([+ L/4., - K*L/2., - L/4.])
    lp1yp = np.array([- L/4., + K*L/2., - L/4.])
    lp2yp = np.array([- L/4., + K*L/2., + L/4.])
    lp3yp = np.array([+ L/4., + K*L/2., + L/4.])
    lp4yp = np.array([+ L/4., + K*L/2., - L/4.])

    lp1zn = np.array([- L/4., - L/4., - K*L/2.])
    lp2zn = np.array([- L/4., + L/4., - K*L/2.])
    lp3zn = np.array([+ L/4., + L/4., - K*L/2.])
    lp4zn = np.array([+ L/4., - L/4., - K*L/2.])
    lp1zp = np.array([- L/4., - L/4., + K*L/2.])
    lp2zp = np.array([- L/4., + L/4., + K*L/2.])
    lp3zp = np.array([+ L/4., + L/4., + K*L/2.])
    lp4zp = np.array([+ L/4., - L/4., + K*L/2.])

    lp1x = np.array([- L/2., 0, 0])
    lp2x = np.array([+ L/2., 0, 0]) 
    lp1y = np.array([0, -L/2, 0])
    lp2y = np.array([0,  L/2, 0])
    lp1z = np.array([0, 0, -L/2])
    lp2z = np.array([0, 0, L/2])

    lp1 = np.array([- L/2., - L/2., - L/2.])
    lp2 = np.array([- L/2., + L/2., - L/2.])
    lp3 = np.array([+ L/2., + L/2., - L/2.])
    lp4 = np.array([+ L/2., - L/2., - L/2.])
    lp5 = np.array([- L/2., - L/2., + L/2.])
    lp6 = np.array([- L/2., + L/2., + L/2.])
    lp7 = np.array([+ L/2., + L/2., + L/2.])
    lp8 = np.array([+ L/2., - L/2., + L/2.]) 

    line1x = line(lp1xn, lp2xp, r) 
    line2x = line(lp2xn, lp1xp, r) 
    line3x = line(lp3xn, lp4xp, r)
    line4x = line(lp4xn, lp3xp, r)
    line1y = line(lp1yn, lp2yp, r)
    line2y = line(lp2yn, lp1yp, r)
    line3y = line(lp3yn, lp4yp, r)
    line4y = line(lp4yn, lp3yp, r)
    line1z = line(lp1zn, lp2zp, r) 
    line2z = line(lp2zn, lp1zp, r) 
    line3z = line(lp3zn, lp4zp, r)
    line4z = line(lp4zn, lp3zp, r)

    linex = line(lp1x, lp2x, r)
    liney = line(lp1y, lp2y, r)
    linez = line(lp1z, lp2z, r)

    line1 = line(lp1, lp7, r)
    line2 = line(lp2, lp8, r)
    line3 = line(lp3, lp5, r)
    line4 = line(lp4, lp6, r)

    RVE = line1x + line2x + line3x + line4x + line1y + line2y + line3y + line4y + line1z + line2z + line3z + line4z + linex + liney + linez + line1 + line2 + line3 + line4
    lp_outer = occ.Box(occ.Pnt(boundN), occ.Pnt(boundP))
    lp_inner = occ.Box(occ.Pnt(boundn), occ.Pnt(boundp))
    GG = lp_outer-lp_inner
    RVE -= GG
    
    minX = -L/2 + 1e-3
    maxX = L/2 - 1e-3  
    trf = occ.gp_Trsf.Translation(L * occ.X)
    RVE.faces[occ.X < minX].Identify(RVE.faces[occ.X > maxX], "periodicX",
                                  IdentificationType.PERIODIC, trf)
  
    minY = -L/2 + 1e-3
    maxY = L/2 - 1e-3  
    trf = occ.gp_Trsf.Translation(L * occ.Y)
    RVE.faces[occ.Y < minY].Identify(RVE.faces[occ.Y > maxY], "periodicY",
                                  IdentificationType.PERIODIC, trf)   
  
    
    minZ = -L/2 + 1e-3
    maxZ = L/2 - 1e-3  
    trf = occ.gp_Trsf.Translation(L * occ.Z)
    RVE.faces[occ.Z < minZ].Identify(RVE.faces[occ.Z > maxZ], "periodicZ",
                                  IdentificationType.PERIODIC, trf)

    posX = occ.X >= L/2.
    negX = occ.X <= -L/2.

    posY = occ.Y >= L/2.
    negY = occ.Y <= -L/2.

    posZ = occ.Z >= L/2.
    negZ = occ.Z <= -L/2.

    RVE.faces[posX].name = 'pos_x'
    RVE.faces[posX].col = (0, 0, 1)
    RVE.faces[negX].name = 'neg_x'
    RVE.faces[negX].col = (0, 0, 1)

    RVE.faces[posY].name = 'pos_y'
    RVE.faces[posY].col = (0, 0, 1)
    RVE.faces[negY].name = 'neg_y'
    RVE.faces[negY].col = (0, 0, 1)

    RVE.faces[posZ].name = 'pos_z'
    RVE.faces[posZ].col = (0, 0, 1)
    RVE.faces[negZ].name = 'neg_z'
    RVE.faces[negZ].col = (0, 0, 1)

    geo = occ.OCCGeometry(RVE)
    mesh = Mesh(geo.GenerateMesh(minh=minh, maxh=maxh))
    mesh.Curve(mesh_order)
    return mesh


def Cell24X(L, r, K, mesh_order=2, minh=0.2, maxh=0.4):

    boundN = np.array([+ L/2. + 100*r, + L/2. + 10*r, - L/2. - 100*r])
    boundP = np.array([- L/2. - 100*r, - L/2. - 10*r, + L/2. + 100*r])
    boundn = np.array([+ L/2., + L/2., - L/2.])
    boundp = np.array([- L/2., - L/2., + L/2.])

    b = L/2 - K
    ksi = 1.2

    P1  = ksi * np.array([- L/2.,   - b, - L/2.])
    P2  = ksi * np.array([- L/2.,     b, - L/2.])
    P3  = ksi * np.array([   - b,  L/2., - L/2.])
    P4  = ksi * np.array([     b,  L/2., - L/2.])
    P5  = ksi * np.array([  L/2.,     b, - L/2.])
    P6  = ksi * np.array([  L/2.,   - b, - L/2.])
    P7  = ksi * np.array([     b,- L/2., - L/2.])
    P8  = ksi * np.array([   - b,- L/2., - L/2.])
    P9  = ksi * np.array([- L/2.,   - b,   L/2.])
    P10 = ksi * np.array([- L/2.,     b,   L/2.])
    P11 = ksi * np.array([   - b,  L/2.,   L/2.])
    P12 = ksi * np.array([     b,  L/2.,   L/2.])
    P13 = ksi * np.array([  L/2.,     b,   L/2.])
    P14 = ksi * np.array([  L/2.,   - b,   L/2.])
    P15 = ksi * np.array([     b,- L/2.,   L/2.])
    P16 = ksi * np.array([   - b,- L/2.,   L/2.])
    P17 = ksi * np.array([- L/2.,- L/2.,     -b])
    P18 = ksi * np.array([- L/2.,  L/2.,     -b])
    P19 = ksi * np.array([  L/2.,  L/2.,     -b])
    P20 = ksi * np.array([  L/2.,- L/2.,     -b])
    P21 = ksi * np.array([- L/2.,- L/2.,      b])
    P22 = ksi * np.array([- L/2.,  L/2.,      b])
    P23 = ksi * np.array([  L/2.,  L/2.,      b])
    P24 = ksi * np.array([  L/2.,- L/2.,      b])
    
    bar1 = line(P1 , P13, r)
    bar2 = line(P2 , P14, r)
    bar3 = line(P3 , P15, r)
    bar4 = line(P4 , P16, r)
    bar5 = line(P5 , P9 , r)
    bar6 = line(P6 , P10, r)
    bar7 = line(P7 , P11, r)
    bar8 = line(P8 , P12, r)
    bar17= line(P17, P23, r)
    bar18= line(P18, P24, r)
    bar19= line(P19, P21, r)
    bar20= line(P20, P22, r)

    mbar1 = line(P13, P1 ,  r)
    mbar2 = line(P14, P2 ,  r)
    mbar3 = line(P15, P3 ,  r)
    mbar4 = line(P16, P4 ,  r)
    mbar5 = line(P9 , P5 ,  r)
    mbar6 = line(P10, P6 ,  r)

    RVE = bar1 +  bar2 + mbar1 +  mbar2+  bar5 +  bar6 #bar3 +  bar4 + \
        #    bar5 +  bar6 +  bar7 +  bar8 + \
        #    bar17 + bar18 + bar19 + bar20
    lp_outer = occ.Box(occ.Pnt(boundN), occ.Pnt(boundP))
    lp_inner = occ.Box(occ.Pnt(boundn), occ.Pnt(boundp))
    GG = lp_outer-lp_inner
    RVE -= GG
    
    # To co je poslední identifikovat s tím co je na začátku - tedy nám jde hlavně o směrovost vektoru
    minX = -L/2 + 1e-3
    maxX = L/2 - 1e-3  
    trf = occ.gp_Trsf.Translation(L * occ.X)
    mtrf = occ.gp_Trsf.Translation(-L * occ.X)    
    # RVE.faces[occ.X > maxX].Identify(RVE.faces[occ.X < minX], "periodicXm",
    #                               IdentificationType.PERIODIC, mtrf)
    RVE.faces[occ.X < minX].Identify(RVE.faces[occ.X > maxX], "periodicXp",
                                  IdentificationType.PERIODIC, trf)
  
    minY = -L/2 + 1e-3
    maxY = L/2 - 1e-3  
    trf = occ.gp_Trsf.Translation(L * occ.Y)
    RVE.faces[occ.Y < minY].Identify(RVE.faces[occ.Y > maxY], "periodicY",
                                  IdentificationType.PERIODIC, trf)   
  
    
    minZ = -L/2 + 1e-3
    maxZ = L/2 - 1e-3  
    trf = occ.gp_Trsf.Translation(L * occ.Z)
    RVE.faces[occ.Z < minZ].Identify(RVE.faces[occ.Z > maxZ], "periodicZ",
                                  IdentificationType.PERIODIC, trf)

    posX = occ.X >= L/2.
    negX = occ.X <= -L/2.

    posY = occ.Y >= L/2.
    negY = occ.Y <= -L/2.

    posZ = occ.Z >= L/2.
    negZ = occ.Z <= -L/2.

    RVE.faces[posX].name = 'pos_x'
    RVE.faces[posX].col = (0, 0, 1)
    RVE.faces[negX].name = 'neg_x'
    RVE.faces[negX].col = (0, 0, 1)

    RVE.faces[posY].name = 'pos_y'
    RVE.faces[posY].col = (0, 0, 1)
    RVE.faces[negY].name = 'neg_y'
    RVE.faces[negY].col = (0, 0, 1)

    RVE.faces[posZ].name = 'pos_z'
    RVE.faces[posZ].col = (0, 0, 1)
    RVE.faces[negZ].name = 'neg_z'
    RVE.faces[negZ].col = (0, 0, 1)

    geo = occ.OCCGeometry(RVE)
    mesh = Mesh(geo.GenerateMesh(minh=minh, maxh=maxh))
    mesh.Curve(mesh_order)
    return mesh


def Cell12Box(L, r, K, mesh_order=2, minh=0.2, maxh=0.4):

 
    P1 = np.array([- L/2.,   - L/2., - L/2.])    
    P2 = np.array([  L/2.,     L/2.,   L/2.])

    RVE = occ.Box(occ.Pnt(P1), occ.Pnt(P2))
    
    
    # To co je poslední identifikovat s tím co je na začátku - tedy nám jde hlavně o směrovost vektoru
    minX = -L/2 + 1e-3
    maxX = L/2 - 1e-3  
    trf = occ.gp_Trsf.Translation(L * occ.X)  
    RVE.faces[occ.X < minX].Identify(RVE.faces[occ.X > maxX], "periodicXp",
                                  IdentificationType.PERIODIC, trf)
  
    minY = -L/2 + 1e-3
    maxY = L/2 - 1e-3  
    trf = occ.gp_Trsf.Translation(L * occ.Y)
    RVE.faces[occ.Y < minY].Identify(RVE.faces[occ.Y > maxY], "periodicY",
                                  IdentificationType.PERIODIC, trf)   
  
    
    minZ = -L/2 + 1e-3
    maxZ = L/2 - 1e-3  
    trf = occ.gp_Trsf.Translation(L * occ.Z)
    RVE.faces[occ.Z < minZ].Identify(RVE.faces[occ.Z > maxZ], "periodicZ",
                                  IdentificationType.PERIODIC, trf)

    posX = occ.X >= L/2.
    negX = occ.X <= -L/2.

    posY = occ.Y >= L/2.
    negY = occ.Y <= -L/2.

    posZ = occ.Z >= L/2.
    negZ = occ.Z <= -L/2.

    RVE.faces[posX].name = 'pos_x'
    RVE.faces[posX].col = (0, 0, 1)
    RVE.faces[negX].name = 'neg_x'
    RVE.faces[negX].col = (0, 0, 1)

    RVE.faces[posY].name = 'pos_y'
    RVE.faces[posY].col = (0, 0, 1)
    RVE.faces[negY].name = 'neg_y'
    RVE.faces[negY].col = (0, 0, 1)

    RVE.faces[posZ].name = 'pos_z'
    RVE.faces[posZ].col = (0, 0, 1)
    RVE.faces[negZ].name = 'neg_z'
    RVE.faces[negZ].col = (0, 0, 1)

    geo = occ.OCCGeometry(RVE)
    mesh = Mesh(geo.GenerateMesh(minh=minh, maxh=maxh))
    mesh.Curve(mesh_order)
    return mesh



def Cell10(L, r, mesh_order=2, minh=0.2, maxh=0.4):
    lp1 = np.array([-L/2., -L/2., -L/2.])
    lp2 = np.array([L/2., L/2., L/2.])
    RVE = line(lp1, lp2, r)

    lp3 = np.array([-L/2., +L/2., -L/2.])
    lp4 = np.array([+L/2., -L/2., +L/2.])
    RVE += line(lp3, lp4, r-1e-10)

    lp5 = np.array([-L/2., -L/2., +L/2.])
    lp6 = np.array([+L/2., +L/2., -L/2.])
    RVE += line(lp5, lp6, r)

    lp7 = np.array([-L/2., +L/2., +L/2.])
    lp8 = np.array([+L/2., -L/2., -L/2.])
    RVE += line(lp7, lp8, r)

    lp_outer = occ.Box(occ.Pnt(lp1-r), occ.Pnt(lp2 + r))
    lp_inner = occ.Box(occ.Pnt(lp1-0.1*r), occ.Pnt(lp2 + 0.1*r))
    GG = lp_outer-lp_inner
    RVE -= GG

    posX = occ.X >= L/2.
    negX = occ.X <= -L/2.

    posY = occ.Y >= L/2.
    negY = occ.Y <= -L/2.

    posZ = occ.Z >= L/2.
    negZ = occ.Z <= -L/2.

    negX_faces = RVE.faces[negX]
    posX_faces = RVE.faces[posX]

    for i in range(len(negX_faces)):
        ic = np.asarray(list(negX_faces[i].center))
        for j in range(len(posX_faces)):
            jc = np.asarray(list(posX_faces[j].center))
            if np.isclose(ic[1], jc[1]) and np.isclose(ic[2], jc[2]):
                negX_faces[i].Identify(posX_faces[j], "periodicX",
                                       IdentificationType.PERIODIC)

                break

    negY_faces = RVE.faces[negY]
    posY_faces = RVE.faces[posY]

    for i in range(len(negY_faces)):
        ic = np.asarray(list(negY_faces[i].center))
        for j in range(len(posY_faces)):
            jc = np.asarray(list(posY_faces[j].center))
            if np.isclose(ic[0], jc[0]) and np.isclose(ic[2], jc[2]):
                negY_faces[i].Identify(posY_faces[j], "periodicY",
                                       IdentificationType.PERIODIC)

                break

    negZ_faces = RVE.faces[negZ]
    posZ_faces = RVE.faces[posZ]
    for i in range(len(negZ_faces)):
        ic = np.asarray(list(negZ_faces[i].center))
        for j in range(len(posZ_faces)):
            jc = np.asarray(list(posZ_faces[j].center))
            if np.isclose(ic[0], jc[0]) and np.isclose(ic[1], jc[1]):
                negZ_faces[i].Identify(posZ_faces[j], "periodicZ",
                                       IdentificationType.PERIODIC)
                break

    RVE.faces[posX].name = 'pos_x'
    RVE.faces[posX].col = (0, 0, 1)
    RVE.faces[negX].name = 'neg_x'
    RVE.faces[negX].col = (0, 0, 1)

    RVE.faces[posY].name = 'pos_y'
    RVE.faces[posY].col = (1, 0, 0)
    RVE.faces[negY].name = 'neg_y'
    RVE.faces[negY].col = (1, 0, 0)

    RVE.faces[posZ].name = 'pos_z'
    RVE.faces[posZ].col = (0, 1, 0)
    RVE.faces[negZ].name = 'neg_z'
    RVE.faces[negZ].col = (0, 1, 0)

    geo = occ.OCCGeometry(RVE)
    mesh = Mesh(geo.GenerateMesh(minh=minh, maxh=maxh))
    mesh.Curve(mesh_order)
    return mesh


def get_engineering_constants(S):
    E1 = 1. / S[0, 0]
    E2 = 1. / S[1, 1]
    E3 = 1. / S[2, 2]

    nu23 = -S[2, 1] / S[1, 1]
    nu32 = -S[1, 2] / S[2, 2]

    nu31 = -S[0, 2] / S[2, 2]
    nu13 = -S[2, 0] / S[0, 0]

    nu12 = -S[1, 0] / S[0, 0]
    nu21 = -S[0, 1] / S[1, 1]

    G23 = 1. / S[3, 3]
    G13 = 1. / S[4, 4]
    G12 = 1. / S[5, 5]
    return {
        'E1': E1, 'E2': E2, 'E3': E3,
        'nu23': nu23, 'nu32': nu32,
        'nu31': nu31, 'nu13': nu13,
        'nu12': nu12, 'nu21': nu21,
        'G23': G23, 'G13': G13, 'G12': G12
    }


def ExternalCellOCC(file_name, mesh_order=2, minh=0.2, maxh=0.4):
    geo = occ.OCCGeometry(file_name)
    RVE = geo.shape
    RVE.faces.Min(occ.X).Identify(RVE.faces.Max(occ.X),
                                  "periodicX",
                                  IdentificationType.PERIODIC)

    RVE.faces.Min(occ.Y).Identify(RVE.faces.Max(occ.Y),
                                  "periodicY",
                                  IdentificationType.PERIODIC)

    RVE.faces.Min(occ.Z).Identify(RVE.faces.Max(occ.Z),
                                  "periodicZ",
                                  IdentificationType.PERIODIC)
    geo = occ.OCCGeometry(RVE)
    mesh = Mesh(geo.GenerateMesh(minh=minh, maxh=maxh))
    mesh.Curve(mesh_order)
    return mesh


def Cell20(L, r, mesh_order=2, minh=0.2, maxh=0.4):
    lp1 = occ.Pnt([-L/2, -L/2, -L/2])
    lp2 = occ.Pnt([L/2, L/2, L/2])
    matrix = occ.Box(lp1, lp2)
    phase = occ.Sphere(occ.Pnt([0., 0., 0.]), r)
    matrix.name = 'matrix'
    phase.name = 'phase'
    RVE = occ.Glue([matrix-phase, phase])

    RVE.faces.Min(occ.X).Identify(RVE.faces.Max(occ.X),
                                  "periodicX",
                                  IdentificationType.PERIODIC)

    RVE.faces.Min(occ.Y).Identify(RVE.faces.Max(occ.Y),
                                  "periodicY",
                                  IdentificationType.PERIODIC)

    RVE.faces.Min(occ.Z).Identify(RVE.faces.Max(occ.Z),
                                  "periodicZ",
                                  IdentificationType.PERIODIC)

    geo = occ.OCCGeometry(RVE)
    mesh = Mesh(geo.GenerateMesh(minh=minh, maxh=maxh))
    mesh.Curve(mesh_order)
    return mesh


def netgen2dolfinx(netgen_mesh):
    GDIM, SHAPE, DEGREE = 3, "tetrahedron", 1

    # only linear elements
    vertices = np.array([list(v.point) for v in netgen_mesh.vertices])

    connectivity = []
    for el in netgen_mesh.Elements(ng.VOL):
        connectivity.append([v.nr for v in el.vertices])

    cell = ufl.Cell(SHAPE, geometric_dimension=GDIM)
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, DEGREE))
    return create_mesh(MPI.COMM_WORLD, np.array(connectivity),
                       vertices, domain)


def netgen2dolfin(netgen_mesh):
    MESH_FILE_NAME = 'mesh.xdmf'
    # only linear elements
    vertices = np.array([list(v.point) for v in netgen_mesh.vertices])

    connectivity = []
    mat_ids = []
    materials = netgen_mesh.GetMaterials()
    for i, el in enumerate(netgen_mesh.Elements(ng.VOL)):
        connectivity.append([v.nr for v in el.vertices])
        mat_ids.append(0)
        for j, material in enumerate(materials):
            if el.mat == material:
                mat_ids[i] = j

    mesh = meshio.Mesh(vertices, {"tetra": connectivity},
                       cell_data={'materials': [mat_ids]})
    mesh.write(MESH_FILE_NAME)

    mesh = df.Mesh()
    with df.XDMFFile(MESH_FILE_NAME) as infile:
        infile.read(mesh)
    return mesh


def macrostrain(v=1.):
    S11 = ng.CoefficientFunction((v, 0, 0,
                                  0, 0, 0,
                                  0, 0, 0), dims=(3, 3))

    S22 = ng.CoefficientFunction((0, 0, 0,
                                  0, v, 0,
                                  0, 0, 0), dims=(3, 3))

    S33 = ng.CoefficientFunction((0, 0, 0,
                                  0, 0, 0,
                                  0, 0, v), dims=(3, 3))

    S23 = ng.CoefficientFunction((0, 0, 0,
                                  0, 0, v/2,
                                  0, v/2, 0), dims=(3, 3))

    S13 = ng.CoefficientFunction((0, 0, v/2,
                                  0, 0, 0,
                                  v/2, 0, 0), dims=(3, 3))

    S12 = ng.CoefficientFunction((0, v/2, 0,
                                  v/2, 0, 0,
                                  0, 0, 0), dims=(3, 3))

    return [S11, S22, S33, S23, S13, S12]


def anisotropy_indicator(C):
    return 2 * C[3, 3] / (C[0, 0] - C[0, 1])


def rotate(C, alpha, beta, gamma):
    R = Rotation.from_euler('xyz', [alpha, beta, gamma])
    return rotate_elastic_constants(C, R.as_matrix())


def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


def polar_plot(C, c_name, res=1000, axis='x'):
    phis = np.linspace(0, 2*np.pi, res)
    rs = np.zeros_like(phis)
    for i, phi in enumerate(phis):
        if axis == 'x':
            S = np.linalg.inv(rotate(C, phi, 0., 0.))
        elif axis == 'y':
            S = np.linalg.inv(rotate(C, 0, phi, 0))
        elif axis == 'z':
            S = np.linalg.inv(rotate(C, 0, 0, phi))

        H = get_engineering_constants(S)
        rs[i] = H[c_name]
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(phis, rs)
    return phis, rs


def material_quadric(C, c_name, res=100, radius=0.5, fixed_axis='x'):
    sphere = pv.Sphere(radius=radius, theta_resolution=res, phi_resolution=res)
    clus = pyacvd.Clustering(sphere)
    clus.cluster(int(len(sphere.points)/5))
    sphere = clus.create_mesh()

    phis, thetas, r = cart2sph(sphere.points[:, 0],
                               sphere.points[:, 1],
                               sphere.points[:, 2])
    N = np.zeros(len(phis))
    for i in trange(len(phis), desc='computing rotations'):
        if fixed_axis=='x':
            S = np.linalg.inv(rotate(C, 0, phis[i], thetas[i]))
        elif fixed_axis=='y':
            S = np.linalg.inv(rotate(C, phis[i], 0, thetas[i]))
        elif fixed_axis=='z':
            S = np.linalg.inv(rotate(C, phis[i], thetas[i], 0))
                
        H = get_engineering_constants(S)
        N[i] = H[c_name]

    r = (N - N.min()) / (N.max()-N.min()) * radius
    x, y, z = sph2cart(phis, thetas, r)
    sphere.points = np.array([x, z, y]).T
    sphere.point_data[c_name] = N
    sphere.smooth()
    return sphere


if __name__ == '__main__':
    # mesh = Cell10(1.0, 0.15, mesh_order=1, minh=0.002, maxh=0.04)
    # mesh = ExternalCellOCC('RECA3D.step', mesh_order=1, minh=0.002, maxh=0.1)
    # mesh = Cell50Block(1, 4, 1, 1.0, 0.15, mesh_order=1, minh=0.004, maxh=0.06)
    # mesh = Cell50(1.0, 0.05, mesh_order=1, minh=0.004, maxh=0.06)    
    mesh = Cell21X(1.0, 0.1, mesh_order=1, minh=0.003, maxh=0.03)
    # mesh = Cell22(1.0, 0.05, mesh_order=1, minh=0.001, maxh=0.1)
    # mesh = Cell23(1.0, 0.05, 1.5, mesh_order=1, minh=0.001, maxh=0.1)
    # mesh = Cell23X(1.0, 0.05, 1.5, mesh_order=1, minh=0.001, maxh=0.1)
    # mesh = Cell20(1., 0.15, mesh_order=1, minh=0.004, maxh=0.06)
    # mesh = Cell24X(1.0, 0.05, 0.16, mesh_order=1, minh=0.001, maxh=0.1)
    # mesh = Cell12Box(1.0, 0.05, 0.16, mesh_order=1, minh=0.001, maxh=0.1)
    mesh = netgen2dolfin(mesh)
    
    domain = ExternalCell2Dolfin(mesh)
    domain.save_mesh('mesh.pvd')
    domain.check_periodic()
    
