import time
import netgen.occ as occ
from netgen.occ import gp_Trsf
from netgen.meshing import IdentificationType
from ngsolve import Mesh
import ngsolve as ng
import numpy as np
import pandas as pd
import dolfin as df
from utils2_a import (Cell10, Cell21X, Cell50, Cell50Block, ExternalCell2Dolfin, ExternalCellOCC,
                   anisotropy_indicator, get_engineering_constants,
                   material_quadric, netgen2dolfin, polar_plot)
from multiprocessing import Pool


class PeriodicBoundary(df.SubDomain):
    def __init__(self, mesh):
        df.SubDomain.__init__(self)
        self.TOL = 1e-15
        self.xmin, self.ymin, self.zmin = mesh.coordinates().min(0)
        self.xmax, self.ymax, self.zmax = mesh.coordinates().max(0)

    def inside(self, x, on_boundary):
        return bool(on_boundary and
                    (
                        (df.near(x[0], self.xmin, self.TOL) and not (
                            df.near(x[1], self.ymax, self.TOL) or df.near(x[2], self.zmax, self.TOL)))
                        or (df.near(x[1], self.ymin, self.TOL) and not (df.near(x[0], self.xmax, self.TOL) or df.near(x[2], self.zmax, self.TOL)))
                        or (df.near(x[2], self.zmin, self.TOL) and not (df.near(x[0], self.xmin, self.TOL) or df.near(x[1], self.ymax, self.TOL)))
                    ))

    def map(self, x, y):
        if df.near(x[0], self.xmin, self.TOL) and df.near(x[1], self.ymin, self.TOL) and df.near(x[2], self.zmax, self.TOL):
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2] - (self.zmax-self.zmin)
        elif df.near(x[0], self.xmin, self.TOL) and df.near(x[1], self.ymax, self.TOL) and df.near(x[2], self.zmin, self.TOL):
            y[0] = x[0]
            y[1] = x[1] - (self.ymax-self.ymin)
            y[2] = x[2]
        elif df.near(x[0], self.xmin, self.TOL) and df.near(x[1], self.ymax, self.TOL) and df.near(x[2], self.zmax, self.TOL):
            y[0] = x[0]
            y[1] = x[1] - (self.ymax-self.ymin)
            y[2] = x[2] - (self.zmax-self.zmin)
        elif df.near(x[0], self.xmax, self.TOL) and df.near(x[1], self.ymin, self.TOL) and df.near(x[2], self.zmin, self.TOL):
            y[0] = x[0] - (self.xmax-self.xmin)
            y[1] = x[1]
            y[2] = x[2]
        elif df.near(x[0], self.xmax, self.TOL) and df.near(x[1], self.ymin, self.TOL) and df.near(x[2], self.zmax, self.TOL):
            y[0] = x[0] - (self.xmax-self.xmin)
            y[1] = x[1]
            y[2] = x[2] - (self.zmax-self.zmin)
        elif df.near(x[0], self.xmax, self.TOL) and df.near(x[1], self.ymax, self.TOL) and df.near(x[2], self.zmin, self.TOL):
            y[0] = x[0] - (self.xmax-self.xmin)
            y[1] = x[1] - (self.ymax-self.ymin)
            y[2] = x[2]
        elif df.near(x[0], self.xmax, self.TOL) and df.near(x[1], self.ymax, self.TOL) and df.near(x[2], self.zmax, self.TOL):
            y[0] = x[0] - (self.xmax-self.xmin)
            y[1] = x[1] - (self.ymax-self.ymin)
            y[2] = x[2] - (self.zmax-self.zmin)
        elif df.near(x[0], self.xmin, self.TOL) and df.near(x[1], self.ymin, self.TOL) and df.near(x[2], self.zmin, self.TOL):
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2]
        # edge
        elif df.near(x[0], self.xmax, self.TOL) and df.near(x[2], self.zmin, self.TOL):
            y[0] = x[0] - (self.xmax-self.xmin)
            y[1] = x[1]
            y[2] = x[2]
        elif df.near(x[0], self.xmax, self.TOL) and df.near(x[2], self.zmax, self.TOL):
            y[0] = x[0] - (self.xmax-self.xmin)
            y[1] = x[1]
            y[2] = x[2] - (self.zmax-self.zmin)
        elif df.near(x[0], self.xmin, self.TOL) and df.near(x[2], self.zmax, self.TOL):
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2] - (self.zmax-self.zmin)
        elif df.near(x[1], self.xmin, self.TOL) and df.near(x[2], self.zmax, self.TOL):
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2] - (self.zmax-self.zmin)
        elif df.near(x[1], self.ymax, self.TOL) and df.near(x[2], self.zmax, self.TOL):
            y[0] = x[0]
            y[1] = x[1] - (self.ymax-self.ymin)
            y[2] = x[2] - (self.zmax-self.zmin)
        elif df.near(x[1], self.ymax, self.TOL) and df.near(x[2], self.zmin, self.TOL):
            y[0] = x[0]
            y[1] = x[1] - (self.ymax-self.ymin)
            y[2] = x[2]
        elif df.near(x[0], self.xmax, self.TOL) and df.near(x[1], self.ymin, self.TOL):
            y[0] = x[0] - (self.xmax-self.xmin)
            y[1] = x[1]
            y[2] = x[2]
        elif df.near(x[0], self.xmax, self.TOL) and df.near(x[1], self.ymax, self.TOL):
            y[0] = x[0] - (self.xmax-self.xmin)
            y[1] = x[1] - (self.ymax-self.ymin)
            y[2] = x[2]
        elif df.near(x[0], 0, self.TOL) and df.near(x[1], self.ymax, self.TOL):
            y[0] = x[0]
            y[1] = x[1] - (self.ymax-self.ymin)
            y[2] = x[2]
        # surface
        elif df.near(x[0], self.xmax, self.TOL):
            y[0] = x[0] - (self.xmax-self.xmin)
            y[1] = x[1]
            y[2] = x[2]
        elif df.near(x[1], self.ymax, self.TOL):
            y[0] = x[0]
            y[1] = x[1] - (self.ymax-self.ymin)
            y[2] = x[2]
        # elif df.near(x[2], self.zmax, self.TOL):
        else:
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2] - (self.zmax-self.zmin)

def contains_zero(matrix, threshold=1e-5):
    """Check if the matrix contains a value close to zero."""
    return np.any(np.abs(matrix) < threshold)

def strain2voigt(eps):
    return df.as_vector([eps[0, 0], eps[1, 1], eps[2, 2], 
                         2*eps[1, 2], 2*eps[0, 2], 2*eps[0, 1]])


def stress2voigt(s):
    return df.as_vector([s[0, 0], s[1, 1], s[2, 2], 
                         s[1, 2], s[0, 2], s[0, 1]])


def voigt2stress(S):
    ss = [[S[0], S[5], S[4]],
          [S[5], S[1], S[3]],
          [S[4], S[3], S[2]]]
    return df.as_tensor(ss)


def macro_strain(i):
    Gamm_Voight = np.zeros(6)
    Gamm_Voight[i] = 1
    return np.array([[Gamm_Voight[0],    Gamm_Voight[5]/2., Gamm_Voight[4]/2.],
                     [Gamm_Voight[5]/2., Gamm_Voight[1],    Gamm_Voight[3]/2.],
                     [Gamm_Voight[4]/2., Gamm_Voight[3]/2., Gamm_Voight[2]]])


def macro_stress(i):
    Gamm_Voight = np.zeros(6)
    Gamm_Voight[i] = 1
    return np.array([[Gamm_Voight[0], Gamm_Voight[5], Gamm_Voight[4]],
                     [Gamm_Voight[5], Gamm_Voight[1], Gamm_Voight[3]],
                     [Gamm_Voight[4], Gamm_Voight[3], Gamm_Voight[2]]])


class MicroFESolver_penalty(object):
    def __init__(self, domain, disp_order=2,
                 FILE_SAVE='homogenized_results.xdmf'):
        self.domain = domain
        self.mesh = domain.mesh
        self.gdim = self.mesh.geometric_dimension()
        Ue = df.VectorElement('CG', self.mesh.ufl_cell(), disp_order)
        Ve = df.FunctionSpace(self.mesh, Ue,
                              constrained_domain=PeriodicBoundary(self.mesh))

        self.du = df.TestFunction(Ve)
        self.u_ = df.TrialFunction(Ve)
        self.u_sol = df.Function(Ve)

        self.file_results = df.XDMFFile(FILE_SAVE)

        self.file_results.parameters["flush_output"] = True
        self.file_results.parameters["functions_share_mesh"] = True

        self.K = df.PETScMatrix()
        self.b = df.PETScVector()
        self.solver = df.LUSolver(self.K, "mumps")
        self.solver.parameters["symmetric"] = True
        self.vol_of_solid = df.assemble(df.Constant(1.0) * df.dx(self.mesh))

    def set_bulk_mat(self, E, nu, rho):
        self.lmbda = E*nu/(1+nu)/(1-2*nu)
        self.mu = E / 2./(1+nu)
        self.rho = rho

    def _eps(self, v):
        return df.sym(df.grad(v))

    def _sigma(self, v, Eps):
        return self.lmbda * df.tr(self._eps(v)
                                  + Eps) * df.Identity(self.gdim) \
            + 2 * self.mu * (self._eps(v) + Eps)

    def init_weak_form(self, penalty):

        self.Gamm_bar = df.Constant(((0, 0, 0), (0, 0, 0), (0, 0, 0)))
        self.a = df.inner(self._sigma(self.u_, self.Gamm_bar),
                          self._eps(self.du)) * df.dx
        self.a += penalty*df.inner(self.u_, self.du) * df.dx
        tic = time.time()
        print('Assembling stiffness...', end=" ", flush=True)
        L_w, self.f_w = df.lhs(self.a), df.rhs(self.a)
        df.assemble(L_w, tensor=self.K)
        tac = time.time()
        print(f'DONE in {(tac-tic):0.3f}s')

    def compute_macro_tangent(self):
        self.C_hom = np.zeros((6, 6))
        for i, case in enumerate(['xx', 'yy', 'zz', 'yz', 'xz', 'xy']):
            self.Gamm_bar.assign(df.Constant(macro_strain(i)))
            df.assemble(self.f_w, tensor=self.b)
            if i == 0:
                print('Matrix factorising...', end=" ", flush=True)
                tic = time.time()
                self.solver.solve(self.u_sol.vector(), self.b)
                tac = time.time()
                print(f'DONE in {(tac-tic):0.3f}s')

            tic = time.time()
            print(f'Solving for {case}...', end=" ", flush=True)
            self.solver.solve(self.u_sol.vector(), self.b)
            tac = time.time()
            print(f'DONE in {(tac-tic):0.3f}s')
            self.u_sol.rename(f'u_{case}', '')
            self.file_results.write(self.u_sol, 0)
            sigma_til = np.zeros((6,))
            for k in range(sigma_til.shape[0]):
                sigma_til[k] = float(df.assemble(stress2voigt(
                    self._sigma(self.u_sol, self.Gamm_bar))[k] * df.dx))
            self.C_hom[i, :] = sigma_til.copy()
        self.file_results.close()
        return self.C_hom / self.vol_of_solid


class MicroFESolver_lagrange(object):
    def __init__(self, domain, disp_order=2):
        self.domain = domain
        self.mesh = domain.mesh
        self.gdim = self.mesh.geometric_dimension()
        Ue = df.VectorElement('CG', self.mesh.ufl_cell(), disp_order)
        Re = df.VectorElement('R', self.mesh.ufl_cell(), 0)
        We = df.MixedElement([Ue, Re])

        Ve = df.FunctionSpace(self.mesh, We,
                              constrained_domain=PeriodicBoundary(self.mesh))

        self.du, self.dlamb = df.TestFunctions(Ve)
        self.u_, self.lamb_ = df.TrialFunctions(Ve)
        self.u_lamb = df.Function(Ve)

        self.file_results = df.XDMFFile('homogen_results.xdmf')

        self.file_results.parameters["flush_output"] = True
        self.file_results.parameters["functions_share_mesh"] = True

        self.K = df.PETScMatrix()
        self.b = df.PETScVector()
        self.solver = df.LUSolver(self.K, "mumps")
        self.solver.parameters["symmetric"] = True
        self.vol_of_solid = df.assemble(df.Constant(1.0) * df.dx(self.mesh))
        self.Gamm_bar = df.Constant(((0, 0, 0), (0, 0, 0), (0, 0, 0)))

    def set_bulk_mat(self, E, nu, rho):
        self.lmbda = E*nu/(1+nu)/(1-2*nu)
        self.mu = E / 2./(1+nu)
        self.rho = rho

    def _eps(self, v):
        return df.sym(df.grad(v))

    def _sigma(self, v, Eps):
        return self.lmbda * df.tr(self._eps(v)
                                  + Eps) * df.Identity(self.gdim) \
            + 2 * self.mu * (self._eps(v) + Eps)

    def init_weak_form(self):
        self.Gamm_bar = df.Constant(((0, 0, 0), (0, 0, 0), (0, 0, 0)))
        self.a = df.inner(self._sigma(self.u_, self.Gamm_bar),
                          self._eps(self.du)) * df.dx
        self.a += (df.inner(self.lamb_, self.du) +
                   df.inner(self.dlamb, self.u_)) * df.dx
        tic = time.time()
        print('Assembling stiffness...', end=" ", flush=True)
        L_w, self.f_w = df.lhs(self.a), df.rhs(self.a)
        df.assemble_mixed(L_w, tensor=self.K)
        tac = time.time()
        print(f'DONE in {(tac-tic):0.3f}s')

    def compute_macro_tangent(self):
        self.C_hom = np.zeros((6, 6))
        for i, case in enumerate(['xx', 'yy', 'zz', 'yz', 'xz', 'xy']):
            self.Gamm_bar.assign(df.Constant(macro_strain(i)))
            df.assemble_mixed(self.f_w, tensor=self.b)
            if i == 0:
                print('Matrix factorising...', end=" ", flush=True)
                tic = time.time()
                self.solver.solve(self.u_lamb.vector(), self.b)
                tac = time.time()
                print(f'DONE in {(tac-tic):0.3f}s')

            tic = time.time()
            print(f'Solving for {case}...', end=" ", flush=True)
            self.solver.solve(self.u_lamb.vector(), self.b)
            tac = time.time()
            print(f'DONE in {(tac-tic):0.3f}s')

            u_, lamb_ = self.u_lamb.split(True)
            u_.rename(f'u_{case}', '')
            self.file_results.write(u_, 0)
            sigma_til = np.zeros((6,))
            for k in range(sigma_til.shape[0]):
                sigma_til[k] = float(df.assemble(stress2voigt(
                    self._sigma(u_, self.Gamm_bar))[k] * df.dx))
            self.C_hom[i, :] = sigma_til.copy()
        self.file_results.close()
        return self.C_hom / self.vol_of_solid


class Deformer(object):
    def __init__(self, mesh, disp_order=1):
        self.domain = domain
        self.mesh = domain.mesh
        self.gdim = self.mesh.geometric_dimension()
        Ue = df.VectorElement('CG', self.mesh.ufl_cell(), disp_order)
        self.Ve = df.FunctionSpace(self.mesh, Ue)

        self.du = df.TestFunction(self.Ve)
        self.u_ = df.TrialFunction(self.Ve)
        self.u_sol = df.Function(self.Ve)

        self.file_results = df.XDMFFile('deformation_results.xdmf')

        self.file_results.parameters["flush_output"] = True
        self.file_results.parameters["functions_share_mesh"] = True

        self.K = df.PETScMatrix()
        self.b = df.PETScVector()
        self.solver = df.LUSolver(self.K, "mumps")
        self.solver.parameters["symmetric"] = True

        self.domain.get_markers()
        self.markers = self.domain.boundaries
        self.BDN = []
        for marker in self.markers:
            self.BDN.append(df.DirichletBC(
                self.Ve, df.Constant((0, 0, 0)), self.markers[marker]))

        self.EPS = 1e-1
        self.X = df.MeshCoordinates(self.mesh)
        self.i = 0

    def set_bulk_mat(self, E, nu, rho):
        self.lmbda = E*nu/(1+nu)/(1-2*nu)
        self.mu = E / 2./(1+nu)
        self.rho = rho

    def _eps(self, v):
        return df.sym(df.grad(v))

    def _sigma(self, v):
        return self.lmbda * df.tr(self._eps(v)) * df.Identity(self.gdim) \
            + 2 * self.mu * (self._eps(v))

    def init_weak_form(self):

        self.a = df.inner(self._sigma(self.u_), self._eps(self.du)) * df.dx
        tic = time.time()
        print('Assembling stiffness...', end=" ", flush=True)
        df.assemble(self.a, tensor=self.K)
        tac = time.time()
        print(f'DONE in {(tac-tic):0.3f}s')

        for bnd in self.BDN:
            bnd.apply(self.K)

    def deform_with(self, bending, torsion, method="analytic", KX=0., KY=0., KZ=0., TX=0., TY=0., TZ=0., charlen=1.):
        b = df.Constant(bending)
        if method == "analytic":
            norX = df.Expression("(pow(charlen,2)-pow(x[0],2))/pow(charlen,2)", charlen=charlen/2, degree=1)
            norY = df.Expression("(pow(charlen,2)-pow(x[1],2))/pow(charlen,2)", charlen=charlen/2, degree=1)
            norZ = df.Expression("(pow(charlen,2)-pow(x[2],2))/pow(charlen,2)", charlen=charlen/2, degree=1)

            expX = df.Expression("exp( -pow(x[0],2)/(1+charlen) ) / (2*pow(charlen,4))", charlen=charlen/2, degree=1)
            expY = df.Expression("exp( -pow(x[1],2)/(1+charlen) ) / (2*pow(charlen,4))", charlen=charlen/2, degree=1)
            expZ = df.Expression("exp( -pow(x[2],2)/(1+charlen) ) / (2*pow(charlen,4))", charlen=charlen/2, degree=1)

            kvXY = df.Expression("(pow(charlen,2)-pow(x[0],2)) * (pow(charlen,2)-pow(x[1],2))", charlen=charlen/2, degree=1)
            kvYZ = df.Expression("(pow(charlen,2)-pow(x[1],2)) * (pow(charlen,2)-pow(x[2],2))", charlen=charlen/2, degree=1)
            kvXZ = df.Expression("(pow(charlen,2)-pow(x[0],2)) * (pow(charlen,2)-pow(x[2],2))", charlen=charlen/2, degree=1)
            # trans_X = df.Expression("0.", degree=1)
            trans_X = df.Expression("TX * norX * ( expY * kvYZ + expZ * kvYZ)", kvYZ=kvYZ, norX=norX, expY=expY, expZ=expZ, TX=TX, charlen=charlen/2, degree=1)
            trans_Y = df.Expression("TY * norY * ( expX * kvXZ + expZ * kvXZ)", kvXZ=kvXZ, norY=norY, expX=expX, expZ=expZ, TY=TY, charlen=charlen/2, degree=1)
            trans_Z = df.Expression("TZ * norZ * ( expX * kvXY + expY * kvXY)", kvXY=kvXY, norZ=norZ, expX=expX, expY=expY, TZ=TZ, charlen=charlen/2, degree=1)
            
            
            # normalizace podle definicniho oboru D = <0;1>
            # normalizace podle maxima fce f(x,y) na definicnim odboru D -> 8/27
            # prepocet na 1 DEG
            extrem01 = 27/8
            phiX = df.Expression("pow(abs(x[1]),2)*pow(charlen-abs(x[2]),2)*pow(charlen-abs(x[1]),2) + (pow(abs(x[2]),2)*pow(charlen-abs(x[2]),2)*pow(charlen-abs(x[1]),2))/pow(charlen,3)",charlen=charlen/2, degree=1)
            phiX = df.Expression("phiX * extrem01 * pi/180",phiX = phiX, extrem01=extrem01, degree=1)
            phiY = df.Expression("pow(abs(x[2]),2)*pow(charlen-abs(x[0]),2)*pow(charlen-abs(x[2]),2) + (pow(abs(x[2]),2)*pow(charlen-abs(x[0]),2)*pow(charlen-abs(x[2]),2))/pow(charlen,3)",charlen=charlen/2, degree=1)
            phiY = df.Expression("phiY * extrem01 * pi/180",phiY = phiY, extrem01=extrem01, degree=1)  
            phiZ = df.Expression("pow(abs(x[0]),2)*pow(charlen-abs(x[0]),2)*pow(charlen-abs(x[1]),2) + (pow(abs(x[1]),2)*pow(charlen-abs(x[0]),2)*pow(charlen-abs(x[1]),2))/pow(charlen,3)",charlen=charlen/2, degree=1)
            phiZ = df.Expression("phiZ * extrem01 * pi/180",phiZ = phiZ, extrem01=extrem01, degree=1)
                      
            


            rot_x = df.Expression((
                    "0",
                    "x[1]-(x[1]*cos(KX*phiX) - x[2]*sin(KX*phiX))",
                    "x[2]-(x[1]*sin(KX*phiX) + x[2]*cos(KX*phiX))"
                ), phiX=phiX, KX=KX, degree=1)
            rot_y = df.Expression((
                    "x[0]-(x[0]*cos(KY*phiY) + x[2]*sin(KY*phiY))",
                    "0",
                    "x[2]-(-x[0]*sin(KY*phiY) + x[2]*cos(KY*phiY))"
                ), phiY=phiY, KY=KY, degree=1)
            rot_z = df.Expression((
                    "x[0]-(x[0]*cos(KZ*phiZ) - x[1]*sin(KZ*phiZ))",
                    "x[1]-(x[0]*sin(KZ*phiZ) + x[1]*cos(KZ*phiZ))",
                    "0"
                ), phiZ=phiZ, KZ=KZ, degree=1)
 
            
            # analytic deformation of original mesh
            # disp = df.Expression(
            #                 ( "x[0]*KX*KY*cos(phiX)*cos(phiY) + x[1]*(KX*KY*KZ*cos(phiX)*sin(phiY)*sin(phiZ)-KX*KZ*sin(phiX)*cos(phiZ)) + x[2]*(KX*KY*KZ*cos(phiX)*sin(phiY)*cos(phiZ)+KX*KZ*sin(phiX)*sin(phiZ))",
            #                   "x[0]*KX*KY*sin(phiX)*cos(phiY) + x[1]*(KX*KY*KZ*sin(phiX)*sin(phiY)*sin(phiZ)+KX*KZ*cos(phiX)*cos(phiZ)) + x[2]*(KX*KY*KZ*sin(phiX)*sin(phiY)*cos(phiZ)-KX*KZ*cos(phiX)*sin(phiZ))",
            #                   "-x[0]*KY*sin(phiY)             + x[1]*KY*KZ*cos(phiY)*sin(phiZ)                                          + x[2]*KY*KZ*cos(phiY)*cos(phiZ)"
            #                  ), phiX=phiX, phiY=phiY, phiZ=phiZ, KX=KX, KY=KY, KZ=KZ, rot_x=rot_x, rot_y=rot_y, rot_z=rot_z, trans_X=trans_X, trans_Y=trans_Y, trans_Z=trans_Z, degree=1)
            # disp = df.Expression(("trans_X", "trans_Y", "trans_Z"), trans_X=trans_X, trans_Y=trans_Y, trans_Z=trans_Z, degree=1)
            # disp = df.Expression(("trans_X", "trans_Y", "trans_Z"), trans_X=trans_X, trans_Y=trans_Y, trans_Z=trans_Z, degree=1)
            disp = rot_x + rot_y + rot_z
            disp = df.Expression(
                                ("rot_x[0] + rot_y[0] + rot_z[0] + trans_X",
                                 "rot_x[1] + rot_y[1] + rot_z[1] + trans_Y",
                                 "rot_x[2] + rot_y[2] + rot_z[2] + trans_Z",
                                ), phiX=phiX, phiY=phiY, phiZ=phiZ, KX=KX, KY=KY, KZ=KZ, rot_x=rot_x, rot_y=rot_y, rot_z=rot_z, trans_X=trans_X, trans_Y=trans_Y, trans_Z=trans_Z, degree = 1)
            df.ALE.move(self.mesh, disp)
            self.i += 1


        elif method == "numerical":  

            norm_z = df.sqrt(self.X[0]**2 + self.X[1]**2) + df.DOLFIN_EPS
            rot_z = df.as_vector(
                (-self.X[1]/norm_z, self.X[0]/norm_z, 0)) * torsion[2]

            norm_y = df.sqrt(self.X[0]**2 + self.X[2]**2) + df.DOLFIN_EPS
            rot_y = df.as_vector(
                (-self.X[2]/norm_y, 0, self.X[0]/norm_y)) * torsion[1]

            norm_x = df.sqrt(self.X[1]**2 + self.X[2]**2) + df.DOLFIN_EPS
            rot_x = df.as_vector(
                (0, -self.X[2]/norm_x, self.X[1]/norm_x)) * torsion[0]

            l = df.dot((b + rot_x + rot_y + rot_z), self.du) * df.dx
            df.assemble(l, tensor=self.b)

            for bnd in self.BDN:
                bnd.apply(self.b)

            if self.i == 0:
                print('Matrix factorising...', end=" ", flush=True)
                tic = time.time()
                self.solver.solve(self.u_sol.vector(), self.b)
                tac = time.time()
                print(f'DONE in {(tac-tic):0.3f}s')
            else:
                tic = time.time()
                print(f'Solving...', end=" ", flush=True)
                self.solver.solve(self.u_sol.vector(), self.b)
                tac = time.time()
                print(f'DONE in {(tac-tic):0.3f}s')

            self.u_sol.rename(f'deformation', '')
            self.file_results.write(self.u_sol)

            df.ALE.move(self.mesh, self.u_sol)
            self.i += 1

    def undeform(self):
        self.u_sol.vector()[:] *= -1
        df.ALE.move(self.mesh, self.u_sol)

class ForgotDeformer(object):
    def __init__(self, mesh, disp_order=1):
        self.domain = domain
        self.mesh = domain.mesh
        self.gdim = self.mesh.geometric_dimension()
        Ue = df.VectorElement('CG', self.mesh.ufl_cell(), disp_order)
        self.Ve = df.FunctionSpace(self.mesh, Ue)

        self.du = df.TestFunction(self.Ve)
        self.u_ = df.TrialFunction(self.Ve)
        self.u_sol = df.Function(self.Ve)

        # self.file_results = df.XDMFFile('deformation_results.xdmf')

        # self.file_results.parameters["flush_output"] = True
        # self.file_results.parameters["functions_share_mesh"] = True

        self.K = df.PETScMatrix()
        self.b = df.PETScVector()
        self.solver = df.LUSolver(self.K, "mumps")
        self.solver.parameters["symmetric"] = True

        self.domain.get_markers()
        self.markers = self.domain.boundaries
        self.BDN = []
        for marker in self.markers:
            self.BDN.append(df.DirichletBC(
                self.Ve, df.Constant((0, 0, 0)), self.markers[marker]))

        self.EPS = 1e-1
        self.X = df.MeshCoordinates(self.mesh)
        self.i = 0

    def set_bulk_mat(self, E, nu, rho):
        self.lmbda = E*nu/(1+nu)/(1-2*nu)
        self.mu = E / 2./(1+nu)
        self.rho = rho

    def _eps(self, v):
        return df.sym(df.grad(v))

    def _sigma(self, v):
        return self.lmbda * df.tr(self._eps(v)) * df.Identity(self.gdim) \
            + 2 * self.mu * (self._eps(v))

    def init_weak_form(self):

        self.a = df.inner(self._sigma(self.u_), self._eps(self.du)) * df.dx
        tic = time.time()
        print('Assembling stiffness...', end=" ", flush=True)
        df.assemble(self.a, tensor=self.K)
        tac = time.time()
        print(f'DONE in {(tac-tic):0.3f}s')

        for bnd in self.BDN:
            bnd.apply(self.K)

    def deform_with(self, bending, torsion, method="analytic", KX=0., KY=0., KZ=0., TX=0., TY=0., TZ=0., charlen=1.):
        b = df.Constant(bending)
        if method == "analytic":
            norX = df.Expression("(pow(charlen,2)-pow(x[0],2))/pow(charlen,2)", charlen=charlen/2, degree=1)
            norY = df.Expression("(pow(charlen,2)-pow(x[1],2))/pow(charlen,2)", charlen=charlen/2, degree=1)
            norZ = df.Expression("(pow(charlen,2)-pow(x[2],2))/pow(charlen,2)", charlen=charlen/2, degree=1)

            expX = df.Expression("exp( -pow(x[0],2)/(1+charlen) ) / (2*pow(charlen,4))", charlen=charlen/2, degree=1)
            expY = df.Expression("exp( -pow(x[1],2)/(1+charlen) ) / (2*pow(charlen,4))", charlen=charlen/2, degree=1)
            expZ = df.Expression("exp( -pow(x[2],2)/(1+charlen) ) / (2*pow(charlen,4))", charlen=charlen/2, degree=1)

            kvXY = df.Expression("(pow(charlen,2)-pow(x[0],2)) * (pow(charlen,2)-pow(x[1],2))", charlen=charlen/2, degree=1)
            kvYZ = df.Expression("(pow(charlen,2)-pow(x[1],2)) * (pow(charlen,2)-pow(x[2],2))", charlen=charlen/2, degree=1)
            kvXZ = df.Expression("(pow(charlen,2)-pow(x[0],2)) * (pow(charlen,2)-pow(x[2],2))", charlen=charlen/2, degree=1)
            # trans_X = df.Expression("0.", degree=1)
            trans_X = df.Expression("TX * norX * ( expY * kvYZ + expZ * kvYZ)", kvYZ=kvYZ, norX=norX, expY=expY, expZ=expZ, TX=TX, charlen=charlen/2, degree=1)
            trans_Y = df.Expression("TY * norY * ( expX * kvXZ + expZ * kvXZ)", kvXZ=kvXZ, norY=norY, expX=expX, expZ=expZ, TY=TY, charlen=charlen/2, degree=1)
            trans_Z = df.Expression("TZ * norZ * ( expX * kvXY + expY * kvXY)", kvXY=kvXY, norZ=norZ, expX=expX, expY=expY, TZ=TZ, charlen=charlen/2, degree=1)
            
            
            # normalizace podle definicniho oboru D = <0;1>
            # normalizace podle maxima fce f(x,y) na definicnim odboru D -> 8/27
            # prepocet na 1 DEG
            extrem01 = 27/8
            phiX = df.Expression("pow(abs(x[1]),2)*pow(charlen-abs(x[2]),2)*pow(charlen-abs(x[1]),2) + (pow(abs(x[2]),2)*pow(charlen-abs(x[2]),2)*pow(charlen-abs(x[1]),2))/pow(charlen,3)",charlen=charlen/2, degree=1)
            phiX = df.Expression("phiX * extrem01 * pi/180",phiX = phiX, extrem01=extrem01, degree=1)
            phiY = df.Expression("pow(abs(x[2]),2)*pow(charlen-abs(x[0]),2)*pow(charlen-abs(x[2]),2) + (pow(abs(x[2]),2)*pow(charlen-abs(x[0]),2)*pow(charlen-abs(x[2]),2))/pow(charlen,3)",charlen=charlen/2, degree=1)
            phiY = df.Expression("phiY * extrem01 * pi/180",phiY = phiY, extrem01=extrem01, degree=1)  
            phiZ = df.Expression("pow(abs(x[0]),2)*pow(charlen-abs(x[0]),2)*pow(charlen-abs(x[1]),2) + (pow(abs(x[1]),2)*pow(charlen-abs(x[0]),2)*pow(charlen-abs(x[1]),2))/pow(charlen,3)",charlen=charlen/2, degree=1)
            phiZ = df.Expression("phiZ * extrem01 * pi/180",phiZ = phiZ, extrem01=extrem01, degree=1)
                      
            


            rot_x = df.Expression((
                    "0",
                    "x[1]-(x[1]*cos(KX*phiX) - x[2]*sin(KX*phiX))",
                    "x[2]-(x[1]*sin(KX*phiX) + x[2]*cos(KX*phiX))"
                ), phiX=phiX, KX=KX, degree=1)
            rot_y = df.Expression((
                    "x[0]-(x[0]*cos(KY*phiY) + x[2]*sin(KY*phiY))",
                    "0",
                    "x[2]-(-x[0]*sin(KY*phiY) + x[2]*cos(KY*phiY))"
                ), phiY=phiY, KY=KY, degree=1)
            rot_z = df.Expression((
                    "x[0]-(x[0]*cos(KZ*phiZ) - x[1]*sin(KZ*phiZ))",
                    "x[1]-(x[0]*sin(KZ*phiZ) + x[1]*cos(KZ*phiZ))",
                    "0"
                ), phiZ=phiZ, KZ=KZ, degree=1)
 
            
            # analytic deformation of original mesh
            # disp = df.Expression(
            #                 ( "x[0]*KX*KY*cos(phiX)*cos(phiY) + x[1]*(KX*KY*KZ*cos(phiX)*sin(phiY)*sin(phiZ)-KX*KZ*sin(phiX)*cos(phiZ)) + x[2]*(KX*KY*KZ*cos(phiX)*sin(phiY)*cos(phiZ)+KX*KZ*sin(phiX)*sin(phiZ))",
            #                   "x[0]*KX*KY*sin(phiX)*cos(phiY) + x[1]*(KX*KY*KZ*sin(phiX)*sin(phiY)*sin(phiZ)+KX*KZ*cos(phiX)*cos(phiZ)) + x[2]*(KX*KY*KZ*sin(phiX)*sin(phiY)*cos(phiZ)-KX*KZ*cos(phiX)*sin(phiZ))",
            #                   "-x[0]*KY*sin(phiY)             + x[1]*KY*KZ*cos(phiY)*sin(phiZ)                                          + x[2]*KY*KZ*cos(phiY)*cos(phiZ)"
            #                  ), phiX=phiX, phiY=phiY, phiZ=phiZ, KX=KX, KY=KY, KZ=KZ, rot_x=rot_x, rot_y=rot_y, rot_z=rot_z, trans_X=trans_X, trans_Y=trans_Y, trans_Z=trans_Z, degree=1)
            # disp = df.Expression(("trans_X", "trans_Y", "trans_Z"), trans_X=trans_X, trans_Y=trans_Y, trans_Z=trans_Z, degree=1)
            # disp = df.Expression(("trans_X", "trans_Y", "trans_Z"), trans_X=trans_X, trans_Y=trans_Y, trans_Z=trans_Z, degree=1)
            disp = rot_x + rot_y + rot_z
            disp = df.Expression(
                                ("rot_x[0] + rot_y[0] + rot_z[0] + trans_X",
                                 "rot_x[1] + rot_y[1] + rot_z[1] + trans_Y",
                                 "rot_x[2] + rot_y[2] + rot_z[2] + trans_Z",
                                ), phiX=phiX, phiY=phiY, phiZ=phiZ, KX=KX, KY=KY, KZ=KZ, rot_x=rot_x, rot_y=rot_y, rot_z=rot_z, trans_X=trans_X, trans_Y=trans_Y, trans_Z=trans_Z, degree = 1)
            df.ALE.move(self.mesh, disp)
            self.i += 1


        elif method == "numerical":  

            norm_z = df.sqrt(self.X[0]**2 + self.X[1]**2) + df.DOLFIN_EPS
            rot_z = df.as_vector(
                (-self.X[1]/norm_z, self.X[0]/norm_z, 0)) * torsion[2]

            norm_y = df.sqrt(self.X[0]**2 + self.X[2]**2) + df.DOLFIN_EPS
            rot_y = df.as_vector(
                (-self.X[2]/norm_y, 0, self.X[0]/norm_y)) * torsion[1]

            norm_x = df.sqrt(self.X[1]**2 + self.X[2]**2) + df.DOLFIN_EPS
            rot_x = df.as_vector(
                (0, -self.X[2]/norm_x, self.X[1]/norm_x)) * torsion[0]

            l = df.dot((b + rot_x + rot_y + rot_z), self.du) * df.dx
            df.assemble(l, tensor=self.b)

            for bnd in self.BDN:
                bnd.apply(self.b)

            if self.i == 0:
                print('Matrix factorising...', end=" ", flush=True)
                tic = time.time()
                self.solver.solve(self.u_sol.vector(), self.b)
                tac = time.time()
                print(f'DONE in {(tac-tic):0.3f}s')
            else:
                tic = time.time()
                print(f'Solving...', end=" ", flush=True)
                self.solver.solve(self.u_sol.vector(), self.b)
                tac = time.time()
                print(f'DONE in {(tac-tic):0.3f}s')

            self.u_sol.rename(f'deformation', '')
            # self.file_results.write(self.u_sol)

            df.ALE.move(self.mesh, self.u_sol)
            self.i += 1

    def undeform(self):
        self.u_sol.vector()[:] *= -1
        df.ALE.move(self.mesh, self.u_sol)

def get_ElastoMatrix_from_pattern (L=1.0, r=0.06, mesh_order=1, minh=0.002, maxh=0.06, 
                                   E=2000.0, nu=0.3, rho=1., 
                                   t_xyz=[0,0,0], r_xyz=[0,0,0],
                                   method='numerical', charlen=6/100):
    """Takes geometric parameters -> generates mesh 
    -> deforms mesh by given t_xyz and r_xyz and method
    -> computes and homogenizes mesh to get matrix of elastic coefficients (pattern to bulk material)
    -> returns matrix of elastic coef"""
    mesh = Cell21X(L, r, mesh_order, minh, maxh)
    mesh = netgen2dolfin(mesh)
    domain = ExternalCell2Dolfin(mesh)
    domain.check_periodic()

    solver = Deformer(domain)
    solver.set_bulk_mat(0.4, -0.3, 1)
    solver.init_weak_form()
    tx = t_xyz[0]/100.
    print(tx)      
    ty = t_xyz[1]/100. 
    tz = t_xyz[2]/100. 
    rx = r_xyz[0]/100. 
    ry = r_xyz[1]/100. 
    rz = r_xyz[2]/100.   
    solver.deform_with([tx,ty,tz], [rx, ry, rz], method, charlen)
    #solver.undeform() ***optimalizovat
    solver = MicroFESolver_penalty(domain)
    # solver = MicroFESolver_lagrange(domain)
    solver.set_bulk_mat(E, nu, rho)
    solver.init_weak_form(1e-7) # smazat pro Lagrange
    # solver.init_weak_form()
    C = solver.compute_macro_tangent()
    return C

def write_main_data_to_file(file, tx, ty, tz, rx, ry, rz, C, duration):
    """Write MAIN data to file. Needs to be open at the satart and closed at the end of your supreme code."""
    file.write(
        f'{tx:7.2f} {ty:7.2f} {tz:7.2f} {rx:7.2f} {ry:7.2f} {rz:7.2f}: '
        f'{C[0,0]:8.4f} {C[1,1]:8.4f} {C[2,2]:8.4f} {C[3,3]:8.4f} {C[4,4]:8.4f} {C[5,5]:8.4f} '
        f'{C[0,1]:8.4f} {C[1,2]:8.4f} {C[2,3]:8.4f} {C[3,4]:8.4f} {C[4,5]:8.4f} '
        f'{C[0,2]:8.4f} {C[1,3]:8.4f} {C[2,4]:8.4f} {C[3,5]:8.4f} '
        f'{C[0,3]:8.4f} {C[1,4]:8.4f} {C[2,5]:8.4f} '
        f'{C[0,4]:8.4f} {C[1,5]:8.4f} {C[0,5]:8.4f} {duration:8.3f}\n'
    )

def write_break_data_to_file(file, tx, ty, tz, rx, ry, rz):
    """Write BREAK data to file. Needs to be open at the satart and closed at the end of your supreme code."""
    file.write(f'{tx:8.2f} {ty:8.2f} {tz:8.2f} {rx:8.2f} {ry:8.2f} {rz:8.2f}\n')

def check_and_break(trsX, trsY, trsZ, C, tx, ty, tz, rx, ry, rz, file2, break_flags):
    if contains_zero(C):
        write_break_data_to_file(file2, tx, ty, tz, rx, ry, rz)
        if trsX == 0:
            break_flags['TX'] = 1
            if trsY == 0:
                break_flags['TY'] = 1
                if trsZ == 0:
                    break_flags['TZ'] = 1        
                    if trsZ == 0:
                        break_flags['RX'] = 1
                        if trsZ == 0:
                            break_flags['RY'] = 1
                            if trsZ == 0:
                                break_flags['RZ'] = 1
        return True
    return False


def perform_homogenization_loops(file1, file2, charlen, break_flags, loop_ranges):
    np.set_printoptions(precision=3, suppress=True)    
    
    for rotZ in range(loop_ranges['rotZ'][0], loop_ranges['rotZ'][1] + 1, loop_ranges['rotZ'][2]):
        for rotY in range(loop_ranges['rotY'][0], loop_ranges['rotY'][1] + 1, loop_ranges['rotY'][2]):
            for rotX in range(loop_ranges['rotX'][0], loop_ranges['rotX'][1] + 1, loop_ranges['rotX'][2]):
                for trsZ in range(loop_ranges['trsZ'][0], loop_ranges['trsZ'][1] + 1, loop_ranges['trsZ'][2]): 
                    for trsY in range(loop_ranges['trsY'][0], loop_ranges['trsY'][1] + 1, loop_ranges['trsY'][2]): 
                        for trsX in range(loop_ranges['trsX'][0], loop_ranges['trsX'][1] + 1, loop_ranges['trsX'][2]): 
                            ticLoop = time.time()
                            tx, ty, tz = trsX / 100, trsY / 100, trsZ / 100
                            rx, ry, rz = rotX / 100, rotY / 100, rotZ / 100         
                            mesh = Cell21X(1.0, 0.06, mesh_order=1, minh=0.002, maxh=0.06)
                            mesh = netgen2dolfin(mesh)
                            domain = ExternalCell2Dolfin(mesh)
                            domain.check_periodic()

                            solver = Deformer(domain)
                            solver.set_bulk_mat(4.0, -0.3, 1)
                            solver.init_weak_form()
                            solver.deform_with([0,0,0], [0,0,0], method='numerical', charlen=charlen)

                            solver = MicroFESolver_penalty(domain)
                            solver.set_bulk_mat(2000, 0.3, 1)
                            solver.init_weak_form(1e-7)
                            C = solver.compute_macro_tangent()

                            if check_and_break(trsX, trsY, trsZ, C, tx, ty, rx, file2, break_flags):
                                break

                            tacLoop = time.time()
                            write_main_data_to_file(file1, tx, ty, tz, rx, ry, rz, C, tacLoop - ticLoop)
                            print(C)

                        if break_flags['TX']:
                            break_flags['TX'] = 2
                            break
                    if break_flags['TY']:
                        break_flags['TY'] = 2
                        break  
                if break_flags['TZ']:
                    break_flags['TZ'] = 2
                    break
            if break_flags['RX']:
                break_flags['RX'] = 2
                break
        if break_flags['RY']:
            break_flags['RY'] = 2
            break

def compute_cases_for_rotZ(rotZ, charlen, mesh_params, break_flags_initial):
    # Each process creates its own mesh and domain
    mesh = Cell21X(*mesh_params)  # Unpack parameters from main
    mesh = netgen2dolfin(mesh)
    domain = ExternalCell2Dolfin(mesh)
    domain.check_periodic()
    results = []
    break_flags = break_flags_initial.copy()  # Make a copy of the initial break flags for each process

    for rotY in range(0, 21, 20):
        if break_flags['RY']: break
        for rotX in range(0, 21, 20):
            if break_flags['RX']: break
            for trsZ in range(0, 21, 10):
                if break_flags['TZ']: break
                for trsY in range(0, 21, 10):
                    if break_flags['TY']: break
                    for trsX in range(0, 21, 10):
                        if break_flags['TX']: break
                        
                        ticLoop = time.time()
                        tx, ty, tz = trsX / 100, trsY / 100, trsZ / 100
                        rx, ry, rz = rotX / 100, rotY / 100, rotZ / 100

                        # Use the passed domain
                        solver = Deformer(domain)
                        solver.set_bulk_mat(4.0, -0.3, 1)
                        solver.init_weak_form()
                        solver.deform_with([tx, ty, tz], [rx, ry, rz], method='numerical', charlen=charlen)

                        solver = MicroFESolver_penalty(domain)
                        solver.set_bulk_mat(2000, 0.3, 1)
                        solver.init_weak_form(1e-7)
                        C = solver.compute_macro_tangent()

                        if check_and_break(trsX, trsY, trsZ, C, tx, ty, tz, rx, ry, rz, break_flags):
                            break

                        tacLoop = time.time()
                        process_time = tacLoop - ticLoop
                        results.append((tx, ty, tz, rx, ry, rz, C, process_time))
                        print(f"--- Process time for rotZ={rotZ}, rotY={rotY}, rotX={rotX}, trsZ={trsZ}, trsY={trsY}, trsX={trsX} = {process_time:.1f} s ---")

                    if break_flags['TX']:
                        break_flags['TX'] = 2
                        break
                if break_flags['TY']:
                    break_flags['TY'] = 2
                    break
            if break_flags['TZ']:
                break_flags['TZ'] = 2
                break
        if break_flags['RX']:
            break_flags['RX'] = 2
            break
    if break_flags['RY']:
        break_flags['RY'] = 2
    
    return results

if __name__ == '__main__':
    file1 = open("Loop_Data.txt", "w")
    file2 = open("Loop_BREAKData.txt", "w")
    charlen = 1.
    thickness = charlen * 6 / 100

    # Pre-generate mesh and domain outside the loop
    mesh = Cell21X(1.0, 0.06, mesh_order=1, minh=0.002, maxh=0.06)
    mesh = netgen2dolfin(mesh)
    domain = ExternalCell2Dolfin(mesh)
    domain.check_periodic()

    # Initial break flags
    break_flags_initial = {'TX': 0, 'TY': 0, 'TZ': 0, 'RX': 0, 'RY': 0, 'RZ': 0}
    np.set_printoptions(precision=3, suppress=True)

    # Define the rotZ values that each process will handle
    rotZ_values = [0, 10, 20, 30]  # Each core will take one of these

    # Mesh setup in main section, pass basic parameters to workers
    mesh_params = (1.0, 0.06)  # Parameters for Cell21X (adjust as needed)
    
    # Using Pool with starmap to pass parameters to each worker
    with Pool(processes=4) as pool:
        results = pool.starmap(
            compute_cases_for_rotZ,
            [(rotZ, charlen, mesh_params, break_flags_initial) for rotZ in rotZ_values]
        )

    # Save the results from all processes
    for core_results in results:
        for tx, ty, tz, rx, ry, rz, C, process_time in core_results:
            write_main_data_to_file(file1, tx, ty, tz, rx, ry, rz, C, process_time)
            print(f"Final result: tx={tx}, ty={ty}, tz={tz}, rx={rx}, ry={ry}, rz={rz}, C={C}, Time={process_time:.3f} s")

    file1.close()
    file2.close()