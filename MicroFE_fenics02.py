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


if __name__ == '__main__':

    charlen = 1.
    thickness = charlen * 6/100
    KX = 0.
    KY = 0.
    KZ = 0.
    TX = 0.
    TY = 0.
    TZ = 0.
    meshSize = 49.

    np.set_printoptions(precision=3, suppress=True)
    # mesh = ExternalCellOCC(
    #     'REC10.step', mesh_order=1, minh=0.05, maxh=0.05)
    # mesh = Cell50(charlen, thickness, mesh_order=1, minh=thickness/6, maxh=thickness/2)
    mesh = Cell21X(1.0, 0.06, mesh_order=1, minh=meshSize/30000, maxh=meshSize/1000)
    # mesh = Cell50Block(nX=3, nY=3, nZ=3, L=charlen*2, r=thickness, mesh_order=1, minh=thickness/3, maxh=thickness/1)
    # mesh = Cell50Block(1, 4, 1, 1.0, 0.15, mesh_order=1, minh=0.004, maxh=0.06)
    # mesh = Cell50(1.0, 0.05, mesh_order=1, minh=0.004, maxh=0.06)
    #mesh = Cell10(1.0, 0.1, mesh_order=1, minh=0.002, maxh=0.01)
    mesh = netgen2dolfin(mesh)
    domain = ExternalCell2Dolfin(mesh)
    domain.check_periodic()

    solver = Deformer(domain)
    solver.set_bulk_mat(4.0, -0.3, 1)
    solver.init_weak_form()

    # tx, ty, tz = 0.3025, 0.0018, 0.0043
    # rx, ry, rz = 0.2048, 0.0995, -0.0004

    tx, ty, tz = 0.0, 0.0, 0.0
    rx, ry, rz = 0.0, 0.0, 0.0
    # solver.deform_with([0.3,0.,0.], [0.4, 0.1, 0.], method='numerical',KX=KX*(2/1), KY=KY*(2/1), KZ=KZ*(2/1), TX=TX/(600+3*(KY+KZ)), TY=TY/(600+3*(KX+KZ)), TZ=TZ/(600+3*(KX+KY)), charlen=charlen)
    solver.deform_with([tx,ty,tz], [rx, ry, rz], method='numerical', KX=KX, KY=KY, KZ=KZ, TX=TX, TY=TY, TZ=TZ, charlen=charlen)
    #solver.undeform()
    solver = MicroFESolver_penalty(domain)
    # solver = MicroFESolver_lagrange(domain)
    solver.set_bulk_mat(2000, 0.3, 1)
    solver.init_weak_form(1e-7) # smazat pro Lagrange
    # solver.init_weak_form()
    C = solver.compute_macro_tangent()
    I = anisotropy_indicator(C)

    name = 'E2'
    m = material_quadric(C, name, res=50, radius=charlen*0.85, fixed_axis='x')
    m.save(f'{name}.vtk')
    d = polar_plot(C, 'E2', axis='x', res=200)
    print(C)
