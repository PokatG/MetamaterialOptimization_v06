import time
import numpy as np
import dolfin as df
import netgen.occ as occ
# from netgen.occ import gp_Trsf
# from netgen.meshing import IdentificationType
# import ngsolve as ng
from ngsolve import Mesh # Import only what is needed

# import pandas as pd

from utils2_a import (Cell10, Cell21X, Cell50, Cell50Block, ExternalCell2Dolfin, ExternalCellOCC,
                   anisotropy_indicator, get_engineering_constants,
                   material_quadric, netgen2dolfin, polar_plot)

class PeriodicBoundary(df.SubDomain):
    """Define periodic boundary conditions for 2D or 3D meshes dynamically."""
    
    def __init__(self, mesh):
        """Initialize the periodic boundary mapping based on mesh coordinates."""
        super().__init__()
        self.TOL = 1e-15
        self.dim = mesh.geometric_dimension()  # Automatically detect mesh dimension

        # Store min/max bounds dynamically
        self.bounds = {
            "xmin": np.min(mesh.coordinates()[:, 0]),
            "ymin": np.min(mesh.coordinates()[:, 1]),
            "xmax": np.max(mesh.coordinates()[:, 0]),
            "ymax": np.max(mesh.coordinates()[:, 1]),
        }
        
        if self.dim == 3:  # Only get z-bounds if the mesh is 3D
            self.bounds["zmin"] = np.min(mesh.coordinates()[:, 2])
            self.bounds["zmax"] = np.max(mesh.coordinates()[:, 2])

        # Compute domain lengths for shifting points
        self.lengths = {
            axis: self.bounds[f"{axis}max"] - self.bounds[f"{axis}min"]
            for axis in ["x", "y", "z"] if f"{axis}max" in self.bounds
        }

    def inside(self, x, on_boundary):
        """
        Check if a point is on a periodic boundary.

        A point is considered inside the periodic boundary if it is located at 
        a minimum boundary (xmin, ymin, zmin) but not at a maximum boundary 
        (xmax, ymax, zmax).
        """
        return bool(on_boundary and (
            df.near(x[0], self.bounds["xmin"], self.TOL) or
            df.near(x[1], self.bounds["ymin"], self.TOL) or
            (self.dim == 3 and df.near(x[2], self.bounds["zmin"], self.TOL))
        ))

    def map(self, x, y):
        """
        Map points from one periodic boundary to the opposite boundary.

        Points at the maximum boundary (xmax, ymax, zmax) are mapped to 
        their corresponding positions at the minimum boundary.
        """
        for axis, coord in zip(["x", "y", "z"], range(self.dim)):
            if df.near(x[coord], self.bounds[f"{axis}max"], self.TOL):
                y[coord] = x[coord] - self.lengths[axis]
            else:
                y[coord] = x[coord]  # Keep it unchanged if not on a boundary


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
    """
    Finite Element Solver using the penalty method.

    This class solves the homogenized behavior of microstructures 
    using the finite element method with a penalty approach. 
    """

    def __init__(self, domain, disp_order=2, FILE_SAVE='homogenized_results.xdmf'):
        """
        Initializes the finite element solver.

        Parameters:
        - domain: The computational domain containing the mesh.
        - disp_order: The order of the displacement finite element space.
        - FILE_SAVE: Name of the output file to store results.
        """

        # Store the computational domain
        self.domain = domain
        self.mesh = domain.mesh  # Extract the mesh
        self.gdim = self.mesh.geometric_dimension()  # Get spatial dimension (2D or 3D)

        # Define function spaces: Continuous Galerkin (CG) for displacement
        Ue = df.VectorElement('CG', self.mesh.ufl_cell(), disp_order)
        Ve = df.FunctionSpace(self.mesh, Ue, constrained_domain=PeriodicBoundary(self.mesh))

        # Define test and trial functions for variational formulation
        self.du = df.TestFunction(Ve)   # Virtual displacement (test function)
        self.u_ = df.TrialFunction(Ve)  # Displacement function to solve for (trial function)
        self.u_sol = df.Function(Ve)    # Solution function

        # Output file for storing results
        self.file_results = df.XDMFFile(FILE_SAVE)
        self.file_results.parameters["flush_output"] = True
        self.file_results.parameters["functions_share_mesh"] = True

        # Initialize matrices and solver
        self.K = df.PETScMatrix()  # Stiffness matrix
        self.b = df.PETScVector()  # Load vector
        self.solver = df.LUSolver(self.K, "mumps")  # Direct solver (MUMPS for efficiency)
        self.solver.parameters["symmetric"] = True  # Matrix symmetry assumption for efficiency

        # Compute the total volume of the solid structure (for normalization later)
        self.vol_of_solid = df.assemble(df.Constant(1.0) * df.dx(self.mesh))

    def set_bulk_mat(self, E, nu, rho):
        """
        Set the material properties for the solver.

        Parameters:
        - E: Young's modulus
        - nu: Poisson's ratio
        - rho: Density
        """
        self.lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))  # First Lame parameter
        self.mu = E / (2.0 * (1 + nu))  # Second Lame parameter (shear modulus)
        self.rho = rho  # Material density

    def _eps(self, v):
        """
        Compute the small-strain tensor (symmetric gradient of displacement).
        
        Parameters:
        - v: A vector function (displacement field)
        
        Returns:
        - The small-strain tensor (symmetric part of the displacement gradient)
        """
        return df.sym(df.grad(v))

    def _sigma(self, v, Eps):
        """
        Compute the Cauchy stress tensor using linear elasticity.
        
        Parameters:
        - v: A vector function (displacement field)
        - Eps: Macro strain applied to the structure
        
        Returns:
        - Stress tensor computed using Hooke's law
        """
        return (self.lmbda * df.tr(self._eps(v) + Eps) * df.Identity(self.gdim) +
                2 * self.mu * (self._eps(v) + Eps))

    def init_weak_form(self, penalty):
        """
        Initialize the weak form of the problem (finite element formulation).

        Parameters:
        - penalty: A small value added to the stiffness to enforce stability.
        """
        # Define applied macroscopic strain as a constant tensor (initially zero)
        self.Gamm_bar = df.Constant(((0, 0, 0), (0, 0, 0), (0, 0, 0)))

        # Define the weak form of the problem (virtual work principle)
        self.a = df.inner(self._sigma(self.u_, self.Gamm_bar), self._eps(self.du)) * df.dx
        self.a += penalty * df.inner(self.u_, self.du) * df.dx  # Penalty term for stability

        # Assemble the system matrix
        tic = time.time()
        print('Assembling stiffness matrix...', end=" ", flush=True)
        L_w, self.f_w = df.lhs(self.a), df.rhs(self.a)  # Get left-hand and right-hand sides
        df.assemble(L_w, tensor=self.K)  # Assemble stiffness matrix
        tac = time.time()
        print(f'DONE in {(tac-tic):0.3f}s')

    def compute_macro_tangent(self):
        """
        Compute the homogenized stiffness matrix for the microstructure.

        This method solves the problem for different macroscopic strain 
        cases and computes the effective material stiffness.
        
        Returns:
        - C_hom: The homogenized stiffness tensor (6x6 matrix)
        """
        self.C_hom = np.zeros((6, 6))  # Initialize homogenized stiffness tensor

        # Iterate over 6 independent strain cases
        for i, case in enumerate(['xx', 'yy', 'zz', 'yz', 'xz', 'xy']):
            # Apply macroscopic strain in the `i`th direction
            self.Gamm_bar.assign(df.Constant(macro_strain(i)))

            # Assemble the right-hand side vector (forcing term)
            df.assemble(self.f_w, tensor=self.b)

            # Factorize matrix once (only needed for the first case)
            if i == 0:
                print('Matrix factorizing...', end=" ", flush=True)
                tic = time.time()
                self.solver.solve(self.u_sol.vector(), self.b)  # Solve system
                tac = time.time()
                print(f'DONE in {(tac-tic):0.3f}s')

            # Solve for each strain case
            tic = time.time()
            print(f'Solving for {case}...', end=" ", flush=True)
            self.solver.solve(self.u_sol.vector(), self.b)
            tac = time.time()
            print(f'DONE in {(tac-tic):0.3f}s')

            # Save solution to file
            self.u_sol.rename(f'u_{case}', '')
            self.file_results.write(self.u_sol, 0)

            # Compute the homogenized stress response for this case
            sigma_til = np.zeros((6,))
            for k in range(6):
                sigma_til[k] = float(df.assemble(stress2voigt(
                    self._sigma(self.u_sol, self.Gamm_bar))[k] * df.dx))

            self.C_hom[i, :] = sigma_til.copy()  # Store computed stiffness

        self.file_results.close()  # Close the results file
        return self.C_hom / self.vol_of_solid  # Normalize by volume


class MicroFESolver_lagrange(object):
    """
    Finite Element Solver using the Lagrange multiplier method.

    This class solves the homogenization problem using the finite element method
    with the Lagrange multiplier approach to enforce periodicity constraints.
    """

    def __init__(self, domain, disp_order=2):
        """
        Initializes the finite element solver with a mixed formulation 
        using Lagrange multipliers.

        Parameters:
        - domain: The computational domain containing the mesh.
        - disp_order: The order of the displacement finite element space.
        """

        # Store the computational domain
        self.domain = domain
        self.mesh = domain.mesh  # Extract the mesh
        self.gdim = self.mesh.geometric_dimension()  # Get spatial dimension (2D or 3D)

        # Define function spaces:
        # Ue: Displacement field using Continuous Galerkin (CG)
        # Re: Lagrange multipliers using a real-valued element (R)
        Ue = df.VectorElement('CG', self.mesh.ufl_cell(), disp_order)
        Re = df.VectorElement('R', self.mesh.ufl_cell(), 0)  # Scalar Lagrange multiplier
        We = df.MixedElement([Ue, Re])  # Mixed element formulation

        # Define the function space for the mixed problem
        Ve = df.FunctionSpace(self.mesh, We, constrained_domain=PeriodicBoundary(self.mesh))

        # Define test and trial functions for variational formulation
        self.du, self.dlamb = df.TestFunctions(Ve)  # Virtual displacements & Lagrange multipliers (test functions)
        self.u_, self.lamb_ = df.TrialFunctions(Ve)  # Displacement & multipliers to solve for (trial functions)
        self.u_lamb = df.Function(Ve)  # Solution function containing both u_ and lamb_

        # Output file for storing results
        self.file_results = df.XDMFFile('homogen_results.xdmf')
        self.file_results.parameters["flush_output"] = True
        self.file_results.parameters["functions_share_mesh"] = True

        # Initialize matrices and solver
        self.K = df.PETScMatrix()  # Stiffness matrix
        self.b = df.PETScVector()  # Load vector
        self.solver = df.LUSolver(self.K, "mumps")  # Direct solver (MUMPS for efficiency)
        self.solver.parameters["symmetric"] = True  # Matrix symmetry assumption for efficiency

        # Compute the total volume of the solid structure (for normalization later)
        self.vol_of_solid = df.assemble(df.Constant(1.0) * df.dx(self.mesh))

        # Define macroscopic strain as a constant tensor (initially zero)
        self.Gamm_bar = df.Constant(((0, 0, 0), (0, 0, 0), (0, 0, 0)))

    def set_bulk_mat(self, E, nu, rho):
        """
        Set the material properties for the solver.

        Parameters:
        - E: Young's modulus
        - nu: Poisson's ratio
        - rho: Density
        """
        self.lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))  # First Lame parameter
        self.mu = E / (2.0 * (1 + nu))  # Second Lame parameter (shear modulus)
        self.rho = rho  # Material density

    def _eps(self, v):
        """
        Compute the small-strain tensor (symmetric gradient of displacement).
        
        Parameters:
        - v: A vector function (displacement field)
        
        Returns:
        - The small-strain tensor (symmetric part of the displacement gradient)
        """
        return df.sym(df.grad(v))

    def _sigma(self, v, Eps):
        """
        Compute the Cauchy stress tensor using linear elasticity.
        
        Parameters:
        - v: A vector function (displacement field)
        - Eps: Macro strain applied to the structure
        
        Returns:
        - Stress tensor computed using Hooke's law
        """
        return (self.lmbda * df.tr(self._eps(v) + Eps) * df.Identity(self.gdim) +
                2 * self.mu * (self._eps(v) + Eps))

    def init_weak_form(self):
        """
        Initialize the weak form of the problem (finite element formulation).

        The weak form consists of:
        - Balance of linear momentum (stress-strain relation)
        - Constraint equations for periodicity enforced via Lagrange multipliers.
        """
        # Reset the applied macroscopic strain to zero
        self.Gamm_bar = df.Constant(((0, 0, 0), (0, 0, 0), (0, 0, 0)))

        # Define the weak form:
        # 1) Stress-strain relation (balance of linear momentum)
        self.a = df.inner(self._sigma(self.u_, self.Gamm_bar), self._eps(self.du)) * df.dx

        # 2) Lagrange multipliers enforce periodicity constraints:
        #    ∫ (lamb * du) dx + ∫ (dlamb * u) dx
        self.a += (df.inner(self.lamb_, self.du) + df.inner(self.dlamb, self.u_)) * df.dx

        # Assemble the system matrix
        tic = time.time()
        print('Assembling stiffness matrix...', end=" ", flush=True)
        L_w, self.f_w = df.lhs(self.a), df.rhs(self.a)  # Get left-hand and right-hand sides
        df.assemble_mixed(L_w, tensor=self.K)  # Assemble mixed formulation
        tac = time.time()
        print(f'DONE in {(tac-tic):0.3f}s')

    def compute_macro_tangent(self):
        """
        Compute the homogenized stiffness matrix for the microstructure.

        This method solves the problem for different macroscopic strain 
        cases and computes the effective material stiffness.
        
        Returns:
        - C_hom: The homogenized stiffness tensor (6x6 matrix)
        """
        self.C_hom = np.zeros((6, 6))  # Initialize homogenized stiffness tensor

        # Iterate over 6 independent strain cases
        for i, case in enumerate(['xx', 'yy', 'zz', 'yz', 'xz', 'xy']):
            # Apply macroscopic strain in the `i`th direction
            self.Gamm_bar.assign(df.Constant(macro_strain(i)))

            # Assemble the right-hand side vector (forcing term)
            df.assemble_mixed(self.f_w, tensor=self.b)

            # Factorize matrix once (only needed for the first case)
            if i == 0:
                print('Matrix factorizing...', end=" ", flush=True)
                tic = time.time()
                self.solver.solve(self.u_lamb.vector(), self.b)  # Solve system
                tac = time.time()
                print(f'DONE in {(tac-tic):0.3f}s')

            # Solve for each strain case
            tic = time.time()
            print(f'Solving for {case}...', end=" ", flush=True)
            self.solver.solve(self.u_lamb.vector(), self.b)
            tac = time.time()
            print(f'DONE in {(tac-tic):0.3f}s')

            # Extract the displacement solution
            u_, lamb_ = self.u_lamb.split(True)

            # Save solution to file
            u_.rename(f'u_{case}', '')
            self.file_results.write(u_, 0)

            # Compute the homogenized stress response for this case
            sigma_til = np.zeros((6,))
            for k in range(6):
                sigma_til[k] = float(df.assemble(stress2voigt(
                    self._sigma(u_, self.Gamm_bar))[k] * df.dx))

            self.C_hom[i, :] = sigma_til.copy()  # Store computed stiffness

        self.file_results.close()  # Close the results file
        return self.C_hom / self.vol_of_solid  # Normalize by volume


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

    def deform_with(self, bending, torsion, method="analytic", KX=0., KY=0., KZ=0., 
                TX=0., TY=0., TZ=0., charlen=1., **kwargs):
        """
        Apply deformation to the structure using either the analytic or numerical method.

        Parameters:
        - bending: Defines the bending applied to the structure.
        - torsion: Defines the torsional deformation.
        - method: Specifies the deformation method. Valid options:
        ["analytic", "analytical", "analyticky", "an al", "a", "A", "numerical"]
        - KX, KY, KZ: Rotational scaling factors.
        - TX, TY, TZ: Translational scaling factors.
        - charlen: Characteristic length parameter.
        - kwargs: Additional keyword arguments (for future expansion).
        """

        # Normalize input to handle different valid method names
        method = method.strip().lower()  # Remove spaces, lowercase
        valid_methods = {"analytic", "analytical", "analyticky", "anal", "a", "A", 
                         "numerical", "numeric", "numericky", "num", "n", "N"}

        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Choose from {valid_methods}")

        # Convert input to a standard format
        method = "analytic" if method in {"analytic", "analytical", "analyticky", "anal", "a", "A"} else "numerical"

        b = df.Constant(bending)  # Convert bending to a Dolfin constant

        if method == "analytic":
            half_charlen = charlen / 2

            # Define common expressions to avoid repetition
            def expr_pow(term):
                return df.Expression(f"(pow(charlen,2) - pow(x[{term}],2)) / pow(charlen,2)", charlen=half_charlen, degree=1)

            def expr_exp(term):
                return df.Expression(f"exp(-pow(x[{term}],2)/(1 + charlen)) / (2 * pow(charlen,4))", charlen=half_charlen, degree=1)

            # Normalized coordinates
            normalizers = {axis: expr_pow(i) for i, axis in enumerate("XYZ")}

            # Exponential decay expressions
            exp_terms = {axis: expr_exp(i) for i, axis in enumerate("XYZ")}

            # Kv terms (combined interaction terms)
            kv_terms = {
                "XY": df.Expression("(pow(charlen,2)-pow(x[0],2)) * (pow(charlen,2)-pow(x[1],2))", charlen=half_charlen, degree=1),
                "YZ": df.Expression("(pow(charlen,2)-pow(x[1],2)) * (pow(charlen,2)-pow(x[2],2))", charlen=half_charlen, degree=1),
                "XZ": df.Expression("(pow(charlen,2)-pow(x[0],2)) * (pow(charlen,2)-pow(x[2],2))", charlen=half_charlen, degree=1)
            }

            # Translation components
            trans_X = df.Expression("TX * norX * (expY * kvYZ + expZ * kvYZ)", 
                                    norX=normalizers["X"], expY=exp_terms["Y"], expZ=exp_terms["Z"], 
                                    kvYZ=kv_terms["YZ"], TX=TX, charlen=half_charlen, degree=1)

            trans_Y = df.Expression("TY * norY * (expX * kvXZ + expZ * kvXZ)", 
                                    norY=normalizers["Y"], expX=exp_terms["X"], expZ=exp_terms["Z"], 
                                    kvXZ=kv_terms["XZ"], TY=TY, charlen=half_charlen, degree=1)

            trans_Z = df.Expression("TZ * norZ * (expX * kvXY + expY * kvXY)", 
                                    norZ=normalizers["Z"], expX=exp_terms["X"], expY=exp_terms["Y"], 
                                    kvXY=kv_terms["XY"], TZ=TZ, charlen=half_charlen, degree=1)
            
            # normalizace podle definicniho oboru D = <0;1>
            # normalizace podle maxima fce f(x,y) na definicnim odboru D -> 8/27
            # prepocet na 1 DEG
            extrem01 = 0.75 # 27/8 * np.pi/180
            sigma = 2.45
            aPower = 2.
            bPower = 1.7
            phiX = df.Expression(
                "(pow(abs(x[1]),2) * pow(charlen-abs(x[2]),2) * pow(charlen-abs(x[1]),2) + "
                "pow(abs(x[2]),2) * pow(charlen-abs(x[2]),2) * pow(charlen-abs(x[1]),2)) / pow(charlen,3) * extrem01",
                charlen=charlen/2, extrem01=extrem01, degree=1)            
            phiY = df.Expression(
                "(pow(abs(x[2]),2) * pow(charlen-abs(x[0]),2) * pow(charlen-abs(x[2]),2) + "
                "pow(abs(x[2]),2) * pow(charlen-abs(x[0]),2) * pow(charlen-abs(x[2]),2)) / pow(charlen,3) * extrem01",
                charlen=charlen/2, extrem01=extrem01, degree=1)
            phiZ = df.Expression(
                "(pow(abs(x[0]),2) * pow(charlen-abs(x[0]),2) * pow(charlen-abs(x[1]),2) + "
                "pow(abs(x[1]),2) * pow(charlen-abs(x[0]),2) * pow(charlen-abs(x[1]),2)) / pow(charlen,3) * extrem01",
                charlen=charlen/2, extrem01=extrem01, degree=1)
            
            # # # # phiX = df.Expression(
            # # # #     "pow((charlen - pow(x[1],a)),b) * pow((charlen - pow(x[2],a)),b) * "
            # # # #     "((pow(x[1],2) / (s * sqrt(2 * pi))) * exp(-((pow(x[1],2) / (2 * pow(s,2))) + (pow(x[2],2) / (2 * pow(s,2))))) + "
            # # # #     "(pow(x[2],2) / (s * sqrt(2 * pi))) * exp(-((pow(x[1],2) / (2 * pow(s,2))) + (pow(x[2],2) / (2 * pow(s,2)))))) * extrem01",
            # # # #     s=sigma, a=aPower, b=bPower, charlen=charlen/2, extrem01=extrem01, pi=np.pi, degree=1)

            # # # # phiY = df.Expression(
            # # # #     "pow((charlen - pow(x[0],a)),b) * pow((charlen - pow(x[2],a)),b) * "
            # # # #     "((pow(x[0],2) / (s * sqrt(2 * pi))) * exp(-((pow(x[0],2) / (2 * pow(s,2))) + (pow(x[2],2) / (2 * pow(s,2))))) + "
            # # # #     "(pow(x[2],2) / (s * sqrt(2 * pi))) * exp(-((pow(x[0],2) / (2 * pow(s,2))) + (pow(x[2],2) / (2 * pow(s,2)))))) * extrem01",
            # # # #     s=sigma, a=aPower, b=bPower, charlen=charlen/2, extrem01=extrem01, pi=np.pi, degree=1)
            # # # # phiZ = df.Expression(
            # # # #     "pow((charlen - pow(x[0],a)),b) * pow((charlen - pow(x[1],a)),b) * "
            # # # #     "((pow(x[0],2) / (s * sqrt(2 * pi))) * exp(-((pow(x[0],2) / (2 * pow(s,2))) + (pow(x[1],2) / (2 * pow(s,2))))) + "
            # # # #     "(pow(x[1],2) / (s * sqrt(2 * pi))) * exp(-((pow(x[0],2) / (2 * pow(s,2))) + (pow(x[1],2) / (2 * pow(s,2)))))) * extrem01",
            # # # #     s=sigma, a=aPower, b=bPower, charlen=charlen/2, extrem01=extrem01, pi=np.pi, degree=1)


            def rotation_matrix(axis):
                if axis == 'x':
                    return (
                        "0",
                        "x[1] * (1 - cos(KX * phiX)) + x[2] * sin(KX * phiX)",
                        "x[2] * (1 - cos(KX * phiX)) - x[1] * sin(KX * phiX)"
                    )
                elif axis == 'y':
                    return (
                        "x[0] * (1 - cos(KY * phiY)) - x[2] * sin(KY * phiY)",
                        "0",
                        "x[2] * (1 - cos(KY * phiY)) + x[0] * sin(KY * phiY)"
                    )
                elif axis == 'z':
                    return (
                        "x[0] * (1 - cos(KZ * phiZ)) + x[1] * sin(KZ * phiZ)",
                        "x[1] * (1 - cos(KZ * phiZ)) - x[0] * sin(KZ * phiZ)",
                        "0"
                    )
 
            rot_x = df.Expression(rotation_matrix('x'),
                      phiX=phiX, KX=KX, degree=1)

            rot_y = df.Expression(rotation_matrix('y'),
                                phiY=phiY, KY=KY, degree=1)

            rot_z = df.Expression(rotation_matrix('z'),
                                phiZ=phiZ, KZ=KZ, degree=1)

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
        """
        Reverts the applied deformation by negating the displacement vector.
        """
        self.u_sol.vector()[:] *= -1
        df.ALE.move(self.mesh, self.u_sol)


if __name__ == '__main__':    

    # === PARAMETERS & NORMALIZATION FACTORS ===
    charlen = 1.0  # DO NOT change this (until it is fixed)
    
    # Rotation and translation factors
    rotational_norm = 1764 / 100
    translational_norm = 35 / 1e+4
    
    # Material & mesh properties
    thickness = charlen * 6 / 100
    meshSize = 60.0

    # Rotation and translation
    KX = 10.
    KY = 0.
    KZ = 0.
    TX = 0.
    TY = 0.
    TZ = 0.

    np.set_printoptions(precision=3, suppress=True)

    # === MESH GENERATION ===
    # Uncomment only ONE of the following mesh configurations:
    # mesh = ExternalCellOCC('REC10.step', mesh_order=1, minh=0.05, maxh=0.05)
    # mesh = Cell50(charlen, thickness, mesh_order=1, minh=thickness/6, maxh=thickness/2)
    mesh = Cell21X(1.0, 0.06, mesh_order=1, minh=meshSize / 30000, maxh=meshSize / 1000)
    # mesh = Cell50Block(nX=3, nY=3, nZ=3, L=charlen*2, r=thickness, mesh_order=1, minh=thickness/3, maxh=thickness/1)
    # mesh = Cell50Block(1, 4, 1, 1.0, 0.15, mesh_order=1, minh=0.004, maxh=0.06)
    # mesh = Cell50(1.0, 0.05, mesh_order=1, minh=0.004, maxh=0.06)
    # mesh = Cell10(1.0, 0.1, mesh_order=1, minh=0.002, maxh=0.01)

    # Convert the Netgen mesh to a Dolfin-compatible format
    mesh = netgen2dolfin(mesh)
    domain = ExternalCell2Dolfin(mesh)
    domain.check_periodic()

    # === DEFORMER INITIALIZATION ===
    solver = Deformer(domain)
    solver.set_bulk_mat(4.0, -0.3, 1)  # Set material properties
    solver.init_weak_form()

    # === DEFORMATION SETTINGS ===
    # Uncomment & modify deformation values if needed:
    # tx, ty, tz = 0.3025, 0.0018, 0.0043
    # rx, ry, rz = 0.2048, 0.0995, -0.0004

    # Define rotation and translation
    rx = 0.
    ry = 0.
    rz = 0.
    tx = 0.
    ty = 0.
    tz = 0.

    # Apply deformation
    solver.deform_with(
        [tx, ty, tz], [rx, ry, rz], method='analytic',
        KX=KX * rotational_norm, KY=KY * rotational_norm, KZ=KZ * rotational_norm,
        TX=TX * translational_norm, TY=TY * translational_norm, TZ=TZ * translational_norm,
        charlen=charlen
    )
    # solver.undeform()  # Uncomment if you need to revert deformation

    # === SOLVER CHOICE: PENALTY OR LAGRANGE ===
    # Uncomment the desired solver:
    solver = MicroFESolver_penalty(domain)  # Penalty method
    # solver = MicroFESolver_lagrange(domain)  # Lagrange method

    solver.set_bulk_mat(2000, 0.3, 1)  # Set material properties

    # Initialize weak form (choose correct method for selected solver)
    solver.init_weak_form(1e-7)  # ✅ Use for penalty solver
    # solver.init_weak_form()  # ✅ Use for Lagrange solver

    # === COMPUTE HOMOGENIZED STIFFNESS & ANISOTROPY ===
    C = solver.compute_macro_tangent()
    I = anisotropy_indicator(C)

    # === POST-PROCESSING: MATERIAL PROPERTIES & POLAR PLOTS ===
    name = 'E2'
    m = material_quadric(C, name, res=50, radius=charlen * 0.85, fixed_axis='x')
    m.save(f'{name}.vtk')
    d = polar_plot(C, 'E2', axis='x', res=200)

    # Print final stiffness matrix
    print(C)
