from dolfin import *
from ufl import Identity, tr, det, grad, inv, derivative
import numpy as np

# Material models
class NeoHookean:
    def __init__(self, mu, kappa):
        self.mu = mu
        self.kappa = kappa

    def strain_energy(self, F):
        C = F.T*F
        J = det(F)
        return self.mu/2*(tr(C) - 3) + self.kappa/2*(J-1)**2

class MooneyRivlin:
    def __init__(self, C10, C01, kappa):
        self.C10 = C10
        self.C01 = C01
        self.kappa = kappa

    def strain_energy(self, F):
        C = F.T*F
        J = det(F)
        I1_bar = J**(-2/3)*tr(C)
        I2_bar = J**(-4/3)*0.5*(tr(C)**2 - tr(C*C))
        return self.C10*(I1_bar - 3) + self.C01*(I2_bar - 3) + self.kappa/2*(J-1)**2

# Stabilized FEM solver
class StabilizedHyperelasticitySolver:
    def __init__(self, mesh, material, order=2):
        self.mesh = mesh
        self.material = material
        self.h = CellDiameter(mesh)
        
        # Mixed function space (equal-order)
        self.V = VectorFunctionSpace(mesh, "CG", order)
        self.Q = FunctionSpace(mesh, "CG", order)
        self.W = self.V * self.Q
        
        # Trial/test functions
        self.w = Function(self.W)
        self.u, self.p = split(self.w)
        self.v, self.q = TestFunctions(self.W)
        
        # Kinematics
        I = Identity(mesh.geometry().dim())
        F = I + grad(self.u)
        C = F.T*F
        J = det(F)
        
        # Material stress
        P = derivative(self.material.strain_energy(F), F)
        
        # Stabilization parameter
        tau = 0.1 * self.h**2 / self.material.mu
        
        # Residuals (Eq. 11-12)
        R_u = div(P)  # Simplified, no body force
        R_p = J - 1
        
        # Weak form (Eq. 13-14)
        Res_u = inner(P, grad(self.v))*dx + tau*inner(R_u, div(self.v))*dx
        Res_p = self.q*(J - 1)*dx + tau*inner(grad(self.q), grad(self.p))*R_p*dx
        self.Res = Res_u + Res_p
        
        # Tangent matrix (Automatic differentiation)
        self.Jacobian = derivative(self.Res, self.w)
        
    def solve(self, bcs, tol=1e-6):
        # Nonlinear solver setup
        problem = NonlinearVariationalProblem(self.Res, self.w, bcs, self.Jacobian)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters["nonlinear_solver"] = "newton"
        solver.parameters["newton_solver"]["linear_solver"] = "gmres"
        solver.parameters["newton_solver"]["preconditioner"] = "petsc_amg"
        solver.parameters["newton_solver"]["absolute_tolerance"] = tol
        solver.solve()
        return self.w.split()