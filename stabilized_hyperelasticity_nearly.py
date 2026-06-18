from dolfin import *
from ufl import Identity, tr, det, grad, inv, derivative, variable
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
    def __init__(self, mesh, boundaries, material, order=2):
        self.mesh = mesh
        self.boundaries = boundaries
        self.material = material
        self.h = CellDiameter(mesh)
        
        # Mixed function space (equal-order)
        V_elem = VectorElement("CG", mesh.ufl_cell(), order)
        Q_elem = FiniteElement("CG", mesh.ufl_cell(), order)
        mixed_element = MixedElement([V_elem, Q_elem])
        self.W = FunctionSpace(mesh, mixed_element)
        
        # 单独的子空间用于设置 BC
        self.V = self.W.sub(0).collapse()
        self.Q = self.W.sub(1).collapse()
        
        # Trial/test functions
        self.w = Function(self.W)
        self.u, self.p = split(self.w)
        self.v, self.q = TestFunctions(self.W)
        
        self.w_trial = TrialFunction(self.W)
        self.du, self.dp = split(self.w_trial)  # 拆分 trial 函数
        
        # Kinematics
        I = Identity(mesh.geometry().dim())
        F = I + grad(self.u)
        F_v = variable(F)
        C = F_v.T*F_v
        J = det(F_v)
        H = J*inv(F_v).T  
        
        # Material stress
        P = diff(self.material.strain_energy(F_v), F_v)
        P = P - self.material.kappa * (J - 1) * H - self.p * H
    
        # Stabilization parameter
        tau = 0.1 * self.h**2 / self.material.mu
        
        # Residuals (Eq. 11-12)
        R_u = div(P)  # Simplified, no body force
        R_p = J - 1 + self.p / self.material.kappa
        
        # ✅ 定义边界测度 ds 和牵引力
        ds = Measure("ds", domain=self.mesh, subdomain_data=self.boundaries)
        ty = Constant(6.25)  # 剪力值
        traction = Constant((0.0, ty))  # 方向向上
        
        # Weak form (Eq. 13-14)
        Res_traction = -inner(traction, self.v) * ds(2)  # ds(2) 表示右边界
        Res_u = inner(P, grad(self.v))*dx + tau*inner(H, grad(self.v))*R_p*dx       
        Res_p = self.q*R_p*dx - tau*inner(R_u, H*grad(self.q))*dx
        self.Res = Res_u + Res_p + Res_traction
        
        # Tangent matrix (Automatic differentiation)
        self.Jacobian = derivative(self.Res, self.w, self.w_trial)
        
    def solve(self, bcs, tol=1e-6):
        # Nonlinear solver setup
        problem = NonlinearVariationalProblem(self.Res, self.w, bcs, self.Jacobian)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters["nonlinear_solver"] = "newton"
        solver.parameters["newton_solver"]["linear_solver"] = "mumps"
        # solver.parameters["newton_solver"]["preconditioner"] = "petsc_amg"
        solver.parameters["newton_solver"]["absolute_tolerance"] = tol
        solver.solve()
        
        u_sol, p_sol = self.w.split(deepcopy=True)
        return u_sol, p_sol