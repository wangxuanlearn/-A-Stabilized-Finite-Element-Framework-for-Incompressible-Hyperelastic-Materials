from dolfin import *
from ufl import Identity, tr, det, grad, inv, derivative, variable, ln
import numpy as np
from petsc4py import PETSc

# Material models
class SimoTaylorNeoHookean:
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu
        self.mu = E / (2 * (1 + nu))
        self.kappa = E / (3 * (1 - 2 * nu))

    def strain_energy(self, F):
        C = F.T * F
        J = det(F)
        # 求解器已经使用 u-p 混合列式，这里只需返回偏应变能(Isochoric部分)
        return self.mu / 2 * (J**(-2/3) * tr(C) - 3)

    def pressure_residual(self, J, p):
        # 对应图中的 Simo-Taylor 体积部分 U(J) = kappa/4 * (J^2 - 1 - 2*ln(J))
        # 求导得: dU/dJ = kappa/2 * (J - 1/J)
        # 我们用 -p 替换 dU/dJ，即 -p = kappa/2 * (J - 1/J)
        # 变形得到 R_p = 0 的方程:
        return 0.5 * (J - 1.0/J) + p / self.kappa

class NeoHookean:
    def __init__(self, mu):
        self.mu = mu

    def strain_energy(self, F):
        C = F.T*F
        J = det(F)
        return self.mu/2*(J**(-2/3)*tr(C) - 3)

class MooneyRivlin:
    def __init__(self, C10, C01):
        self.C10 = C10
        self.C01 = C01

    def strain_energy(self, F):
        C = F.T*F
        J = det(F)
        I1_bar = J**(-2/3)*tr(C)
        I2_bar = J**(-4/3)*0.5*(tr(C)**2 - tr(C*C))
        return self.C10*(I1_bar - 3) + self.C01*(I2_bar - 3)

# Stabilized FEM solver
class StabilizedHyperelasticitySolver:
    def __init__(self, mesh, boundaries, material, u_order=2, p_order=1):
        self.mesh = mesh
        self.boundaries = boundaries
        self.material = material
        self.h = CellDiameter(mesh)
        
        # Mixed function space
        V_elem = VectorElement("CG", mesh.ufl_cell(), u_order)
        Q_elem = FiniteElement("CG", mesh.ufl_cell(), p_order)
        mixed_element = MixedElement([V_elem, Q_elem])
        self.W = FunctionSpace(mesh, mixed_element)
        
        self.V = self.W.sub(0).collapse()
        self.Q = self.W.sub(1).collapse()
        
        self.w = Function(self.W)
        self.u, self.p = split(self.w)
        self.v, self.q = TestFunctions(self.W)
        
        self.w_trial = TrialFunction(self.W)
        self.du, self.dp = split(self.w_trial)
        
        I = Identity(mesh.geometry().dim())
        F = I + grad(self.u)
        F_v = variable(F)
        C = F_v.T*F_v
        J = det(F_v)
        H = J*inv(F_v).T  
        
        # 统一做法: 不管什么材料，应变能提供的是 P_iso，体积项由 p 替代
        P = diff(self.material.strain_energy(F_v), F_v) - self.p * H
    
        tau = 0.1 * self.h**2 / self.material.mu
        
        R_u = div(P)
        
        # 根据材料对象是否定义了特殊的体积残差 (如 Simo-Taylor)
        if hasattr(self.material, 'pressure_residual'):
            R_p = self.material.pressure_residual(J, self.p)
        elif hasattr(self.material, 'kappa'):
            # 退化为最初的罚函数模型 R_p
            R_p = J - 1 + self.p / self.material.kappa
        else:
            R_p = J - 1

        Res_u = inner(P, grad(self.v))*dx #+ tau*inner(H, grad(self.v))*R_p*dx       
        Res_p = self.q*R_p*dx #- tau*inner(R_u, H*grad(self.q))*dx
        self.Res = Res_u + Res_p
        
        self.Jacobian = derivative(self.Res, self.w, self.w_trial)
    
    def solve(self, bcs, tol=1e-8):
        problem = NonlinearVariationalProblem(self.Res, self.w, bcs, self.Jacobian)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters["nonlinear_solver"] = "newton"
        solver.parameters["newton_solver"]["linear_solver"] = "mumps"
        solver.parameters["newton_solver"]["absolute_tolerance"] = tol
        solver.solve()
        
        u_sol, p_sol = self.w.split(deepcopy=True)
        return u_sol, p_sol
