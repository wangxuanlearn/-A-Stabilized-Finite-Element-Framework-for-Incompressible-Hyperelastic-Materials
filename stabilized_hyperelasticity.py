from dolfin import *
from ufl import Identity, tr, det, grad, inv, derivative, variable, ln, conditional
import numpy as np
from petsc4py import PETSc

# Material models
class SimoTaylorNeoHookean:
    def __init__(self, E):
        self.E = E
        # 对于完全不可压缩，nu = 0.5
        self.nu = 0.5
        self.mu = E / (2 * (1 + self.nu))

    def strain_energy(self, F):
        C = F.T * F
        J = det(F)
        # 完全不可压缩情况下的偏应变能部分
        return self.mu / 2 * (J**(-2/3) * tr(C) - 3)

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
    
        tau = 0.2 * self.h**2 / self.material.mu
        
        R_u = div(P)
        
        # 根据材料对象是否定义了特殊的体积残差 (如 Simo-Taylor)
        if hasattr(self.material, 'pressure_residual'):
            R_p = self.material.pressure_residual(J, self.p)
        elif hasattr(self.material, 'kappa'):
            # 退化为最初的罚函数模型 R_p
            R_p = J - 1 + self.p / self.material.kappa
        else:
            R_p = J - 1

        Res_u = inner(P, grad(self.v))*dx + tau*inner(H, grad(self.v))*R_p*dx       
        Res_p = self.q*R_p*dx - tau*inner(R_u, H*grad(self.q))*dx
        self.Res = Res_u + Res_p
        
        self.Jacobian = derivative(self.Res, self.w, self.w_trial)
    
class DynamicStabilizedHyperelasticitySolver:
    def __init__(self, mesh, boundaries, material, dt, rho=1.0, u_order=2, p_order=1, c3=0.1, c4=0.2):
        self.mesh = mesh
        self.boundaries = boundaries
        self.material = material
        self.h = CellDiameter(mesh)
        self.dt = Constant(dt)
        self.rho = Constant(rho)
        self.u_order = u_order
        self.p_order = p_order
        
        # Taylor-Hood (u_order >= 2, p_order = u_order-1): inf-sup 稳定，无需稳定项
        self._use_stab = not (u_order >= 2 and p_order == u_order - 1)
        
        # Mixed function space
        V_elem = VectorElement("CG", mesh.ufl_cell(), u_order)
        Q_elem = FiniteElement("CG", mesh.ufl_cell(), p_order)
        mixed_element = MixedElement([V_elem, Q_elem])
        self.W = FunctionSpace(mesh, mixed_element)
        
        self.V = self.W.sub(0).collapse()
        self.Q = self.W.sub(1).collapse()
        
        self.w = Function(self.W)
        self.w_n = Function(self.W)
        self.w_nn = Function(self.W)
        
        self.u, self.p = split(self.w)
        self.u_n, self.p_n = split(self.w_n)
        self.u_nn, self.p_nn = split(self.w_nn) 
        
        self.v, self.q = TestFunctions(self.W)
        
        self.w_trial = TrialFunction(self.W)
        
        I = Identity(mesh.geometry().dim())
        F = I + grad(self.u)
        F_v = variable(F)
        C = F_v.T*F_v
        J = det(F_v)
        H = J*inv(F_v).T  
        
        dt_val = self.dt
        a = (self.u - 2.0*self.u_n + self.u_nn) / (dt_val**2)

        P = diff(self.material.strain_energy(F_v), F_v) - self.p * H
    
        # 材料参数
        self.mu = material.mu
        
        # LSIC 稳定参数（动量方程）
        tau_p = c3 * self.mu * self.h**2
        # PSPG 稳定参数: τ = (c_tau/2)·max(dt_mu/100, min(dt_mu, Δt))
        #   dt_mu = h / c_mu  剪切波穿越单元时间
        tau_pspg = c4 * self.h**2 / material.mu
        if hasattr(self.material, 'pressure_residual'):
            R_p = self.material.pressure_residual(J, self.p)
        elif hasattr(self.material, 'kappa'):
            R_p = J - 1 + self.p / self.material.kappa
        else:
            R_p = J - 1

        # 完整动态残差（含惯性项）
        R_u = div(P)
        
        # GLS 变分形式
        Res_u = inner(self.rho * a, self.v)*dx + inner(P, grad(self.v))*dx
        Res_p = self.q*R_p*dx
        
        if self._use_stab:
            Res_u += tau_p*inner(H, grad(self.v))*R_p*dx
            Res_p += - tau_pspg*inner(R_u, H*grad(self.q))*dx
        
        self.Res = Res_u + Res_p
        self.Jacobian = derivative(self.Res, self.w, self.w_trial)
        
    def solve(self, bcs, tol=1e-8, use_predictor=True):
        if use_predictor:
            self.w.vector().zero()
            self.w.vector().axpy(2.0, self.w_n.vector())
            self.w.vector().axpy(-1.0, self.w_nn.vector())
            for bc in bcs:
                bc.apply(self.w.vector())
        
        # P2/P1 (Taylor-Hood): 天然 inf-sup → 全 Newton
        # P1/P1 + GLS: 稳定强度与静态一致 → 全 Newton 可行
        relax = 1.0
        
        problem = NonlinearVariationalProblem(self.Res, self.w, bcs, self.Jacobian)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters["nonlinear_solver"] = "newton"
        solver.parameters["newton_solver"]["linear_solver"] = "mumps"
        solver.parameters["newton_solver"]["absolute_tolerance"] = tol
        solver.parameters["newton_solver"]["maximum_iterations"] = 25
        solver.parameters["newton_solver"]["relaxation_parameter"] = relax
        solver.parameters["newton_solver"]["error_on_nonconvergence"] = True
        solver.solve()
        
        u_sol, p_sol = self.w.split(deepcopy=True)
        return u_sol, p_sol


    def dynamic_solve(self, bcs, num_steps, tol=1e-8):
        u_history = []
        p_history = []
        for step in range(num_steps):
            use_pred = (step > 0)
            if step > 0:
                self.w_nn.assign(self.w_n)
                self.w_n.assign(self.w)

            self.solve(bcs, tol, use_predictor=use_pred)

            u_sol, p_sol = self.w.split(deepcopy=True)
            u_history.append(u_sol)
            p_history.append(p_sol)
        return u_history, p_history