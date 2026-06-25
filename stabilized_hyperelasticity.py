from dolfin import *
from ufl import Identity, tr, det, grad, inv, derivative, variable, ln, conditional, as_vector, sqrt
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

class StandardReinforcedModel:
    def __init__(self, G, Gf, f0):
        self.G = G
        self.Gf = Gf
        self.f0 = f0
        self.mu = G  # 基体剪切模量，供稳定化参数使用
        
    def strain_energy(self, F):
        C = F.T*F
        J = det(F)
        I1_bar = J**(-2/3)*tr(C)
        f0_vec = as_vector(self.f0)
        I4f = inner(C * f0_vec, f0_vec)
        # 光滑 max(I4f, 1.0): P1/P2 均兼容
        I4f_smooth = 0.5 * (I4f + 1.0 + sqrt((I4f - 1.0)**2 + 1e-12))
        return self.G/2*(I1_bar - 3) + self.Gf/2 * (I4f_smooth - 1.0)**2

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


# =====================================================================
# 三变量 (u, v, p) 动态求解器 —— 严格不可压缩
# 约束: H:∇v = 0  (J̇ = 0 的率形式，等价于 J=1 恒定)
# =====================================================================
class DynamicThreeFieldSolver:
    def __init__(self, mesh, boundaries, material, dt, rho=1.0,
                 u_order=2, p_order=1, c3=0.1, c4=0.2):
        self.mesh = mesh
        self.boundaries = boundaries
        self.material = material
        self.h = CellDiameter(mesh)
        self.dt_c = Constant(dt)
        self.rho_c = Constant(rho)
        self._dt = dt
        self.u_order = u_order
        self.p_order = p_order

        # Taylor-Hood 组合免稳定项
        self._use_stab = not (u_order >= 2 and p_order == u_order - 1)

        # 三变量混合空间: 位移 u × 速度 v × 压力 p
        Vu = VectorElement("CG", mesh.ufl_cell(), u_order)
        Vv = VectorElement("CG", mesh.ufl_cell(), u_order)
        Q  = FiniteElement("CG", mesh.ufl_cell(), p_order)
        self.W = FunctionSpace(mesh, MixedElement([Vu, Vv, Q]))

        self.w = Function(self.W)
        self.w_n = Function(self.W)      # t_{n}
        self.w_nn = Function(self.W)     # t_{n-1}

        u, v, p = split(self.w)
        u_n, v_n, p_n = split(self.w_n)
        u_nn, v_nn, p_nn = split(self.w_nn)

        w_u, w_v, q = TestFunctions(self.W)
        self.w_trial = TrialFunction(self.W)

        dim = mesh.geometry().dim()
        I = Identity(dim)
        F = I + grad(u)
        F_v = variable(F)
        J = det(F_v)
        H = J * inv(F_v).T

        # 第一 Piola-Kirchhoff 应力
        P = diff(self.material.strain_energy(F_v), F_v) - p * H

        # BDF2 加速度 (位移用二阶，速度用一阶向后 Euler)
        a_u = (u - 2.0*u_n + u_nn) / (self.dt_c**2)
        a_v = (v - v_n) / self.dt_c

        # 不可压缩约束（率形式）:  R_inc = H : ∇v
        R_inc = inner(H, grad(v))

        # 稳定参数
        mu = material.mu
        tau_p = c3 * mu * self.h**2          # LSIC
        tau_c = c4 * self.h**2 / mu          # PSPG

        # ---- 弱形式 ----
        # (1) 运动学:  u̇ = v  →  ( (u-u_n)/Δt - v , w_u )
        Res_u = inner((u - u_n) / self.dt_c - v, w_u) * dx

        # (2) 动量:  ρ v̇ - ∇·P = 0  + LSIC 稳定
        Res_v = (inner(self.rho_c * a_v, w_v) * dx
                 + inner(P, grad(w_v)) * dx)

        # (3) 不可压缩:  H:∇v = 0  + PSPG 稳定
        Res_p = inner(R_inc, q) * dx

        if self._use_stab:
            Res_v += tau_p * inner(H, grad(w_v)) * R_inc * dx
            Res_p += - tau_c * inner(div(P), dot(H, grad(q))) * dx

        self.Res = Res_u + Res_v + Res_p
        self.Jacobian = derivative(self.Res, self.w, self.w_trial)

    def solve(self, bcs, tol=1e-8, use_predictor=True):
        if use_predictor:
            # 线性预测 u_pred = 2*u_n - u_nn
            u_n_vec = self.w_n.sub(0, deepcopy=True).vector()
            u_nn_vec = self.w_nn.sub(0, deepcopy=True).vector()
            v_n_vec = self.w_n.sub(1, deepcopy=True).vector()
            v_nn_vec = self.w_nn.sub(1, deepcopy=True).vector()

            self.w.vector().zero()
            # u ← 2*u_n - u_nn
            self.w.sub(0).vector().axpy(2.0, u_n_vec)
            self.w.sub(0).vector().axpy(-1.0, u_nn_vec)
            # v ← 2*v_n - v_nn
            self.w.sub(1).vector().axpy(2.0, v_n_vec)
            self.w.sub(1).vector().axpy(-1.0, v_nn_vec)

            for bc in bcs:
                bc.apply(self.w.vector())

        problem = NonlinearVariationalProblem(self.Res, self.w, bcs, self.Jacobian)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters["nonlinear_solver"] = "newton"
        solver.parameters["newton_solver"]["linear_solver"] = "mumps"
        solver.parameters["newton_solver"]["absolute_tolerance"] = tol
        solver.parameters["newton_solver"]["maximum_iterations"] = 25
        solver.parameters["newton_solver"]["relaxation_parameter"] = 1.0
        solver.parameters["newton_solver"]["error_on_nonconvergence"] = True
        solver.solve()

        u_sol, v_sol, p_sol = self.w.split(deepcopy=True)
        return u_sol, v_sol, p_sol

    def dynamic_solve(self, bcs, num_steps, tol=1e-8):
        u_history, v_history, p_history = [], [], []
        for step in range(num_steps):
            use_pred = (step > 0)
            if step > 0:
                self.w_nn.assign(self.w_n)
                self.w_n.assign(self.w)
            self.solve(bcs, tol, use_predictor=use_pred)
            u_sol, v_sol, p_sol = self.w.split(deepcopy=True)
            u_history.append(u_sol)
            v_history.append(v_sol)
            p_history.append(p_sol)
        return u_history, v_history, p_history


# =====================================================================
# 各向异性纤维增强动态求解器 (u, p) —— GLS 稳定化
# =====================================================================
class DynamicAnisotropicSolver:
    def __init__(self, mesh, boundaries, material, dt, rho=1.0,
                 u_order=2, p_order=1, c3=0.1, c4=0.1):
        self.mesh = mesh
        self.boundaries = boundaries
        self.material = material
        self.h = CellDiameter(mesh)
        self.dt = Constant(dt)
        self.rho = Constant(rho)
        self.u_order = u_order
        self.p_order = p_order

        self._use_stab = not (u_order >= 2 and p_order == u_order - 1)

        V_elem = VectorElement("CG", mesh.ufl_cell(), u_order)
        Q_elem = FiniteElement("CG", mesh.ufl_cell(), p_order)
        self.W = FunctionSpace(mesh, MixedElement([V_elem, Q_elem]))

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
        J = det(F_v)
        H = J * inv(F_v).T

        a = (self.u - 2.0 * self.u_n + self.u_nn) / (self.dt ** 2)
        P = diff(self.material.strain_energy(F_v), F_v) - self.p * H

        # 瞬时切线刚度: ∂ψ/∂I1_bar + ∂ψ/∂I4f (光滑化)
        f0_vec = as_vector(self.material.f0)
        C_bar = F_v.T * F_v
        I4f_val = inner(C_bar * f0_vec, f0_vec)
        I4f_smooth = 0.5 * (I4f_val + 1.0 + sqrt((I4f_val - 1.0)**2 + 1e-12))
        mu = self.material.G / 2.0 + self.material.Gf * (I4f_smooth - 1.0)

        # LSIC: c3·μ·h² (正比于刚度)  |  PSPG: c4·h²/μ (正比于柔度)
        tau_p = c3 * mu * self.h ** 2
        tau_pspg = c4 * self.h ** 2 / mu

        if hasattr(self.material, 'pressure_residual'):
            R_p = self.material.pressure_residual(J, self.p)
        elif hasattr(self.material, 'kappa'):
            R_p = J - 1 + self.p / self.material.kappa
        else:
            R_p = J - 1

        R_u = self.rho*a - div(P)
        Res_u = inner(self.rho * a, self.v) * dx + inner(P, grad(self.v)) * dx
        Res_p = self.q * R_p * dx

        if self._use_stab:
            Res_u += tau_p * inner(H, grad(self.v)) * R_p * dx
            Res_p += - tau_pspg * inner(div(P), dot(H, grad(self.q))) * dx

        self.Res = Res_u + Res_p
        self.Jacobian = derivative(self.Res, self.w, self.w_trial)

    def solve(self, bcs, tol=1e-8, use_predictor=True):
        if use_predictor:
            self.w.assign(self.w_n)
            for bc in bcs:
                bc.apply(self.w.vector())

        problem = NonlinearVariationalProblem(self.Res, self.w, bcs, self.Jacobian)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters["nonlinear_solver"] = "newton"
        solver.parameters["newton_solver"]["linear_solver"] = "mumps"
        solver.parameters["newton_solver"]["absolute_tolerance"] = tol
        solver.parameters["newton_solver"]["maximum_iterations"] = 25
        solver.parameters["newton_solver"]["relaxation_parameter"] = 1.0
        solver.parameters["newton_solver"]["error_on_nonconvergence"] = True
        solver.solve()

        u_sol, p_sol = self.w.split(deepcopy=True)
        return u_sol, p_sol

    def dynamic_solve(self, bcs, num_steps, tol=1e-8):
        u_history, p_history = [], []
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


