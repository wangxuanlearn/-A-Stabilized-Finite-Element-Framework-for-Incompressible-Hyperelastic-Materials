from dolfin import *
from ufl import Identity, tr, det, grad, inv, derivative, variable, ln, conditional, as_vector, sqrt, Max, Min
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
    def __init__(self, mesh, boundaries, material, dt, rho=1.0, u_order=2, p_order=1, c1=1, c2=0.1, c3=1):
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
        
        # LSIC 稳定参数（动量方程）—— 正比于 μ·h²
        tau_p = c3 * self.mu * self.h**2
        # PSPG 稳定参数 —— 时间-空间调和尺度
        tau_u = 1.0 / (c1 * self.rho / self.dt**2 + c2 * self.mu / self.h**2)
        
        if hasattr(self.material, 'pressure_residual'):
            R_p = self.material.pressure_residual(J, self.p)
        elif hasattr(self.material, 'kappa'):
            R_p = J - 1 + self.p / self.material.kappa
        else:
            R_p = J - 1

        # GLS 变分形式 —— PSPG 用 div(P)，惯性项 ρa 由 Galerkin 质量矩阵处理
        Res_u = inner(self.rho * a, self.v)*dx + inner(P, grad(self.v))*dx
        Res_p = self.q*R_p*dx
        
        if self._use_stab:
            Res_u += tau_p*inner(H, grad(self.v))*R_p*dx
            Res_p += - tau_u*inner(div(P), H*grad(self.q))*dx
        
        self.Res = Res_u + Res_p
        self.Jacobian = derivative(self.Res, self.w, self.w_trial)
        
    def solve(self, bcs, tol=1e-8, use_predictor=True):
        if use_predictor:
            self.w.vector().zero()
            self.w.vector().axpy(2.0, self.w_n.vector())
            self.w.vector().axpy(-1.0, self.w_nn.vector())
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

        # tau_p = c3 * mu * self.h ** 2
        tau_p = c3 * self.h ** 2 / mu
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
            # Res_u += tau_p * inner(H, grad(self.v)) * R_p * dx
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
        solver.parameters["newton_solver"]["relaxation_parameter"] = 0.7
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


# =====================================================================
# (v,p) + 显式 u + 完整 R_mom PSPG —— GLS 严格一致
# 惯性 Jacobian ~ τ·ρ/Δt ≪ τ·ρ/Δt²，自然压低 100 倍
# =====================================================================
class DynamicConsistentGLSSolver:
    def __init__(self, mesh, boundaries, material, dt, rho=1.0,
                 v_order=2, p_order=1, c3=0.3, c4=0.2, force_stab=False):
        self.mesh = mesh; self.boundaries = boundaries
        self.material = material; self.h = CellDiameter(mesh)
        self.dt_c = Constant(dt); self.rho_c = Constant(rho)
        self._dt = dt; self.v_order = v_order; self.p_order = p_order
        self._use_stab = not (v_order >= 2 and p_order == v_order - 1) or force_stab

        Vv = VectorElement("CG", mesh.ufl_cell(), v_order)
        Q  = FiniteElement("CG", mesh.ufl_cell(), p_order)
        self.W = FunctionSpace(mesh, MixedElement([Vv, Q]))
        self.w = Function(self.W); self.w_n = Function(self.W)
        v, p = split(self.w); v_n, p_n = split(self.w_n)
        self.w_v, q = TestFunctions(self.W)
        self.w_trial = TrialFunction(self.W)

        Vu = VectorFunctionSpace(mesh, "CG", v_order)
        self.u_n = Function(Vu)
        u = self.u_n + self.dt_c * v

        dim = mesh.geometry().dim(); I = Identity(dim)
        F = I + grad(u); F_v = variable(F)
        J = det(F_v); H = J * inv(F_v).T
        P = diff(self.material.strain_energy(F_v), F_v) - p * H

        R_mom = self.rho_c * (v - v_n) / self.dt_c- div(P) 
        mu = material.mu
        tau_p = c3 * mu           #* self.h ** 2
        tau_c = c4 * self.h ** 2 / mu
        # tau_c = 1.0 / ( c1*self.rho_c/self.dt_c + c2*mu/self.h**2 )

        if hasattr(self.material, 'pressure_residual'):
            R_p = self.material.pressure_residual(J, p)
        elif hasattr(self.material, 'kappa'):
            R_p = J - 1 + p / self.material.kappa
        else:
            R_p = inner(H, grad(v))

        Res_v = (inner(self.rho_c * (v - v_n) / self.dt_c, self.w_v) * dx
                 + inner(P, grad(self.w_v)) * dx)
        Res_p = inner(R_p, q) * dx
        if self._use_stab:
            Res_v += tau_p * inner(H, grad(self.w_v)) * R_p * dx
            Res_p += tau_c * inner(R_mom, dot(H, grad(q))) * dx
        self.Res = Res_v + Res_p
        self.Jacobian = derivative(self.Res, self.w, self.w_trial)

    def solve(self, bcs, tol=1e-8, use_predictor=True):
        if use_predictor:
            self.w.assign(self.w_n)
            for bc in bcs: bc.apply(self.w.vector())
        problem = NonlinearVariationalProblem(self.Res, self.w, bcs, self.Jacobian)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters["nonlinear_solver"] = "newton"
        solver.parameters["newton_solver"]["linear_solver"] = "mumps"
        solver.parameters["newton_solver"]["absolute_tolerance"] = tol
        solver.parameters["newton_solver"]["maximum_iterations"] = 25
        solver.parameters["newton_solver"]["relaxation_parameter"] = 0.7
        solver.parameters["newton_solver"]["error_on_nonconvergence"] = True
        solver.solve()
        v_sol, p_sol = self.w.split(deepcopy=True)
        self.u_n.vector().axpy(self._dt, v_sol.vector())
        return Function(self.u_n.function_space(), self.u_n.vector()), v_sol, p_sol


class DynamicConsistentGLSBDF2Solver:
    """BDF2 时间积分的一致 GLS 稳定化求解器 (v, p 混合形式).

    统一的系数形式:
      加速度: a = (α1·v - α2·v_n + α3·v_nn) / (α0·Δt)
      位移:   u = (β1·u_n - β2·u_nn + β3·Δt·v) / β0

    第一步用 Backward Euler:       α=[1,1,0,1], β=[1,0,1,1]
    第二步及以后用 BDF2:          α=[3,4,1,2], β=[4,1,2,3]
    """
    def __init__(self, mesh, boundaries, material, dt, rho=1.0,
                 v_order=2, p_order=1, c3=0.3, c4=0.2, force_stab=False):
        self.mesh = mesh; self.boundaries = boundaries
        self.material = material; self.h = CellDiameter(mesh)
        self.dt_c = Constant(dt); self.rho_c = Constant(rho)
        self._dt = dt; self.v_order = v_order; self.p_order = p_order
        self._use_stab = not (v_order >= 2 and p_order == v_order - 1) or force_stab

        # ---- 统一的符号系数（运行时可通过 assign 切换 BE / BDF2） ----
        self.alpha_0 = Constant(1.0)   # Δt 分母
        self.alpha_1 = Constant(1.0)   # v 系数
        self.alpha_2 = Constant(1.0)   # v_n 系数
        self.alpha_3 = Constant(0.0)   # v_nn 系数
        self.beta_0  = Constant(1.0)   # u 分母
        self.beta_1  = Constant(1.0)   # u_n 系数
        self.beta_2  = Constant(0.0)   # u_nn 系数
        self.beta_3  = Constant(1.0)   # Δt·v 系数

        Vv = VectorElement("CG", mesh.ufl_cell(), v_order)
        Q  = FiniteElement("CG", mesh.ufl_cell(), p_order)
        self.W = FunctionSpace(mesh, MixedElement([Vv, Q]))
        self.w = Function(self.W)
        self.w_n = Function(self.W)      # t^n
        self.w_nn = Function(self.W)     # t^{n-1}
        v, p = split(self.w)
        v_n, p_n = split(self.w_n)
        v_nn, p_nn = split(self.w_nn)
        self.w_v, q = TestFunctions(self.W)
        self.w_trial = TrialFunction(self.W)

        # 统一加速度: (α1·v - α2·v_n + α3·v_nn) / (α0·Δt)
        a_gen = (self.alpha_1 * v - self.alpha_2 * v_n
                 + self.alpha_3 * v_nn) / (self.alpha_0 * self.dt_c)

        Vu = VectorFunctionSpace(mesh, "CG", v_order)
        self.u_n = Function(Vu)
        self.u_nn = Function(Vu)
        # 统一位移: u = (β1·u_n - β2·u_nn + β3·Δt·v) / β0
        u = (self.beta_1 * self.u_n - self.beta_2 * self.u_nn
             + self.beta_3 * self.dt_c * v) / self.beta_0

        dim = mesh.geometry().dim(); I = Identity(dim)
        F = I + grad(u); F_v = variable(F)
        J = det(F_v); H = J * inv(F_v).T
        P = diff(self.material.strain_energy(F_v), F_v) - p * H

        R_mom = self.rho_c * a_gen - div(P)
        mu = material.mu
        # tau_p = c3 * mu
        # tau_c = c4 * self.h ** 2 / mu
        mu = material.mu
        c_mu = sqrt(mu / self.rho_c)            # 剪切波速度 c_mu^e = sqrt(mu/rho)
        dt_mu = self.h / c_mu                    # Δt_mu = h^e / c_mu^e

        # === 网格特征时间诊断 ===
        c_mu_val = float(sqrt(mu / rho))
        h_min = MPI.min(MPI.comm_world, mesh.hmin())
        h_max = MPI.max(MPI.comm_world, mesh.hmax())
        dt_mu_min = h_min / c_mu_val
        dt_mu_max = h_max / c_mu_val
        if MPI.comm_world.rank == 0:
            print(f"\nCFL 诊断: dt={self._dt:.6f}, "
                  f"dt_mu ∈ [{dt_mu_min:.6f}, {dt_mu_max:.6f}]")
            print(f"  h ∈ [{h_min:.4f}, {h_max:.4f}], "
                  f"c_mu = {c_mu_val:.4f}  (mu={float(mu):.2f}, rho={rho})")
            if self._dt > dt_mu_min:
                ratio = self._dt / dt_mu_min
                print(f"  ⚠ dt 超出 dt_mu_min {ratio:.1f}x "
                      f"— 隐式 BDF2 仍可稳定，tau=Min(dt_mu,dt) 已自动适配")
                if ratio > 10.0:
                    print(f"  ⚠ 比值过大，建议减小 dt 或粗化网格")
            print()

        self.c_tau = Constant(0.01)              # c_tau ∈ [0.01, 0.03]，可作为构造参数暴露出来
        tau = (self.c_tau / 2.0) * Max(dt_mu / 100.0, Min(dt_mu, self.dt_c))

        tau_p = tau
        tau_c = tau

        if hasattr(self.material, 'pressure_residual'):
            R_p = self.material.pressure_residual(J, p)
        elif hasattr(self.material, 'kappa'):
            R_p = J - 1 + p / self.material.kappa
        else:
            R_p = inner(H, grad(v))

        Res_v = (inner(self.rho_c * a_gen, self.w_v) * dx
                 + inner(P, grad(self.w_v)) * dx)
        Res_p = inner(R_p, q) * dx
        if self._use_stab:
            Res_v += tau_p * inner(H, grad(self.w_v)) * R_p * dx
            Res_p += tau_c * inner(R_mom, dot(H, grad(q))) * dx
        self.Res = Res_v + Res_p
        self.Jacobian = derivative(self.Res, self.w, self.w_trial)

        # 初始默认 BE
        self._set_backward_euler()

    # ---------- 系数切换 ----------
    def _set_backward_euler(self):
        """第一步：一阶 Backward Euler"""
        self.alpha_0.assign(1.0); self.alpha_1.assign(1.0)
        self.alpha_2.assign(1.0); self.alpha_3.assign(0.0)
        self.beta_0.assign(1.0);  self.beta_1.assign(1.0)
        self.beta_2.assign(0.0);  self.beta_3.assign(1.0)
        self._is_bdf2 = False

    def _set_bdf2(self):
        """第二步及以后：二阶 BDF2"""
        self.alpha_0.assign(2.0); self.alpha_1.assign(3.0)
        self.alpha_2.assign(4.0); self.alpha_3.assign(1.0)
        self.beta_0.assign(3.0);  self.beta_1.assign(4.0)
        self.beta_2.assign(1.0);  self.beta_3.assign(2.0)
        self._is_bdf2 = True

    # ---------- 位移更新（统一公式） ----------
    def _update_displacement(self, v_sol_vec):
        """u^{n+1} = (β1·u^n - β2·u^{n-1} + β3·Δt·v) / β0"""
        b0 = self.beta_0.values()[0]
        b1 = self.beta_1.values()[0]
        b2 = self.beta_2.values()[0]
        b3 = self.beta_3.values()[0]

        # u_tmp = β1 * u_n
        u_new = self.u_n.vector().copy()
        u_new *= b1
        # u_tmp -= β2 * u_nn
        u_new.axpy(-b2, self.u_nn.vector())
        # u_tmp += β3 * Δt * v_sol
        u_new.axpy(b3 * self._dt, v_sol_vec)
        # u_tmp /= β0
        u_new /= b0

        # 更新历史: u_nn ← u_n, u_n ← u_new
        self.u_nn.assign(self.u_n)
        self.u_n.vector().zero()
        self.u_n.vector().axpy(1.0, u_new)

    # ---------- 求解 ----------
    def solve(self, bcs, tol=1e-8, use_predictor=True, first_step=False):
        # 1. 设置时间积分系数
        if first_step:
            self._set_backward_euler()
        else:
            self._set_bdf2()

        # 2. 预测器
        if use_predictor:
            self.w.vector().zero()
            if self._is_bdf2:
                # BDF2: w_pred = 2·w_n - w_nn
                self.w.vector().axpy(2.0, self.w_n.vector())
                self.w.vector().axpy(-1.0, self.w_nn.vector())
            else:
                # BE: w_pred = w_n
                self.w.vector().axpy(1.0, self.w_n.vector())
            for bc in bcs:
                bc.apply(self.w.vector())

        # 3. 求解非线性问题（低松弛防止大载荷步Newton发散）
        problem = NonlinearVariationalProblem(self.Res, self.w, bcs, self.Jacobian)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters["nonlinear_solver"] = "newton"
        solver.parameters["newton_solver"]["linear_solver"] = "mumps"
        solver.parameters["newton_solver"]["absolute_tolerance"] = tol
        solver.parameters["newton_solver"]["maximum_iterations"] = 25
        solver.parameters["newton_solver"]["relaxation_parameter"] = 1.0
        solver.parameters["newton_solver"]["error_on_nonconvergence"] = True
        solver.solve()
        v_sol, p_sol = self.w.split(deepcopy=True)

        # 4. 位移更新 (BDF2 或 BE 统一公式)
        self._update_displacement(v_sol.vector())
        return Function(self.u_n.function_space(), self.u_n.vector()), v_sol, p_sol


