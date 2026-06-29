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
    def __init__(self, G, Gf, f0, kappas, c_init, epsilon, phi0, phi_crit, K0):
        self.G = G; self.Gf = Gf; self.f0 = f0
        self.mu = G
        self.kappas = kappas
        self.c = Constant(c_init)  # 动态罚参数，由 solver 每步更新
        self.epsilon = epsilon
        self.phi0 = phi0
        self.phi_crit = phi_crit
        self.K0 = K0  # 渗透率（标量或 2×2 列表）
        
    def strain_energy(self, F):
        C = F.T*F; J = det(F)
        I1_bar = J**(-2/3)*tr(C)
        f0_vec = as_vector(self.f0)
        I4f = inner(C * f0_vec, f0_vec)
        I4f_smooth = 0.5*(I4f + 1.0 + sqrt((I4f - 1.0)**2 + 1e-12))
        return self.G/2*(I1_bar - 3) + self.Gf/2*(I4f_smooth - 1.0)**2
    
    def PV(self, m):
        return self.kappas * m
    
    def PC(self, m):
        return self.c * self.epsilon / (self.epsilon**2 + (m + self.phi0 - self.phi_crit)**2)

class MooneyRivlin:
    def __init__(self, C10, C01, kappas, c_init, epsilon, phi0, phi_crit, K0):
        self.C10 = C10
        self.C01 = C01
        self.mu = 2.0 * (C10 + C01)   # 小变形剪切模量，供稳定化参数使用
        self.kappas = kappas
        self.c = Constant(c_init)  # 动态罚参数，由 solver 每步更新
        self.epsilon = epsilon
        self.phi0 = phi0
        self.phi_crit = phi_crit
        self.K0 = K0  # 渗透率（标量或 2×2 列表）

    def strain_energy(self, F):
        C = F.T*F
        J = det(F)
        I1_bar = J**(-2/3)*tr(C)
        I2_bar = J**(-4/3)*0.5*(tr(C)**2 - tr(C*C))
        return self.C10*(I1_bar - 3) + self.C01*(I2_bar - 3)
    
    def PV(self, m):
        return self.kappas * m
    
    def PC(self, m):
        return self.c * self.epsilon / (self.epsilon**2 + (m + self.phi0 - self.phi_crit)**2)

class DynamicStabilizedPoroelasticitySolver:
    def __init__(self, mesh, boundaries, material, dt, rho0_f=1.0, rho=1.0,
                 u_order=2, p_order=1, m_order=1, c1=1, c2=4, c3=0.1,
                 beta=0.0):
        self.mesh = mesh; self.boundaries = boundaries
        self.material = material; self.h = CellDiameter(mesh)
        self.dt = Constant(dt); self.rho = Constant(rho)
        self.rho0_f = Constant(rho0_f)
        self._dt = dt; self.beta = beta
        self.u_order = u_order; self.p_order = p_order; self.m_order = m_order
        
        self._use_stab = not (u_order >= 2 and p_order == u_order - 1)

        Vu = VectorElement("CG", mesh.ufl_cell(), u_order)
        Qp = FiniteElement("CG", mesh.ufl_cell(), p_order)
        Qm = FiniteElement("CG", mesh.ufl_cell(), m_order)
        self.W = FunctionSpace(mesh, MixedElement([Vu, Qp, Qm]))

        self.w = Function(self.W); self.w_n = Function(self.W); self.w_nn = Function(self.W)
        u, p, m = split(self.w)
        u_n, p_n, m_n = split(self.w_n)
        u_nn, p_nn, m_nn = split(self.w_nn)
        w_u, q, r = TestFunctions(self.W)
        self.w_u = w_u
        self.w_trial = TrialFunction(self.W)

        dim = mesh.geometry().dim(); I = Identity(dim)
        F = I + grad(u); F_v = variable(F)
        J = det(F_v); H = J * inv(F_v).T
        a = (u - 2.0*u_n + u_nn) / (self.dt**2)
        P = diff(self.material.strain_energy(F_v), F_v) - p * H

        # 渗透率张量
        K0_val = self.material.K0
        K0_tensor = K0_val * I if np.isscalar(K0_val) else \
                    as_matrix([[K0_val[0], K0_val[1]], [K0_val[2], K0_val[3]]])
        K_perm = J * inv(F_v) * K0_tensor * inv(F_v).T

        # 罚参数 c = 上一时刻 max|p+PV+PC|, 初始值从 material.c 读取
        dm = m + self.material.phi0 - self.material.phi_crit
        PV_expr = self.material.kappas * m
        PC_expr = self.material.c * self.material.epsilon / (self.material.epsilon**2 + dm**2)
        # K_m = K * (∂PV/∂m + ∂PC/∂m)
        dPV_dm = self.material.kappas
        dPC_dm = -2.0 * self.material.c * self.material.epsilon * dm / \
                 (self.material.epsilon**2 + dm**2)**2
        K_m = K_perm * (dPV_dm + dPC_dm)
        # S = -β·(p + PV + PC)
        S = - Constant(beta) * (p + PV_expr + PC_expr)

        mu = material.mu
        tau_p = c3 * mu * self.h**2
        tau_u = 1.0 / (c1 * self.rho / self.dt**2 + c2 * mu / self.h**2)
        tau_m = 0.2 * self.h**2 / mu

        R_inc = J - 1.0 - m / self.rho0_f
        R_darcy = (m - m_n) / self.dt - div(K_perm * grad(p)) - div(K_m * grad(m)) - S

        Res_u = inner((self.rho + m) * a, w_u) * dx + inner(P, grad(w_u)) * dx
        Res_p = q * R_inc * dx
        Res_m = r * R_darcy * dx

        if self._use_stab:
            Res_u += tau_p * inner(H, grad(w_u)) * R_inc * dx
            Res_p += - tau_u * inner(div(P), dot(H, grad(q))) * dx
            Res_m += tau_m * dot(K_perm * grad(r), grad(p)) * dx

        self.Res = Res_u + Res_p + Res_m
        self.Jacobian = derivative(self.Res, self.w, self.w_trial)

        # 保存 PV+PC 的计算函数以供后处理
        self._PV_expr = PV_expr
        self._PC_expr = PC_expr
        
    def _update_c_penalty(self):
        """用当前解的 max(p + PV + PC) 更新 material.c"""
        u, p, m = self.w.split(deepcopy=True)
        Qm = self.W.sub(2).collapse()
        
        # 用 deepcopy 出的 Function 直接计算 PV 和 PC，避免 UFL 符号引用错误
        dm = m + self.material.phi0 - self.material.phi_crit
        PV = self.material.kappas * m
        PC = self.material.c * self.material.epsilon**2 / (self.material.epsilon**2 + dm**2)
        total = project(p + PV + PC, Qm)
        
        # 取最大值（不是绝对值）
        local_max = max(total.vector().get_local())
        global_max = MPI.max(MPI.comm_world, local_max)
        
        # 下限保护，避免 c 太小
        self.material.c.assign(max(global_max, 1.0))
        
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
        solver.parameters["newton_solver"]["relaxation_parameter"] = 0.5
        solver.parameters["newton_solver"]["error_on_nonconvergence"] = True
        solver.solve()
        
        u_sol, p_sol, m_sol = self.w.split(deepcopy=True)
        self._update_c_penalty()
        return u_sol, p_sol, m_sol


    def dynamic_solve(self, bcs, num_steps, tol=1e-8):
        u_hist, p_hist, m_hist = [], [], []
        for step in range(num_steps):
            use_pred = (step > 0)
            if step > 0:
                self.w_nn.assign(self.w_n)
                self.w_n.assign(self.w)
            self.solve(bcs, tol, use_predictor=use_pred)
            u_sol, p_sol, m_sol = self.w.split(deepcopy=True)
            u_hist.append(u_sol); p_hist.append(p_sol); m_hist.append(m_sol)
        return u_hist, p_hist, m_hist
