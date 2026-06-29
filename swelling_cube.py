from dolfin import *
from meshes import create_swellingcube_mesh
from stabilized_poroelasticity import MooneyRivlin, DynamicStabilizedPoroelasticitySolver
from mpi4py import MPI as pyMPI
import os, time, csv, glob
import numpy as np

if not os.path.exists("results"):
    os.makedirs("results")
if not os.path.exists("results/swelling_poroelastic"):
    os.makedirs("results/swelling_poroelastic")

# # 清理上次运行可能残留的损坏 HDF5/XDMF 文件
if MPI.comm_world.rank == 0:
    for pattern in ["results/swelling_poroelastic/*.h5",
                    "results/swelling_poroelastic/*.xdmf"]:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
                print(f"  已清理旧文件: {f}")
            except OSError:
                pass
MPI.comm_world.barrier()

t_total_start = time.time()
timings = {}

# Parameters
C1 = 2e4       
C2 = 330    
kappas = 2e4    
beta_val = 0
epsilon = 0.001; phi0 = 0.1; phi_crit = 0.001
K0 = 1e-4       # *cm3.s/g
rho0_f = 1.0   # kg/m³
dt = 0.01; 
rho = 1.0      # kg/m³
num_steps = 200
c_init = 1.0
u_order = 2; p_order = 1; m_order = 1

# Mesh
t0 = time.time()
mesh, boundaries = create_swellingcube_mesh(mesh_size=0.25)
timings['网格生成'] = time.time() - t0
if MPI.comm_world.rank == 0:
    print(f"Mesh: {timings['网格生成']:.2f}s")

# Solver
t0 = time.time()
material = MooneyRivlin(C1, C2, kappas, c_init, epsilon, phi0, phi_crit, K0)
solver = DynamicStabilizedPoroelasticitySolver(
    mesh, boundaries, material, dt=dt, rho0_f=rho0_f, rho=rho,
    u_order=u_order, p_order=p_order, m_order=m_order, c1=1, c2=4, c3=0.1,
    beta=beta_val
)
timings['求解器初始化'] = time.time() - t0
if MPI.comm_world.rank == 0:
   print(f"求解器初始化: {timings['求解器初始化']:.2f} 秒")

# BCs: 位移约束（防止刚体运动）+ 流体质量边界条件
bc_u_left  = DirichletBC(solver.W.sub(0).sub(0), Constant(0.0), boundaries, 1)   # 左面 u_x=0
bc_u_bottom = DirichletBC(solver.W.sub(0).sub(1), Constant(0.0), boundaries, 3)  # 底面 u_y=0
bc_u_back   = DirichletBC(solver.W.sub(0).sub(2), Constant(0.0), boundaries, 5)  # 后面 u_z=0

bcs = [bc_u_left, bc_u_bottom, bc_u_back]

# 时变流体质量：左端面 m(t) = 0.5 * (1 - exp(-t²/0.25)) * rho0_f
t_const = Constant(0.0)
m_left = Expression("0.5 * (1 - exp(-pow(t, 2) / 0.25)) * rho0_f",
                    degree=4, t=t_const, rho0_f=rho0_f)
bc_m_left  = DirichletBC(solver.W.sub(2), m_left, boundaries, 1)
bc_m_right = DirichletBC(solver.W.sub(2), Constant(0.0), boundaries, 2)
bcs.extend([bc_m_left, bc_m_right])

# Traction on top [5,15]
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
solver.Jacobian = derivative(solver.Res, solver.w, solver.w_trial)

# XDMF 输出
xdmf_u = XDMFFile(MPI.comm_world, "results/swelling_poroelastic/swelling_displacement.xdmf")
xdmf_p = XDMFFile(MPI.comm_world, "results/swelling_poroelastic/swelling_pressure.xdmf")
xdmf_m = XDMFFile(MPI.comm_world, "results/swelling_poroelastic/swelling_fluid_mass.xdmf")
xdmf_u.parameters["flush_output"] = True
xdmf_p.parameters["flush_output"] = True
xdmf_m.parameters["flush_output"] = True

if MPI.comm_world.rank == 0:
    print(f"\nSolving {num_steps} steps...")
t0 = time.time()
time_points = []
uy_vals = []
point = Point(0, 0, 0)  # 左下角监测点
point_mid = Point(0.5, 0.5, 0.5)  # 中心监测点
point_top = Point(1.0, 1.0, 1.0)  # 顶面中心监测点

# 监测点记录：存储 m 和 p+PV+PC
point_names = ['left_bottom', 'center', 'top']
points_list = [point, point_mid, point_top]
monitor_m = {name: [] for name in point_names}
monitor_p_total = {name: [] for name in point_names}

try:
    for step in range(num_steps):
        t = step * dt
        t_const.assign(t)   # 更新时间常数供 Expression 使用
        if MPI.comm_world.rank == 0 and step % 20 == 0:
            print(f"Step {step+1}/{num_steps}, t={t:.3f}")

        u_sol, p_sol, m_sol = solver.solve(bcs, tol=1e-8, use_predictor=(step>0))
        if step > 0: solver.w_nn.assign(solver.w_n)
        solver.w_n.assign(solver.w)

        # 监测三个点的 m 和 p+PV+PC
        Qm = solver.W.sub(2).collapse()
        dm_sol = m_sol + solver.material.phi0 - solver.material.phi_crit
        PV_val = solver.material.kappas * m_sol
        PC_val = solver.material.c * solver.material.epsilon / (
            solver.material.epsilon**2 + dm_sol**2)
        p_total = project(p_sol + PV_val + PC_val, Qm)
        for name, pt in zip(point_names, points_list):
            m_val, m_cnt = 0.0, 0
            p_val, p_cnt = 0.0, 0
            try:
                m_val = m_sol(pt); m_cnt = 1
            except RuntimeError:
                pass
            try:
                p_val = p_total(pt); p_cnt = 1
            except RuntimeError:
                pass
            m_val = pyMPI.COMM_WORLD.allreduce(m_val, op=pyMPI.SUM)
            m_cnt = pyMPI.COMM_WORLD.allreduce(m_cnt, op=pyMPI.SUM)
            p_val = pyMPI.COMM_WORLD.allreduce(p_val, op=pyMPI.SUM)
            p_cnt = pyMPI.COMM_WORLD.allreduce(p_cnt, op=pyMPI.SUM)
            monitor_m[name].append(m_val / m_cnt if m_cnt > 0 else 0.0)
            monitor_p_total[name].append(p_val / p_cnt if p_cnt > 0 else 0.0)

        u_sol.rename("displacement", "displacement")
        p_sol.rename("pressure", "pressure")
        m_sol.rename("fluid_mass", "fluid_mass")
        xdmf_u.write(u_sol, t)
        xdmf_p.write(p_sol, t)
        xdmf_m.write(m_sol, t)

        uy = 0.0; cnt = 0
        try: uy = u_sol(point)[1]; cnt = 1
        except RuntimeError: pass
        uy = pyMPI.COMM_WORLD.allreduce(uy, op=pyMPI.SUM)
        cnt = pyMPI.COMM_WORLD.allreduce(cnt, op=pyMPI.SUM)
        time_points.append(t); uy_vals.append(uy/cnt if cnt>0 else 0.0)

    if MPI.comm_world.rank == 0:
        print(f"Done! Final uy = {uy_vals[-1]:.6f}")
        with open(f"results/swelling_poroelastic/uy_u{u_order}_p{p_order}.csv", 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['time','uy'])
            for ti, uyi in zip(time_points, uy_vals): w.writerow([ti, uyi])
        # 保存监测点数据
        for name in point_names:
            with open(f"results/swelling_poroelastic/monitor_{name}.csv", 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['time', 'm', 'p+PV+PC'])
                for ti, mi, pi in zip(time_points, monitor_m[name], monitor_p_total[name]):
                    w.writerow([ti, mi, pi])
        print("Monitoring data saved to results/swelling_poroelastic/monitor_*.csv")
    if MPI.comm_world.rank == 0:
        print("Dynamic solution completed!")
    xdmf_u.close(); xdmf_p.close(); xdmf_m.close()
except Exception as e:
    if MPI.comm_world.rank == 0:
        print(f"Failed: {e}")
    pyMPI.COMM_WORLD.Abort(1)

timings['求解'] = time.time() - t0
if MPI.comm_world.rank == 0:
    print(f"Total time: {time.time()-t_total_start:.1f}s")
t_total = time.time() - t_total_start
if MPI.comm_world.rank == 0:
    print(f"\n{'='*50}")
    print(f"{'Timing':^50}")
    print(f"{'='*50}")
    for name, elapsed in timings.items():
        percent = 100 * elapsed / t_total if t_total > 0 else 0
        print(f"  {name:<15}: {elapsed:>8.2f} s ({percent:>5.1f}%)")
    print(f"  {'-'*46}")
    print(f"  {'Total':<15}: {t_total:>8.2f} s (100.0%)")
    print(f"{'='*50}")
