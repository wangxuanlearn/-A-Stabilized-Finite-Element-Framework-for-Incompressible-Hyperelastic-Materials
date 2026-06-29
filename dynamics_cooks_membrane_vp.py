from dolfin import *
from meshes import create_cooks_mesh
from stabilized_hyperelasticity import SimoTaylorNeoHookean, DynamicConsistentGLSBDF2Solver
import os, time, csv, glob
import matplotlib.pyplot as plt
from mpi4py import MPI as pyMPI

if not os.path.exists("results"):
    os.makedirs("results")
if not os.path.exists("results/dynamic_cooks_vp"):
    os.makedirs("results/dynamic_cooks_vp")

# 清理上次运行可能残留的损坏 HDF5/XDMF 文件
if MPI.comm_world.rank == 0:
    for pattern in ["results/dynamic_cooks_vp/*.h5",
                    "results/dynamic_cooks_vp/*.xdmf"]:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
                print(f"  已清理旧文件: {f}")
            except OSError:
                pass
MPI.comm_world.barrier()

timings = {
    '网格生成': 0.0,
    '求解器初始化': 0.0,
    '求解': 0.0,
    '保存': 0.0,
}

t_total_start = time.time()

# Parameters
E = 250.0
dt = 0.01
rho = 1.0
num_steps = 700
v_order = 1
p_order = 1
stab_label = "BDF2(Stab)"  # 用于文件命名和图表标题

# 1. 网格生成
t0 = time.time()
mesh, boundaries = create_cooks_mesh(nx=32, ny=26)
timings['网格生成'] = time.time() - t0
if MPI.comm_world.rank == 0:
    print(f"网格生成: {timings['网格生成']:.2f} 秒")

# 2. 求解器初始化
t0 = time.time()
material = SimoTaylorNeoHookean(E)
solver = DynamicConsistentGLSBDF2Solver(
    mesh, boundaries, material, dt=dt, rho=rho,
    v_order=v_order, p_order=p_order, c3=0.1, c4=0.2
)
timings['求解器初始化'] = time.time() - t0
if MPI.comm_world.rank == 0:
    print(f"求解器初始化: {timings['求解器初始化']:.2f} 秒")

# 边界条件: 左边界固定 (位移 u=0 → 速度 v=0)
bc = DirichletBC(solver.W.sub(0), Constant((0.0, 0.0)), boundaries, 1)
bcs = [bc]

# 右边界施加牵引力
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
traction = Constant((0.0, 6.25))
solver.Res += - inner(traction, solver.w_v) * ds(2)
solver.Jacobian = derivative(solver.Res, solver.w, solver.w_trial)

# 3. 求解
if MPI.comm_world.rank == 0:
    print(f"\nSolving dynamics for {num_steps} steps...")
t0 = time.time()

xdmf_file_u = XDMFFile(MPI.comm_world, "results/dynamic_cooks_vp/dynamics_BDF2_displacement.xdmf")
xdmf_file_p = XDMFFile(MPI.comm_world, "results/dynamic_cooks_vp/dynamics_pressure.xdmf")
xdmf_file_u.parameters["flush_output"] = True
xdmf_file_p.parameters["flush_output"] = True

time_points = []
uy_values = []
point = Point(4.8, 6.0)  # 右上角监测点

csv_filename = f"results/dynamic_cooks_vp/BDF2_uy_v{v_order}_p{p_order}_{stab_label}.csv"

try:
    for step in range(num_steps):
        t = step * dt
        if MPI.comm_world.rank == 0 and step % 50 == 0:
            print(f"Time step {step+1}/{num_steps}, t={t:.3f}")

        current_force = 6.25 * min(t / 0.1, 1.0)
        traction.assign(Constant((0.0, current_force)))

        use_pred = (step > 0)
        is_first = (step == 0)
        u_sol, v_sol, p_sol = solver.solve(bcs, tol=1e-8, use_predictor=use_pred, first_step=is_first)
        # BDF2: 更新两步历史 w_nn ← w_n, w_n ← w
        if step > 0:
            solver.w_nn.assign(solver.w_n)
        solver.w_n.assign(solver.w)

        u_sol.rename("displacement", "displacement")
        p_sol.rename("pressure", "pressure")

        xdmf_file_u.write(u_sol, t)
        xdmf_file_p.write(p_sol, t)

        # 监测点竖向位移
        uy_local = 0.0
        count_local = 0
        try:
            uy_local = u_sol(point)[1]
            count_local = 1
        except RuntimeError:
            pass

        uy_sum = pyMPI.COMM_WORLD.allreduce(uy_local, op=pyMPI.SUM)
        count_sum = pyMPI.COMM_WORLD.allreduce(count_local, op=pyMPI.SUM)
        uy = uy_sum / count_sum if count_sum > 0 else 0.0

        time_points.append(t)
        uy_values.append(uy)

    if MPI.comm_world.rank == 0 and len(time_points) > 0:
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'uy'])
            for ti, uyi in zip(time_points, uy_values):
                writer.writerow([ti, uyi])
        print(f"CSV saved: {csv_filename}")

    if MPI.comm_world.rank == 0:
        print("Dynamic solution completed!")
    xdmf_file_u.close()
    xdmf_file_p.close()

except Exception as e:
    if MPI.comm_world.rank == 0:
        print(f"Solver failed: {e}")
    pyMPI.COMM_WORLD.Abort(1)

timings['求解'] = time.time() - t0
if MPI.comm_world.rank == 0:
    print(f"求解耗时: {timings['求解']:.2f} 秒")

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

if MPI.comm_world.rank == 0 and len(time_points) > 0:
    print(f"Final u_y = {uy_values[-1]:.6f} m")

    plt.figure(figsize=(10, 6))
    plt.plot(time_points, uy_values, 'b-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Vertical displacement uy (m)')
    plt.title(f'Cook Membrane — Consistent GLS (v,p) BDF2, V{v_order}P{p_order}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'results/dynamic_cooks_vp/displacement_vs_time_{stab_label}.png', dpi=300)
    print("Plot saved.")
else:
    if MPI.comm_world.rank == 0:
        print("No data collected.")
