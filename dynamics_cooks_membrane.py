from dolfin import *
from meshes import create_cooks_mesh
from stabilized_hyperelasticity import SimoTaylorNeoHookean, DynamicStabilizedHyperelasticitySolver
import os
import time
import matplotlib.pyplot as plt

if not os.path.exists("results"):
    os.makedirs("results")

# 计时字典（预先初始化）
timings = {
    '网格生成': 0.0,
    '求解器初始化': 0.0,
    '求解': 0.0,
    '保存': 0.0,
}

t_total_start = time.time()

# Parameters
E = 250.0  # N/m2
# 完全不可压缩因此不需要 nu (本构中默认为 0.5)
dt = 0.01
rho = 1.0
num_steps = 700
u_order = 1
p_order = 1
stab_label = "GLS"  # 稳定化标签: "GLS" 或 "noStab"

# 1. 网格生成
t0 = time.time()
mesh, boundaries = create_cooks_mesh(nx=16, ny=13)
timings['网格生成'] = time.time() - t0
if MPI.comm_world.rank == 0:
    print(f"网格生成: {timings['网格生成']:.2f} 秒")

# 2. 求解器初始化
t0 = time.time()
material = SimoTaylorNeoHookean(E)
solver = DynamicStabilizedHyperelasticitySolver(
    mesh, boundaries, material, dt=dt, rho=rho, u_order=u_order, p_order=p_order, c1=1, c2=4, c3=0.1
)
timings['求解器初始化'] = time.time() - t0
if MPI.comm_world.rank == 0:
    print(f"求解器初始化: {timings['求解器初始化']:.2f} 秒")

# 边界条件
bc = DirichletBC(solver.W.sub(0), Constant((0.0, 0.0)), boundaries, 1)
bcs = [bc]

# 添加原有的 Cook's 膜右端外力牵引
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
# 将载荷设为随时间平滑增加，避免物理上的瞬间无穷大加速度冲击
traction = Constant((0.0, 6.25))

# 将外力加载到残差上
solver.Res -= inner(traction, solver.v) * ds(2)
solver.Jacobian = derivative(solver.Res, solver.w, solver.w_trial)

# 3. 求解
if MPI.comm_world.rank == 0:
    print(f"\nSolving dynamics for {num_steps} steps...")
t0 = time.time()

xdmf_file_u = XDMFFile(MPI.comm_world, "results/dynamic_cooks/dynamics_displacement.xdmf")
xdmf_file_p = XDMFFile(MPI.comm_world, "results/dynamic_cooks/dynamics_pressure.xdmf")
xdmf_file_u.parameters["flush_output"] = True
xdmf_file_p.parameters["flush_output"] = True

time_points = []
uy_values = []

point = Point(4.8, 6.0)  # 指定读取位移的点位

# 构建 CSV 文件名: uy_u{p}_p{q}_{stab}.csv
csv_filename = f"results/dynamic_cooks/uy_u{u_order}_p{p_order}_{stab_label}.csv"

try:
    for step in range(num_steps):
        t = step * dt
        if MPI.comm_world.rank == 0:
            print(f"Time step {step+1}/{num_steps}, t={t:.3f}")
        
        # 稳定加载，前 0.1s 缓慢加力到 6.25，避免激波引发发散
        current_force = 6.25 * min(t / 0.1, 1.0)
        traction.assign(Constant((0.0, current_force)))
        
        # 第一步无历史数据，不启用预测器
        use_pred = (step > 0)
        solver.solve(bcs, tol=1e-8, use_predictor=use_pred)
        
        # 更新前两个时间步状态
        if step > 0:
            solver.w_nn.assign(solver.w_n)
        solver.w_n.assign(solver.w)
        
        u, p = solver.w.split(deepcopy=True)
        u.rename("displacement", "displacement")
        p.rename("pressure", "pressure")
        
        xdmf_file_u.write(u, t)
        xdmf_file_p.write(p, t)
        
        # 并行安全地获取指定点的竖向位移
        uy_local = 0.0
        count_local = 0
        try:
            uy_local = u(point)[1]
            count_local = 1
        except RuntimeError:
            pass  # 该点不在当前进程的网格分区上
            
        from mpi4py import MPI as pyMPI
        uy_sum = pyMPI.COMM_WORLD.allreduce(uy_local, op=pyMPI.SUM)
        count_sum = pyMPI.COMM_WORLD.allreduce(count_local, op=pyMPI.SUM)
        uy = uy_sum / count_sum if count_sum > 0 else 0.0
        
        time_points.append(t)
        uy_values.append(uy)
    
    # 保存 CSV 文件（仅 rank 0）
    if MPI.comm_world.rank == 0 and len(time_points) > 0:
        import csv
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'uy'])
            for ti, uyi in zip(time_points, uy_values):
                writer.writerow([ti, uyi])
        print(f"✓ CSV 已保存: {csv_filename}")
        
    if MPI.comm_world.rank == 0:
        print("✓ Dynamic solution completed!")
except Exception as e:
    if MPI.comm_world.rank == 0:
        print(f"✗ Solver failed: {e}")
    exit(1)

timings['求解'] = time.time() - t0
if MPI.comm_world.rank == 0:
    print(f"求解耗时: {timings['求解']:.2f} 秒")

# 总计时
t_total = time.time() - t_total_start

# 输出统计
if MPI.comm_world.rank == 0:
    print(f"\n{'='*50}")
    print(f"{'计时统计':^50}")
    print(f"{'='*50}")
    for name, elapsed in timings.items():
        percent = 100 * elapsed / t_total if t_total > 0 else 0
        print(f"  {name:<15}: {elapsed:>8.2f} 秒 ({percent:>5.1f}%)")
    print(f"  {'-'*46}")
    print(f"  {'总计':<15}: {t_total:>8.2f} 秒 (100.0%)")
    print(f"{'='*50}")

if MPI.comm_world.rank == 0 and len(time_points) > 0:
    print(f"Final u_y = {uy_values[-1]} m")
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, uy_values, 'b-', linewidth=2, label='Vertical displacement at (48, 60)')
    plt.xlabel('Time (s)')
    plt.ylabel('Vertical displacement (m)')
    plt.title('Vertical Displacement vs Time')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    # 保存为 PNG
    plt.savefig('vertical_displacement_vs_time.png', dpi=300)
    print("✓ 图像已保存为 'vertical_displacement_vs_time.png'")
    
    # 如果还想显示图像，取消注释下一行
    # plt.show()
else:
    if MPI.comm_world.rank == 0:
        print("⚠ No data collected, skipping plot.")
        
# 结束程序，提取图表曲线结果
if MPI.comm_world.rank == 0 and len(uy_values) > 0:
    print(f"End u_y at (4.8, 6.0) = {uy_values[-1]:.6f} m")