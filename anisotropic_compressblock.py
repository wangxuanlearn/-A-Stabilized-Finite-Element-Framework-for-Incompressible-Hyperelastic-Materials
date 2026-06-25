from dolfin import *
from meshes import create_anisotropic_compressblock_mesh
from stabilized_hyperelasticity import StandardReinforcedModel, DynamicAnisotropicSolver
from mpi4py import MPI as pyMPI
import os, time, csv
import matplotlib.pyplot as plt

if not os.path.exists("results"):
    os.makedirs("results")
if not os.path.exists("results/anisotropic_compressblock"):
    os.makedirs("results/anisotropic_compressblock")

# 计时字典（预先初始化）
timings = {
    '网格生成': 0.0,
    '求解器初始化': 0.0,
    '求解': 0.0,
    '保存': 0.0,
}

t_total_start = time.time()

# Parameters
G = 80.194   # N/m2
Gf = 80.194  # N/m2
f0 = [0.866, 0.5]  # 纤维方向
dt = 0.01
rho = 1.0
T = 20.0
num_steps = int(T/dt)
# 注意: 2000步×P2/P1×20×10网格极为耗时(>2小时)，建议先串行小步测试
u_order = 2
p_order = 1
stab_label = "noStab"  # 稳定化标签，求解器按单元阶次自动决定是否开启
scale = 1.0
pressure = 200  # 顶部压力 (N/m2)

# 1. 网格生成
t0 = time.time()
mesh, boundaries = create_anisotropic_compressblock_mesh(nx=20, ny=10, scale=scale)
timings['网格生成'] = time.time() - t0
if MPI.comm_world.rank == 0:
    print(f"网格生成: {timings['网格生成']:.2f} 秒")

# 2. 求解器初始化
t0 = time.time()
material = StandardReinforcedModel(G, Gf, f0)
solver = DynamicAnisotropicSolver(
    mesh, boundaries, material, dt=dt, rho=rho, u_order=u_order, p_order=p_order, c3=0.1, c4=0.1
)
timings['求解器初始化'] = time.time() - t0
if MPI.comm_world.rank == 0:
    print(f"求解器初始化: {timings['求解器初始化']:.2f} 秒")

# 边界条件: 顶部全域 u_x=0 (标记 3+5), 底部 u_y=0 (标记 2)
# 左(1)右(4)边界自由 (traction-free)
bc_top_x_3 = DirichletBC(solver.W.sub(0).sub(0), Constant(0.0), boundaries, 3)
bc_top_x_5 = DirichletBC(solver.W.sub(0).sub(0), Constant(0.0), boundaries, 5)
bc_bottom_y = DirichletBC(solver.W.sub(0).sub(1), Constant(0.0), boundaries, 2)
bcs = [bc_top_x_3, bc_top_x_5, bc_bottom_y]

# 顶部 [5,15] 标记 5 施加恒定向下压力
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
traction = Constant((0.0, -pressure))
solver.Res -= inner(traction, solver.v) * ds(5)
solver.Jacobian = derivative(solver.Res, solver.w, solver.w_trial)

# 3. 求解
if MPI.comm_world.rank == 0:
    print(f"\nSolving dynamics for {num_steps} steps...")
t0 = time.time()

xdmf_file_u = XDMFFile(MPI.comm_world, "results/anisotropic_compressblock/dynamics_displacement.xdmf")
xdmf_file_p = XDMFFile(MPI.comm_world, "results/anisotropic_compressblock/dynamics_pressure.xdmf")
xdmf_file_u.parameters["flush_output"] = True
xdmf_file_p.parameters["flush_output"] = True

time_points = []
uy_values = []

point = Point(10.0 * scale, 10.0 * scale)  # 右上角位移监测点

# 构建 CSV 文件名: uy_u{p}_p{q}_{stab}.csv
csv_filename = f"results/anisotropic_compressblock/compressblock_uy_u{u_order}_p{p_order}_{stab_label}.csv"

try:
    for step in range(num_steps):
        t = step * dt
        if MPI.comm_world.rank == 0 and step % 100 == 0:
            print(f"Time step {step+1}/{num_steps}, t={t:.3f}")
        
        # 恒定载荷 200 dyne/cm²
        traction.assign(Constant((0.0, -pressure)))
        
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
            pass
            
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
    xdmf_file_u.close()
    xdmf_file_p.close()
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
    plt.plot(time_points, uy_values, 'b-', linewidth=2, label='Vertical displacement at (10, 10)')
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
    print(f"End u_y at (10.0, 10.0) = {uy_values[-1]:.6f} m")