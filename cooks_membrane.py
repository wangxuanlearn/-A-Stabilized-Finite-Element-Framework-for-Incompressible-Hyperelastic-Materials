from dolfin import *
from meshes import create_cooks_mesh
from stabilized_hyperelasticity import NeoHookean, StabilizedHyperelasticitySolver
import os
import time

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
mu = 80.0
# 完全不可压缩不再需要 nu 和 kappa (理论上 kappa 趋于无限大)

# 1. 网格生成
t0 = time.time()
mesh, boundaries = create_cooks_mesh(nx=32, ny=26)
timings['网格生成'] = time.time() - t0
print(f"网格生成: {timings['网格生成']:.2f} 秒")

# 2. 求解器初始化
t0 = time.time()
material = NeoHookean(mu)
solver = StabilizedHyperelasticitySolver(
    mesh, boundaries, material, u_order=1, p_order=1
)
timings['求解器初始化'] = time.time() - t0
print(f"求解器初始化: {timings['求解器初始化']:.2f} 秒")

# 边界条件
bc = DirichletBC(solver.W.sub(0), Constant((0.0, 0.0)), boundaries, 1)
bcs = [bc]

# 添加原有的 Cook's 膜右端外力牵引
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
traction = Constant((0.0, 6.25))
solver.Res -= inner(traction, solver.v) * ds(2)
solver.Jacobian = derivative(solver.Res, solver.w, solver.w_trial)


# 3. 求解（用 finally 保证 timings['求解'] 被赋值）
print("\nSolving...")
t0 = time.time()
try:
    u, p = solver.solve(bcs, tol=1e-8)
    print("✓ Solution converged!")
except Exception as e:
    print(f"✗ Solver failed: {e}")
    exit(1)
timings['求解'] = time.time() - t0
print(f"求解耗时: {timings['求解']:.2f} 秒")

# 4. 保存结果
t0 = time.time()
File("results/cook_mesh.pvd") << mesh
u.rename("displacement", "displacement")
File("results/displacement.pvd") << u
p.rename("pressure", "pressure")
File("results/pressure.pvd") << p
timings['保存'] = time.time() - t0
print(f"保存结果: {timings['保存']:.2f} 秒")

# 总计时
t_total = time.time() - t_total_start

# 输出统计
print(f"\n{'='*50}")
print(f"{'计时统计':^50}")
print(f"{'='*50}")
for name, elapsed in timings.items():
    percent = 100 * elapsed / t_total if t_total > 0 else 0
    print(f"  {name:<15}: {elapsed:>8.2f} 秒 ({percent:>5.1f}%)")
print(f"  {'-'*46}")
print(f"  {'总计':<15}: {t_total:>8.2f} 秒 (100.0%)")
print(f"{'='*50}")