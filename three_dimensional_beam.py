from dolfin import *
from meshes import create_three_dimensional_beam_mesh
from stabilized_hyperelasticity import SimoTaylorNeoHookean, StabilizedHyperelasticitySolver
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
E = 250.0  # N/m2
# 完全不可压

# 1. 网格生成
t0 = time.time()
mesh, boundaries = create_three_dimensional_beam_mesh(mesh_size=0.125)
timings['网格生成'] = time.time() - t0
print(f"网格生成: {timings['网格生成']:.2f} 秒")

# 2. 求解器初始化
t0 = time.time()
material = SimoTaylorNeoHookean(E)
solver = StabilizedHyperelasticitySolver(
    mesh, boundaries, material, u_order=1, p_order=1
)
timings['求解器初始化'] = time.time() - t0
print(f"求解器初始化: {timings['求解器初始化']:.2f} 秒")

# 边界条件
bcs = []

# BC 1: X=3 界面, u_x = 0 (因为 X=3 是网格内部节点，不能用面查找，需使用 pointwise)
bc_x = DirichletBC(solver.W.sub(0).sub(0), Constant(0.0), "near(x[0], 3.0)", method="pointwise")
bcs.append(bc_x)

# BC 2: X=0 端面中性轴, u_y = 0 (线约束，需使用 pointwise)
bc_y_left = DirichletBC(solver.W.sub(0).sub(1), Constant(0.0), "near(x[0], 0.0) && near(x[1], 0.0)", method="pointwise")
bcs.append(bc_y_left)

# BC 3: X=6 端面中性轴, u_y = 0 (线约束，需使用 pointwise)
bc_y_right = DirichletBC(solver.W.sub(0).sub(1), Constant(0.0), "near(x[0], 6.0) && near(x[1], 0.0)", method="pointwise")
bcs.append(bc_y_right)

# BC 4: 点约束, u_z = 0
bc_z_left = DirichletBC(solver.W.sub(0).sub(2), Constant(0.0), "near(x[0], 0.0) && near(x[1], 0.0) && near(x[2], 0.0)", method="pointwise")
bcs.append(bc_z_left)

# BC 5: 点约束, u_z = 0
bc_z_right = DirichletBC(solver.W.sub(0).sub(2), Constant(0.0), "near(x[0], 6.0) && near(x[1], 0.0) && near(x[2], 0.0)", method="pointwise")
bcs.append(bc_z_right)

class Pressure(UserExpression):
        def eval(self, values, x):
            values[0] = 50.0 * x[1]  # p = 50*y
            
        def value_shape(self):
            return ()
    
pressure = Pressure(degree=1)

# ====== 添加压力外力边界条件 ======
n = FacetNormal(mesh)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# 左端面 (x=0): 法向量为 (-1, 0, 0)
# 右端面 (x=L): 法向量为 (1, 0, 0)
N = FacetNormal(mesh)
traction_left = -N * pressure 
traction_right = -N * pressure
solver.Res -= inner(traction_left, solver.v) * ds(6) + inner(traction_right, solver.v) * ds(7)
# 更新 Jacobian 矩阵
solver.Jacobian = derivative(solver.Res, solver.w, solver.w_trial)
# ==================================

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
File("results/three_dimensional_beam/mesh.pvd") << mesh
u.rename("displacement", "displacement")
File("results/three_dimensional_beam/displacement.pvd") << u
p.rename("pressure", "pressure")
File("results/three_dimensional_beam/pressure.pvd") << p
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


# 获取中点竖向挠度
uy_mid = u(Point(3.0, 0.0, 0.0))
print(f"u_y = {uy_mid} m")