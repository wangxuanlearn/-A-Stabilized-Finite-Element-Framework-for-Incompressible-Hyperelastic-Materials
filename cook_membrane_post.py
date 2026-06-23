from dolfin import *
import numpy as np
from meshes import create_cooks_mesh
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 重新生成网格（与计算时完全一致）
# ==========================================
nx = 32
ny = 26
mesh, boundaries = create_cooks_mesh(nx=nx, ny=ny)
dt = 0.01  # 与计算时相同

# ==========================================
# 2. 定义函数空间（与计算时完全一致）
# ==========================================
V = VectorFunctionSpace(mesh, "CG", 2)
u = Function(V)

# ==========================================
# 3. 目标点（注意单位：米）
# ==========================================
# Cook膜右上角点 A：宽 48mm → 0.048m，高 44mm → 0.044m
point = Point(4.8, 6.0)  # 根据你的几何调整

# ==========================================
# 4. 从 XDMF 中读取数据
# ==========================================
time_points = []
uy_values = []

os.makedirs('results/dynamic_cooks', exist_ok=True)

with XDMFFile("dynamics_displacement.xdmf") as infile:
    step = 0
    while True:
        try:
            # ★ 方案一：用 read() ★
            infile.read(u, step)
            t = step * dt
        except RuntimeError:
            # 如果上面的方法不行，尝试方案二
            try:
                infile.read_checkpoint(u, "displacement", step)
                t = step * dt
            except RuntimeError:
                break
        
        try:
            uy = u[1](point)
        except RuntimeError:
            # 如果点不在网格内，用最近节点替代
            coordinates = V.tabulate_dof_coordinates()
            target = np.array([4.8, 6.0, 0.0])
            distances = np.linalg.norm(coordinates - target, axis=1)
            nearest_dof = np.argmin(distances)
            uy = u.vector()[nearest_dof * 2 + 1]
            print(f"Warning: point not found, using nearest node at t={t}")
        
        time_points.append(t)
        uy_values.append(uy)
        print(f"Step {step}: t={t:.4f}s, uy={uy:.6f} m")
        step += 1

print(f"✓ 共读取 {len(time_points)} 个时间步")

# ==========================================
# 5. 保存为 CSV
# ==========================================
if len(time_points) > 0:
    data = np.column_stack((time_points, uy_values))
    np.savetxt('results/dynamic_cooks/displacement_y_at_A.csv', data,
               delimiter=',',
               header='Time (s), Vertical Displacement (m)',
               comments='',
               fmt='%.8f')
    print("✓ 数据已保存为 'results/dynamic_cooks/displacement_y_at_A.csv'")

    # ==========================================
    # 6. 绘制并保存图像
    # ==========================================
    plt.figure(figsize=(12, 7))
    plt.plot(time_points, uy_values, 'b-', linewidth=2, label='Vertical displacement at A')
    plt.xlabel('Time (s)', fontsize=13)
    plt.ylabel('Vertical displacement (m)', fontsize=13)
    plt.title('Vertical Displacement vs Time at Point A (Cook\'s membrane)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=11)
    plt.xlim(left=0)
    plt.tight_layout()
    plt.savefig('results/dynamic_cooks/displacement_y_at_A.png', dpi=300)
    print("✓ 图像已保存为 'results/dynamic_cooks/displacement_y_at_A.png'")

    # ==========================================
    # 7. 统计信息
    # ==========================================
    print("\n=== 统计信息 ===")
    print(f"  最大竖向位移: {max(uy_values):.6f} m")
    print(f"  最小竖向位移: {min(uy_values):.6f} m")
    print(f"  最终位移 (t={time_points[-1]:.2f}s): {uy_values[-1]:.6f} m")
else:
    print("⚠ 没有数据点，无法绘图")