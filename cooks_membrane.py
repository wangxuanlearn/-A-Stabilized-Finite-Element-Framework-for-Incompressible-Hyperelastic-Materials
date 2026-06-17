from dolfin import *
from meshes import create_cooks_mesh
from stabilized_hyperelasticity import NeoHookean, StabilizedHyperelasticitySolver
import os

# 创建 results 目录
if not os.path.exists("results"):
    os.makedirs("results")
    
# Parameters
mu = 80.0  # MPa
nu = 0.4999
kappa = 2*mu*(1+nu)/(3*(1-2*nu))  # Bulk modulus

# Mesh and material
mesh, boundaries = create_cooks_mesh(nx=32, ny=26)
material = NeoHookean(mu, kappa)
solver = StabilizedHyperelasticitySolver(mesh, boundaries, material, order=1)

# Boundary conditions
#左边固定
bc = DirichletBC(solver.W.sub(0), Constant((0.0, 0.0)), boundaries, 1)
bcs = [bc]

# Solve
print("\nSolving...")
try:
    u, p = solver.solve(bcs, tol=1e-8)
    print("✓ Solution converged!")
except Exception as e:
    print(f"✗ Solver failed: {e}")
    exit()

# 保存网格
File("results/cook_mesh.pvd") << mesh

# 保存位移（向量场）
u.rename("displacement", "displacement")
File("results/displacement.pvd") << u

# 保存压力（标量场）
p.rename("pressure", "pressure")
File("results/pressure.pvd") << p