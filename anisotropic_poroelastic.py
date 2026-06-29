from dolfin import *
from meshes import create_anisotropic_compressblock_mesh
from stabilized_poroelasticity import StandardReinforcedModel, DynamicStabilizedPoroelasticitySolver
from mpi4py import MPI as pyMPI
import os, time, csv, glob
import numpy as np

if not os.path.exists("results"):
    os.makedirs("results")
if not os.path.exists("results/anisotropic_poroelastic"):
    os.makedirs("results/anisotropic_poroelastic")

# 清理上次运行可能残留的损坏 HDF5/XDMF 文件
if MPI.comm_world.rank == 0:
    for pattern in ["results/anisotropic_poroelastic/*.h5",
                    "results/anisotropic_poroelastic/*.xdmf"]:
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
G = 80.194; Gf = 80.194
f0 = [0.866, 0.5]
kappas = 2e4; beta_val = 1e-5
epsilon = 0.001; phi0 = 0.1; phi_crit = 0.001
K0 = 1e-5        # cm⁴/(dyne·s)
rho0_f = 1.0     # g/cm³
dt = 0.01; rho = 1.0
num_steps = 100
c_init = 1.0
u_order = 2; p_order = 1; m_order = 1
scale = 1.0; pressure = 200

# Mesh
t0 = time.time()
mesh, boundaries = create_anisotropic_compressblock_mesh(nx=40, ny=20, scale=scale)
timings['网格生成'] = time.time() - t0
if MPI.comm_world.rank == 0:
    print(f"Mesh: {timings['网格生成']:.2f}s")

# Solver
t0 = time.time()
material = StandardReinforcedModel(G, Gf, f0, kappas, c_init, epsilon, phi0, phi_crit, K0)
solver = DynamicStabilizedPoroelasticitySolver(
    mesh, boundaries, material, dt=dt, rho0_f=rho0_f, rho=rho,
    u_order=u_order, p_order=p_order, m_order=m_order, c1=1, c2=4, c3=0.1,
    beta=beta_val
)
timings['求解器初始化'] = time.time() - t0
if MPI.comm_world.rank == 0:
   print(f"求解器初始化: {timings['求解器初始化']:.2f} 秒")

# BCs: top u_x=0 (3+5), bottom u_y=0 (2)
bc_top_3 = DirichletBC(solver.W.sub(0).sub(0), Constant(0.0), boundaries, 3)
bc_top_5 = DirichletBC(solver.W.sub(0).sub(0), Constant(0.0), boundaries, 5)
bc_bot   = DirichletBC(solver.W.sub(0).sub(1), Constant(0.0), boundaries, 2)
bcs = [bc_top_3, bc_top_5, bc_bot]

# Traction on top [5,15]
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
traction = Constant((0.0, -pressure))
solver.Res += - inner(traction, solver.w_u) * ds(5)
solver.Jacobian = derivative(solver.Res, solver.w, solver.w_trial)

# XDMF 输出
xdmf_u = XDMFFile(MPI.comm_world, "results/anisotropic_poroelastic/displacement.xdmf")
xdmf_p = XDMFFile(MPI.comm_world, "results/anisotropic_poroelastic/pressure.xdmf")
xdmf_m = XDMFFile(MPI.comm_world, "results/anisotropic_poroelastic/fluid_mass.xdmf")
xdmf_u.parameters["flush_output"] = True
xdmf_p.parameters["flush_output"] = True
xdmf_m.parameters["flush_output"] = True

if MPI.comm_world.rank == 0:
    print(f"\nSolving {num_steps} steps...")
t0 = time.time()
time_points = []
uy_vals = []
point = Point(10.0*scale, 10.0*scale)

try:
    for step in range(num_steps):
        t = step * dt
        if MPI.comm_world.rank == 0 and step % 20 == 0:
            print(f"Step {step+1}/{num_steps}, t={t:.3f}")

        traction.assign(Constant((0.0, -pressure)))
        u_sol, p_sol, m_sol = solver.solve(bcs, tol=1e-8, use_predictor=(step>0))
        if step > 0: solver.w_nn.assign(solver.w_n)
        solver.w_n.assign(solver.w)

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
        with open(f"results/anisotropic_poroelastic/uy_u{u_order}_p{p_order}.csv", 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['time','uy'])
            for ti, uyi in zip(time_points, uy_vals): w.writerow([ti, uyi])
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
