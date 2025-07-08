from dolfin import *
from stabilized_hyperelasticity import NeoHookean, StabilizedHyperelasticitySolver

# Parameters
mu = 80.0  # MPa
nu = 0.4999
kappa = 2*mu*(1+nu)/(3*(1-2*nu))  # Bulk modulus

# Mesh and material
mesh = Mesh("cooks_mesh.xml")  # Pre-generated mesh
material = NeoHookean(mu, kappa)
solver = StabilizedHyperelasticitySolver(mesh, material, order=2)

# Boundary conditions
left = CompiledSubDomain("near(x[0], 0)")
bc = DirichletBC(solver.W.sub(0), Constant((0,0)), left)

# Solve
u, p = solver.solve([bc])

# Output results
File("results/displacement.pvd") << u
File("results/pressure.pvd") << p