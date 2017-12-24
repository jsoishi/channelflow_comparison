"""channel.py

Dedalus script to compare against channelflow.

Solves 

∂_t u + z ∂_x u + w x̂ + u.∇u = -∇p + ∇^2 u/Re

∇.u = 0

where u = (u, v, w).  I've swapped y and z in order to have the Chebyshev basis in the last position.

See Gibson, Halcrow, and Cvitanović (2008, JFM),
http://www.cns.gatech.edu/~gibson/publications/GibsonJFM08.pdf
"""
import numpy as np
import dedalus.public as de

# W03 cell
α = 1.14
γ = 2.5

Re  = 400

Lx = 2*np.pi/α
Ly = 2*np.pi/γ

x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', ny, interval=(0, Ly), dealias=3/2)
z_basis = de.Chebyshev('z', nz, interval=(-1,1),dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)

problem = de.IVP(domain, variables=['u', 'v', 'w', 'uz', 'vz', 'wz', 'p'])

problem.parameters['Re'] = Re

problem.substitutions['ugrad(A, Az)'] = 'u*dx(A) + v*dy(A) + w*Az'
problem.substitutions['Lap(A, Az)'] = 'dx(dx(A)) + dy(dy(A)) + dz(Az)'

problem.add_equation("dt(u) + dx(p) - Lap(u, uz)/Re + w + z*dx(u) = ugrad(u,uz)")
problem.add_equation("dt(v) + dy(p) - Lap(v, vz)/Re + z*dx(v) = ugrad(v,vz)")
problem.add_equation("dt(w) + dz(p) - Lap(w, wz)/Re + z*dx(w) = ugrad(w,wz)")
problem.add_equation("dx(u) + dy(v) + wz = 0")
problem.add_equation("dz(u) + uz = 0")
problem.add_equation("dz(v) + vz = 0")
problem.add_equation("dz(w) + wz = 0")

problem.add_bc("left(u) = 0")
problem.add_bc("right(u) = 0")
problem.add_bc("left(v) = 0")
problem.add_bc("right(v) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(w) = 0")
problem.add_bc("right(w) = 0", condition="(nx != 0) or (ny != 0)")
problem.add_bc("p = 0", condition="(nx == 0) and (ny == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.RK443)
logger.info('Solver built')

# load data


