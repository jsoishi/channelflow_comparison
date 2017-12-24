import dedalus.public as de
import numpy as np

import logging
logger = logging.getLogger(__name__)

class Channel:
    def __init__(self, nx, ny, nz, Lx, Ly, Re):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.Lx = Lx
        self.Ly = Ly
        self.Re = Re

        self.build_domain()
        self.build_solver()

    def build_domain(self):
        self.x_basis = de.Fourier('x', self.nx, interval=(0, self.Lx), dealias=3/2)
        self.y_basis = de.Fourier('y', self.ny, interval=(0, self.Ly), dealias=3/2)
        self.z_basis = de.Chebyshev('z', self.nz, interval=(-1,1))
        self.domain = de.Domain([self.x_basis, self.y_basis, self.z_basis], grid_dtype=np.float64)

    def build_solver(self):
        self.problem = de.IVP(self.domain, variables=['u', 'v', 'w', 'uz', 'vz', 'wz', 'p'])
        self.problem.parameters['Re'] = self.Re
        self.problem.substitutions['ugrad(A, Az)'] = 'u*dx(A) + v*dy(A) + w*Az'
        self.problem.substitutions['Lap(A, Az)'] = 'dx(dx(A)) + dy(dy(A)) + dz(Az)'

        self.problem.add_equation("dt(u) + dx(p) - Lap(u, uz)/Re + w + z*dx(u) = ugrad(u,uz)")
        self.problem.add_equation("dt(v) + dy(p) - Lap(v, vz)/Re + z*dx(v) = ugrad(v,vz)")
        self.problem.add_equation("dt(w) + dz(p) - Lap(w, wz)/Re + z*dx(w) = ugrad(w,wz)")
        self.problem.add_equation("dx(u) + dy(v) + wz = 0")
        self.problem.add_equation("dz(u) + uz = 0")
        self.problem.add_equation("dz(v) + vz = 0")
        self.problem.add_equation("dz(w) + wz = 0")

        self.problem.add_bc("left(u) = 0")
        self.problem.add_bc("right(u) = 0")
        self.problem.add_bc("left(v) = 0")
        self.problem.add_bc("right(v) = 0")
        self.problem.add_bc("left(w) = 0")
        self.problem.add_bc("right(w) = 0", condition="(nx != 0) or (ny != 0)")
        self.problem.add_bc("left(p) = 0", condition="(nx == 0) and (ny == 0)")

        self.solver = self.problem.build_solver(de.timesteppers.RK443)
        logger.info('Solver built')
