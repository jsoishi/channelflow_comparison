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

from equations import Channel

import logging
logger = logging.getLogger(__name__)

# W03 cell
α = 1.14
γ = 2.5

Re  = 400

channel = Channel(nx, ny, nz, α, γ, Re)


