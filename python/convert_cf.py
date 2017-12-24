"""convert_cf.py Convert data from channelflow.org to Dedalus file format.

"""
from pathlib import Path
import sys
import numpy as np
import dedalus.public as de

from equations import Channel

def load_geom(datadir):
    geomfile = datadir / "UB.geom"
    with geomfile.open() as geo:
        g = geo.readlines()
    geom = {}
    for l in g:
        value, key = l.split("%")
        value = value.strip()
        key  = key.strip()
        if '=' in key:
            key = key.split('=')[0]

        # rely on data having . for floats only!
        if '.' in value:
            value = float(value)
        else:
            value = int(value)
        geom[key] = value

    return geom

def load_cf_data(datadir, channel):
    datafile = datadir / "UB.asc"
    state = channel.solver.state
    i = 0 # component
    size = state.fields[0]['g'].size
    nx, ny, nz = state.fields[0]['g'].shape
    with datafile.open() as df:
        for n in range(size):
            i = n // (ny*nz)
            k = n % nz
            j = (n - k - (ny*nz*i))//nz
            state['u']['g'][i,j,k] = float(df.readline())
            state['w']['g'][i,j,k] = float(df.readline())
            state['v']['g'][i,j,k] = float(df.readline())

    # take derivatives
    state['u'].differentiate('z', out=state['uz'])
    state['v'].differentiate('z', out=state['vz'])
    state['w'].differentiate('z', out=state['wz'])
    
if __name__ == "__main__":
    datadir = Path(sys.argv[-1])

    geom = load_geom(datadir)

    # we swap y and z
    nx = geom['Nx']
    ny = geom['Nz']
    nz = geom['Ny']
    Lx = geom['Lx']
    Ly = geom['Lz']

    Re = 400.
    channel = Channel(nx, ny, nz, Lx, Ly, Re)
    load_cf_data(datadir, channel)

    state = channel.solver.state

    check = channel.solver.evaluator.add_file_handler(datadir/'checkpoints', iter=1)
    check.add_system(channel.solver.state)
    dt = 0.05 # guess from channelflow.org
    channel.solver.evaluator.evaluate_scheduled(0,0,0,world_time=0,timestep=dt)
