import sys

import psutil
sys.path.append("./Algorithms")

# from LSQR import *

from LSQRLP import *
from LSQRMP import *

from CGLSLP import *
from CGLSMP import *

# CIL core components needed
from cil.processors import Normaliser, TransmissionAbsorptionConverter, Padder, CentreOfRotationCorrector
from cil.framework import AcquisitionGeometry, AcquisitionData, BlockDataContainer

# CIL optimisation algorithms and linear operators
# from cil.optimisation.algorithms import CGLS
from cil.optimisation.operators import BlockOperator, IdentityOperator

# CIL example synthetic test image
from cil.utilities import dataexample

# Forward/backprojector from CIL ASTRA plugin
from cil.plugins.astra import ProjectionOperator

# Third-party imports
import scipy
import numpy as np    
import os
import argparse
from threading import Event, Thread
import time
import datetime as datetime_module


# region command line parsing
parser = argparse.ArgumentParser("Memory Profiling Script For CGLS & LSQR")
parser.add_argument("algorithm", help="The algorithm that will be run [Options: CGLS_LP, CGLS_MP, LSQR_LP, LSQR_MP,  \
                    cgls_lp_tik_block, 'cgls_mp_tik_block', lsqr_lp_tik_block, lsqr_tik_lp, lsqr_tik_mp", type=str)
parser.add_argument("--track-peak", help="Track the peak memory usage in a separate thread", default=False, action='store_true')
args = parser.parse_args()
algorithm = args.algorithm.lower()

if algorithm not in ['cgls_mp', 'cgls_lp', 'lsqr_mp', 'lsqr_lp',
                     'cgls_lp_tik_block', 'cgls_mp_tik_block',
                     'lsqr_lp_tik_block', 'lsqr_mp_tik_block',
                     'lsqr_tik_lp', 'lsqr_tik_mp'
                     ]:
    raise ValueError()

if args.track_peak:
    print("Tracking Peak Usage")
# endregion

# region Setting up the data:
dataexample.SANDSTONE.download_data(data_dir='..', prompt=False)
datapath = '../sandstone'
filename = "slice_0270_data.mat"
padsize = 600

all_data = scipy.io.loadmat(os.path.join(datapath,filename))
sandstone = all_data['X_proj'].astype(np.float32)
flats = all_data['X_flat'].astype(np.float32)
darks = all_data['X_dark'].astype(np.float32)

ag = AcquisitionGeometry.create_Parallel2D()  \
        .set_panel(num_pixels=(2560))        \
        .set_angles(angles=np.linspace(0,180,1500,endpoint=False)) \
        .set_labels(['horizontal','angle'])
sandstone = AcquisitionData(sandstone, geometry=ag, deep_copy=False)

sandstone.reorder('astra')
sandstone_norm = Normaliser(flat_field=flats.mean(axis=1),
                dark_field=darks.mean(axis=1))(sandstone)
sandstone_norm = TransmissionAbsorptionConverter()(sandstone_norm)
sandstone_pad = Padder.edge(pad_width={'horizontal': padsize})(sandstone_norm)
sandstone_cor = CentreOfRotationCorrector.image_sharpness(backend='astra', search_range=100, tolerance=0.1)(sandstone_pad)
sandstone_cor.geometry.get_centre_of_rotation(distance_units='pixels')

background_counts = 10000
counts = background_counts * np.exp(-sandstone_cor.as_array())
noisy_counts = np.random.poisson(counts)
sand_noisy_data = -np.log(noisy_counts/background_counts)

sandstone_noisy = sandstone_cor.geometry.allocate()
sandstone_noisy.fill(sand_noisy_data)

sandstone_noisy.reorder('astra')
ig = sandstone_noisy.geometry.get_ImageGeometry()
ag = sandstone_noisy.geometry # ig and ag need to be same
# endregion

# region Set up algorithm:
device = "gpu"
A = ProjectionOperator(ig, ag, device)
initial = ig.allocate(0)
N = 10
itsAtATime = 1


if algorithm in ['cgls_lp_tik_block', 'cgls_mp_tik_block', 'lsqr_lp_tik_block', 'lsqr_mp_tik_block', 'lsqr_tik_lp', 'lsqr_tik_mp']:
    L = IdentityOperator(ig)
    alpha = 0.1
    if algorithm in ['cgls_lp_tik_block', 'cgls_mp_tik_block', 'lsqr_lp_tik_block', 'lsqr_mp_tik_block']:
        operator_block =  BlockOperator(A, alpha*L)
        zero_data = L.range.allocate(0)
        data_block = BlockDataContainer(sandstone_noisy, zero_data)
     
# endregion


def psutil_track(pid, stop):
    process = psutil.Process(pid)    
    while not stop.is_set() and process.is_running():
        try:
            mem_info = process.memory_info()
            # timestamp = pycompat.time_isoformat(
            #     datetime_module.datetime.now().time(),
            #     timespec='microseconds')
            timestamp = datetime_module.datetime.now().time()

            print(f"Memory Usage Log (Time, Memory in MB): {timestamp}, {mem_info.rss/(1024 * 1024):.2f} MB\n")
            time.sleep(0.01) #0.01
        except psutil.NoSuchProcess:
            break

algorithm_map = {
    'cgls_mp': CGLS_MP,
    'cgls_lp': CGLS_LP,
    'lsqr_mp': LSQR_MP,
    'lsqr_lp': LSQR_LP,
    'cgls_lp_tik_block': CGLS_LP,
    'cgls_mp_tik_block': CGLS_MP,
    'lsqr_lp_tik_block': LSQR_LP,
    'lsqr_mp_tik_block': LSQR_MP,
    'lsqr_tik_lp': LSQR_LP,
    'lsqr_tik_mp': LSQR_MP}

if args.track_peak:
    pid = os.getpid()
    process = psutil.Process(pid)
    stop_event = Event()
    thread = Thread(target = psutil_track, args = (pid, stop_event))
    thread.start()

if algorithm in algorithm_map:
    algorithm_class = algorithm_map[algorithm]

    if algorithm in ['lsqr_tik_lp', 'lsqr_tik_mp']:
        solver = algorithm_class(initial=initial, operator=A, data=sandstone_noisy, alpha=alpha) # Tik: alpha != None
    elif algorithm in ['cgls_lp_tik_block', 'cgls_mp_tik_block', 'lsqr_lp_tik_block', 'lsqr_mp_tik_block']:
        solver = algorithm_class(initial=initial, operator=operator_block, data=data_block)
    else:
        solver = algorithm_class(initial=initial, operator=A, data=sandstone_noisy)

    for ii in range(N):
        print(f"run {ii+1}")
        solver.run(itsAtATime, verbose=False)

if args.track_peak:
    stop_event.set()
    thread.join()
