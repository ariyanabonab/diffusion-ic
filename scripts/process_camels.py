import os
import readgadget
import numpy as np
import MAS_library as MASL
import argparse


# Constants
GRID = 64
BOX_SIZE = 25
MAS = 'CIC'
VERBOSE = False

# Get lhid
parser = argparse.ArgumentParser()
parser.add_argument('--suite', type=str)
parser.add_argument('--idx', type=int)
args = parser.parse_args()

suite = args.suite
idx = args.idx

# setup in/output paths
suite_path = f'/work/hdd/bdne/maho3/CAMELS/Sims/IllustrisTNG/{suite}'
sim_list = sorted(os.listdir(suite_path))

snapshot_path = f'{suite_path}/{sim_list[idx]}'
outpath = f'/work/hdd/bdne/for_ariyana_from_matt/CAMELS/Sims/IllustrisTNG/{suite}/ic_{sim_list[idx]}.npy'

print(f"Processing suite={suite} idx={idx} file={sim_list[idx]}...")

# 1. Check if the source file exists
# 2. Check if the output file does NOT exist
if not os.path.exists(snapshot_path):
    raise ValueError(f"Input path doesn't exist: {snapshot_path}")

if os.path.exists(outpath):
    raise ValueError(f"Output file already exists: {outpath}")

try:
    # Note: readgadget usually expects the prefix without the '.0'
    # if it's a multi-part file, but since we verified ics.0:
    snapshot_prefix = f'{snapshot_path}/ICs/ics'

    ptype = [0, 1]  # gas and dm
    pos = readgadget.read_block(snapshot_prefix, "POS ", ptype) / 1e3

    delta = np.zeros((GRID, GRID, GRID), dtype=np.float32)
    MASL.MA(pos, delta, BOX_SIZE, MAS, verbose=VERBOSE)

    np.save(outpath, delta)

except Exception as e:
    print(f"Error processing idx={idx}: {e}")
    raise e
