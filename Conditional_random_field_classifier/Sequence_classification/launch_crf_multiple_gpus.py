import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import os
import subprocess
import sys

# Number of processes (should equal the number of GPUs available)
world_size = 4
# Path to the script to be executed
script_path = 'crf_classifier_pytorch.py'

# Launch a subprocess for each process
for i in range(world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = str(i)
    os.environ['WORLD_SIZE'] = str(world_size)
    subprocess.Popen([sys.executable, script_path])
