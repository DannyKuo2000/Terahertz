import os
import torch

print(
    f"START rank={os.environ.get('RANK')} "
    f"local_rank={os.environ.get('LOCAL_RANK')}",
    flush=True
)

local_rank = int(os.environ["LOCAL_RANK"])

torch.cuda.set_device(local_rank)

print(
    f"AFTER rank={os.environ.get('RANK')} "
    f"local_rank={local_rank} "
    f"current={torch.cuda.current_device()}",
    flush=True
)