"""
Util functions for setting up distributed training.
Credit: https://github.com/mlfoundations/open_clip/blob/main/src/training/distributed.py
"""
import os
import torch
from datetime import timedelta

def is_global_master(args):
    return args.rank == 0

def is_local_master(args):
    return args.local_rank == 0

def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)

def is_using_distributed():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"]) > 1
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"]) > 1
    return False

def world_info_from_env():
    local_rank = 0
    for v in (
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size

# def init_distributed_device(args):
#     # Distributed training = training on more than one GPU.
#     # Works in both single and multi-node scenarios.
#     args.distributed = False
#     args.world_size = 1
#     args.rank = 0  # global rank
#     args.local_rank = 0

#     if is_using_distributed():
#         if "SLURM_PROCID" in os.environ:
#             # DDP via SLURM
#             args.local_rank, args.rank, args.world_size = world_info_from_env()
#             # SLURM var -> torch.distributed vars in case needed
#             os.environ["LOCAL_RANK"] = str(args.local_rank)
#             os.environ["RANK"] = str(args.rank)
#             os.environ["WORLD_SIZE"] = str(args.world_size)
#             torch.distributed.init_process_group(
#                 backend=args.dist_backend,
#                 init_method=args.dist_url,
#                 world_size=args.world_size,
#                 rank=args.rank,
#                 timeout=timedelta(seconds=36000000)
#             )
#         else:
#             # DDP via torchrun, torch.distributed.launch
#             args.local_rank, _, _ = world_info_from_env()
#             torch.distributed.init_process_group(
#                 backend=args.dist_backend, init_method=args.dist_url, timeout=timedelta(seconds=36000000)
#             )
#             args.world_size = torch.distributed.get_world_size()
#             args.rank = torch.distributed.get_rank()
#         args.distributed = True
#     else:
#         # needed to run on single gpu
#         torch.distributed.init_process_group(
#             backend=args.dist_backend,
#             init_method=args.dist_url,
#             world_size=1,
#             rank=0,
#             timeout=timedelta(seconds=36000000)
#         )

#     if torch.cuda.is_available():
#         if args.distributed and not args.no_set_device_rank:
#             device = "cuda:%d" % args.local_rank
#         else:
#             device = "cuda:0"
#         torch.cuda.set_device(device)
#     else:
#         device = "cpu"
#     args.device = device
#     device = torch.device(device)
#     return device

def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    from pdb import set_trace
    import datetime
    if is_using_distributed():
        if "SLURM_PROCID" in os.environ:
            # DDP via SLURM
            args.local_rank, args.rank, args.world_size = world_info_from_env()
            # SLURM var -> torch.distributed vars in case needed
            os.environ["LOCAL_RANK"] = str(args.local_rank)
            os.environ["RANK"] = str(args.rank)
            os.environ["WORLD_SIZE"] = str(args.world_size)
            os.environ['NCCL_BLOCKING_WAIT'] = '0'  # not to enforce timeout
            torch.distributed.init_process_group(
                # timeout=timedelta(seconds=7200000), # was 1800000
                timeout=datetime.timedelta(seconds=7200),
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank,
            )
        else:
            # DDP via torchrun, torch.distributed.launch
            os.environ['NCCL_BLOCKING_WAIT'] = '0'  # not to enforce timeout
            args.local_rank, _, _ = world_info_from_env()
            torch.distributed.init_process_group(
                # timeout=timedelta(seconds=7200000), # was 18000
                timeout=datetime.timedelta(seconds=17200),
                backend=args.dist_backend, 
                init_method=args.dist_url
            )
            args.world_size = torch.distributed.get_world_size()
            args.rank = torch.distributed.get_rank()
        args.distributed = True
    else:
        # needed to run on single gpu
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=1,
            rank=0,
        )

    if torch.cuda.is_available():
        if args.distributed and not args.no_set_device_rank:
            device = "cuda:%d" % args.local_rank
        else:
            device = "cuda:0"
        torch.cuda.set_device(device)
    else:
        device = "cpu"
    args.device = device
    device = torch.device(device)
    return device
    