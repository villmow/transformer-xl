#!/usr/bin/env python
"""Launch training on AWS with 8 GPUs."""

import argparse
import ncluster
from ncluster import aws_util

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='txl',
                    help="name of the current run, used for machine naming and tensorboard visualization")
parser.add_argument('--machines', type=int, default=1,
                    help="how many machines to use")
parser.add_argument('--instance_type', type=str, default="p3.16xlarge",
                    help="how many machines to use")
parser.add_argument('--nospot', action='store_true',
                    help='Use more expensive on-demand instance')
parser.add_argument('--ncluster_backend', type=str, default='aws',
                    help='Use spot instance')

parser.add_argument('--skip_setup', action='store_true',
                    help='Make startup slightly faster by skiping various initialization tasks, like '
                         'tmux/efs setup. Only use on reruns.')

# environment/dependencies
parser.add_argument('--image_name', type=str, default='reference03',
                    help="name of AMI to use ")
parser.add_argument('--conda_env', type=str, default='pytorch_p36',
                    help='which conda env to use')

# network settings
parser.add_argument('--num_rings', type=int, default=16)


def get_nccl_params(_num_tasks, _num_gpus):
    params = f'NCCL_DEBUG=VERSION '

    if args.machines > 1:
        params += f'NCCL_MIN_NRINGS={args.num_rings} NCCL_MAX_NRINGS={args.num_rings} '
    assert aws_util.instance_supports_100gbps_network(args.instance_type), \
        "set this to proper socket name for non p3dn instances"
    params += f'NCCL_SOCKET_IFNAME=ens5 '
    return params


# reference schedule for 1 GPU, batch size 64 run
lr_schedules = {1: {'base_lr':  0.00025*5/3},  # ben-big-lr.09
                2: {'base_lr': 0.00025*5/3},    # yaro-two.07
                8: {'base_lr': 0.00025}}


def main(_unused_args_using_global_args_instead):
    ncluster.set_backend(args.ncluster_backend)
    ncluster.set_logdir_root('/ncluster/runs.new')  # TODO(y): /ncluster/runs
    job = ncluster.make_job(name=args.name,
                            run_name=f"{args.name}",
                            num_tasks=args.machines,
                            image_name=args.image_name,
                            instance_type=args.instance_type,
                            spot=not args.nospot,
                            skip_setup=args.skip_setup)

    # Uncomment if using regular DLAMI which doesn't have these installed
    #  'pip install -U protobuf' # tensorflow/models/issues/3995

    job.rsync('.')

    job.run(f'killall python || echo failed && '  # kill previous run
            f'source activate {args.conda_env} && ' +
            f'pip install -r requirements.txt')

    # Training script args
    default_params = [
        '--logdir', job.logdir,
        '--distributed',
    ]

    num_gpus = ncluster.aws_backend.INSTANCE_INFO[args.instance_type]['gpus']
    # gpu_mem_gb = ncluster.aws_backend.INSTANCE_INFO[args.instance_type]['gpu_mem_gb']

    local_batch = 64
    #    base_lr = 0.00025  # reference lr for 1 worker, batch size 64
    base_lr = lr_schedules[args.machines]['base_lr']

    num_workers = num_gpus * args.machines
    global_batch = local_batch * num_workers
    print("using global batch ", global_batch)  # 512=8*32*2*1

    if '24x' in args.instance_type:  # special config for increased batch size
        local_batch = 96  # nonlinear bs scaling
        _global_batch = local_batch * num_workers
        base_lr = base_lr * 1.5  # linear scaling for 64 -> 96 batch size
        
    lr = base_lr * num_workers

    # todo(y): consistency with - and _ in args
    # Based on run_wt103_base.sh
    # todo(y): data should be baked into image rather than EFS
    training_params = [
        '--seed', 1111,
        '--data', '/ncluster/data/transformer-xl-data/wikitext-103',
        '--dataset', 'wt103',
        '--dist-backend', 'nccl',
        '--adaptive',
        '--log-interval', 100,
        '--n_layer', 16,
        '--d_model', 410,
        '--n_head', 10,
        '--d_head', 41,
        '--d_inner', 2100,
        '--dropout', 0.1,
        '--dropatt', 0.0,
        '--optim', 'lamb',
        '--lr', lr,
        '--wd', 0,
        '--warmup_tokens', 0,
        '--max_tokens', int(1.8e9),
        '--tgt_len', 128,
        '--mem_len', 128,
        '--eval_tgt_len', 128,
        '--batch_size', local_batch,  # per-gpu batch size
        '--eval-interval', 4000,
        # '--scheduler', 'finder', # Use max_tokens 2e7 and log-interval 10
    ]

    training_params = default_params + training_params
    training_params = ' '.join(str(p) for p in training_params)
    nccl_params = get_nccl_params(args.machines, num_gpus)

    for i, task in enumerate(job.tasks):
        dist_params = \
            f'--nproc_per_node={num_gpus} ' \
            f'--nnodes={args.machines} --node_rank={i} ' \
            f'--master_addr={job.tasks[0].ip} --master_port={6016}'
        cmd = f'{nccl_params} python -m torch.distributed.launch {dist_params} train.py {training_params}'
        task.run(f'echo {cmd} > {job.logdir}/task-{i}.cmd')  # save command-line
        task.run(cmd, non_blocking=True)

    print(f"Logging to {job.logdir}")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
