#!/usr/bin/env python
"""Launch training on AWS with 8 GPUs."""

import argparse
import re

import ncluster
from ncluster import aws_util

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='txl',
                    help="name of the current run, used for machine naming and tensorboard visualization")
parser.add_argument('--machines', type=int, default=0,
                    help="how many machines to use")
parser.add_argument('--instance_type', type=str, default='',
                    help="how many machines to use")

parser.add_argument('--nospot', action='store_true',
                    help='Use more expensive on-demand instance')

parser.add_argument('--skip_setup', action='store_true',
                    help='Make startup slightly faster by skiping various initialization tasks, like '
                         'tmux/efs setup. Only use on reruns.')

# environment/dependencies
parser.add_argument('--image_name', type=str, default='reference03',
                    help="name of AMI to use ")
parser.add_argument('--conda_env', type=str, default='pytorch_p36',
                    help='which conda env to use')

parser.add_argument('--checkpoint-each-epoch', type=int, default=0,
                    help='whether to save checkpoint at each epoch')

# network settings
parser.add_argument('--num_rings', type=int, default=16)


# learning settings
parser.add_argument('--config', type=str, default='',
                    help='which training config to use')


# 'base_lr': learning rate for BASE_LR_BATCHSIZE, actual learning rate will apply linear scaling according to
#   the actual batch size used
# local_batch: per-GPU batch size
BASE_LR_BATCHSIZE = 32

one_gpu = {
    # 24x smaller batch than ben-big-lr.09, use 5x more agressive learning rate
    'base_lr': 0.000125*5/3*5,   
    'local_batch': 32,
    'instance_type': 'p3.2xlarge',
    'machines': 1
}

one_machine = {
    'base_lr': 0.000125*5/3,  # ben-big-lr.09
    'instance_type': 'p3dn.24xlarge',
    'local_batch': 96,
    'machines': 1,
}

two_machines = {23
    'base_lr': 0.000125*5/3,    # yaro-two.07
    'instance_type': 'p3dn.24xlarge',
    'local_batch': 96,
    'machines': 2,
}

eight_machines = {
    'base_lr': 0.000125,
    'instance_type': 'p3dn.24xlarge',
    'local_batch': 96,
    'machines': 8,
}


def get_nccl_params(_num_tasks, _num_gpus):
    params = f'NCCL_DEBUG=VERSION '

    if args.machines > 1:
        params += f'NCCL_MIN_NRINGS={args.num_rings} NCCL_MAX_NRINGS={args.num_rings} '
    return params


def main(_unused_args_using_global_args_instead):
    ncluster.set_backend('aws')
    ncluster.set_logdir_root('/ncluster/runs.new')  # TODO(y): /ncluster/runs

    if args.config:
        assert not args.instance_type, "specify instance_type as part of config"
        assert not args.machines, "specify number of machines as part of config"
        assert re.match('\\w+', args.config)
        assert args.config in globals()
        schedule = eval(args.config)
        args.machines = schedule['machines']

    else:  # legacy way of setting config
        assert args.instance_type
        assert args.machines
        schedule = {
            'base_lr': 0.000125*5/3,  # ben-big-lr.09
            'local_batch': 96,
        }
        schedule['instance_type'] = args.instance_type
        schedule['machines'] = args.machines
        

    args.instance_type = schedule['instance_type']
    num_gpus = ncluster.aws_backend.INSTANCE_INFO[args.instance_type]['gpus']
    job = ncluster.make_job(name=args.name,
                            run_name=f"{args.name}",
                            num_tasks=schedule['machines'],
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

    local_batch = schedule['local_batch']
    base_lr = schedule['base_lr']

    num_workers = num_gpus * args.machines
    global_batch = local_batch * num_workers
    print("using global batch ", global_batch)  # 512=8*32*2*1

    #    if '24x' in args.instance_type:  # special config for increased batch size
    #        local_batch = 96  # nonlinear bs scaling
    #        _global_batch = local_batch * num_workers
    #        base_lr = base_lr * 3  # linear scaling for 32 -> 96 batch size

    # linear LR scaling (https://arxiv.org/abs/1706.02677)
    lr = base_lr * (global_batch / BASE_LR_BATCHSIZE)

    # todo(y): consistency with - and _ in args
    # Based on run_wt103_base.sh, tweaked to be multiples of 8
    # todo(y): data should be baked into image rather than EFS
    training_params = [
        '--seed', 1111,
        '--data', '/ncluster/data/transformer-xl-data/wikitext-103',
        '--dataset', 'wt103',
        '--dist-backend', 'nccl',
        '--adaptive',
        '--log-interval', 100,
        '--n_layer', 16,
        '--d_model', 512,
        '--n_head', 8,
        '--d_head', 48,
        '--d_inner', 2048,
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

    # todo(y) rename to params, change to extend
    training_params = default_params + training_params

    # pass through command-line launcher arguments to the worker
    user_params = ['--checkpoint-each-epoch', args.checkpoint_each_epoch]

    training_params.extend(user_params)
    
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
