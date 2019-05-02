#!/usr/bin/env python
"""Launch training on AWS with 8 GPUs."""


from attrdict import AttrDict
import argparse
import ncluster

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='txl',
                    help="name of the current run, used for machine naming and logging directory")
parser.add_argument('--config', type=str, default='',
                    help='which training config to use')
parser.add_argument('--nospot', action='store_true',
                    help='Use more expensive on-demand instance')
parser.add_argument('--skip_setup', action='store_true',
                    help='Make startup faster by skiping various initialization tasks, like '
                         'tmux/efs setup. Only use on reruns.')

# Flags that affect all configs
parser.add_argument('--num_rings', type=int, default=16)
parser.add_argument('--image_name', type=str, default='reference03',
                    help="use custom AMI ")
parser.add_argument('--conda_env', type=str, default='pytorch_p36',
                    help='use custom conda env')
args = parser.parse_args()

# Config notes:
# 'base_lr': gives learning rate relative to  BASE_LR_BATCHSIZE, actual
# learning rate will be applied by scaling by global_batch_size/BASE_LR_BATCHSIZE
# local_batch_size: per-GPU batch size

BASE_LR_BATCHSIZE = 32

# logs: yaro-1gpu
one_gpu = {
    # 24x smaller batch than ben-big-lr.09, use 5x more agressive learning rate
    'base_lr': 0.000125 * 5 / 3 * 5,
    'local_batch_size': 32,
    'instance_type': 'p3.2xlarge',
    'machines': 1
}

# Logs: yaro-fp16
one_machine_fp16 = {
    'base_lr': 0.000125 * 5 / 3,  # from ben-big-lr.09
    'instance_type': 'p3dn.24xlarge',
    'local_batch_size': 96,
    'machines': 1,
}


# /ncluster/runs.new/yaro-two-fp16.04 (with checkpoints)
two_machines_fp16 = {
    'base_lr': 0.000125 * 5 / 3,  # from ben-big-lr.09
    'instance_type': 'p3dn.24xlarge',
    'local_batch_size': 96,
    'machines': 2,
}

# yaro-four
four_machines = {
    'base_lr': 0.000125,  # remove ben's 5/3 tweak, and additional penalty of 2x
    'instance_type': 'p3dn.24xlarge',
    'local_batch_size': 96,
    'machines': 4,
}

# logs: yaro-eight.03
eight_machines = {
    'base_lr': 0.000125/2,  # remove ben's 5/3 tweak, and additional penalty of 2x
    'instance_type': 'p3dn.24xlarge',
    'local_batch_size': 96,
    'machines': 8,
    'checkpoint': '/ncluster/runs.new/yaro-one.08/model-1.pt',
}


if __name__ == '__main__':
    assert args.config in globals(), f"unknown config {args.config}"
    config = AttrDict(eval(args.config))  # easier access to dictionary entries

    config.image_name = args.image_name
    config.conda_env = args.conda_env

    job = ncluster.make_job(name=args.name,
                            run_name=f"{args.name}",
                            num_tasks=config.machines,
                            image_name=config.image_name,
                            instance_type=config.instance_type,
                            spot=not args.nospot,
                            skip_setup=args.skip_setup)

    job.rsync('.')
    job.run(f'killall python || echo failed && '  # kill previous run
            f'source activate {config.conda_env} && ' +
            f'pip install -r requirements.txt')

    instance_info = ncluster.aws_backend.INSTANCE_INFO[config.instance_type]
    num_gpus_per_machine = instance_info['gpus']

    total_gpus = num_gpus_per_machine * config.machines
    global_batch_size = config.local_batch_size * total_gpus

    # linear LR scaling (https://arxiv.org/abs/1706.02677)
    lr = config.base_lr * (global_batch_size / BASE_LR_BATCHSIZE)

    # worker parameters with training setup
    worker_params = [
        '--seed', 1111,
        '--data', '/ncluster/data/transformer-xl-data/wikitext-103',
        '--dataset', 'wt103',
        '--adaptive',
        '--log_interval', 100,
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
        '--max_tokens', int(1.8e9),
        '--tgt_len', 128,
        '--mem_len', 128,
        '--eval_tgt_len', 128,
        '--batch_size', config.local_batch_size,  # per-gpu batch size
        '--eval_interval', 4000,
        '--fp16',
        '--dynamic_loss_scale',
        '--distributed',
        '--logdir', job.logdir,
    ]

    nccl_params = f'NCCL_DEBUG=VERSION NCCL_MIN_NRINGS={args.num_rings} '

    for i, task in enumerate(job.tasks):
        dist_params = \
            f'--nproc_per_node={num_gpus_per_machine} ' \
            f'--nnodes={config.machines} --node_rank={i} ' \
            f'--master_addr={job.tasks[0].ip} --master_port={6016}'
        cmd = f'{nccl_params} python -m torch.distributed.launch {dist_params} ' \
              f'train.py {worker_params}'
        task.run(f'echo {cmd} > {job.logdir}/task-{i}.cmd')  # save command-line
        task.run(cmd, non_blocking=True)

    print(f"Logging to {job.logdir}")
