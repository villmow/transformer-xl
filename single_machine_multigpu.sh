export DATA=data/wikitext-103
NCCL_DEBUG=VERSION python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=6016 train.py \
    --logdir /tmp/txl \
    --seed 1111 \
    --data ${DATA} \
    --dataset wt103 \
    --adaptive \
    --log_interval 100 \
    --n_layer 16 \
    --d_model 512 \
    --n_head 8 \
    --d_head 48 \
    --d_inner 2048 \
    --dropout 0.1 \
    --dropatt 0.0 \
    --optim lamb \
    --lr 0.0010416666666666667 \
    --wd 0 \
    --warmup_tokens 0 \
    --max_tokens 1800000000 \
    --tgt_len 128 \
    --mem_len 128 \
    --eval_tgt_len 128 \
    --batch_size 32 \
    --eval_interval 4000 \
    --checkpoint_each_epoch 0
