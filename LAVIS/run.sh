GPU_NUM=$1
CONFIG_PATH=$2

srun -p basemodel_lm --quotatype=auto --gres=gpu:$GPU_NUM -N 1 python -m torch.distributed.run --nproc_per_node=$GPU_NUM train.py  --cfg-path $CONFIG_PATH
