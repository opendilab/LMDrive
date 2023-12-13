GPU_NUM=8
DATASET_ROOT='/path/to/your/dataset'

srun -p two_ibs -N 1 --gres=gpu:$GPU_NUM --quotatype=auto ./distributed_pretrain.sh $GPU_NUM $DATASET_ROOT  --dataset carla --train-towns 1 2 3 4 5 6 7 10  --val-towns 1 5 7\
    --train-weathers 0 1 2 3 4 5 6 7 8 9 10 11 14 15 16 17 18 19 --val-weathers  12 13 20\
    --model memfuser_baseline_e1d3 --sched cosine --epochs 25 --warmup-epochs 5 --lr 0.00075 --batch-size 24  -j 16 --no-prefetcher --eval-metric l1_error \
    --opt adamw --opt-eps 1e-8 --weight-decay 0.05 \
    --scale 0.9 1.1 --saver-decreasing --clip-grad 5 --freeze-num -1 \
    --with-backbone-lr --backbone-lr 0.0003 \
    --multi-view --with-lidar --multi-view-input-size 3 128 128 \
    --smoothed_l1 \
    --experiment memfuser_e1d3 \
    --pretrained \
