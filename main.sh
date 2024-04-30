# export CUDA_HOME=/mnt/cephfs/smil/cuda/cuda-10.2
source /mnt/cephfs/home/cascol/anaconda3/bin/activate bbt

export CUDA_VISIBLE_DEVICES=5
# export HOME=/chenguohao

python3 main.py \
    --batch_size 64 \
    --workers 8 \
    --data /mnt/cephfs/dataset/TTA/imagenet \
    --data_v2 /mnt/cephfs/dataset/TTA/ImageNetV2 \
    --data_sketch /mnt/cephfs/dataset/TTA/imagenet-sketch \
    --data_adv /mnt/cephfs/dataset/TTA/imagenet-a \
    --data_corruption /mnt/cephfs/dataset/TTA/imagenet-c \
    --data_rendition /mnt/cephfs/dataset/TTA/imagenet-r \
    --output ./outputs \
    --algorithm 'lame' \
    --tag '_bs64_test'

# --tag '_bs64_gauss_offlineloss_alpha0.9_noshuffle'