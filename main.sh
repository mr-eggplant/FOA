export CUDA_VISIBLE_DEVICES=0

python3 main.py \
    --batch_size 64 \
    --workers 8 \
    --data /mnt/cephfs/dataset/TTA/imagenet \
    --data_v2 /mnt/cephfs/dataset/TTA/ImageNetV2 \
    --data_sketch /mnt/cephfs/dataset/TTA/imagenet-sketch \
    --data_corruption /mnt/cephfs/dataset/TTA/imagenet-c \
    --data_rendition /mnt/cephfs/dataset/TTA/imagenet-r \
    --output ./outputs \
    --algorithm 'foa' \
    --tag '_bs64_test'

