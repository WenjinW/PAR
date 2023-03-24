id=${1:-"0"}
cd src
python run.py \
    --task_seed=0 \
    --seed=0 \
    --experiment=cifar100_10 \
    --approach=par_v2 \
    --id=${id} \
    --model=par \
    --sample_epochs=100 \
    --epochs=100 \
    --batch=64 \
    --lr=0.01 \
    --lamb=0.003 \
    --lr_search=0.01 \
    --num_layers=4 \
    --init_channel=64 \
    --coefficient_kl=1 \
    --lr_patience=20 \
    --nas="mdl" \
    --task_relatedness_method="mean" \
    --reuse_threshold=0.5 \
    --reuse_cell_threshold=0.75 \
    --pretrained_feat_extractor="resnet18" \
    --task_dist_image_size="(8,6)" \
    --num_workers=4 \
