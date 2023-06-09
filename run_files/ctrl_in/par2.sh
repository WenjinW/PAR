cd src
python run.py \
    --task_seed=0 \
    --seed=0 \
    --experiment="ctrl_s_in" \
    --approach=par_v2 \
    --id=500e3e-2p100factor3e-1minlr1e-6bound25 \
    --model=par \
    --sample_epochs=50 \
    --epochs=500 \
    --batch=32 \
    --eval_batch=1024  \
    --lr=0.01 \
    --lr_scheduler="reduce" \
    --lr_patience=30 \
    --lr_factor=0.3 \
    --lamb=0.03 \
    --lr_search=0.01 \
    --num_layers=4 \
    --init_channel=64 \
    --coefficient_kl=1 \
    --nas="mdl" \
    --task_relatedness_method="mean" \
    --reuse_threshold=0.5 \
    --reuse_cell_threshold=0.75 \
    --pretrained_feat_extractor="resnet18" \
    --task_dist_image_size="(8,6)" \
    --num_workers=4 \
