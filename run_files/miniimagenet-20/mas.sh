cd src
python run.py \
    --mode=search \
    --task_seed=0 \
    --seed=0 \
    --experiment="miniimagenet" \
    --approach=mas \
    --model=resnet \
    --id=0 \
    --lr=0.025 \
    --lamb=0.001 \
    --c_mas=300 \
    --epochs=50 \
    --batch=64 \
