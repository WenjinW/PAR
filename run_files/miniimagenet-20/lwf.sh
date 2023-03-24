cd src
python run.py \
    --mode=search \
    --task_seed=0 \
    --seed=0 \
    --experiment="miniimagenet" \
    --approach=lwf \
    --model=resnet \
    --id=0 \
    --lr=0.025 \
    --lamb=0.001 \
    --c_lwf=2 \
    --epochs=50 \
    --batch=64 \
