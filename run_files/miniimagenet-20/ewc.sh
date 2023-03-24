cd src
python run.py\
    --mode=search \
    --task_seed=0 \
    --seed=0 \
    --experiment="miniimagenet" \
    --approach=ewc \
    --model=resnet \
    --id=0 \
    --lr=0.025 \
    --lamb=0.001 \
    --lamb_ewc=3000 \
    --epochs=50 \
    --batch=64

