cd src
python run.py \
    --task_seed=0\
    --seed=0 \
    --experiment="miniimagenet" \
    --approach=ltg \
    --id=0 \
    --model=resnet \
    --o_epochs=50  \
    --epochs=100  \
    --batch=64  \
    --lr=0.01  \
    --lamb=0.003  \
    --o_batch=64  \
    --o_lr=0.01  \
    --o_lamb=0.003  \
    --o_lr_a=0.003  \
    --o_lamb_a=0.003 \
    --o_lamb_size=0  \
    
