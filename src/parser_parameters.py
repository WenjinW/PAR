import sys, os, argparse, time
from utils import str2bool
# from queuer import set_config_gpus, Config


def get_parser():
    # Arguments
    parser=argparse.ArgumentParser(description='xxx')
    parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
    parser.add_argument('--task_seed', type=int, default=0, help='(default=%(default)d)')
    parser.add_argument('--experiment', default='cifar10', type=str,
                        help='(default=%(default)s)')
    # parser.add_argument('--ctrl_stream', default='s_minus', type=str, required=False)
    parser.add_argument('--approach', default='TALL', type=str,
                        # choices=['single', 'individual', 'ewc', 'ltg', 'pn',
                        # 'TALL', 'TALL_one', 'lwf', 'imm_mean', 'imm_mode', 'mas',
                        # "TALL_v2", "TALL_v3", "TALL_v4", 'ltg_finetune', 'expert_gate', 'TALL_darts',
                        # 'TALL_no_bound', 'expert_gate_no_bound',
                        # 'par'],
                        help='(default=%(default)s)')
    # mode: training or search the best hyper-parameter
    parser.add_argument('--mode', default='train', type=str, required=False, choices=['train', 'search'],
                        help='(default=%(default)s)')
    # parser.add_argument('--dir', default='~/LifelongSimilarity', type=str, required=True,
    #                 help='(default=%(default)s)')
    # if debug is true, only use a small dataset
    parser.add_argument('--debug', default='False', type=str2bool, required=False, choices=['False', 'True'],
                        help='(default=%(default)s)')
    parser.add_argument('--location', default='local', type=str, required=False, choices=['local', 'polyaxon'],
                        help='(default=%(default)s)')
    # model: the basic model
    parser.add_argument('--model',default='auto', type=str, required=False,
                        choices=['alexnet', 'resnet', 'mlp', 'par', 'auto',
                            'par_resnet_18'],
                        help='(default=%(default)s)')
    parser.add_argument('--output',default='', type=str, required=False, help='(default=%(default)s)')
    # parser.add_argument('--device', type=str, default='0', help='choose the device')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--id', type=str, default='0', help='the id of experiment')
    parser.add_argument('--metric', type=str, default='top1')
    parser.add_argument('--search_layers', default=6, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--eval_layers', default=20, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--num_layers', default=5, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--init_channel', default=64, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--schedule', default='no', type=str, 
                        choices=['no', 'general', 'softmax'],
                        help='use schedule in TALL')
    # hyper parameters in cell search stage
    parser.add_argument('--c_epochs', default=100, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--c_batch', default=256, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--c_lr', default=0.025, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--c_lr_a', default=0.01, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--c_lamb', default=0.0003, type=float, required=False, help='(default=%(default)f)')

    # hyper parameters in operation search stage
    parser.add_argument('--o_epochs', default=100, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--o_batch', default=256, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--o_lr', default=0.025, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--o_lr_a', default=0.01, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--o_lamb', default=0.0003, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--o_lamb_a', default=0.0003, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--o_lamb_size', default=1, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--o_size', default=0, type=int, required=False,
    help="the initial number of epochs for previous EUs")

    # hyper parameters in training stage
    parser.add_argument('--epochs', default=50, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--batch', default=128, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--eval_batch', default=0, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--lr', default=0.025, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--lamb', default=0.0003, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--lamb_ewc', default=10000, type=float, required=False, help='(default=%(default)f)')

    parser.add_argument('--lr_patience', default=20, type=int, required=False, help="Learning rate patience")
    parser.add_argument('--lr_factor', default=0.1, type=float, required=False, help="Learning rate patience")
    parser.add_argument('--lr_scheduler', default='cos', type=str, 
                        help='use schedule in TALL')
    parser.add_argument('--clipgrad', default=10, type=float, required=False, help="Learning rate patience")
    parser.add_argument('--coefficient_kl', default=1, type=float, required=False,
        help="Coefficient of distillation loss in par")
    
    parser.add_argument('--sample_epochs', default=50, type=int, required=False,
                        help='sampling epochs in par')
    parser.add_argument('--task_relatedness_method', default='mean', type=str, required=False,
                        choices=['mean', 'min', 'max'],
                        )
    parser.add_argument('--resnet_version', default='MNTDP', type=str, required=False,
                        choices=['MNTDP', 'original'],
                        )
    parser.add_argument('--pretrained_feat_extractor', default='alexnet', type=str, required=False,
                        choices=['alexnet', 'resnet18', ''],
                        )   
    parser.add_argument('--lr_search', default=0.01, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--reuse_threshold', default=0.01, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--reuse_cell_threshold', default=1, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--task_dist_image_size', default='(8,6)', type=str, required=False)    
    

    # hyper parameters in LwF
    parser.add_argument('--c_lwf', default=1.0, type=float, required=False,
                        help='Coefficient of distillation loss in LwF')

    # hyper parameters in IMM
    parser.add_argument('--c_l2', default=0.0001, type=float, required=False,
                        help='Coefficient of L2 loss in IMM')

    # hyper parameters in MAS
    parser.add_argument('--c_mas', default=1, type=float, required=False,
                        help='Coefficient of importance loss in MAS')

    parser.add_argument('--lr_encoder', default=0.001, type=float, required=False,
                        help='learning rate of encoder')
    parser.add_argument('--epochs_encoder', default=50, type=int, required=False,
                        help='train epochs of encoder')
    parser.add_argument('--batch_encoder', default=512, type=int, required=False,
                        help='train batch of encoder')
    parser.add_argument('--input_dim_encoder', default=256, type=int, required=False,
                        help='input dim of encoder')
    parser.add_argument('--s_finetune', default=0.9, type=float, required=False)
    parser.add_argument('--s_reuse_g', default=0.0, type=float, required=False)
    parser.add_argument('--c_finetune', default=1, type=float, required=False)
    parser.add_argument('--eval_no_bound', action='store_true', required=False)

    parser.add_argument('--nas', default='mdl', type=str, required=False)
    parser.add_argument('--pre_cell', action="store_true", required=False)

    parser.add_argument("--local_rank", default=-1, type=int)

    parser.add_argument("--relatedness_type", default='reconstruct_error', type=str, required=False,
                        choices=['reconstruct_error', 'tvMF'])
    parser.add_argument("--tvMF_k", default=0, type=int, required=False)

    parser.add_argument("--num_workers", default=0, type=int, required=False)

    parser.add_argument("--gpus", type=str, default="0",
                    help="cuda devices, seperated by `,` with no spaces")
 
    args = parser.parse_args()

    args.dir = os.path.expanduser("~/LifelongSimilarity")

    # gpu_config = set_config_gpus(Config())
    # args.gpus = gpu_config.default_device
    # args.gpus = "cuda:4"

    return args
