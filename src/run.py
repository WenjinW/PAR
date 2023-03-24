from email import policy
import sys, os, argparse, time
sys.path.append('.')
import logging
import json
import numpy as np
import pickle
import torch
import torch.distributed as dist
from torch.utils.tensorboard.writer import SummaryWriter

import wandb

import random

import utils
from parser_parameters import get_parser

from metric import get_metric
from utils_draw import draws


def main(args):
    tstart=time.time()

    # Set the result path
    if args.approach in ['TALL', 'TALL_one'] and args.o_size != 50:
        approach_name = args.approach + str(args.o_size)
    else:
        approach_name = args.approach
    exp_name = args.experiment+'__'+approach_name+'__'+args.model+'__tseed'+str(args.task_seed)+'__id_'+args.id
    
    wandb.init(project="LifelongSimilarity", name=exp_name)
    wandb.config.update(args)

    result_path = "res/{}/{}/{}".format(
            args.experiment, approach_name, exp_name)
    result_path = os.path.join(args.dir, result_path)

    args.result_path = result_path
    utils.mkdir(result_path)

    # define logger
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler(result_path+'/log.log', mode='w')
    handler = logging.FileHandler(os.path.join(wandb.run.dir, 'log.log'), mode='w')
    logger.addHandler(handler)
    args.logger = logger

    # Set Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(gpu_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends
        # device = torch.device('cuda:0')
        device = torch.device(args.gpus)
    else:
        device = torch.device('cpu')
        logger.info('[CUDA unavailable]')
        sys.exit()

    args.device = device

    logger.info('='*100)
    logger.info('Arguments =')
    for arg in vars(args):
        logger.info('{}:{}'.format(arg, getattr(args, arg)))
    logger.info('='*100)

    # Args -- Experiment
    if args.experiment == 'pmnist':
        from dataloaders import pmnist as dataloader
    elif args.experiment == 'cifar':
        from dataloaders import cifar as dataloader
    elif args.experiment == 'cifar10':
        from dataloaders import cifar10 as dataloader
    elif args.experiment in ['cifar100_20', 'cifar100_coarse', 'cifar100_10',
        "cifar100__carnivores__omnivores_and_herbivores",
        "cifar100__carnivores__flowers",
        "cifar100__flowers__fruits_and_vegetables",
        "cifar100__flowers__large_man-made_outdoor_things",
        "cifar100__aquatic_mammals__medium-sized_mammals",
        "cifar100__aquatic_mammals__household_furniture",
        ]:
        from dataloaders import cifar100 as dataloader
    elif args.experiment in ['ctrl_s_out', 'ctrl_s_in', 'ctrl_s_pl', 'ctrl_s_long',
                            'ctrl_s_minus', 'ctrl_s_plus']:
        from dataloaders import ctrl as dataloader
    elif args.experiment == 'mixture':
        from dataloaders import mixture as dataloader
    elif args.experiment == 'core50':
        from dataloaders import core50 as dataloader
    elif args.experiment == 'medmnist':
        from dataloaders import medmnist as dataloader
    elif args.experiment == 'tiny_imagenet':
        from dataloaders import imagenet as dataloader
    elif args.experiment == 'miniimagenet':
        from dataloaders import miniimagenet as dataloader
    elif args.experiment in ['mix_cifar100_celeba', 'mix_cifar100-20_celeba']:
        from dataloaders import mix_cifar100_celeba as dataloader
    else:
        raise Exception("Unknown experiment name")

    # Args -- Approach
    if args.approach == 'single':
        from approaches import single as approach
    elif args.approach == 'individual':
        from approaches import individual as approach
    elif args.approach == 'ewc':
        from approaches import ewc as approach
    elif args.approach in ['expert_gate', 'expert_gate_no_bound']:
        from approaches import expert_gate as approach
    elif args.approach == 'mas':
        from approaches import mas as approach
    elif args.approach == 'lwf':
        from approaches import lwf as approach
    elif args.approach == 'imm_mean':
        from approaches import imm_mean as approach
    elif args.approach == 'imm_mode':
        from approaches import imm_mode as approach
    elif args.approach == 'ltg':
        from approaches import ltg as approach
    elif args.approach == 'ltg_finetune':
        from approaches import ltg_finetune_L2 as approach
    elif args.approach == 'pn':
        from approaches import progressive as approach
    elif args.approach == 'TALL':
        if args.parallel:
            from approaches.TALL_parallel import Appr as approach
        else:
            from approaches.TALL import Appr as approach
    elif args.approach == 'TALL_one':
        from approaches.TALL_onesearch import Appr as approach
    elif args.approach in ['TALL_v2', 'TALL_v3', 'TALL_v4', 'TALL_darts', 'TALL_no_bound']:
        from approaches.TALL_v2 import Appr as approach
    elif args.approach in ['par']:
        from approaches.par import Appr as approach
    elif args.approach in ['par_v2']:
        from approaches.par_v2 import Appr as approach

    # Args -- Network
    if args.model == 'resnet':
        if 'ltg' in args.approach:
            from models import resnet_ltg as network
        elif args.approach in ['par']:
            from models import PARModel as network
        elif args.approach == 'pn':
            from models import resnet_pn as network
        elif args.approach in ['expert_gate', 'expert_gate_no_bound']:
            from models import expert_gate_model as network
        elif args.approach in ['individual']:
            from models import individual_model as network
        else:  # ewc
            from models import resnet as network
    elif args.model == "par":
        from models import PARModel as network
    elif args.model == "par_resnet_18":
        from models import par_resnet_18 as network
    elif args.model == 'auto':
        network = None

    writer = SummaryWriter(log_dir='../logs/{}/'.format(args.experiment) + exp_name)

    # Load date
    logger.info('Load data...')
    time1 = time.time()
    data, taskcla, inputsize = dataloader.get(path='../dat', seed=args.task_seed, args=args)
    time2 = time.time()
    logger.info('Input size = {}'.format(inputsize))
    logger.info('Task info = {}'.format(taskcla))
    logger.info("Time of loading data: {:.3f}s".format(time2-time1))

    # Inits
    logger.info("Experiment: {}".format(exp_name))
    logger.info('Inits...')
    appr = None
    if args.approach in ['individual', 'single', 'ewc', 'pn', 'lwf', 'imm_mean', 'imm_mode', 'mas']:
        net = network.Net(inputsize, taskcla).to(device=device)
        utils.print_model_report(net, logger=args.logger)
        appr = approach.Appr(net, lr=args.lr, device=device, writer=writer, exp_name=exp_name,
                            args=args)
    elif args.approach in ['par']:
        net = network(inputsize, taskcla, args).to(device=device)
        utils.print_model_report(net, logger=args.logger)
        appr = approach(net, input_size=inputsize, task_class_num=taskcla, device=device, writer=writer, exp_name=exp_name,
                        args=args)
    elif args.approach in ['par_v2']:
        appr = approach(input_size=inputsize, task_class_num=taskcla, device=device, writer=writer, exp_name=exp_name,
                        args=args)
    elif args.approach in ['expert_gate', 'expert_gate_no_bound']:
        net = network.Net(inputsize, taskcla, args.input_dim_encoder).to(device=device)
        utils.print_model_report(net, logger=args.logger)
        appr = approach.Appr(net, lr=args.lr, device=device, writer=writer, exp_name=exp_name,
                            args=args)
    elif args.approach in ['ltg', 'ltg_finetune']:
        net = network.Net(inputsize, taskcla).to(device=device)
        utils.print_model_report(net, logger=args.logger)
        appr = approach.Appr(net, lr=args.lr, device=device, writer=writer, exp_name=exp_name,
                            args=args)
    elif args.approach in ['TALL', 'TALL_one', 'TALL_v2', 'TALL_v3', 'TALL_v4', 'TALL_darts', 'TALL_no_bound']:
        appr = approach(input_size=inputsize, task_class_num=taskcla, lr=args.lr, device=device,
                            writer=writer, exp_name=exp_name, args=args)


    # Loop tasks
    val_acc_m = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    val_acc_5_m = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    test_acc_m = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    test_acc_5_m = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    model_size = []

    for t, ncla in taskcla:
        logger.info('*'*100)
        logger.info('Task {:2d} ({:s})'.format(t, data[t]['name']))
        logger.info('*'*100)
        time1 = time.time()
        # get dataset

        train_data = data[t]['train']
        valid_data = data[t]['val']

        time2 = time.time()
        # Train
        appr.learn(t, train_data, valid_data, device=device)

        time3 = time.time()
        logger.info("Train time of task {}: {}".format(t, time3-time2))
        logger.info('-'*100)

        # eval
        for u, num_class in taskcla:
            val_batch = args.batch if args.eval_batch == 0 else args.eval_batch
            val_data = data[u]['val']
            val_loader = torch.utils.data.DataLoader(
                val_data, batch_size=val_batch, shuffle=False, pin_memory=True, num_workers=args.num_workers)
            if u > t and args.approach in ['par', 'par_v2', "ltg"]:
                val_loss, val_acc, val_acc_5 = 0.0, 1 / num_class, 0.0
            else:
                val_loss, val_acc, val_acc_5 = appr.eval(u, val_loader, mode='test', device=device)
            logger.info('>>> Valid on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}%, acc5={:5.1f}% <<<'.format(
                u, data[u]['name'], val_loss, 100*val_acc, 100*val_acc_5))
            val_acc_m[t, u] = val_acc
            wandb.log({f"Eval/Val/Task {u+1} ACC": val_acc, 'task': t+1})

            test_batch = args.batch if args.eval_batch == 0 else args.eval_batch
            test_data = data[u]['test']
            test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=test_batch, shuffle=False, pin_memory=True, num_workers=args.num_workers)
            if u > t and args.approach in ['par', 'par_v2', "ltg"]:
                test_loss, test_acc, test_acc_5 = 0.0, 1 / num_class, 0.0
            else:
                test_loss, test_acc, test_acc_5 = appr.eval(u, test_loader, mode='test', device=device)
            logger.info('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}%, acc5={:5.1f}% <<<'.format(
                u, data[u]['name'], test_loss, 100*test_acc, 100*test_acc_5))
            test_acc_m[t, u] = test_acc
            test_acc_5_m[t, u] = test_acc_5
            wandb.log({f"Eval/Test/Task {u+1} ACC": test_acc, 'task': t+1})

        model_size.append(utils.get_model_size(appr.model, mode='M'))

    # Done, logging the experiment results
    logger.info('*'*100)
    logger.info('Val Accuracies:')
    logger.info("{}".format(val_acc_m))
    logger.info('*'*100)
    logger.info('*'*100)
    logger.info('Test Accuracies:')
    logger.info("{}".format(test_acc_m))
    logger.info('*'*100)

    # store the result of experiment
    total_time = time.time()-tstart
    average_time = total_time / len(taskcla)
    total_time = total_time/(60*60)
    average_time = average_time/(60*60)

    if args.approach == 'pn':
        model_size = appr.model.get_model_size()

    result = {
        "model_size": model_size,
        "val_acc_m": val_acc_m,
        "test_acc_m": test_acc_m,
        "time": total_time
    }

    if args.approach == 'TALL':
        result['archi'] = appr.archis
    if args.approach in ["par_v2", "par"]:
        result['info'] = appr.model.get_info()
    metrics = get_metric(result, method=args.approach)
    logger.info("Metrics: {}".format(metrics))


    metrics_path = os.path.join(args.result_path, "metrics.json")
    # metrics_path = result_path + '.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    wandb.save(metrics_path, args.result_path, policy='end')
    for k, v in metrics.items():
        if not isinstance(v, list):
            wandb.run.summary[k] = v

    logger.info("[Total time = {:.1f} h, Average time = {:.1f} h]".format(total_time, average_time))


if __name__ == "__main__":
    args = get_parser()
    # set_config_gpus(Config())
    main(args)
