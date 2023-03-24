from copy import deepcopy
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from rsa import sign
import scipy.stats
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture as GMM
import torch
from torch.distributions import MultivariateNormal
from tqdm import tqdm


def get_model_size(model, mode=None):
    count = 0
    for n, p in model.named_parameters():
        # print(n)
        # print(p.size(), np.prod(p.size()))
        count += np.prod(p.size())
    human_count = None
    if mode is None:
        human_count = human_format(count)
    elif mode == 'M':
        human_count = human_format_m(count)

    return human_count


def print_model_report(model, logger):
    logger.info('-'*100)
    logger.info("model: {}".format(model))
    count = 0
    for p in model.parameters():
        # print(p.size(), end=' ')
        count += np.prod(p.size())
    logger.info('Num parameters = {}'.format(human_format(count)))
    logger.info('-'*100)
    return count


def human_format_m(num):

    return num / 1000000.0


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def print_optimizer_config(optim):
    if optim is None:
        print(optim)
    else:
        print(optim,'=',end=' ')
        opt=optim.param_groups[0]
        for n in opt.keys():
            if not n.startswith('param'):
                print(n+':',opt[n],end=', ')
        print()
    return


def get_model(model):
    return deepcopy(model.state_dict())


def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return


def modify_model(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True
    return


def freeze_parameter(params):
    for param in params.values():
        param.requires_grad = False


def unfreeze_parameter(params):
    for param in params.values():
        param.requires_grad = True


def model_size(model):
    total_size = 0.0
    for param in model.parameters():
        size_list = param.size()
        size = 1
        for i in range(len(size_list)):
            size *= size_list[i]
        total_size += size
    log_size = math.log(1+total_size, 10)

    return log_size


def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))


def compute_mean_std_dataset(dataset):
    # dataset already put ToTensor
    mean=0
    std=0
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for image, _ in loader:
        mean+=image.mean(3).mean(2)
    mean /= len(dataset)

    mean_expanded=mean.view(mean.size(0),mean.size(1),1,1).expand_as(image)
    for image, _ in loader:
        std+=(image-mean_expanded).pow(2).sum(3).sum(2)

    std=(std/(len(dataset)*image.size(2)*image.size(3)-1)).sqrt()

    return mean, std


def fisher_matrix_diag(t, train_loader, model, criterion, device, sbatch=20):
    # Init
    fisher = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            fisher[n] = 0 * p.data
    # Compute
    model.prepare(t)
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        # Forward and backward
        model.zero_grad()
        output = model.forward(x)
        loss = criterion(t, output, y)
        loss.backward()

        # Get gradients
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += sbatch * p.grad.data.pow(2)
    # Mean
    for n, _ in model.named_parameters():
        fisher[n] = fisher[n] / len(train_loader)

    return fisher


def cross_entropy(outputs, targets, exp=1, size_average=True,eps=1e-5):
    # print(outputs.size())
    # print(targets.size())
    out=torch.nn.functional.softmax(outputs, dim=1)
    tar=torch.nn.functional.softmax(targets, dim=1)
    if exp!=1:
        out = out.pow(exp)
        out = out / out.sum(1).view(-1,1).expand_as(out)
        tar = tar.pow(exp)
        tar = tar/tar.sum(1).view(-1,1).expand_as(tar)
    out = out + eps/out.size(1)
    out = out / out.sum(1).view(-1,1).expand_as(out)
    
    ce=-(tar*out.log()).sum(1)
    # print(ce.size())
    if size_average:
        ce=ce.mean()
    return ce


def t_vMF(x1, x2, k):
    """compute the t-vMF similarity
    Params:
        x1: tensor
        x2: tensor
        k: kappa parameter
    """
    # normalize x1 and x2 by L2 norm
    x1 /= torch.sqrt(torch.sum(x1 ** 2))
    x2 /= torch.sqrt(torch.sum(x2 ** 2))
    # compute cosine
    cosine_x1_x2 = torch.sum(x1 * x2)
    # t-vMF similarity
    s = (1 + cosine_x1_x2) / (1 + k * (1 - cosine_x1_x2)) - 1

    return s

def cosine_similarity(x1, x2):
    # normalize x1 and x2 by L2 norm
    x1 /= torch.sqrt(torch.sum(x1 ** 2))
    x2 /= torch.sqrt(torch.sum(x2 ** 2))
    # compute cosine
    cosine_x1_x2 = torch.sum(x1 * x2)

    return cosine_x1_x2


def set_req_grad(layer, req_grad):
    if hasattr(layer, 'weight'):
        layer.weight.requires_grad=req_grad
    if hasattr(layer, 'bias'):
        layer.bias.requires_grad=req_grad
    return


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def mkdir(path):
    path=path.strip()
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print("Create " + path)
        return True
    else:
        print(path + ' already exists')
        return False

def gaussian_mixture(X, n_components, path, file_type):
    gmm = GMM(n_components=n_components, random_state=0).fit(X)

    labels = gmm.predict(X)

    pca = PCA(n_components=50)
    x_reduce = pca.fit_transform(X)

    tsne = TSNE(learning_rate='auto', init='pca')
    X_embedded = tsne.fit_transform(x_reduce)

    sns.set()
    plt.figure()
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, s=2, cmap='viridis')
    plt.savefig("{}.{}".format(path, file_type), bbox_inches = 'tight')
    
    return gmm

def gmm_js_old(gmm_p, gmm_q, n_samples=10**5):
    X, _ = gmm_p.sample(n_samples)
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    log_mix_X = np.logaddexp(log_p_X, log_q_X)

    Y, _ = gmm_q.sample(n_samples)
    log_p_Y = gmm_p.score_samples(Y)
    log_q_Y = gmm_q.score_samples(Y)
    log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)

    return (log_p_X.mean() - (log_mix_X.mean() - np.log(2))
            + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2

def gmm_js(gmm_p, gmm_q, feat):
    log_p = gmm_p.score_samples(feat)
    log_q = gmm_q.score_samples(feat)

    log_mix = np.logaddexp(log_p, log_q)

    return 0.5*scipy.stats.entropy(log_p, log_mix) + 0.5 * scipy.stats.entropy(log_q, log_mix)


def gaussian_kl(u1, cov1, u2, cov2):
    """
    Params:
        cov1: D x D
    """
    u1 = torch.reshape(u1, (-1, 1)) # D x 1
    u2 = torch.reshape(u2, (-1, 1)) # D x 1
    
    print(f"u1.shape: {u1.shape}")
    print(f"u2.shape: {u2.shape}")
    print(f"cov1.shape: {cov1.shape}")
    print(f"cov2.shape: {cov2.shape}")

    a = (u1 - u2).T @ torch.linalg.inv(cov2) @ (u1 - u2)

    print(a)

    b = torch.log(torch.linalg.det(torch.linalg.inv(cov2) @ cov1))

    print(b)
    c = torch.trace(torch.linalg.inv(cov2) @ cov1)

    print(c)
    d = cov1.shape[0]
    print(d)
    return 0.5 * (a - b + c - d)


def dual_gaussian_kl(u1, cov1, u2, cov2):

    kl12 = gaussian_kl(u1, cov1, u2, cov2)
    kl21 = gaussian_kl(u2, cov2, u1, cov1)

    return (kl12 + kl21) / 2


def gaussian_mean_cov(X):
    """mean and covariance of Guassian distribution
    
    Params:
        X: N x D
    """
    device = X.device
    N, D = X.shape[0], X.shape[1]
    u = torch.mean(X, dim=0)
    u_row = torch.reshape(u, (1, -1))  # 1 x D
    cov = torch.matmul(X.T, X) - N * torch.matmul(u_row.T, u_row)  # D x D
    cov = cov / (N - 1)

    cov = cov * torch.diag(torch.ones(D)).to(X.device) + (torch.diag(torch.ones(D))).to(X.device)

    return u, cov


def gaussian_wasserstein(u1, cov1, u2, cov2):
    dist = torch.sum((u1 - u2) ** 2)
    dist += torch.trace(cov1)
    dist += torch.trace(cov2)
    dist -= 2 * torch.trace((cov1 ** 0.5 * cov2 * cov1 ** 0.5) ** 0.5)

    return dist


def gaussian_kl_divergence(X1, u1, cov1, X2, u2, cov2):

    p1 = MultivariateNormal(u1, cov1)
    p2 = MultivariateNormal(u2, cov2)

    # ep1 = p1.cdf(X1)

    kl_12 = torch.distributions.kl.kl_divergence(p1, p2)
    kl_21 = torch.distributions.kl.kl_divergence(p2, p1)

    kl_mean = (kl_12 + kl_21) / 2

    return kl_mean


def linear_CKA(X, Y):

    XY = torch.sum(torch.matmul(Y.T, X) ** 2)
    XX = torch.sum(torch.matmul(X.T, X) ** 2) ** 0.5
    YY = torch.sum(torch.matmul(Y.T, Y) ** 2) ** 0.5
    dist = XY / (XX * YY)

    return dist

def str2bool(str):
    # convert a string variable to a bool variable (support parser) 

    return True if str.lower() == 'true' else False
