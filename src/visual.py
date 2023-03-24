import matplotlib.pyplot as plt
import os
import seaborn as sns
import torch
from sklearn.manifold import TSNE

import utils
from dataloaders import cifar100 as dataloader
from parser_parameters import get_parser


def main(args):

    data, taskcla, inputsize = dataloader.get(path='../dat', seed=0, args=args)

    path = f"./tsne_{args.experiment}"
    file_type = "png"

    valid_data_1 = data[0]['val']
    valid_data_2 = data[1]['val']

    val_loader_1 = torch.utils.data.DataLoader(
        valid_data_1, batch_size=len(valid_data_1), shuffle=False, pin_memory=True, num_workers=args.num_workers)

    val_loader_2 = torch.utils.data.DataLoader(
        valid_data_2, batch_size=len(valid_data_2), shuffle=False, pin_memory=True, num_workers=args.num_workers)

    for x, y in val_loader_1:
        x1 = x.reshape(x.shape[0], -1)
        y1 = y
    for x, y in val_loader_2:
        x2 = x.reshape(x.shape[0], -1)
        y2 = y
    
    x1x2 = torch.sum(torch.matmul(x2.T, x1) ** 2)
    x1x1 = torch.sum(torch.matmul(x1.T, x1) ** 2) ** 0.5
    x2x2 = torch.sum(torch.matmul(x2.T, x2) ** 2) ** 0.5
    dist = x1x2 / (x1x1 * x2x2)

    print(dist)

    # all_x = torch.cat([x1, x2], dim=0)
    # all_y = torch.cat([torch.zeros_like(y1), torch.ones_like(y2)], dim=0)



    # tsne = TSNE(learning_rate='auto', init="pca")
    # X_embedded = tsne.fit_transform(all_x)

    # sns.set()
    # plt.figure()
    # plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=all_y, s=2, cmap='viridis')
    # plt.savefig(f"{path}.{file_type}", bbox_inches = 'tight')
    # print("Main")
    return


if __name__ == "__main__":
    args = get_parser()
    main(args)
