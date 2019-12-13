#!/usr/bin/env python
"""main function for reconstruction"""

__author__ = "Qiaoying Huang"
__date__ = "04/08/2019"
__institute__ = "Rutgers University"


import time
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tensorboardX import SummaryWriter
from utils import get_rmse, roll, get_psnr, get_ssim
import shutil
import argparse
import torchvision.utils as vutils
from loss import Percetual
from generatekdata import gaussiansample
from networks import MICCAN, MICCANlong
import random


class KdataDataset(Dataset):
    def __init__(self, root='./data/'):
        self.kdataroot = root + 'kdata'
        self.maskroot = root + 'mask'
        self.labelroot = root + 'image'
        self.fullyroot = root + 'fully'
        self.list = []

        for root, dirs, files in os.walk(self.kdataroot, topdown=False):
            for file in files:
                self.list.append(file)


    def __len__(self):
        return len(self.list)


    def __getitem__(self, idx):
        kdata = np.load(os.path.join(self.kdataroot, self.list[idx]))
        mask = np.load(os.path.join(self.maskroot, self.list[idx]))
        label = np.load(os.path.join(self.labelroot, self.list[idx]))
        fully = np.load(os.path.join(self.fullyroot, self.list[idx]))

        kdata = np.expand_dims(kdata, axis=0)
        mask = np.expand_dims(mask, axis=0)
        label = np.expand_dims(label, axis=0)
        fully = np.expand_dims(fully, axis=0)

        # seperate complex data to two channels data(real and imaginary)
        kdata_real = kdata.real
        kdata_imag = kdata.imag
        kdata = np.concatenate((kdata_real, kdata_imag), axis=0)

        # seperate complex data to two channels data(real and imaginary)
        fully_real = fully.real
        fully_imag = fully.imag
        fully = np.concatenate((fully_real, fully_imag), axis=0)
        return torch.from_numpy(kdata), torch.from_numpy(mask), torch.from_numpy(label), torch.from_numpy(fully)


class AverageMeter(object):
    def __init__(self):
        self.reset()


    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def save_checkpoint(state, save_path, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))


def train(train_loader, model, cri, optimizer, epoch, train_writer, device):
    global n_iter
    batch_time = AverageMeter()
    losses = AverageMeter()

    epoch_size = len(train_loader)

    # switch to train mode
    model.train()

    i = 0
    for kdata, mask, image, fully in train_loader:
        kdata = kdata.float().to(device)
        image = image.float().to(device)
        fully = fully.float().to(device)

        _, _, w, h = kdata.size()

        # generate undersampling mask on the fly
        idx = gaussiansample(h, int(h/4), np.int(np.floor(h * 0.125)))
        randmask = torch.zeros((kdata.size(0), 1, w, h)).float().to(device)
        randmask[:, :, :, idx] = 1

        # ifftshift
        randmask = roll(randmask, int(w/2), 2)
        randmask = roll(randmask, int(h/2), 3)

        # generate undersampled k-space data
        randkdata = fully * randmask

        # model forward
        reconsimage = model(randkdata, randmask)

        # calculate loss
        loss = cri(reconsimage[-1], image)

        # record loss
        losses.update(loss.item(), kdata.size(0))
        train_writer.add_scalar('train_loss', loss.item(), n_iter)

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_iter += 1

        if i % 3 == 0:
            print('Epoch: [{:02d}][{:04d}]\t Loss {:.5f}'.format(epoch, i, losses.avg))

        if i >= epoch_size:
            break

        i = i+1

    return losses.avg


def validate(val_loader, model, cri, epoch, output_writers, device):
    val_loss = AverageMeter()

    # switch to evaluate mode
    model.eval()

    allrmse = []
    allpsnr = []
    allssim = []
    i = 0
    with torch.no_grad():
        for kdata, mask, image, fully in val_loader:
            kdata = kdata.float().to(device)
            mask = mask.float().to(device)
            image = image.float().to(device)

            _, _, w, h = kdata.size()

            # ifftshift
            kdata = roll(kdata, int(w/2), 2)
            kdata = roll(kdata, int(h/2), 3)
            mask = roll(mask, int(w/2), 2)
            mask = roll(mask, int(h/2), 3)

            # model forward
            reconsimage = model(kdata, mask)

            # calculate loss
            loss = cri(reconsimage[-1], image)

            # calculate score
            rmse = get_rmse(reconsimage[-1], image)
            psnr = get_psnr(reconsimage[-1], image)
            ssim = get_ssim(reconsimage[-1], image)
            allrmse.append(rmse.item())
            allpsnr.append(psnr)
            allssim.append(ssim.item())

            # record validation loss
            val_loss.update(loss.item(), kdata.size(0))

            # display results in tensorboard
            if 1 < i < 4:
                image = vutils.make_grid(image, normalize=True, scale_each=True)
                output_writers[i].add_image('gt image', image, 0)
                rec1 = vutils.make_grid(reconsimage[-1], normalize=True, scale_each=True)
                output_writers[i].add_image('reconstruction image 1 ', rec1, epoch)
            i = i+1

        # print out average scores
        print(' * Average Validation Loss {:.3f}'.format(val_loss.avg))
        print(' * Average RMSE {:.4f}'.format(np.mean(np.asarray(allrmse))))
        print(' * Average PSNR {:.4f}'.format(np.mean(np.asarray(allpsnr))))
        print(' * Average SSIM {:.4f}'.format(np.mean(np.asarray(allssim))))
        return val_loss.avg


parser = argparse.ArgumentParser(description='Main function arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train', default=True, help='train or test')
parser.add_argument('--multi', default=False, help='train on multi gpu or not')
parser.add_argument('--seed', default=6, type=int, help='random seed')
parser.add_argument('--loss', default='percetual', type=str, help='loss function')
parser.add_argument('--blocktype', default='UCA', type=str, help='model')
parser.add_argument('--nblock', default=5, type=int, help='number of block')
parser.add_argument('--model', default='long', type=str, help='model')
parser.add_argument('--gpuid', default='0', type=str, help='gpu id')
parser.add_argument('--bs', default=16, type=int, help='batchsize')
parser.add_argument('--epoch', default=50, type=int, help='number of epoch')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--savepath', default='main_out', type=str, help='save file path')


# global variables
n_iter = 0
best_loss = -1

# main function
def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    global save_path, n_iter, best_loss

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # save model in save_path
    save_path = args.savepath
    print('=> will save everything to {}'.format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save training data in train and val folder
    train_writer = SummaryWriter(os.path.join(save_path, "train"))
    val_writer = SummaryWriter(os.path.join(save_path, "val"))
    output_writers = []
    for i in range(4):
        output_writers.append(SummaryWriter(os.path.join(save_path, "val", str(i))))

    # initialize input data
    trainloader = DataLoader(KdataDataset(), batch_size=args.bs, shuffle=True)
    validloder = DataLoader(KdataDataset(), batch_size=1, shuffle=False)

    # specify network structure
    if args.model == 'nolong':
        network = MICCAN(2, 2, args.nblock, args.blocktype)
    if args.model == 'long':
        network = MICCANlong(2, 2, args.nblock, args.blocktype)

    # whether is using multiple gpu
    if args.multi is not False:
        network = nn.DataParallel(network)

    # specify loss function
    if args.loss == 'percetual':
        loss = Percetual()
    if args.loss == 'l2':
        loss = nn.MSELoss()
    if args.loss == 'l1':
        loss == nn.L1Loss()

    # initialize optimizer and schedule loss decay
    optimizer = Adam(network.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 20, 35], gamma=0.5)

    # initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network.to(device)

    # training
    if args.train == True:
        for epoch in range(args.epoch):

            train_loss = train(trainloader, network, loss, optimizer, epoch, train_writer, device)
            train_writer.add_scalar("Aver loss", train_loss, epoch)

            val_loss = validate(validloder, network, loss, epoch, output_writers, device)
            val_writer.add_scalar('Aver loss', val_loss, epoch)

            scheduler.step()

            if best_loss < 0:
                best_loss = val_loss

            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model,
                'best_valloss': best_loss,
                'state_dict': network.state_dict(),
            }, save_path, is_best)
    # testing
    else:
        network.load_state_dict(torch.load(args.savepath + '/model_best.pth.tar')['state_dict'])
        validate(validloder, network, loss, 0, output_writers, device)


if __name__ == '__main__':
    main()



