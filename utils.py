import math
import os
import torch
import torch.nn as nn
import numpy as np
import glob
import random
import cv2


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        nn.init.constant_(m.bias.data, 0.0)


def set_random_set(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def calc_psnr(im, recon):
    im = np.squeeze(im)
    recon = np.squeeze(recon)
    if im.ndim == 2:
        mse = np.sum((im - recon) ** 2) / np.prod(im.shape)
        max_val = 1.0  # np.max(im)
        # max_val = np.max(im)
        psnr = 10 * np.log10(max_val ** 2 / mse)
        return psnr
    elif im.ndim == 3:
        c, w, h = im.shape
        prod = w * h
        mpsnr = 0.0
        for i in range(c):
            mse = np.sum((im[i] - recon[i]) ** 2) / prod
            max_val = 1.0  # np.max(im)
            # max_val = np.max(im)
            mpsnr += 10 * np.log10(max_val ** 2 / mse)
        return mpsnr / c


def calc_ssim(im, recon):
    def ssim(img1, img2):
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    if im.ndim == 2:
        return ssim(im, recon)
    elif im.ndim == 3:
        c, h, w = im.shape
        mssim = 0.0
        for i in range(c):
            mssim += ssim(im[i], recon[i])
        return mssim / c


def reconsturction_loss(distance='l1'):
    if distance == 'l1':
        dist = nn.L1Loss()
    elif distance == 'l2':
        dist = nn.MSELoss()
    else:
        raise ValueError(f"unidentified value {distance}")

    return dist


def get_criterion(losses_types, factors):
    """
    Build Loss
        total_loss = sum_i factor_i * loss_i(results, targets)
    Args:
        factors(list): scales for each loss.
        losses(list): loss to apply to each result, target element
    """
    losses = []
    for loss_type in losses_types:
        losses.append(reconsturction_loss(loss_type))

    # if use_cuda:
    #   losses = [l.cuda() for l in losses]

    def total_loss(results, targets):
        """Cacluate total loss
            total_loss = sum_i losses_i(results_i, targets_i)
        Args:
            results(tensor): nn outputs.
            targets(tensor): targets of resluts.

        """
        loss_acc = 0
        for fac, loss in zip(factors, losses):
            _loss = loss(results, targets)
            loss_acc += _loss * fac
        return loss_acc

    return total_loss


def save_train(path, model, optimizer, epoch=None):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if epoch is not None:
        state['epoch'] = epoch
    # `_use_new_zipfile...=False` support pytorch version < 1.6
    torch.save(state, os.path.join(path, 'epoch_{}'.format(epoch)))
    return os.path.join(path, 'epoch_{}'.format(epoch))


def init_exps(exp_root_dir):
    if not os.path.exists(exp_root_dir):
        os.makedirs(exp_root_dir)
    all_exps = glob.glob(f'{exp_root_dir}/exp*')

    cur_exp_id = None
    if len(all_exps) == 0:
        cur_exp_id = 0
    else:
        exp_ids = [int(os.path.basename(s).split('_')[1]) for s in all_exps]
        exp_ids.sort()
        cur_exp_id = exp_ids[-1] + 1

    log_dir = f'{exp_root_dir}/exp_{cur_exp_id}'
    os.makedirs(log_dir)

    return log_dir


def normalize(x, mean, std):
    """
    x: (c, h, w)
    """
    return (x - mean) / std


def denormalize(x, mean, std):
    """
    x: (c, h, w)
    """
    return x * std + mean


