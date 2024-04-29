import argparse
import sys
sys.path.append("./")
import scipy.io as scio

import blobfile as bf
import numpy as np
import torch as th
import yaml
from functools import partial
import matplotlib.pyplot as plt
import random

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    create_gaussian_diffusion,
    diffusion_defaults,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.prior_model import PriorModel

from unmixing_utils import UnmixingUtils


def setup_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True


def load_data(data_dir):
    data = np.load(data_dir)
    W = data["W"]
    H = data["H"]
    Y = data["Y"]
    X = data["X"]
    A = data["A"]
    sigma = data["sigma"]
    return W, H, X, sigma, Y, th.from_numpy(A).to(dist_util.dev()).float()


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    filename = args.input_hsi
    filename = filename.split("/")[-1]
    filename = filename.split(".")[:-1]
    filename = ".".join(filename)
    logger.configure(dir=bf.join(args.save_dir, f"{filename}_wo"))
    logger.log(args)

    logger.log("creating model...")
    DiffUn = create_gaussian_diffusion(
        **args_to_dict(args, diffusion_defaults().keys())
    )

    W_t, H_t, X_t, sigma, Y, library = load_data(args.input_hsi)
    model = PriorModel(library, DiffUn.alphas_cumprod)
    model.to(dist_util.dev())
    model.eval()

    hyper_utils = UnmixingUtils(W_t.T, H_t)

    logger.log("unmixing...")
    R = W_t.shape[0]
    
    sample, H = DiffUn.unmixing(
        model,
        R,
        224,
        Y,
        cache_H=args.cache_H,
        progress=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12,5))

    Distance, meanDistance, P = hyper_utils.hyperSAD(sample.T)
    axes[0].plot(sample.T @ P.T)
    axes[1].plot(W_t.T)
    plt.savefig(bf.join(logger.get_dir(), f"W.png"), dpi=500)
    rmse, armse = hyper_utils.hyperRMSE(H, P)

    output_data = "L21: SAD ", str(meanDistance), str(Distance), "aRMSE: ", str(armse), str(rmse)
    logger.log(output_data)
    SRE = 10*np.log10(np.sum((X_t)**2)/np.sum((H@sample - Y)**2))
    logger.log("SRE: ", SRE)
    SAM = np.mean(np.arccos(np.clip(np.sum(X_t * (H@sample), axis=1)/np.linalg.norm(X_t, axis=1)/np.linalg.norm(H@sample, axis=1), -1, 1)))
    logger.log("SAM: ", SAM)

    logger.log("sampling complete")

    np.savez(bf.join(logger.get_dir(), "result.npz"), H=H, W=sample)


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        range_t=0,
        input_hsi="",
        model_path="",
        save_dir="",
        cache_H=False,
    )
    defaults.update(diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def upgrade_by_config(args):
    model_config = load_yaml(args.model_config)
    for k, v in model_config.items():
        setattr(args, k, v)


def prepare_data(dataFile):
    data = scio.loadmat(dataFile)
    X = data['x_n']
    A = data['A']
    s = data['s']
    return X, A, s


if __name__ == "__main__":
    main()