import argparse
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

from unmixing_utils import UnmixingUtils, cal_conditional_gradient_W, vca


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
    return {"ref_img": th.from_numpy(Y).to(dist_util.dev()).float()}, W, H, X, sigma, Y, th.from_numpy(A).to(dist_util.dev()).float()


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    filename = args.base_samples
    filename = filename.split("/")[-1]
    filename = filename.split(".")[:-1]
    filename = ".".join(filename)
    logger.configure(dir=bf.join(args.save_dir, f"{filename}_wo"))
    logger.log(args)

    logger.log("creating model...")
    diffusion = create_gaussian_diffusion(
        **args_to_dict(args, diffusion_defaults().keys())
    )

    model_kwargs, W_t, H_t, X_t, sigma, Y, library = load_data(args.base_samples)
    model = PriorModel(library, diffusion.alphas_cumprod)
    model.to(dist_util.dev())
    model.eval()

    hyper_utils = UnmixingUtils(W_t.T, H_t)

    logger.log("creating samples...")
    repeat = 0
    SRE = - np.inf
    R = W_t.shape[0]

    while repeat < 20:
        W0, _, _ = vca(Y.T, R)
        W0 = th.from_numpy(W0.T).to(dist_util.dev()).float()[:, None]
        repeat += 1
        _samplek, _H = diffusion.p_sample_loop(
            model,
            (R, 1, 224),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            range_t=args.range_t,
            progress=False,
            logger=logger,
            measure_sigma=sigma,
            measurement=partial(cal_conditional_gradient_W, type="diffun"),
            W0=W0,
            t0=200,
        )
        _H = _H.cpu().detach().numpy()[:, 0]
        _sample = _samplek.cpu().detach().numpy()[:, 0]
        _sample = (_sample + 1)/2
        _sre = 10*np.log10(np.sum((Y)**2)/np.sum((_H@_sample - Y)**2))
        print(f"repeat {repeat}: {_sre}")
        if _sre > SRE:
            SRE = _sre
            sample = _sample
            H = _H

    print("repeat", repeat)

    fig, axes = plt.subplots(1, 2, figsize=(12,5))

    Distance, meanDistance, P = hyper_utils.hyperSAD(sample.T)
    axes[0].plot(sample.T @ P.T)
    axes[1].plot(W_t.T)
    plt.savefig(bf.join(logger.get_dir(), f"W.png"), dpi=500)
    rmse, armse = hyper_utils.hyperRMSE(H, P)

    output_data = "L21: SAD ", str(meanDistance), str(Distance), "RMSE: ", str(armse), str(rmse)
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
        base_samples="",
        model_path="",
        save_dir="",
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