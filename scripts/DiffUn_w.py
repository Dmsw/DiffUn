import sys
sys.path.append("./")

import argparse

import blobfile as bf
import numpy as np
import torch as th
import yaml
import matplotlib.pyplot as plt

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

from unmixing_utils import UnmixingUtils, denoising_fn


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
    if args.model_config is not None:
        upgrade_by_config(args)

    dist_util.setup_dist()
    filename = args.input_hsi
    filename = filename.split("/")[-1]
    filename = filename.split(".")[:-1]
    filename = ".".join(filename)
    logger.configure(dir=bf.join(args.save_dir, f"{filename}"))
    logger.log(args)

    logger.log("creating model...")
    model, DiffUn = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    W_t, H_t, X_t, sigma, Y, _ = load_data(args.input_hsi)
    hyper_utils = UnmixingUtils(W_t.T, H_t)

    logger.log("creating samples...")
    SRE = 0
    R = 6
    
    sample, H = DiffUn.unmixing(
        model,
        R,
        224,
        Y,
        denoising_fn=denoising_fn(),
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(12,5))

    Distance, meanDistance, P = hyper_utils.hyperSAD(sample.T)
    axes[0].plot(sample.T @ P.T)
    axes[1].plot(W_t.T)
    plt.savefig(bf.join(logger.get_dir(), f"W.png"), dpi=500)
    rmse = hyper_utils.hyperRMSE(H, P)

    output_data = "L21: SAD ", str(meanDistance), str(Distance), "aRMSE: ", str(rmse)
    logger.log(output_data)
    SRE = 10*np.log10(np.sum((X_t)**2)/np.sum((H@sample - Y)**2))
    logger.log("SNR: ", SRE)
    SAM = np.mean(np.arccos(np.sum(X_t * (H@sample), axis=1)/np.linalg.norm(X_t, axis=1)/np.linalg.norm(H@sample, axis=1)))
    logger.log("SAM: ", SAM)

    logger.log("sampling complete")

    np.savez(bf.join(logger.get_dir(), "result.npz"), H=H, W=sample)


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        range_t=0,
        use_ddim=False,
        input_hsi="",
        model_path="",
        save_dir="",
        model_config=None,
        save_latents=False,
        input_size=512,
        dmps=False,
    )
    defaults.update(model_and_diffusion_defaults())
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


if __name__ == "__main__":
    main()