This project is the source code for the article __Diffusion Model Based Hyperspectral Unmixing Using Spectral Prior Distribution__.

It is highly dependent on guided-diffusion-model (https://github.com/openai/guided-diffusion.git)

Run the run/DiffUn_w.sh and run/DiffUn_wo.sh for unmixing with and without training, respectively. Or use the function guided_diffusion.guided_diffusion.GaussianDiffusion.unmixing().

Some important files are listed here:
```
config/  
    model_config_1d.yaml            # model configuration  
guided_diffusion/
    gaussian_diffusion.py           # define the forward and reverse process of DiffUn
    spectral_datasets.py            # spectral dataset
    prior_model.py                  # spectral prior model for DiffUn w/o
    unet.py                         # spectral prior model for DiffUn w/
    train_util.py                   # train the DiffUn w/
run/
    DiffUn_w.sh                     # shell to run the DiffUn w/
    DiffUn_wo.sh                    # shell to run the DiffUn w/o
    train.sh                        # shell to run the training
scripts/
    DiffUn_w.py
    DiffUn_wo.py
    train.py
unmixing_utils.py                   # some auxiliary function for unmixing
```


```shell
@ARTICLE{10545540,
  author={Deng, Keli and Qian, Yuntao and Nie, Jie and Zhou, Jun},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Diffusion Model Based Hyperspectral Unmixing Using Spectral Prior Distribution}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Libraries;Hyperspectral imaging;Image restoration;Noise;Task analysis;Probabilistic logic;Noise reduction;Hyperspectral unmixing;diffusion model;spectral library},
  doi={10.1109/TGRS.2024.3408475}}

```
