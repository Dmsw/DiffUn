This project is the source code for the article __Diffusion Model Based Hyperspectral Unmixing Using Spectral Prior Distribution__.

It is highly dependent on guided-diffusion-model (https://github.com/openai/guided-diffusion.git)

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
    DiffUn_w.py                     # shell to run the DiffUn w/
    DiffUn_wo.py                    # shell to run the DiffUn w/o
    train.py                        # shell to run the training
scripts/
    DiffUn_w.py
    DiffUn_wo.py
    train.py
unmixing_utils.py                   # some auxiliary function for unmixing
```
