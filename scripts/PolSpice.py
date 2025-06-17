import os
import yaml
import numpy as np
import healpy as hp
import heracles
import heracles.dices as dices
from heracles.io import read, write


# Config
config_path = "./sims_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
n = config['nsims']
nside = config['nside']
lmax = config['lmax']
mode = config['mode']  # "lognormal" or "gaussian"
path = f"../{mode}_sims"
mask_cls = heracles.read(f"{path}/cls/cls_mask.fits")

for i in range(1, n+1):
    print(f"Unmixing sim {i}", end='\r')
    # Load cls
    data_cls = heracles.read(f"{path}/cls/cls_data_{i}.fits")

    # PolSpice
    nu_cls = heracles.PolSpice(data_cls, mask_cls, mode='natural', patch_hole=True)
    pp_cls = heracles.PolSpice(data_cls, mask_cls, mode='plus', patch_hole=True)
    pm_cls = heracles.PolSpice(data_cls, mask_cls, mode='minus', patch_hole=True)

    # Save cls
    heracles.write(f"{path}/cls/cls_data_nu_{i}.fits", nu_cls)
    heracles.write(f"{path}/cls/cls_data_pp_{i}.fits", pp_cls)
    heracles.write(f"{path}/cls/cls_data_pm_{i}.fits", pm_cls)
