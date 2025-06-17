import os
import yaml
import numpy as np
import healpy as hp
import heracles
from heracles.healpy import HealpixMapper
from heracles.fields import Positions, Shears, Visibility, Weights
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

mapper = HealpixMapper(nside=nside, lmax=lmax)
fields = {
    "POS": Positions(mapper, mask="VIS"),
    "SHE": Shears(mapper, mask="WHT"),
    "VIS": Visibility(mapper),
    "WHT": Weights(mapper),
}
mask_cls = heracles.read(f"{path}/cls/cls_mask.fits")
theory_cls = heracles.read(f"{path}/cls/cls_theory.fits")

# Format theory cls
# Bit hardcoded for now, but could be generalized
ls = np.arange(lmax + 1)
fl = -np.sqrt((ls+2)*(ls+1)*ls*(ls-1))
fl /= np.clip(ls*(ls+1), 1, None)
_theory_cls = {}
_theory_cls[("POS", "POS", 1, 1)] = heracles.Result(theory_cls["W1xW1"].array, ell=ls)
_theory_cls[("POS", "POS", 1, 2)] = heracles.Result(theory_cls["W1xW2"].array, ell=ls)
_theory_cls[("POS", "POS", 2, 2)] = heracles.Result(theory_cls["W2xW2"].array, ell=ls)

c = np.zeros((2, 2, lmax+1))
c[0, 0, :] = theory_cls["W3xW3"].array* fl**2
_theory_cls[("SHE", "SHE", 1, 1)] = heracles.Result(c, ell=ls)

c = np.zeros((2, 2, lmax+1))
c[0, 0, :] = theory_cls["W3xW4"].array* fl**2
_theory_cls[("SHE", "SHE", 1, 2)] = heracles.Result(c, ell=ls)

c = np.zeros((2, 2, lmax+1))
c[0, 0, :] = theory_cls["W4xW4"].array* fl**2
_theory_cls[("SHE", "SHE", 2, 2)] = heracles.Result(c, ell=ls)

c = np.zeros((2, lmax+1))
c[0, :] = theory_cls["W1xW3"].array* fl
_theory_cls[("POS", "SHE", 1, 1)] = heracles.Result(c, ell=ls)

c = np.zeros((2, lmax+1))
c[0, :] = theory_cls["W2xW3"].array* fl
_theory_cls[("POS", "SHE", 1, 2)] = heracles.Result(c, ell=ls)

c = np.zeros((2, lmax+1))
c[0, :] = theory_cls["W3xW2"].array* fl
_theory_cls[("POS", "SHE", 2, 1)] = heracles.Result(c, ell=ls)

c = np.zeros((2, lmax+1))
c[0, :] = theory_cls["W2xW4"].array* fl
_theory_cls[("POS", "SHE", 2, 2)] = heracles.Result(c, ell=ls)

for i in range(1, n+1):
    print(f"Unmixing sim {i}", end='\r')
    # Load cls
    data_cls = heracles.read(f"{path}/cls/cls_data_{i}.fits")

    # Compute mixing matrix
    mms = heracles.mixing_matrices(
        fields,
        mask_cls,
    )

    # direc inversion
    inv_cls = heracles.inversion(data_cls, mms)

    # Save cls
    heracles.write(f"{path}/cls/cls_data_inv_{i}.fits", inv_cls)
