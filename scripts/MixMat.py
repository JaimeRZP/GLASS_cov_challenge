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

for i in range(1, n+1):
    folname = f"{mode}_sim_{i}"
    print(f"Unmixing sim {i} in {folname}", end='\r')
    # Load cls
    data_cls = heracles.read(f"{path}/{folname}/cls_data_wmask.fits")
    mask_cls = heracles.read(f"{path}/{folname}/cls_mask.fits")
    theory_cls = heracles.read(f"{path}/{folname}/theory_cls.fits")
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

    # Compute mixing matrix
    mms = heracles.mixing_matrices(
        fields,
        mask_cls,
    )

    # forwards
    f_cls = heracles.forwards(_theory_cls, mms)
    # direc inversion
    i_cls = heracles.inversion(data_cls, mms)

    # Save cls
    output_path = f"{path}/{folname}/"
    heracles.write(output_path + "theory_cls_f.fits", f_cls)
    heracles.write(output_path + "cls_data_i.fits", i_cls)
