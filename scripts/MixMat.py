import os
import yaml
import numpy as np
import healpy as hp
import heracles
from heracles.healpy import HealpixMapper
from heracles.fields import Positions, Shears, Visibility, Weights
import heracles.dices as dices
from heracles.result import Result


def inversion(d, M):
    """
    Inversion model for the unmixing E/B modes.
    Args:
        d: Data Cl
        M: Mixing matrix
        Returns:
        inversion_cls: inverted Cl
    """
    inversion_cls = {}
    for key in list(d.keys()):
        a, b, i, j = key
        _d = np.atleast_2d(d[key])
        _M = M[key].array
        *_, _n, _m = _M.shape
        if a == b == "SHE":
            _M_EB = _M[2]
            _M_EE = np.hstack((_M[0], _M[1]))
            _M_BB = np.hstack((_M[1], _M[0]))
            _M_EEBB = np.vstack((_M_EE, _M_BB))
            _inv_M_EEBB = np.linalg.pinv(_M_EEBB)
            _inv_M_EB = np.linalg.pinv(_M_EB)
            _d_EEBB = np.hstack((_d[0, 0, :], _d[1, 1, :]))
            _id_EEBB = _inv_M_EEBB @ _d_EEBB
            _id_EE = _id_EEBB[:_m][:_n]
            _id_BB = _id_EEBB[_m:][:_n]
            _id_EB = _inv_M_EB @ _d[0, 1, :]
            _id_BE = _inv_M_EB @ _d[1, 0, :]
            _id_EB = _id_EB[:_n]
            _id_BE = _id_BE[:_n]
            _id = np.array([[_id_EE, _id_EB], [_id_BE, _id_BB]])
        else:
            _inv_M = np.linalg.pinv(_M)
            _id = np.array([_inv_M @ __d.T for __d in _d])
            _id = _id[:, :_n]
        if len(_id) == 1:
            _id = _id[0]
        inversion_cls[key] = Result(_id, axis=d[key].axis, ell=d[key].ell)
    return inversion_cls


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

c = np.zeros((2, 2, lmax+1))
c[0, 0, :] = theory_cls["W2xW2"].array * fl**2
_theory_cls[("SHE", "SHE", 1, 1)] = heracles.Result(c, ell=ls)

c = np.zeros((2, lmax+1))
c[0, :] = theory_cls["W1xW2"].array * fl
_theory_cls[("POS", "SHE", 1, 1)] = heracles.Result(c, ell=ls)

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
    inv_cls = inversion(data_cls, mms)

    # Save cls
    heracles.write(f"{path}/cls_inv/cls_data_inv_{i}.fits", inv_cls)
