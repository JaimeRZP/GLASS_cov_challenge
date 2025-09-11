import os
import yaml
import numpy as np
import healpy as hp
import heracles
from heracles.healpy import HealpixMapper
from heracles.fields import Positions, Shears, Visibility, Weights
import heracles.dices as dices
from heracles.result import Result


def invert_mixing_matrix(M, rtol=1e-5):
    """
    Inversion model for the unmixing E/B modes.
    Args:
        M: Mixing matrix
        Returns:
        inversion_cls: inverted Cl
    """
    inv_M = {}
    for key in M.keys():
        a, b, i, j = key
        _M = M[key].array
        *_, _n, _m = _M.shape
        if a == b == "SHE":
            _inv_m = np.linalg.pinv(
                np.vstack((np.hstack((_M[0], _M[1])), np.hstack((_M[1], _M[0])))),
                rtol=rtol,
            )
            _inv_M_EEEE = _inv_m[:_m, :_n]
            _inv_M_EEBB = _inv_m[_m:, _n:]
            _inv_M_EBEB = np.linalg.pinv(_M[2], rtol=rtol)
            _inv_M = np.array([_inv_M_EEEE, _inv_M_EEBB, _inv_M_EBEB])
        else:
            _inv_M = np.linalg.pinv(_M, rtol=rtol)
        inv_M[key] = Result(_inv_M, axis=M[key].axis, ell=M[key].ell)
    return inv_M


def apply_mixing_matrix(d, M):
    """
    Apply mixing matrix to the data Cl.
    Args:
        d: Data Cl
        M: Mixing matrix
        Returns:
        corr_d: Corrected Cl
    """
    corr_d = {}
    for key in d.keys():
        a, b, i, j = key
        dtype = d[key].array.dtype
        ell = d[key].ell
        axis = d[key].axis
        _d = np.atleast_2d(d[key].array)
        _M = M[key].array
        *_, _n, _m = _M.shape
        if a == b == "SHE":
            _corr_d_EE = _M[0] @ _d[0, 0] + _M[1] @ _d[1, 1]
            _corr_d_BB = _M[1] @ _d[0, 0] + _M[0] @ _d[1, 1]
            _corr_d_EB = _M[2] @ _d[1, 1]
            _corr_d = np.array([[_corr_d_EE, _corr_d_EB], [_corr_d_EB, _corr_d_BB]])
        else:
            _corr_d = []
            for cl in _d:
                _corr_d.append(_M @ cl)
            _corr_d = np.squeeze(_corr_d)
        _corr_d = np.array(list(_corr_d), dtype=dtype)
        corr_d[key] = Result(_corr_d, axis=axis, ell=ell)
    return corr_d


# Config
config_path = "./sims_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
n = config['nsims']
nside = config['nside']
lmax = config['lmax']
nlbins = config.get('nlbins', 20)  # Default to 20 if not specified
mode = config['mode']  # "lognormal" or "gaussian"
mask_type = config.get('mask_type', 'Patch')  # Default to 'Patch' if not specified
path = f"../{mask_type}"
rtol=1e-2

# Format theory cls
# Bit hardcoded for now, but could be generalized
ls = np.arange(lmax + 1)
ledges = np.logspace(np.log10(10), np.log10(lmax), nlbins + 1)
lgrid = (ledges[1:] + ledges[:-1]) / 2

# Compute mixing matrix
mms = heracles.read(f"{path}/mixmat.fits")
# Invert the mixing matrix
inv_mms = invert_mixing_matrix(mms, rtol=rtol)

for i in range(1, n+1):
    print(f"Unmixing sim {i}", end='\r')
    # Load cls
    data_cls = heracles.read(f"{path}/cls/cls_data_{i}.fits")
    # binned data cls
    #data_cqs = heracles.binned(data_cls, ledges)
    # direc inversion
    #inv_cls = inversion(data_cqs, mmqs)
    inv_cls = apply_mixing_matrix(data_cls, inv_mms)
    # binned inversion cls
    #inv_cqs = heracles.binned(inv_cls, ledges)
    # Save cls
    heracles.write(f"{path}/cls_inv/cls_data_inv_{i}.fits", inv_cls)
