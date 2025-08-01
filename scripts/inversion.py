import os
import yaml
import numpy as np
import healpy as hp
import heracles
from heracles.healpy import HealpixMapper
from heracles.fields import Positions, Shears, Visibility, Weights
import heracles.dices as dices
from heracles.result import Result


def apply_matrix(cls, M, rtol=1e-5):
    """
    Inversion model for the unmixing E/B modes.
    Args:
        d: Data Cl
        M: Mixing matrix
        Returns:
        inversion_cls: inverted Cl
    """
    _cls = {}
    for key in list(cls.keys()):
        a, b, i, j = key
        c = np.atleast_2d(cls[key])
        _M = M[key].array
        *_, _n, _m = _M.shape
        ell = np.arange(_m)
        if a == b == "SHE":
            _M_EB = _M[2]
            _M_EE = np.hstack((_M[0], _M[1]))
            _M_BB = np.hstack((_M[1], _M[0]))
            _M_EEBB = np.vstack((_M_EE, _M_BB))
            _d_EEBB = np.hstack((c[0, 0, :], c[1, 1, :]))
            _c_EEBB = _M_EEBB @ _d_EEBB
            _c_EE = _c_EEBB[:_m]
            _c_BB = _c_EEBB[_m:]
            _c_EB = _M_EB @ c[0, 1, :]
            _c_BE = _M_EB @ c[1, 0, :]
            _c = np.array([[_c_EE, _c_EB], [_c_BE, _c_BB]])
        else:
            _c = np.array([_M @ __c.T for __c in c])
        if len(_c) == 1:
            _c = _c[0]
        _cls[key] = Result(_c, axis=cls[key].axis, ell=ell)
    return _cls


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
mms = heracles.read(f"{path}/cls/mixmat.fits")
# Invert the mixing matrix
inv_mms_pppp = np.linalg.pinv(mms["POS", "POS", 1, 1].array, rtol=rtol).T
inv_mms_pepe = np.linalg.pinv(mms["POS", "SHE", 1, 1].array, rtol=rtol).T
inv_mms_ebeb = np.linalg.pinv(mms['SHE', 'SHE', 1, 1][2], rtol=rtol).T
mms_eeee = np.hstack((mms['SHE', 'SHE', 1, 1][0], mms['SHE', 'SHE', 1, 1][1]))
mms_eebb = np.hstack((mms['SHE', 'SHE', 1, 1][1], mms['SHE', 'SHE', 1, 1][0]))
mms_she = np.vstack((mms_eeee, mms_eebb))
inv_mms_she = np.linalg.pinv(mms_she, rtol=rtol).T
inv_mms_eeee = inv_mms_she[:lmax+1, :lmax+1]
inv_mms_eebb = inv_mms_she[lmax+1:, :lmax+1]
inv_mms = {
    ("POS", "POS", 1, 1): Result(inv_mms_pppp, axis=mms["POS", "POS", 1, 1].axis, ell=ls),
    ("POS", "SHE", 1, 1): Result(inv_mms_pepe, axis=mms["POS", "SHE", 1, 1].axis, ell=ls),
    ("SHE", "SHE", 1, 1): Result(np.array([inv_mms_eeee, inv_mms_eebb, inv_mms_ebeb]), axis=mms["SHE", "SHE", 1, 1].axis, ell=ls),
    }
# Save inverted mixing matrix
heracles.write(f"{path}/cls/inv_mixmat.fits", inv_mms)

# binned mixing matrix
mmqs = heracles.binned(mms, ledges)
# binned inverted mixing matrix
inv_mmqs_pppp = np.linalg.pinv(mmqs["POS", "POS", 1, 1].array, rtol=rtol).T
inv_mmqs_pepe = np.linalg.pinv(mmqs["POS", "SHE", 1, 1].array, rtol=rtol).T
inv_mmqs_ebeb = np.linalg.pinv(mmqs['SHE', 'SHE', 1, 1][2], rtol=rtol).T
mmqs_eeee = np.hstack((mmqs['SHE', 'SHE', 1, 1][0], mmqs['SHE', 'SHE', 1, 1][1]))
mmqs_eebb = np.hstack((mmqs['SHE', 'SHE', 1, 1][1], mmqs['SHE', 'SHE', 1, 1][0]))
mmqs_she = np.vstack((mmqs_eeee, mmqs_eebb))
inv_mmqs_she = np.linalg.pinv(mmqs_she, rtol=rtol).T
inv_mmqs_eeee = inv_mmqs_she[:nlbins, :lmax+1]
inv_mmqs_eebb = inv_mmqs_she[nlbins:, :lmax+1]
inv_mmqs = {
    ("POS", "POS", 1, 1): Result(inv_mmqs_pppp, axis=mmqs["POS", "POS", 1, 1].axis, ell=lgrid),
    ("POS", "SHE", 1, 1): Result(inv_mmqs_pepe, axis=mmqs["POS", "SHE", 1, 1].axis, ell=lgrid),
    ("SHE", "SHE", 1, 1): Result(np.array([inv_mmqs_eeee, inv_mmqs_eebb, inv_mmqs_ebeb]), axis=mmqs["SHE", "SHE", 1, 1].axis, ell=lgrid),
}
#save binned inverted mixing matrix
heracles.write(f"{path}/cls/inv_mixmat_binned.fits", inv_mmqs)

# Compute inversion kernel
inv_kk_pppp = inv_mms_pppp @ mms['POS', 'POS', 1, 1].array
inv_kk_ppss = inv_mms_pepe @ mms['POS', 'SHE', 1, 1].array
inv_kk_ebeb = inv_mms_ebeb @ mms['SHE', 'SHE', 1, 1][2]
inv_kk_ssss = inv_mms_she @ mms_she
inv_kk_eeee = inv_kk_ssss[:lmax+1, :lmax+1]
inv_kk_eebb = inv_kk_ssss[lmax+1:, :lmax+1]
inv_kk = {
    ("POS", "POS", 1, 1): Result(inv_kk_pppp, axis=mms["POS", "POS", 1, 1].axis, ell=ls),
    ("POS", "SHE", 1, 1): Result(inv_kk_ppss, axis=mms["POS", "SHE", 1, 1].axis, ell=ls),
    ("SHE", "SHE", 1, 1): Result(np.array([inv_kk_eebb, inv_kk_ebeb, inv_kk_ebeb]), axis=mms["SHE", "SHE", 1, 1].axis, ell=ls),
}
# Save inversion kernel
heracles.write(f"{path}/cls_inv/inv_kernel.fits", inv_kk)

# Compute binned inversion kernel
inv_kkq_pppp = inv_mmqs_pppp @ mms['POS', 'POS', 1, 1].array
inv_kkq_ppss = inv_mmqs_pepe @ mms['POS', 'SHE', 1, 1].array
inv_kkq_ebeb = inv_mmqs_ebeb @ mms['SHE', 'SHE', 1, 1][2]
inv_kkq_ssss = inv_mmqs_she @ mms_she
inv_kkq_eeee = inv_kkq_ssss[:nlbins, :lmax+1]
inv_kkq_eebb = inv_kkq_ssss[nlbins:, :lmax+1]
inv_kkq_binned = {
    ("POS", "POS", 1, 1): Result(inv_kkq_pppp, axis=mmqs["POS", "POS", 1, 1].axis, ell=lgrid),
    ("POS", "SHE", 1, 1): Result(inv_kkq_ppss, axis=mmqs["POS", "SHE", 1, 1].axis, ell=lgrid),
    ("SHE", "SHE", 1, 1): Result(np.array([inv_kkq_eebb, inv_kkq_ebeb, inv_kkq_ebeb]), axis=mmqs["SHE", "SHE", 1, 1].axis, ell=lgrid),
}
# Save binned inversion kernel
heracles.write(f"{path}/cls_inv/inv_kernel_binned.fits", inv_kkq_binned)

for i in range(1, n+1):
    print(f"Unmixing sim {i}", end='\r')
    # Load cls
    data_cls = heracles.read(f"{path}/cls/cls_data_{i}.fits")
    # binned data cls
    #data_cqs = heracles.binned(data_cls, ledges)
    # direc inversion
    #inv_cls = inversion(data_cqs, mmqs)
    inv_cls = apply_matrix(data_cls, inv_mms)
    # binned inversion cls
    #inv_cqs = heracles.binned(inv_cls, ledges)
    # Save cls
    heracles.write(f"{path}/cls_inv/cls_data_inv_{i}.fits", inv_cls)
