import yaml
import numpy as np
import heracles
from heracles.fields import Positions, Shears, Visibility, Weights
from heracles.healpy import HealpixMapper
from heracles.transforms import cl2corr, corr2cl, _cached_gauss_legendre
from heracles.result import Result

# Config
config_path = "./sims_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
n = config['nsims']
nside = config['nside']
lmax = config['lmax']
lmax_mask = config['lmax_mask']
mode = config['mode']  # "lognormal" or "gaussian"
mask_type = config['mask_type'] # Default to 'Patch' if not specified
path = f"../{mask_type}/"
mask_cls = heracles.read(f"{path}/cls/cls_mask_lmax_{lmax_mask}.fits")

mask_mapper = HealpixMapper(nside=nside, lmax=lmax, deconvolve=False)
mask_fields = {
    "POS": Positions(mask_mapper, mask="VIS"),
    "SHE": Shears(mask_mapper, mask="WHT"),
    "VIS": Visibility(mask_mapper),
    "WHT": Weights(mask_mapper),
}


def __natural_unmixing(d, wm):
    """
    Natural unmixing of the data Cl.
    Args:
        d: Data Cl
        m: mask cls
        patch_hole: If True, apply the patch hole correction
    Returns:
        corr_d: Corrected Cl
    """
    corr_d = {}
    d_keys = list(d.keys())
    wm_keys = list(wm.keys())
    for d_key, wm_key in zip(d_keys, wm_keys):
        a, b, i, j = d_key
        _wm = wm[wm_key]
        # Grab metadata
        dtype = d[d_key].array.dtype
        ell = d[d_key].ell
        axis = d[d_key].axis
        # Check if ell is None
        if ell is None:
            ell = np.arange(len(_wm))
        _d = np.atleast_2d(d[d_key])
        if a == b == "SHE":
            __d = np.array(
                [
                    np.zeros_like(_d[0, 0]),
                    _d[0, 0],  # EE like spin-2
                    _d[1, 1],  # BB like spin-2
                    np.zeros_like(_d[0, 0]),
                ]
            )
            __id = np.array(
                [
                    np.zeros_like(_d[0, 0]),
                    -_d[0, 1],  # EB like spin-0
                    _d[1, 0],  # EB like spin-0
                    np.zeros_like(_d[0, 0]),
                ]
            )
            # Correct by alpha
            wd = cl2corr(__d.T).T + 1j * cl2corr(__id.T).T
            corr_wd = (wd * _wm).real
            icorr_wd = (wd * _wm).imag
            # Transform back to Cl
            __corr_d = corr2cl(corr_wd.T).T
            __icorr_d = corr2cl(icorr_wd.T).T
            # reorder
            _corr_d = np.zeros_like(_d)
            _corr_d[0, 0] = __corr_d[1]  # EE like spin-2
            _corr_d[1, 1] = __corr_d[2]  # BB like spin-2
            _corr_d[0, 1] = -__icorr_d[1]  # EB like spin-0
            _corr_d[1, 0] = __icorr_d[2]  # EB like spin-0
        else:
            # Treat everything as spin-0
            _corr_d = []
            for cl in _d:
                wd = cl2corr(cl).T
                corr_wd = wd * _wm
                # Transform back to Cl
                __corr_d = corr2cl(corr_wd.T).T
                _corr_d.append(__corr_d[0])
            # remove extra axis
            _corr_d = np.squeeze(_corr_d)
        # Add metadata back
        _corr_d = np.array(list(_corr_d), dtype=dtype)
        corr_d[d_key] = Result(_corr_d, axis=axis, ell=ell)
    return corr_d

inv_mask_corr = {}
inv_mask_cls = {}

m_keys = list(mask_cls.keys())
for m_key in m_keys:
    _m = mask_cls[m_key]
    _wm = heracles.transforms.cl2corr(_m)
    # Smooth wm
    _wm = _wm.T[0]
    _wm *= heracles.unmixing.logistic(np.log10(abs(_wm)), x0=-3, k=50)
    inv_wm = 1/_wm
    # Interpolate 
    xmask, _ = _cached_gauss_legendre(lmax_mask+1)
    xcls, _ = _cached_gauss_legendre(lmax+1)
    inv_wm = np.interp(xcls, xmask, inv_wm)
    inv_mask_corr[m_key] = inv_wm
    # Compute inv mask cls
    _inv_wm = np.zeros((4, len(inv_wm)))
    _inv_wm[0] = inv_wm
    _inv_mask_cls = heracles.transforms.corr2cl(_inv_wm.T).T[0]
    inv_mask_cls[m_key] = heracles.Result(_inv_mask_cls,
                                          axis=mask_cls[m_key].axis,
                                          ell=mask_cls[m_key].ell)

# inv Mixing matrices
unmms = heracles.mixing_matrices(
    mask_fields,
    inv_mask_cls,
    l1max=lmax,
    l2max=lmax_mask,)
heracles.write(path+f"/unmixmat_l1max_{lmax}_l2max_{lmax_mask}.fits", unmms)

for i in range(1, n+1):
    print(f"Unmixing sim {i}", end='\r')
    # Load cls
    data_cls = heracles.read(f"{path}/cls/cls_data_{i}_lmax_{lmax}.fits")
    # PolSpice
    nu_cls = __natural_unmixing(data_cls, inv_mask_corr)
    # Save cls
    heracles.write(f"{path}/cls_nu/cls_data_nu_{i}_l1max_{lmax}_l2max_{lmax_mask}.fits", nu_cls)
