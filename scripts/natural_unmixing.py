import yaml
import numpy as np
import heracles
from heracles.fields import Positions, Shears, Visibility, Weights
from heracles.healpy import HealpixMapper

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
x0_nu = config.get('x0_nu', -3.5)
x0_inv = config.get('x0_inv', -6.5)

mask_mapper = HealpixMapper(nside=nside, lmax=lmax, deconvolve=False)
mask_fields = {
    "POS": Positions(mask_mapper, mask="VIS"),
    "SHE": Shears(mask_mapper, mask="WHT"),
    "VIS": Visibility(mask_mapper),
    "WHT": Weights(mask_mapper),
}

mask_corr = {}
inv_mask_cls = {}

m_keys = list(mask_cls.keys())
for m_key in m_keys:
    _m = mask_cls[m_key]
    _wm = heracles.transforms.cl2corr(_m)
    # Smooth wm
    _wm = _wm.T[0]
    _wm *= heracles.unmixing.logistic(np.log10(abs(_wm)), x0=x0_nu)
    mask_corr[m_key] = _wm
    __inv_wm = np.zeros((4, len(_wm)))
    __inv_wm[0] = 1/_wm
    _inv_mask_cls = heracles.transforms.corr2cl(__inv_wm.T).T[0]
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
    nu_cls = heracles.unmixing._natural_unmixing(data_cls, mask_corr)
    # Save cls
    heracles.write(f"{path}/cls_nu/cls_data_nu_{i}_l1max_{lmax}_l2max_{lmax_mask}.fits", nu_cls)
