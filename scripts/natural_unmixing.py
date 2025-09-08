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
mode = config['mode']  # "lognormal" or "gaussian"
mask_type = config['mask_type'] # Default to 'Patch' if not specified
path = f"../{mask_type}/"
mask_cls = heracles.read(f"{path}/cls/cls_mask.fits")

mask_mapper = HealpixMapper(nside=nside, lmax=lmax, deconvolve=False)
mask_fields = {
    "POS": Positions(mask_mapper, mask="VIS"),
    "SHE": Shears(mask_mapper, mask="WHT"),
    "VIS": Visibility(mask_mapper),
    "WHT": Weights(mask_mapper),
}

wm = {}
inv_mask_cls = {}

m_keys = list(mask_cls.keys())
for m_key in m_keys:
    _m = mask_cls[m_key]
    _wm = heracles.transforms.cl2corr(_m)
    inv_wm = 1/_wm
    # Smooth wm
    __wm = _wm.T[0]
    __wm *= heracles.unmixing.logistic(np.log10(abs(__wm)), x0=-3, k=50)
    wm[m_key] = __wm
    # Compute inv mask cls
    _inv_mask_cls = heracles.transforms.corr2cl(inv_wm).T[0]
    inv_mask_cls[m_key] = heracles.Result(_inv_mask_cls,
                                          axis=mask_cls[m_key].axis,
                                          ell=mask_cls[m_key].ell)

# inv Mixing matrices
unmms = heracles.mixing_matrices(
    mask_fields,
    inv_mask_cls,
    l1max=lmax,)
heracles.write(path+f"/unmixmat.fits", unmms)

for i in range(1, n+1):
    print(f"Unmixing sim {i}", end='\r')
    # Load cls
    data_cls = heracles.read(f"{path}/cls/cls_data_{i}.fits")
    # PolSpice
    nu_cls = heracles.unmixing._natural_unmixing(data_cls, wm)
    # Save cls
    heracles.write(f"{path}/cls_nu/cls_data_nu_{i}.fits", nu_cls)
