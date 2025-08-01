import yaml
import heracles
import numpy as np

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

wm = {}
inv_wm = {}
m_keys = list(mask_cls.keys())
for m_key in m_keys:
    _m = mask_cls[m_key].array
    _wm = heracles.transforms.cl2corr(_m).T[0]
    _wm *= heracles.unmixing.logistic(np.log10(abs(_wm)), x0=-2, k=50)
    wm[m_key] = _wm
    inv_wm[m_key] = 1/_wm



for i in range(1, n+1):
    print(f"Unmixing sim {i}", end='\r')
    # Load cls
    data_cls = heracles.read(f"{path}/cls/cls_data_{i}.fits")
    # PolSpice
    nu_cls = heracles.unmixing._natural_unmixing(data_cls, wm, patch_hole=True)
    #pp_cls = heracles.unmixing.PolSpice(data_cls, mask_cls, mode='plus', patch_hole=True)
    #pm_cls = heracles.unmixing.PolSpice(data_cls, mask_cls, mode='minus', patch_hole=True)
    # Save cls
    heracles.write(f"{path}/cls_nu/cls_data_nu_{i}.fits", nu_cls)
    #heracles.write(f"{path}/cls_pp/cls_data_pp_{i}.fits", pp_cls)
    #heracles.write(f"{path}/cls_pm/cls_data_pm_{i}.fits", pm_cls)
