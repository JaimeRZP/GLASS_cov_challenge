import yaml
import heracles
import numpy as np
from heracles.result import truncated
from heracles.twopoint import invert_mixing_matrix, apply_mixing_matrix

# Config
config_path = "./sims_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
n = config['nsims']
nside = config['nside']
lmax = config['lmax']
lmax_mask = config['lmax_mask']
nlbins = config.get('nlbins', 20)  # Default to 20 if not specified
mode = config['mode']  # "lognormal" or "gaussian"
mask_type = config.get('mask_type', 'Patch')  # Default to 'Patch' if not specified
path = f"../{mask_type}"
rtol_inv = config.get('rtol_inv', 1e-2)
rtol_nu = config.get('rtol_nu', 1e-3)
rtol_nu = np.float32(rtol_nu)
rtol_inv = np.float32(rtol_inv)

# Compute mixing matrix
mms = heracles.read(f"{path}/mixmat_l1max_{lmax}_l2max_{lmax_mask}.fits")
# Invert the mixing matrix
inv_mms = invert_mixing_matrix(mms, rtol=rtol_inv)
heracles.write(path+f"/inv_mixmat_l1max_{lmax}_l2max_{lmax_mask}.fits", inv_mms)

for i in range(1, n+1):
    print(f"Unmixing sim {i}", end='\r')
    # Load cls
    data_cls = heracles.read(f"{path}/cls/cls_data_{i}_lmax_{lmax}.fits")
    inv_cls = apply_mixing_matrix(data_cls, inv_mms)
    inv_cls = truncated(inv_cls, lmax)
    heracles.write(f"{path}/cls_inv/cls_data_inv_{i}_l1max_{lmax}_l2max_{lmax_mask}.fits", inv_cls)
