import os
import yaml
import numpy as np
import heracles
import heracles.dices as dices

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
path = f"../{mask_type}/"


ls = np.arange(lmax + 1)
ledges = np.logspace(np.log10(10), np.log10(lmax), nlbins + 1)
lgrid = (ledges[1:] + ledges[:-1]) / 2

ls = np.arange(lmax + 1)
cls = {}
for i in range(1, n+1):
    print(f"Loading sim {i}", end='\r')
    cls[i] = heracles.read(path+f"cls/cls_data_{i}.fits")
# Binning cls
cqs = heracles.binned(cls, ledges)

# Covariance
cls_cov = dices.jackknife_covariance(cls, nd=0)
cqs_cov = dices.jackknife_covariance(cqs, nd=0)

# Save
heracles.write(path+"covs/cov_cls.fits", cls_cov)
heracles.write(path+"covs/cov_cqs.fits", cqs_cov)
print("Done")
