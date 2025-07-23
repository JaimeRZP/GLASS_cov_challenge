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

ledges = np.logspace(np.log10(10), np.log10(lmax), nlbins + 1)
lgrid = (ledges[1:] + ledges[:-1]) / 2

i_cqs = {}
nu_cls = {}
pp_cls = {}
pm_cls = {}
for i in range(1, n+1):
    print(f"Loading sim {i}", end='\r')
    i_cqs[i] = heracles.read(path+f"cls_inv/cqs_data_inv_{i}.fits")
    nu_cls[i] = heracles.read(path+f"cls_nu/cls_data_nu_{i}.fits")
    pp_cls[i] = heracles.read(path+f"cls_pp/cls_data_pp_{i}.fits")
    pm_cls[i] = heracles.read(path+f"cls_pm/cls_data_pm_{i}.fits")
# Binning
nu_cqs = heracles.binned(nu_cls, ledges)
pp_cqs = heracles.binned(pp_cls, ledges)
pm_cqs = heracles.binned(pm_cls, ledges)
# Covariance
nu_cls_cov = dices.jackknife_covariance(nu_cls, nd=0)
pp_cls_cov = dices.jackknife_covariance(pp_cls, nd=0)
pm_cls_cov = dices.jackknife_covariance(pm_cls, nd=0)
i_cqs_cov = dices.jackknife_covariance(i_cqs, nd=0)
nu_cqs_cov = dices.jackknife_covariance(nu_cqs, nd=0)
pp_cqs_cov = dices.jackknife_covariance(pp_cqs, nd=0)
pm_cqs_cov = dices.jackknife_covariance(pm_cqs, nd=0)
# Save
heracles.write(path+"covs/cov_nu_cls.fits", nu_cls_cov)
heracles.write(path+"covs/cov_pp_cls.fits", pp_cls_cov)
heracles.write(path+"covs/cov_pm_cls.fits", pm_cls_cov)
heracles.write(path+"covs/cov_inv_cqs.fits", i_cqs_cov)
heracles.write(path+"covs/cov_nu_cqs.fits", nu_cqs_cov)
heracles.write(path+"covs/cov_pp_cqs.fits", pp_cqs_cov)
heracles.write(path+"covs/cov_pm_cqs.fits", pm_cqs_cov)
print("Done")
