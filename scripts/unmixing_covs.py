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
print(f"Using {len(lgrid)} bins for the covariance matrix.")

nmt_cqs = {}
i_cls = {}
nu_cls = {}
#pp_cls = {}
#pm_cls = {}
pols_cls = {}
#dpols_cls = {}
for i in range(1, n+1):
    print(f"Loading sim {i}", end='\r')
    nmt_cqs[i] = heracles.read(path+f"cls_nmt/cqs_data_nmt_np_{i}.fits")
    i_cls[i] = heracles.read(path+f"cls_inv/cls_data_inv_{i}.fits")
    nu_cls[i] = heracles.read(path+f"cls_nu/cls_data_nu_{i}.fits")
    #pp_cls[i] = heracles.read(path+f"cls_pp/cls_data_pp_{i}.fits")
    #pm_cls[i] = heracles.read(path+f"cls_pm/cls_data_pm_{i}.fits")
    pols_cls[i] = heracles.read(path+f"cls_pols/cls_data_pols_{i}.fits")
    #dpols_cls[i] = heracles.read(path+f"cls_pols/cls_data_decoupled_pols_{i}.fits")
# Binning
i_cqs = heracles.binned(i_cls, ledges)
nu_cqs = heracles.binned(nu_cls, ledges)
#pp_cqs = heracles.binned(pp_cls, ledges)
#pm_cqs = heracles.binned(pm_cls, ledges)
pols_cqs = heracles.binned(pols_cls, ledges)
#dpols_cqs = heracles.binned(dpols_cls, ledges)
# Covariance
i_cls_cov = dices.jackknife_covariance(i_cls, nd=0)
nu_cls_cov = dices.jackknife_covariance(nu_cls, nd=0)
#pp_cls_cov = dices.jackknife_covariance(pp_cls, nd=0)
#pm_cls_cov = dices.jackknife_covariance(pm_cls, nd=0)
pols_cls_cov = dices.jackknife_covariance(pols_cls, nd=0)
#dpols_cls_cov = dices.jackknife_covariance(dpols_cls, nd=0)
# Covariance for binned cls
i_cqs_cov = dices.jackknife_covariance(i_cqs, nd=0)
nu_cqs_cov = dices.jackknife_covariance(nu_cqs, nd=0)
#pp_cqs_cov = dices.jackknife_covariance(pp_cqs, nd=0)
#pm_cqs_cov = dices.jackknife_covariance(pm_cqs, nd=0)
nmt_cqs_cov = dices.jackknife_covariance(nmt_cqs, nd=0)
pols_cqs_cov = dices.jackknife_covariance(pols_cqs, nd=0)
#dpols_cqs_cov = dices.jackknife_covariance(dpols_cqs, nd=0)

# Save
heracles.write(path+"covs/cov_inv_cls.fits", i_cls_cov)
heracles.write(path+"covs/cov_nu_cls.fits", nu_cls_cov)
#heracles.write(path+"covs/cov_pp_cls.fits", pp_cls_cov)
#heracles.write(path+"covs/cov_pm_cls.fits", pm_cls_cov)
heracles.write(path+"covs/cov_pols_cls.fits", pols_cls_cov)
#eracles.write(path+"covs/cov_dpols_cls.fits", dpols_cls_cov)
heracles.write(path+"covs/cov_inv_cqs.fits", i_cqs_cov)
heracles.write(path+"covs/cov_nu_cqs.fits", nu_cqs_cov)
#heracles.write(path+"covs/cov_pp_cqs.fits", pp_cqs_cov)
#heracles.write(path+"covs/cov_pm_cqs.fits", pm_cqs_cov)
heracles.write(path+"covs/cov_nmt_cqs.fits", nmt_cqs_cov)
heracles.write(path+"covs/cov_pols_cqs.fits", pols_cqs_cov)
#heracles.write(path+"covs/cov_dpols_cqs.fits", dpols_cqs_cov)
print("Done")
