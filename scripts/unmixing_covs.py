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
lmax_mask = config.get('lmax_mask', lmax)  # Default to lmax if not specified
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
pols_cls = {}
for i in range(1, n+1):
    print(f"Loading sim {i}", end='\r')
    nmt_cqs[i] = heracles.read(path+f"cls_nmt/cqs_data_nmt_np_{i}_lmax_{lmax}.fits")
    i_cls[i] = heracles.read(path+f"cls_inv/cls_data_inv_{i}_l1max_{lmax}_l2max_{lmax_mask}.fits")
    nu_cls[i] = heracles.read(path+f"cls_nu/cls_data_nu_{i}_l1max_{lmax}_l2max_{lmax_mask}.fits")
    pols_cls[i] = heracles.read(path+f"cls_pols/cls_data_pols_{i}_lmax_{lmax}.fits")
# Binning
i_cqs = heracles.binned(i_cls, ledges)
nu_cqs = heracles.binned(nu_cls, ledges)
pols_cqs = heracles.binned(pols_cls, ledges)
# Covariance
i_cls_cov = dices.jackknife_covariance(i_cls, nd=0)
nu_cls_cov = dices.jackknife_covariance(nu_cls, nd=0)
pols_cls_cov = dices.jackknife_covariance(pols_cls, nd=0)
# Covariance for binned cls
i_cqs_cov = dices.jackknife_covariance(i_cqs, nd=0)
nu_cqs_cov = dices.jackknife_covariance(nu_cqs, nd=0)
nmt_cqs_cov = dices.jackknife_covariance(nmt_cqs, nd=0)
pols_cqs_cov = dices.jackknife_covariance(pols_cqs, nd=0)

# Save
heracles.write(path+f"covs/cov_inv_cls_l1max_{lmax}_l2max_{lmax_mask}.fits", i_cls_cov)
heracles.write(path+f"covs/cov_nu_cls_l1max_{lmax}_l2max_{lmax_mask}.fits", nu_cls_cov)
heracles.write(path+f"covs/cov_pols_cls_l1max_{lmax}_l2max_{lmax_mask}.fits", pols_cls_cov)
heracles.write(path+f"covs/cov_inv_cqs_l1max_{lmax}_l2max_{lmax_mask}.fits", i_cqs_cov)
heracles.write(path+f"covs/cov_nu_cqs_l1max_{lmax}_l2max_{lmax_mask}.fits", nu_cqs_cov)
heracles.write(path+f"covs/cov_nmt_cqs_l1max_{lmax}_l2max_{lmax_mask}.fits", nmt_cqs_cov)
heracles.write(path+f"covs/cov_pols_cqs_l1max_{lmax}_l2max_{lmax_mask}.fits", pols_cqs_cov)
print("Done")
