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
mode = config['mode']  # "lognormal" or "gaussian"

nlbins = 10
ls = np.arange(lmax + 1)
ledges = np.logspace(np.log10(10), np.log10(lmax), nlbins + 1)
lgrid = (ledges[1:] + ledges[:-1]) / 2

ls = np.arange(lmax + 1)
path = f"../{mode}_sims/"
cls = {}
cls_wmask = {}
f_cls = {}
i_cls = {}
nu_cls = {}
pp_cls = {}
pm_cls = {}
for i in range(1, n+1):
    print(f"Loading sim {i}", end='\r')
    if os.path.exists(path+mode+f"_sim_{i}/cls_data.fits"):
        cls[i] = heracles.read(path+mode+f"_sim_{i}/cls_data.fits")
        cls_wmask[i] = heracles.read(path+mode+f"_sim_{i}/cls_data_wmask.fits")

cqs = heracles.binned(cls, ledges)
cqs_wmask = heracles.binned(cls_wmask, ledges)

# Covariance
cls_cov = dices.jackknife_covariance(cls, nd=0)
cls_wmask_cov = dices.jackknife_covariance(cls_wmask, nd=0)


cqs_cov = dices.jackknife_covariance(cqs, nd=0)
cqs_wmask_cov = dices.jackknife_covariance(cqs_wmask, nd=0)

# Save
heracles.write(path+"cov_cls.fits", cls_cov)
heracles.write(path+"cov_cls_wmask.fits", cls_wmask_cov)
heracles.write(path+"cov_cqs.fits", cqs_cov)
heracles.write(path+"cov_cqs_wmask.fits", cqs_wmask_cov)
print("Done")
