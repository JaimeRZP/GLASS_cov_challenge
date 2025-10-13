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
rtol_nu = config.get('rtol_nu', 1e-3)
rtol_inv = config.get('rtol_inv', 1e-2)
rtol_nu = np.float32(rtol_nu)
rtol_inv = np.float32(rtol_inv)

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
    _wm = heracles.transforms.cl2corr(_m).T[0]
    mask_corr[m_key] = _wm
    # psuedoinverse wm
    cutoff = rtol_nu * np.max(np.abs(_wm))
    _wm *= heracles.unmixing.logistic(np.log10(abs(_wm)), x0=np.log10(cutoff))
    inv_wm = 1 / _wm
    #inv_wm = np.array([1/wi if abs(wi) > cutoff else 0 for wi in _wm])
    # transform back to cls
    __inv_wm = np.zeros((4, len(_wm)))
    __inv_wm[0] = inv_wm
    _inv_mask_cls = heracles.transforms.corr2cl(__inv_wm.T).T[0]
    inv_mask_cls[m_key] = heracles.Result(_inv_mask_cls,
                                          axis=mask_cls[m_key].axis,
                                          ell=mask_cls[m_key].ell)

# inv Mixing matrices
unmms = heracles.mixing_matrices(
    mask_fields,
    inv_mask_cls,
    l1max=lmax,
    l2max=lmax,
    l3max=lmax_mask)
heracles.write(path+f"/unmixmat_l1max_{lmax}_l2max_{lmax_mask}.fits", unmms)

cls = {}
for i in range(1, n+1):
    print(f"Unmixing sim {i}", end='\r')
    # Load cls
    data_cls = heracles.read(f"{path}/cls/cls_data_{i}_lmax_{lmax}.fits")
    # PolSpice
    nu_cls = heracles.unmixing._natural_unmixing(data_cls, mask_corr)
    # Save cls
    cls[i] = nu_cls
    heracles.write(f"{path}/cls_nu/cls_data_nu_{i}_l1max_{lmax}_l2max_{lmax_mask}.fits", nu_cls)
print("Done")

# Binning cls
nlbins = config.get('nlbins', 20)  # Default to 20 if not specified
ls = np.arange(lmax + 1)
ledges = np.logspace(np.log10(10), np.log10(lmax), nlbins + 1)
lgrid = (ledges[1:] + ledges[:-1]) / 2
print(f"Using {len(lgrid)} bins for the covariance matrix.")
nu_cqs = heracles.binned(cls, ledges)

# compute covariances
print("Computing covariances")
nu_cls_cov = heracles.dices.jackknife_covariance(cls, nd=0)
nu_cqs_cov = heracles.dices.jackknife_covariance(nu_cqs, nd=0)

# Save
print("Saving covariances")
heracles.write(path+f"covs/cov_nu_cls_l1max_{lmax}_l2max_{lmax_mask}.fits", nu_cls_cov)
heracles.write(path+f"covs/cov_nu_cqs_l1max_{lmax}_l2max_{lmax_mask}.fits", nu_cqs_cov)
