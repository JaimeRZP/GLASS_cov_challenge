import yaml
import numpy as np
import healpy as hp
import heracles
import pymaster as nmt
import heracles.dices as dices


# conda activate nmt
def compute_master(f_a, f_b, wsp):
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    return cl_decoupled

# Config
config_path = "./sims_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
n = config['nsims']
nside = config['nside']
lmax = config['lmax']
nlbins = config.get('nlbins', 20)  # Default to 20 if not specified
mode = config['mode']  # "lognormal" or "gaussian"
mask_type = config['mask_type']  # Default to 'Patch' if not specified
path = f"../{mask_type}"

ls = np.arange(lmax + 1)
ledges = np.logspace(np.log10(10), np.log10(lmax), nlbins + 1)
lgrid = ledges[:-1] + np.diff(ledges)/2
b = nmt.NmtBin.from_edges(ledges[:-1].astype(int), ledges[1:].astype(int))
b.lmax = lmax

mask = hp.read_map(f"{path}/mask.fits")
mask_apo = nmt.mask_apodization(mask, 0.0, apotype='C1')

# workspaces
sim_path = f"../{mode}_sims/{mode}_sim_1_nside_{nside}"
POS1 = heracles.read_maps(f"{sim_path}/POS_1.fits")
SHE1 = heracles.read_maps(f"{sim_path}/SHE_1.fits")
map_t = POS1['POS', 1]
map_q = SHE1['SHE', 1][0]
map_u = SHE1['SHE', 1][1]
print(map_t.shape, map_q.shape, map_u.shape)
f0 = nmt.NmtField(mask_apo, [map_t], lmax=lmax)
f2 = nmt.NmtField(mask_apo, [map_q, map_u], lmax=lmax)
w00 = nmt.NmtWorkspace.from_fields(f0, f0, b)
w02 = nmt.NmtWorkspace.from_fields(f0, f2, b)
w22 = nmt.NmtWorkspace.from_fields(f2, f2, b)

cls = {}
for i in range(1, n+1):
    print(f"Unmixing sim {i}", end='\r')
    # Load maps
    sim_path = f"../{mode}_sims/{mode}_sim_{i}_nside_{nside}"
    POS1 = heracles.read_maps(f"{sim_path}/POS_1.fits")
    SHE1 = heracles.read_maps(f"{sim_path}/SHE_1.fits")
    map_t = POS1['POS', 1]
    map_q = SHE1['SHE', 1][0]
    map_u = SHE1['SHE', 1][1]
    # Make fields
    f0 = nmt.NmtField(mask_apo, [map_t])
    f2 = nmt.NmtField(mask_apo, [map_q, map_u])
    # Compute cls
    cls_00 = compute_master(f0, f0, w00)
    cls_02 = compute_master(f0, f2, w02)
    cls_22 = compute_master(f2, f2, w22)
    # reorder
    _cls_nmt= {}
    #_cls_yp = {}
    _cls_nmt['POS', 'POS', 1, 1] = heracles.Result(cls_00[0], axis=(0,), ell=lgrid)
    _cls_nmt['POS', 'SHE', 1, 1] = heracles.Result(cls_02, axis=(1,), ell=lgrid)
    _cls_nmt['SHE', 'SHE', 1, 1] = heracles.Result(
        np.array([[cls_22[0], cls_22[1]],
                 [cls_22[2], cls_22[3]]]),
        axis=(2,), ell=lgrid)
    # Save results
    cls[i] = _cls_nmt
    heracles.write(f"{path}/cls_nmt/cqs_data_nmt_np_{i}_lmax_{lmax}.fits", _cls_nmt)
print("Done")

# Compute covariance
print("Computing covariance")
nmt_cqs_cov = dices.jackknife_covariance(cls, nd=0)

# Save
print("Saving covariance")
heracles.write(path+f"/covs/cov_nmt_cqs_l1max_{lmax}.fits", nmt_cqs_cov)