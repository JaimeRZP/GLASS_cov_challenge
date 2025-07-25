import yaml
import numpy as np
import healpy as hp
import heracles
import pymaster as nmt


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
mask_type = config.get('mask_type', 'Patch')  # Default to 'Patch' if not specified
path = f"../{mask_type}"

ls = np.arange(lmax + 1)
ledges = np.logspace(np.log10(10), np.log10(lmax), nlbins + 1)
lgrid = ledges[:-1] + np.diff(ledges)/2
b = nmt.NmtBin.from_edges(ledges[:-1].astype(int), ledges[1:].astype(int))
b.lmax = lmax

mask = hp.read_map(f"{path}/mask.fits")
msk_apo = nmt.mask_apodization(mask, 10.0, apotype='C1')

# workspaces
sim_path = f"../{mode}_sims/{mode}_sim_1"
POS1 = heracles.read_maps(f"{sim_path}/POS_1.fits")
SHE1 = heracles.read_maps(f"{sim_path}/SHE_1.fits")
map_t = POS1['POS', 1]
map_q = SHE1['SHE', 1][0]
map_u = SHE1['SHE', 1][1]
f0 = nmt.NmtField(msk_apo, [map_t], lmax=lmax)
f2_np = nmt.NmtField(msk_apo, [map_q, map_u], lmax=lmax)
#f2_yp = nmt.NmtField(msk_apo, [map_q, map_u], purify_e=True, purify_b=True, lmax=lmax)
w00 = nmt.NmtWorkspace.from_fields(f0, f0, b)
w02_np = nmt.NmtWorkspace.from_fields(f0, f2_np, b)
#w02_yp = nmt.NmtWorkspace.from_fields(f0, f2_yp, b)
w22_np = nmt.NmtWorkspace.from_fields(f2_np, f2_np, b)
#w22_yp = nmt.NmtWorkspace.from_fields(f2_yp, f2_yp, b)

for i in range(1, n+1):
    print(f"Unmixing sim {i}", end='\r')
    # Load maps
    sim_path = f"../{mode}_sims/{mode}_sim_{i}"
    POS1 = heracles.read_maps(f"{sim_path}/POS_1.fits")
    SHE1 = heracles.read_maps(f"{sim_path}/SHE_1.fits")
    map_t = POS1['POS', 1]
    map_q = SHE1['SHE', 1][0]
    map_u = SHE1['SHE', 1][1]
    # Make fields
    f0 = nmt.NmtField(msk_apo, [map_t])
    f2_np = nmt.NmtField(msk_apo, [map_q, map_u])
    #f2_yp = nmt.NmtField(msk_apo, [map_q, map_u], purify_e=True, purify_b=True)
    # Compute cls
    cls_00 = compute_master(f0, f0, w00)
    cls_02_np = compute_master(f0, f2_np, w02_np)
    cls_22_np = compute_master(f2_np, f2_np, w22_np)
    #cls_02_yp = compute_master(f0, f2_yp, w02_yp)
    #cls_22_yp = compute_master(f2_yp, f2_yp, w22_yp)
    # reorder
    _cls_np = {}
    #_cls_yp = {}
    _cls_np['POS', 'POS', 1, 1] = heracles.Result(cls_00[0], axis=(0,), ell=lgrid)
    _cls_np['POS', 'SHE', 1, 1] = heracles.Result(cls_02_np, axis=(1,), ell=lgrid)
    _cls_np['SHE', 'SHE', 1, 1] = heracles.Result(
        np.array([[cls_22_np[0], cls_22_np[1]],
                 [cls_22_np[2], cls_22_np[3]]]),
        axis=(2,), ell=lgrid)
    #_cls_yp['POS', 'POS', 1, 1] = heracles.Result(cls_00, axis=(0,), ell=ls)
    #_cls_yp['POS', 'SHE', 1, 1] = heracles.Result(cls_02_yp, axis=(1,), ell=ls)
    #_cls_yp['SHE', 'SHE', 1, 1] = heracles.Result(
    #    np.array([[cls_22_yp[0], cls_22_yp[1]]],
    #             [cls_22_yp[2], cls_22_yp[3]]),
    #    axis=(2,), ell=ls)
    # Save results
    heracles.write(f"{path}/cls_nmt/cqs_data_nmt_np_{i}.fits", _cls_np)
    #heracles.write(f"{path}/cls_nmt/cls_data_nmt_yp_{i}.fits", _cls_yp)
