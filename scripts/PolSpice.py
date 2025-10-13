import yaml
import numpy as np
import healpy as hp
import heracles
import ispice


# Config
# export HEALPIX=/home/jaimerzp/Documents/UCL/Healpix_3.83_2024Nov13/Healpix_3.83/
config_path = "./sims_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
n = config['nsims']
nside = config['nside']
lmax = config['lmax']
mode = config['mode']  # "lognormal" or "gaussian"
mask_type = config.get('mask_type', 'Patch')  # Default to 'Patch' if not specified
path = f"../{mask_type}"
l = np.arange(lmax + 1)

for i in range(1, n+1):
    print(f"Unmixing sim {i}", end='\r')
    # Load maps
    sim_path = f"../{mode}_sims/{mode}_sim_{i}"
    POS1 = heracles.read_maps(f"{sim_path}/POS_1.fits")
    SHE1 = heracles.read_maps(f"{sim_path}/SHE_1.fits")
    maps_spice = [POS1['POS', 1], SHE1['SHE', 1][0], SHE1['SHE', 1][1]]
    hp.write_map(f"{sim_path}/maps_spice.fits", maps_spice, overwrite=True)
    ispice.ispice(f"{sim_path}/maps_spice.fits",
                f"{path}/cls_pols/cls_data_pols.fits",
                maskfile1=f"{path}/mask.fits",
                nlmax=lmax,
                polarization=True,
                decouple=True,
                symmetric_cl=True,
                covfileout=f"{path}/cls_pols/cov_data_pols_{i}_l1max_{lmax}.fits",
                kernelsfileout=f"{path}/cls_pols/kernels_pols_{i}_l1max_{lmax}.fits",
                apodizesigma=10.0,
                pixelfile=False,
                #cl_outmask_file=f"{path}/cls_pols/cls_mask_pols_decoupled_{i}.fits",
                #cl_outmap_file=f"{path}/cls_pols/cls_data_pols_raw_{i}.fits",
                #corfile=f"{path}/cls_pols/corr_data_pols_{i}.fits",
                binpath="/home/jaimerzp/Documents/UCL/PolSpice_v03-08-03/bin/spice")

    # reformat cls
    pols_cl = hp.read_cl(path+f"/cls_pols/cls_data_pols.fits")
    _pols_cl = {}
    _pols_cl['POS', 'POS', 1, 1] = heracles.Result(pols_cl[0], axis=(0,), ell=l)
    _pols_cl['POS', 'SHE', 1, 1] = heracles.Result(
        np.array([pols_cl[3], pols_cl[4]]), axis=(1,), ell=l)
    _pols_cl['SHE', 'SHE', 1, 1] = heracles.Result(
        np.array([[pols_cl[1], pols_cl[5]],
                  [pols_cl[5], pols_cl[2]]]),
        axis=(2,), ell=l)
    # Save cls
    heracles.write(f"{path}/cls_pols/cls_data_pols_{i}_lmax_{lmax}.fits", _pols_cl)