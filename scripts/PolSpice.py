import yaml
import healpy as hp
import heracles
import ispice


# Config
config_path = "./sims_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
n = config['nsims']
nside = config['nside']
lmax = config['lmax']
mode = config['mode']  # "lognormal" or "gaussian"
mask_type = config.get('mask_type', 'Patch')  # Default to 'Patch' if not specified
path = f"../{mode}_sims"
mask_cls = heracles.read(f"{path}/cls/cls_mask.fits")

for i in range(1, n+1):
    print(f"Unmixing sim {i}", end='\r')
    # Load maps
    data_maps = {}
    sim_path = f"{path}/{mode}_sim_{i}"
    POS1 = heracles.read_maps(f"{sim_path}/POS_1.fits")
    SHE1 = heracles.read_maps(f"{sim_path}/SHE_1.fits")
    maps_spice = [POS1['POS', 1], SHE1['SHE', 1][0], SHE1['SHE', 1][1]]
    hp.write_map(f"{sim_path}/maps_spice.fits", maps_spice, overwrite=True)
    ispice.ispice(f"{sim_path}/maps_spice.fits",
                  f"{path}/cls_pols/cls_data_pols_{i}.fits",
                  maskfile1=f"{path}/dummy_mask.fits",
                  nlmax=lmax,
                  polarization=True,
                  decouple=True,
                  symmetric_cl=True,
                  cl_outmask_file=f"{path}/cls_pols/cls_mask_pols_{i}.fits",
                  cl_outmap_file=f"{path}/cls_pols/cls_data_pols_raw_{i}.fits",
                  corfile=f"{path}/cls_pols/corr_data_pols_{i}.fits",
                  binpath="/home/jaimerzp/Documents/UCL/PolSpice_v03-08-03/bin/spice")
