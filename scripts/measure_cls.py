import yaml
import numpy as np
import healpy as hp
import heracles
import heracles.dices as dices
from heracles.fields import Positions, Shears, Visibility, Weights
from heracles import transform
from heracles.healpy import HealpixMapper


# Config
config_path = "./sims_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
n = config['nsims']
nside = config['nside']
lmax = config['lmax']
mode = config['mode']  # "lognormal" or "gaussian"
path = f"../{mode}_sims/"
# Fields
mapper = HealpixMapper(nside=nside, lmax=lmax)
fields = {
    "POS": Positions(mapper, mask="VIS"),
    "SHE": Shears(mapper, mask="WHT"),
    "VIS": Visibility(mapper),
    "WHT": Weights(mapper),
}
mask_mapper = HealpixMapper(nside=2 * nside, lmax=2 * lmax)
mask_fields = {
    "POS": Positions(mapper, mask="VIS"),
    "SHE": Shears(mapper, mask="WHT"),
    "VIS": Visibility(mapper),
    "WHT": Weights(mapper),
}

# vamp
ref_map = hp.read_map(path+f"{mode}_sim_1/POS_1.fits")
mask = np.ones_like(ref_map)
pixel_theta, pixel_phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
mask_type = 'Patch'

if mask_type == 'Patch':
    mask[np.pi/3 > pixel_theta] = 0.0
    mask[pixel_theta > 2*np.pi/3] = 0.0
    mask[pixel_phi > np.pi/2] = 0.0
    mask[np.pi/8> pixel_phi] = 0.0

if mask_type == 'One third cover':
    mask[np.pi/3 > pixel_theta] = 0.0

if mask_type == 'Half cover':
        mask[np.pi/2 > pixel_theta] = 0.0

if mask_type == 'Two thirds cover':
        mask[2*np.pi/3 > pixel_theta] = 0.0
else:
    print("Unknown mask type, using full sky mask")
hp.write_map(path+f"mask.fits", mask, overwrite=True)

# mask cls
vmaps = {}
vmaps[("VIS", 1)] = mask
vmaps[("WHT", 1)] = mask
mask_alms = heracles.transform(mask_fields, vmaps)
mask_cls = heracles.angular_power_spectra(mask_alms)
heracles.write(path+f"cls/cls_mask.fits", mask_cls)

for i in range(1, n+1):
    print(f"Loading sim {i}", end='\r')
    data_maps = {}
    sim_path = f"{path}/{mode}_sim_{i}"
    POS1 = heracles.read_maps(f"{sim_path}/POS_1.fits")
    SHE1 = heracles.read_maps(f"{sim_path}/SHE_1.fits")

    # Full sky
    data_maps[("POS", 1)] = POS1[('POS', 1)]
    data_maps[("SHE", 1)] = SHE1[('SHE', 1)]
    # Masked
    data_maps[("POS", 1)] *= mask
    data_maps[("SHE", 1)] *= mask

    alms = transform(fields, data_maps)
    cls = heracles.angular_power_spectra(alms)
    heracles.write(path+f"cls/cls_data_{i}.fits", cls)
print("Done")
