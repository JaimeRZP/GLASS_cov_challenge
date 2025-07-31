import yaml
import fitsio
import numpy as np
import healpy as hp
import heracles
import heracles.dices as dices
from heracles.fields import Positions, Shears, Visibility, Weights
from heracles import transform
from heracles.healpy import HealpixMapper

def _read_map(path, nside, *, nest=False):
    """
    Read a HEALPix map in "partial" format from *path* and return it at
    resolution *nside*.

    The returned NSIDE cannot be larger than the NSIDE of the stored
    map.

    If *nest* is true, returns the map in NESTED ordering.
    """
    data, header = fitsio.read(path, header=True)
    nside_in = header["NSIDE"]
    fact = (nside_in // nside) ** 2
    if fact == 0:
        raise ValueError(
            f"requested NSIDE={nside} greater than map NSIDE={nside_in}"
        )
    out = np.zeros(12 * nside**2)
    ipix, wht = data["PIXEL"], data["WEIGHT"]
    order = header["ORDERING"]
    if order == "RING":
        ipix = hp.ring2nest(nside, ipix)
    elif order != "NESTED":
        raise ValueError(f"unknown pixel ordering {order} in map")
    ipix = ipix // fact
    if not nest:
        ipix = hp.nest2ring(nside, ipix)
    np.add.at(out, ipix, wht / fact)
    return out

# Config
config_path = "./sims_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
n = config['nsims']
nside = config['nside']
lmax = config['lmax']
mode = config['mode']  # "lognormal" or "gaussian"
mask_type = config['mask_type']  # Default to 'Patch' if not specified
path = f"../{mask_type}/"
# Fields
mapper = HealpixMapper(nside=nside, lmax=lmax, deconvolve=False)
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
mask = np.ones(hp.nside2npix(nside))
pixel_theta, pixel_phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))

if mask_type == 'Patch':
    mask[np.pi/3 > pixel_theta] = 0.0
    mask[pixel_theta > 2*np.pi/3] = 0.0
    mask[pixel_phi > np.pi/2] = 0.0
    mask[np.pi/8> pixel_phi] = 0.0
if mask_type == 'One third cover':
    mask[np.pi/3 > pixel_theta] = 0.0
if mask_type == 'half_sky':
        mask[np.pi/2 > pixel_theta] = 0.0
if mask_type == 'Two thirds cover':
        mask[2*np.pi/3 > pixel_theta] = 0.0
if mask_type == 'planck':
    path_mask = f"../{mode}_sims/{mask_type}_mask.fits"
    mask = hp.read_map(path_mask)
    mask = hp.ud_grade(mask, nside_out=nside)
if mask_type == 'rr2':
    path_mask = f"../{mode}_sims/{mask_type}_mask.fits"
    mask = _read_map(path_mask, nside)
    mask = hp.ud_grade(mask, nside_out=nside)
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
    sim_path = f"../{mode}_sims/{mode}_sim_{i}"
    POS1 = heracles.read_maps(f"{sim_path}/POS_1.fits")
    SHE1 = heracles.read_maps(f"{sim_path}/SHE_1.fits")
    # Full sky
    data_maps[("POS", 1)] = POS1[('POS', 1)]
    data_maps[("SHE", 1)] = SHE1[('SHE', 1)]
    # Masked
    data_maps[("POS", 1)] *= mask
    data_maps[("SHE", 1)] *= mask
    # Compute Cls
    alms = transform(fields, data_maps)
    cls = heracles.angular_power_spectra(alms)
    heracles.write(path+f"cls/cls_data_{i}.fits", cls)
print("Done")
