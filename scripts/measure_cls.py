import numpy as np
import math
import matplotlib.pyplot as plt
import dices
import yaml
import healpy as hp
import heracles
import camb
import camb.correlations
import skysegmentor
from heracles.fields import Positions, Shears, Visibility, Weights
from heracles import transform
from heracles.healpy import HealpixMapper

with open("../dices_conf.yaml", mode="r") as file:
    config = yaml.safe_load(file)

path = "../gaussian_sims"
mode = "gaussian"
nside = config["Nside"]
lmax = config["bins"]["Lmax"]
mapper = HealpixMapper(nside=nside, lmax=lmax)

# Fields
Nbins = int(config["Fields"]["Nbins"])
fields = {
    "POS": Positions(mapper, mask="VIS"),
    "SHE": Shears(mapper, mask="WHT"),
}

vmap = hp.read_map("../masks/vmap.fits")
r = hp.Rotator(coord=['G','E']) 
vmap = r.rotate_map_pixel(vmap)
vmap = np.abs(hp.ud_grade(vmap, config["Nside"]))
vmap[vmap <= 1] = 0.0
vmap[vmap != 0] = vmap[vmap != 0] / vmap[vmap != 0]
vmap[vmap == 0] = 2.0
vmap[vmap == 1] = 0.0
vmap[vmap == 2] = 1.0

l = np.arange(lmax + 1)
clss = {}
for i in range(1, 100+1):
    print(f"Loading sim {i}", end='\r')
    data_maps = {}
    sim_path = f"{path}/{mode}_sim_{i}"
    POS1 = heracles.read_maps(f"{sim_path}/POS_1.fits")
    SHE1 = heracles.read_maps(f"{sim_path}/SHE_1.fits")
    POS2 = heracles.read_maps(f"{sim_path}/POS_2.fits")
    SHE2 = heracles.read_maps(f"{sim_path}/SHE_2.fits")
    data_maps[("POS", 1)] = POS1[("POS", 1)]*vmap
    data_maps[("POS", 2)] = POS2[("POS", 2)]*vmap
    data_maps[("SHE", 1)] = SHE1[("SHE", 1)]*vmap
    data_maps[("SHE", 2)] = SHE2[("SHE", 2)]*vmap

    alms = transform(fields, data_maps)
    cls = heracles.angular_power_spectra(alms)
    _cls = dices.compsep_Cls(cls)
    heracles.write(f"../gaussian_sims/gaussian_sim_{i}/measured_cls.fits", _cls)

    for key in list(_cls.keys()):
        cl = _cls[key].__array__()
        if i==1:
            clss[key] = [cl]
        else:
            clss[key] = clss[key]+[cl]

cls_m = {}
cls_cov = {}
for key in list(clss.keys()):
    print(f"Measuring {key}")
    cl = np.array(clss[key])
    cls_m[key] = heracles.Result(np.mean(cl, axis=0), ell=l)
    cls_cov[key] = heracles.Result(np.cov(cl.T), ell=(l,l))
heracles.write("../gaussian_sims/mean_cls.fits", cls_m)
heracles.write("../gaussian_sims/cov_cls.fits", cls_cov)
