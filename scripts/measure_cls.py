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

nside = config["Nside"]
lmax = config["bins"]["Lmax"]
mapper = HealpixMapper(nside=nside, lmax=lmax)

# Fields
Nbins = int(config["Fields"]["Nbins"])
Keys_Pos_Ra = str(config["Fields"]["Pos"]["Ra"])
Keys_Pos_Dec = str(config["Fields"]["Pos"]["Dec"])
Pos_lonlat = (Keys_Pos_Ra, Keys_Pos_Dec)
Keys_She_Ra = str(config["Fields"]["She"]["Ra"])
Keys_She_Dec = str(config["Fields"]["She"]["Dec"])
She_lonlat = (Keys_She_Ra, Keys_She_Dec)
Keys_She_E1 = str(config["Fields"]["She"]["E1"])
Keys_She_E2 = str(config["Fields"]["She"]["E2"])
Keys_She_Weights = str(config["Fields"]["She"]["Weights"])

fields = {
    "POS": Positions(mapper, *Pos_lonlat, mask="VIS"),
    "SHE": Shears(
        mapper,
        *She_lonlat,
        Keys_She_E1,
        Keys_She_E2,
        Keys_She_Weights,
        mask="WHT",
    ),
}

l = np.arange(lmax + 1)
clss = {}
for i in range(1, 100+1):
    print(f"Loading sim {i}", end='\r')
    data_maps = {}
    POS1 = heracles.read_maps(f"../gaussian_sims/gaussian_sim_{i}/POS_1.fits")
    SHE1 = heracles.read_maps(f"../gaussian_sims/gaussian_sim_{i}/SHE_1.fits")
    POS2 = heracles.read_maps(f"../gaussian_sims/gaussian_sim_{i}/POS_2.fits")
    SHE2 = heracles.read_maps(f"../gaussian_sims/gaussian_sim_{i}/SHE_2.fits")
    data_maps[("POS", 1)] = POS1[("POS", 1)]
    data_maps[("POS", 2)] = POS2[("POS", 2)]
    data_maps[("SHE", 1)] = SHE1[("SHE", 1)]
    data_maps[("SHE", 2)] = SHE2[("SHE", 2)]

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
