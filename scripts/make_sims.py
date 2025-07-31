import yaml
import os
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import glass
import glass.ext.camb
import camb
import camb.sources
import heracles
from cosmology import Cosmology

# Config
config_path = "./sims_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
n = config['nsims']
nside = config['nside']
lmax = config['lmax']
mode = config['mode']  # "lognormal" or "gaussian"
nbins = 2
path = f"../{mode}_sims"

# Load nzs
nzs = np.load(f"{path}/nzs.npz")
z = nzs['z']
nz_1 = nzs['nz_1']
nz_2 = nzs['nz_2']

# Load theory cls
cls = heracles.read(f"{path}/cls_theory.fits")
cls = [cls[f"W{i+1}xW{j+1}"].array for i, j in glass.spectra_indices(nbins)]

# Make GLASS cls
shells_1 = [
    glass.RadialWindow(z, nz_i, np.trapezoid(z * nz_i, z) / np.trapezoid(nz_i, z)) for nz_i in nz_1
]
shells_2 = [
    glass.RadialWindow(z, nz_i, np.trapezoid(z * nz_i, z) / np.trapezoid(nz_i, z)) for nz_i in nz_2
]

# Make fields
if mode == "gaussian":
    # density
    fields_1 = glass.gaussian_fields(shells_1)
    # convergence
    fields_2 = glass.gaussian_fields(shells_2)
elif mode == "lognormal":
    # density
    fields_1 = glass.lognormal_fields(shells_1)
    # convergence
    fields_2 = glass.lognormal_fields(shells_2, glass.lognormal_shift_hilbert2011)
else:
    raise ValueError(f"Unknown mode: {mode}")

# Solve for spectra
fields = fields_1 + fields_2
gls = glass.solve_gaussian_spectra(fields, cls)

if mode == "lognormal":
    print("Regularizing")
    gls = glass.regularized_spectra(gls)

# Check if folder exists
for i in range(1, n+1):
    folname = f"{mode}_sim_{i}"
    print(f"Making sim {i} in folder {folname}", end='\r')
    if not os.path.exists(f"{path}/{folname}"):
        os.makedirs(f"{path}/{folname}")
        # Generate maps
        rng = np.random.default_rng(seed=i)
        maps = list(glass.generate(fields, gls, nside))
        POS1 = maps[0]
        KAPPA1 = maps[1]
        Q1, U1 = glass.shear_from_convergence(KAPPA1)
        SHE1 = np.array([Q1, U1])

        fsky = 1.0
        wmean = 0.0
        w2mean = 0.0
        var = 0.0
        variance = 0.0
        bias = 0.0
        npix = hp.nside2npix(nside)

        ngal = np.sum(POS1)
        nbar = (ngal * wmean) / fsky / npix
        heracles.update_metadata(POS1,
                                nside=nside,
                                lmax=lmax,
                                ngal=ngal,
                                nbar=nbar,
                                wmean=wmean,
                                bias=bias,
                                var=var,
                                variance=variance,
                                neff=ngal/(4*np.pi*fsky),
                                fsky=fsky,
                                spin=0)

        ngal = np.sum(SHE1)
        nbar = (ngal * wmean) / fsky / npix
        heracles.update_metadata(SHE1,
                                nside=nside,
                                lmax=lmax,
                                ngal=ngal,
                                nbar=nbar,
                                wmean=wmean,
                                bias=bias,
                                var=var,
                                variance=variance,
                                neff=ngal/(2*np.pi*fsky),
                                fsky=fsky,
                                spin=2)


        # Write maps
        filename = "POS_1.fits"
        data = {("POS", 1): POS1}
        heracles.write_maps(f"{path}/{folname}/{filename}", data, clobber=True)

        filename = "SHE_1.fits"
        data = {("SHE", 1): SHE1}
        heracles.write_maps(f"{path}/{folname}/{filename}", data, clobber=True)
