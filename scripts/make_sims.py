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

path = f"../{mode}_sims"
l = np.arange(0, lmax + 1)
nbins = 4
h = 0.7
Oc = 0.25
Ob = 0.05

# make nz's
z = np.arange(0.0, 5.01, 0.01)
dndz = glass.smail_nz(z, 1.0, 1.5, 2.0)
zbins = glass.equal_dens_zbins(z, dndz, nbins)
nz = glass.tomo_nz_gausserr(z, dndz, 0.05, zbins)
nz_1 = nz[:2]
nz_2 = nz[2:]

print(f"saving nz's to {path}/nzs.npy")
np.savez(
    f"{path}/nzs.npz",
    z=z,
    nz_1=nz_1,
    nz_2=nz_2,
    )

# make a cosmology
pars = camb.set_params(
    H0=100 * h,
    omch2=Oc * h**2,
    ombh2=Ob * h**2,
    NonLinear=camb.model.NonLinear_both,
)
pars.set_accuracy(AccuracyBoost=2.0, lAccuracyBoost=2.0, lSampleBoost=2.0)
pars.Want_CMB = False
pars.Want_CMB_lensing = False
pars.min_l = 1
pars.set_for_lmax(2 * lmax)

pars.SourceWindows = [
    camb.sources.SplinedSourceWindow(z=z, W=nz_i, source_type="counts") for nz_i in nz_1
] + [
    camb.sources.SplinedSourceWindow(z=z, W=nz_i, source_type="lensing") for nz_i in nz_2
]

# Make theory cls
cls_dict = camb.get_results(pars).get_source_cls_dict(lmax=lmax, raw_cl=True)
cls = [cls_dict[f"W{i+1}xW{j+1}"] for i, j in glass.spectra_indices(nbins)]

# Turn into heracles results
results = {}
for key in cls_dict.keys():
    results[key] = heracles.Result(cls_dict[key], ell=l)
heracles.write(f"{path}/cls/cls_theory.fits", results)

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
        POS2 = maps[1]
        KAPPA1 = maps[2]
        KAPPA2 = maps[3]
        Q1, U1 = glass.shear_from_convergence(KAPPA1)
        Q2, U2 = glass.shear_from_convergence(KAPPA2)
        SHE1 = np.array([Q1, U1])
        SHE2 = np.array([Q2, U2])

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

        ngal = np.sum(POS2)
        nbar = (ngal * wmean) / fsky / npix
        heracles.update_metadata(POS2,
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

        ngal = np.sum(SHE2)
        nbar = (ngal * wmean) / fsky / npix
        heracles.update_metadata(SHE2,
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

        filename = "POS_2.fits"
        data = {("POS", 2): POS2}
        heracles.write_maps(f"{path}/{folname}/{filename}", data, clobber=True)

        filename = "SHE_1.fits"
        data = {("SHE", 1): SHE1}
        heracles.write_maps(f"{path}/{folname}/{filename}", data, clobber=True)

        filename = "SHE_2.fits"
        data = {("SHE", 2): SHE2}
        heracles.write_maps(f"{path}/{folname}/{filename}", data, clobber=True)
