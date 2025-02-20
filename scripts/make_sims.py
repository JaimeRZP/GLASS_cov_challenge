import healpy as hp
import os
import matplotlib.pyplot as plt
import numpy as np
import glass
import glass.ext.camb
import camb
import camb.sources
import heracles
from cosmology import Cosmology

#Config
n = 100
path = "../gaussian_sims"
mode = "gaussian"
nside = 256
lmax = nside
nbins = 4
h = 0.7
Oc = 0.25
Ob = 0.05

print("Config:")
print(f"path: {path}")
print(f"mode: {mode}")
print(f"nside: {nside}")
print(f"lmax: {lmax}")
print(f"nbins: {nbins}")
print(f"h: {h}")
print(f"Oc: {Oc}")
print(f"Ob: {Ob}")

# make nz's
z = np.arange(0.0, 5.01, 0.01)
dndz = glass.smail_nz(z, 1.0, 1.5, 2.0)
zbins = glass.equal_dens_zbins(z, dndz, nbins)
nz = glass.tomo_nz_gausserr(z, dndz, 0.05, zbins)
nz_1 = nz[:2]
nz_2 = nz[2:]

# Check if folder exists
for i in range(1, n+1):
    folname = f"{mode}_sim_{i}"
    print(f"Making sim {i} in folder {folname}", end='\r')
    if not os.path.exists(f"{path}/{folname}"):
        os.makedirs(f"{path}/{folname}")
        # make a cosmology
        rng = np.random.default_rng(seed=i)
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

        # Generate maps
        maps = list(glass.generate(fields, gls, nside))

        # Write maps
        filename = "POS_1.fits"
        data = {("POS", 1): maps[0]}
        heracles.write_maps(f"{path}/{folname}/{filename}", data, clobber=True)

        filename = "POS_2.fits"
        data = {("POS", 2): maps[1]}
        heracles.write_maps(f"{path}/{folname}/{filename}", data, clobber=True)

        filename = "SHE_1.fits"
        data = {("SHE", 1): maps[2]}
        heracles.write_maps(f"{path}/{folname}/{filename}", data, clobber=True)

        filename = "SHE_2.fits"
        data = {("SHE", 2): maps[3]}
        heracles.write_maps(f"{path}/{folname}/{filename}", data, clobber=True)