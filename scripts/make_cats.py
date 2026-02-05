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


def spectra_indices(n):
    i, j = np.tril_indices(n)
    return np.transpose([i, i - j])
    
# Config
config_path = "./sims_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
n = config['nsims']
nside = config['nside']
lmax = config['lmax']
mode = config['mode']  # "lognormal" or "gaussian"
nbins = 2
path = f"/pscratch/sd/j/jaimerz/{mode}_sims"
mask_type = "tr1"

# Load nzs
nzs = np.load(f"{path}/nzs.npz")
z = nzs['z']
nz_1 = nzs['nz_1']
nz_2 = nzs['nz_2']

# Load theory cls
cls = heracles.read(f"{path}/cls_theory_lmax_{lmax}.fits")
cl_pp = cls["W1xW1"].array
cl_ep = cls["W2xW1"].array
cl_pe = cls["W1xW2"].array
cl_ee = cls["W2xW2"].array

cls = [cls[f"W{i+1}xW{j+1}"].array for i, j in spectra_indices(nbins)]

# Make GLASS cls
shells_1 = [
    glass.RadialWindow(z, nz_1, np.trapezoid(z * nz_1, z) / np.trapezoid(nz_1, z))
]
shells_2 = [
    glass.RadialWindow(z, nz_2, np.trapezoid(z * nz_2, z) / np.trapezoid(nz_2, z))
]

# vamp
if mask_type != "dummy":
    path_mask = f"../masks/{mask_type}_mask_nside_{nside}.fits"
    mask = hp.read_map(path_mask)
else:
    mask = np.ones(hp.nside2npix(nside))
print("computed mask")
# Add spin information to mask
heracles.core.update_metadata(mask, spin=0)

# Config
# galaxy density (using 1/100 of the expected galaxy number density for Stage-IV)
n_arcmin2 = 0.3

# true redshift distribution following a Smail distribution
z = np.arange(0.0, 3.0, 0.01)
dndz = glass.smail_nz(z, z_mode=0.9, alpha=2.0, beta=1.5)
dndz *= n_arcmin2

# distribute dN/dz over the radial window functions
ngal = glass.partition(z, dndz, shells)

# compute tomographic redshift bin edges with equal density
nbins = 10
zbins = glass.equal_dens_zbins(z, dndz, nbins=nbins)

# photometric redshift error
sigma_z0 = 0.03

# constant bias parameter for all shells
bias = 1.2

# ellipticity standard deviation as expected for a Stage-IV survey
sigma_e = 0.27

# Check if folder exists
for i in range(1, n+1):
    # Generate maps
    rng = np.random.default_rng(seed=i)
    folname = f"{mode}_sim_{i}_nside_{nside}"
    print(f"Making cat {i} in folder {folname}", end='\r')
    
    file_name = f"{path}/{folname}/{filename}/POS_1_cat.fits"
    if not os.path.exists(f"file_name"):
        # Load Maps
        POS1 = heracles.read_maps(f"{path}/{folname}/POS_1.fits")[("POS", 1)]

        # Generate catalog
        POS_catalogue = np.empty(
            0,
            dtype=[
                ("RA", float),
                ("DEC", float),
                ("Z_TRUE", float),
                ("PHZ", float),
                ("ZBIN", int),
            ],
        )
        
        # generate galaxy positions from the matter density contrast
        for gal_lon, gal_lat, gal_count in glass.positions_from_delta(
            ngal[i],
            POS1,
            bias,
            mask,
            rng=rng,
        ):
            # generate random redshifts over the given shell
            gal_z = glass.redshifts(gal_count, shells_1, rng=rng)
    
            # generator photometric redshifts using a Gaussian model
            gal_phz = glass.gaussian_phz(gal_z1, sigma_z0, rng=rng)
    
            # attach tomographic bin IDs to galaxies, based on photometric redshifts
            gal_zbin = np.digitize(gal_phz, np.unique(zbins)) - 1
    
            # make a mini-catalogue for the new rows
            rows = np.empty(gal_count, dtype=catalogue.dtype)
            rows["RA"] = gal_lon
            rows["DEC"] = gal_lat
            rows["Z_TRUE"] = gal_z
            rows["PHZ"] = gal_phz
            rows["ZBIN"] = gal_zbin
    
            # add the new rows to the catalogue
            POS_catalogue = np.append(catalogue, rows)
        
        print(f"Total number of galaxies sampled: {len(POS_catalogue):,}")
        glass.write_catalog(filename, POS_catalogue)

    file_name = f"{path}/{folname}/{filename}/SHE_1_cat.fits"
    if not os.path.exists(f"file_name"):
        # Load Maps
        SHE1 = heracles.read_maps(f"{path}/{folname}/SHE_1.fits")[("SHE", 1)]
        
        # Generate catalog
        SHE_catalogue = np.empty(
            0,
            dtype=[
                ("RA", float),
                ("DEC", float),
                ("Z_TRUE", float),
                ("PHZ", float),
                ("ZBIN", int),
                ("G1", float),
                ("G2", float),
            ],
        )
        
        # generate galaxy positions from the matter density contrast
        for gal_lon, gal_lat, gal_count in glass.positions_from_delta(
            ngal[i],
            SHE1,
            bias,
            mask,
            rng=rng,
        ):
            # generate random redshifts over the given shell
            gal_z = glass.redshifts(gal_count, shells_2, rng=rng)
    
            # generator photometric redshifts using a Gaussian model
            gal_phz = glass.gaussian_phz(gal_z2, sigma_z0, rng=rng)
    
            # attach tomographic bin IDs to galaxies, based on photometric redshifts
            gal_zbin = np.digitize(gal_phz, np.unique(zbins)) - 1

            # generate galaxy ellipticities from the chosen distribution
            gal_eps = glass.ellipticity_intnorm(gal_count, sigma_e, rng=rng, xp=np)

            # apply the shear fields to the ellipticities
            gal_she = glass.galaxy_shear(
                gal_lon,
                gal_lat,
                gal_eps,
                kappa_i,
                gamm1_i,
                gamm2_i,
            )
    
            # make a mini-catalogue for the new rows
            rows = np.empty(gal_count, dtype=catalogue.dtype)
            rows["RA"] = gal_lon
            rows["DEC"] = gal_lat
            rows["Z_TRUE"] = gal_z
            rows["PHZ"] = gal_phz
            rows["ZBIN"] = gal_zbin
            rows["E1"] = gal_she.real
            rows["E2"] = gal_she.imag
    
            # add the new rows to the catalogue
            SHE_catalogue = np.append(catalogue, rows)

        print(f"Total number of galaxies sampled: {len(SHE_catalogue):,}")
        glass.write_catalog(file_name, SHE_catalogue)

    file_name = f"{path}/{folname}/{filename}/SHE_1_wb_cat.fits"
    if not os.path.exists(f"file_name"):
        # Load Maps
        SHE1_wb = heracles.read_maps(f"{path}/{folname}/SHE_1_wb.fits")[("SHE", 1)]

        # Generate catalog
        SHE_catalogue_wb = np.empty(
            0,
            dtype=[
                ("RA", float),
                ("DEC", float),
                ("Z_TRUE", float),
                ("PHZ", float),
                ("ZBIN", int),
                ("G1", float),
                ("G2", float),
            ],
        )
        
        # generate galaxy positions from the matter density contrast
        for gal_lon, gal_lat, gal_count in glass.positions_from_delta(
            ngal[i],
            SHE1_wb,
            bias,
            mask,
            rng=rng,
        ):
            # generate random redshifts over the given shell
            gal_z = glass.redshifts(gal_count, shells_2, rng=rng)
    
            # generator photometric redshifts using a Gaussian model
            gal_phz = glass.gaussian_phz(gal_z2, sigma_z0, rng=rng)
    
            # attach tomographic bin IDs to galaxies, based on photometric redshifts
            gal_zbin = np.digitize(gal_phz, np.unique(zbins)) - 1

            # generate galaxy ellipticities from the chosen distribution
            gal_eps = glass.ellipticity_intnorm(gal_count, sigma_e, rng=rng, xp=np)

            # apply the shear fields to the ellipticities
            gal_she = glass.galaxy_shear(
                gal_lon,
                gal_lat,
                gal_eps,
                kappa_i,
                gamm1_i,
                gamm2_i,
            )
    
            # make a mini-catalogue for the new rows
            rows = np.empty(gal_count, dtype=catalogue.dtype)
            rows["RA"] = gal_lon
            rows["DEC"] = gal_lat
            rows["Z_TRUE"] = gal_z
            rows["PHZ"] = gal_phz
            rows["ZBIN"] = gal_zbin
            rows["E1"] = gal_she.real
            rows["E2"] = gal_she.imag
    
            # add the new rows to the catalogue
            SHE_catalogue_wb = np.append(catalogue, rows)
        
        print(f"Total number of galaxies sampled: {len(SHE_catalogue_wb):,}")
        glass.write_catalog(file_name, SHE_catalogue_wb)
