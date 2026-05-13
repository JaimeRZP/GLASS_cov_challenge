import os
import numpy as np
import os
import glass
import glass.ext.pkdgrav
import healpy as hp
import heracles


# config
nside = 2048
npix = 12 * nside**2

bias = 1.2
sigma_e = 0.26
ngal = 3.5

mask_type = "tr1"

# Load nzs
path = f"/pscratch/sd/j/jaimerz/gatti_sims"
nzs = np.load(f"{path}/nzs.npz")
z = nzs['z']
nz_1 = nzs['nz_1']
nz_2 = nzs['nz_2']
nzs = np.array([nz_1, nz_2])
nzs = nzs / np.trapz(nzs, axis=1)[:, None]  # normalize
nzs[0] = nzs[0] * ngal
nzs[1] = nzs[1] * ngal

# Mask
if mask_type == "fullsky":
    mask = np.ones(hp.nside2npix(nside))
else:
    path_mask = f"/pscratch/sd/j/jaimerz/masks/{mask_type}_mask_nside_{nside}.fits"
    mask = hp.read_map(path_mask)

# Stack these nzs to generate fields for both [P1P1] , P1P2 and P2P2
# as well as W1W1, W1W2 and [W2W2]

for nsim in np.arange(1, 200+1):
    #rng 
    rng = np.random.default_rng(seed=42+nsim)

    cats_path = f"/pscratch/sd/j/jaimerz/gatti_cats/{mask_type}/sim_{nsim}_nside_{nside}.fits"
    path_simulation = f'/global/cfs/cdirs/m5099/GowerSt2/Fiducial/{nsim}_big/'
    print(f"processing sim {nsim}")
    if not os.path.exists(cats_path):
        print(f"computing cat for sim {nsim}")
        os.makedirs(cats_path)

        # load simulation
        sim = glass.ext.pkdgrav.load(f"{path_simulation}/control.par")
        cosmo = sim.cosmology
        shells = glass.tophat_windows(sim.redshifts)

        # this will load a GowerSt simulation iteratively
        # up to redshift 2 and rescaled to nside
        matter = glass.ext.pkdgrav.read_gowerst(sim, path_simulation, zmax=3.0, nside=nside, format="parquet")
        print("Computed the matter field")

        # this will compute the convergence field iteratively
        convergence = glass.MultiPlaneConvergence(cosmo)
        print("Computed the convergence field")
        # Compute the ngal
        # Compute ngal for the stack instead
        ngals = glass.partition(z, nzs, shells)
        print("Computed the Ngal field")

        # generate galaxy positions uniformly over the sphere
        with glass.write_catalog(cats_path) as out:
            for i, delta_i in enumerate(matter):
                # compute the lensing maps for this shell
                convergence.add_window(delta_i, shells[i])
                kappa_i = convergence.kappa
                gamma1_i, gamma2_i = glass.shear_from_convergence(kappa_i)
                for gal_lon, gal_lat, gal_count in glass.positions_from_delta(
                        ngals[i],
                        delta_i,
                        bias,
                        mask,
                        rng=rng,
                    ):
                        # this shold become glass.redshifts(shells[i])
                        gal_z = glass.redshifts(gal_count, shells[i], rng=rng)
                        gal_eps = glass.ellipticity_intnorm(
                            gal_count,
                            sigma_e,
                            rng=rng,
                        )
                        gal_she = glass.galaxy_shear(
                            gal_lon,
                            gal_lat,
                            gal_eps,
                            kappa_i,
                            gamma1_i,
                            gamma2_i,
                            reduced_shear=False,
                        )
                        out.write(
                            RA=gal_lon,
                            DEC=gal_lat,
                            Z=gal_z,
                            E1=gal_she.real,
                            E2=gal_she.imag,
                            TOMBINID=np.repeat([1,2], gal_count),
                        )
    print("Done")
