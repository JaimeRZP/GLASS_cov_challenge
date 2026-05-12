import os
import numpy as np
import os
import glass
import glass.ext.pkdgrav
import healpy as hp
import heracles


# config
baryonification = False
nside_baryonification = 2048
nside_maps = 2048
nside = 2048
npix = 12 * nside**2
rng = np.random.default_rng(seed=42)

# Load nzs
path = f"/pscratch/sd/j/jaimerz/gatti_sims"
nzs = np.load(f"{path}/nzs.npz")
z = nzs['z']
nz_1 = nzs['nz_1']
nz_2 = nzs['nz_2']
# Stack these nzs to generate fields for both [P1P1] , P1P2 and P2P2
# as well as W1W1, W1W2 and [W2W2]

for nsim in np.arange(1, 200+1):
    maps_path = f"/pscratch/sd/j/jaimerz/gatti_sims/gatti_sim_{nsim}_nside_{nside_maps}"
    path_simulation = f'/global/cfs/cdirs/m5099/GowerSt2/Fiducial/{nsim}_big/'
    print(f"processing sim {nsim}")
    if not os.path.exists(maps_path):
        print(f"computing map for sim {nsim}")
        os.makedirs(maps_path)
        
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
        ngal_1 = glass.partition(z, nz_1, shells)
        ngal_2 = glass.partition(z, nz_2, shells)
        print("Computed the Ngal field")
        
        pos = np.zeros(12 * nside**2)
        she = np.zeros(12 * nside**2, complex)
        # generate galaxy positions uniformly over the sphere
        for i, delta_i in enumerate(matter):
            print(i)
            # compute the lensing maps for this shell
            convergence.add_window(delta_i, shells[i])
            kappa = convergence.kappa
            gamma, = glass.from_convergence(kappa, shear=True)

            # add to mean maps
            pos += ngal_1[i] * delta_i
            she += ngal_2[i] * gamma
        # normalise mean maps
        pos /= np.sum(ngal_1)
        she /= np.sum(ngal_2)

        # separate she components
        she = np.array([she.real, she.imag])

        # Save file
        heracles.update_metadata(
            pos,
            nside=nside_maps,
            spin=0)
        data = {("POS", 1): pos}
        heracles.write_maps(maps_path+"/POS_1.fits", data, clobber=True)
        heracles.update_metadata(
            she,
            nside=nside_maps,
            spin=2)
        data = {("SHE", 1): she}
        heracles.write_maps(maps_path+"/SHE_1.fits", data, clobber=True)


    
