import yaml
import os
import argparse
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import glass
import glass.ext.camb
import camb
import camb.sources
import heracles
from cosmology import Cosmology
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d

# Note: We do not model source clustering
# This makes comparing to predictions easier.
# B-modes shouldn't care about this either.
# However, it migth be a problem in the future.


# Per-mask-type galaxy number densities (per sq. arcmin) and ellipticity sigmas.
# Keep these at module level so it's easy to add new mask types.
NGALS = {
    "tr1": {
        1: 3.630431,
        2: 3.369029,
        3: 3.417767,
        4: 3.333166,
        5: 3.442296,
        6: 3.253078,
    },
    "dr1_south": {
        1: 4.049680,
        2: 3.800670,
        3: 3.881422,
        4: 3.787431,
        5: 3.929530,
        6: 3.767061,
    },
}

SIGMA_E = {
    "tr1": {
        1: 0.2672282382294292,
        2: 0.2662526016273488,
        3: 0.2590877242553745,
        4: 0.2591418343365571,
        5: 0.2550252629962767,
        6: 0.2641187068627740,
    },
    "dr1_south": {
        1: 0.2676301,
        2: 0.2665834,
        3: 0.2592022,
        4: 0.2592046,
        5: 0.2551500,
        6: 0.2645000,
    },
}


def main():
    # Config from command line
    parser = argparse.ArgumentParser(description="Generate tomographic shear catalogs from sim maps.")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["gaussian", "lognormal",],
        help="sim type.",
    )
    parser.add_argument(
        "--mask_type", 
        type=str,
        required=True,
        choices=["dr1_south", "tr1"],
        help="mask type.",
    )
    parser.add_argument(
        "--nsims",
        type=int,
        default=None,
        help="Number of sims to process.",
    )
    args = parser.parse_args()
    print(f"Using mode: {args.mode}, mask_type: {args.mask_type}")

    # Config
    n = args.nsims
    nside = 1024
    lmax =  3000
    mode = args.mode
    mask_type = args.mask_type
    path = f"/pscratch/sd/j/jaimerz/{mode}_sims"

    # Validate: we have per-bin inputs for this mask type
    if mask_type not in NGALS or mask_type not in SIGMA_E:
        raise ValueError(
            f"No ngals/sigma_e table defined for mask_type={mask_type!r}. "
            f"Available: {sorted(set(NGALS) & set(SIGMA_E))}."
        )
    ngals = NGALS[mask_type]
    sigma_e = SIGMA_E[mask_type]

    # Mask
    if mask_type == "fullsky":
        mask = np.ones(hp.nside2npix(nside))
    else:
        path_mask = f"/pscratch/sd/j/jaimerz/masks/{mask_type}_mask_nside_{nside}.fits"
        mask = hp.read_map(path_mask)

    print("computed mask")
    # Add spin information to mask
    heracles.core.update_metadata(mask, spin=0)

    # Load nzs
    z = np.linspace(0.0, 3.0, 3000)
    nzs_wl_hdul = fits.open("/pscratch/sd/j/jaimerz/lognormal_sims/TR1_v1_Nz_WL_C2020_sel_pv.fits")
    dndz = gaussian_filter1d(nzs_wl_hdul[1].data["N_Z"].T, 10, axis=0)
    n_bins = 6

    # Check if folder exists
    for i in range(1, n + 1):
        # Generate maps
        rng = np.random.default_rng(seed=i)
        folname = f"{mode}_sim_{i}_nside_{nside}"
        print(f"Making cat {i} in folder {folname}", end="\r")
        # Generate folder
        if not os.path.exists(f"{path}/{mask_type}/cats/{folname}"):
            os.makedirs(f"{path}/{mask_type}/cats/{folname}")
        maps_path = f"{path}/fullsky/maps/{folname}/SHE.fits"
        cat_path = f"{path}/{mask_type}/cats/{folname}/SHE.fits"
        if not os.path.exists(cat_path):
            with glass.write_catalog(cat_path) as out:
                for j in range(1, n_bins + 1):
                    SHE = heracles.read_maps(maps_path)[("SHE", j)]
                    Q1, U1 = SHE
                    for gal_lon, gal_lat, gal_count in glass.positions_from_delta(
                        ngals[j],
                        np.zeros_like(Q1), # Gower St processing: delta_i
                        0.0, # b=1.2 for SLC, but we ignore SLC for now
                        mask,
                        rng=rng,
                    ):  
                        # this shold become glass.redshifts(shells[i])
                        gal_z = glass.redshifts_from_nz(
                            gal_count,
                            z,
                            dndz.T[j - 1],
                            rng=rng,
                            warn=False,
                        )
                        gal_eps = glass.ellipticity_intnorm(
                            gal_count,
                            sigma_e[j],
                            rng=rng,
                        )
                        gal_she = glass.galaxy_shear(
                            gal_lon,
                            gal_lat,
                            gal_eps,
                            np.zeros_like(Q1),
                            Q1,
                            U1,
                            reduced_shear=False,
                        )
                        out.write(
                            RA=gal_lon,
                            DEC=gal_lat,
                            Z=gal_z,
                            E1=gal_she.real,
                            E2=gal_she.imag,
                            TOMBINID=np.full(gal_count, j, dtype=np.int32),
                        )
    print("Done")


if __name__ == "__main__":
    main()
