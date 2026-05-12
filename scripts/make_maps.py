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


def spectra_indices(n):
    i, j = np.tril_indices(n)
    return np.transpose([i, i - j])


def main():
    # Config from command line
    parser = argparse.ArgumentParser(description="Generate GLASS shear maps")
    parser.add_argument(
        "--nsims",
        type=int,
        default=None,
        help="Number of sims to process.",
    )
    parser.add_argument(
        "--recompute",
        default="True",
        help="recompute cls."
    )
    args = parser.parse_args()
    recompute = args.recompute

    # Config
    config_path = "./sims_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    nsims = args.nsims
    nside = config["nside"]
    lmax = 3000
    mode = config["mode"]  # "lognormal" or "gaussian"
    nbins = 6
    path = f"/pscratch/sd/j/jaimerz/{mode}_sims"

    # Load nzs
    z = np.linspace(0.0, 3.0, 3000)
    nzs_wl_hdul = fits.open(
        "/pscratch/sd/j/jaimerz/lognormal_sims/TR1_v1_Nz_WL_C2020_sel_pv.fits"
    )
    nzs = gaussian_filter1d(nzs_wl_hdul[1].data["N_Z"].T, 10, axis=0)

    # Load theory cls
    theory_file = "theory_tommo_kappa_MDB.fits"
    cls_dict = heracles.read(f"{path}/{theory_file}")
    print(cls_dict)
    cls = [cls_dict[("KAP", "KAP", j+1, i+1)][0, 0][:lmax+1] for i, j in spectra_indices(nbins)]

    # Make GLASS cls
    shells = [
        glass.RadialWindow(z, nz, np.trapezoid(z * nz, z) / np.trapezoid(nz, z))
        for nz in nzs.T
    ]

    # Make fields
    fields = glass.lognormal_fields(shells, glass.lognormal_shift_hilbert2011)

    # Solve for spectra
    gls = glass.solve_gaussian_spectra(fields, cls)
    gls = glass.regularized_spectra(gls, niter=200)

    # Output
    out_filename = "SHE.fits"
    folname_template = f"/fullsky/maps/{mode}_sim_{{i}}_nside_{nside}"

    # Generate maps
    for i in range(1, nsims + 1):
        folname = folname_template.format(i=i)
        full_folder = f"{path}/{folname}"
        out_path = f"{full_folder}/{out_filename}"
        print(f"Making sim {i} in folder {folname}", end="\r")

        if os.path.exists(out_path) and recompute == "False":
            continue
        os.makedirs(full_folder, exist_ok=True)

        rng = np.random.default_rng(seed=i)
        maps = list(glass.generate(fields, gls, nside, rng=rng))

        SHEs = {}
        for j, _map in enumerate(maps):
            Q1, U1 = glass.shear_from_convergence(_map)
            SHE = np.array([Q1, U1])
            heracles.update_metadata(SHE, nside=nside, fsky=1.0, spin=2)
            SHEs["SHE", j + 1] = SHE

        # Write maps
        heracles.write_maps(out_path, SHEs, clobber=True)


if __name__ == "__main__":
    main()
