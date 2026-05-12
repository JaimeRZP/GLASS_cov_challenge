import os
import yaml
import fitsio
import argparse
import numpy as np
import healpy as hp
import heracles
import heracles.dices as dices
from heracles.fields import Positions, Shears, Visibility, Weights
from heracles import transform
from heracles.healpy import HealpixMapper


def main():
    # Config from command line
    parser = argparse.ArgumentParser(description="Mask type")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["gaussian", "lognormal", "gatti"],
        help="sim type."
    )
    parser.add_argument(
        "--mask_type",
        type=str,
        required=True,
        choices=["dr1_south", "fullsky", "tr1"],
        help="mask type."
    )
    parser.add_argument(
        "--tomo",
        default="False",
        help="recompute cls."
    )
    parser.add_argument(
        "--recompute",
        default="True",
        help="recompute cls."
    )
    parser.add_argument(
        "--nsims",
        type=int,
        default=None,
        help="Number of sims to process.",
    )
    args = parser.parse_args()
    print(f"Using method: {args.mask_type}")
    
    # Config
    config_path = "./sims_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    nsims = args.nsims #config['nsims']
    nside = 1024 #config['nside']
    lmax_partial = config['lmax_partial']
    lmax_full = config['lmax_full']
    lmin = config['lmin']
    lmax_mask = config['lmax_mask']
    mode = args.mode  # "lognormal" or "gaussian"
    mask_type = args.mask_type  # Default to 'Patch' if not specified
    path = f"/pscratch/sd/j/jaimerz/{mode}_sims/{mask_type}/"
    recompute = args.recompute
    tomo = args.tomo
    print("Tomographic analysis: ", tomo)
    
    # vamp
    if mask_type != "fullsky":
        path_mask = f"/pscratch/sd/j/jaimerz/masks/{mask_type}_mask_nside_{nside}.fits"
        mask = hp.read_map(path_mask)
    else:
        mask = np.ones(hp.nside2npix(nside))
    print("computed mask")
    # Add spin information to mask
    heracles.core.update_metadata(mask, spin=0)
    
    # Fields
    mapper = HealpixMapper(nside=nside, lmax=lmax_partial, deconvolve=False)
    fields = {
        "SHE": heracles.Shears(
            mapper,
            "RA",
            "DEC",
            "E1",
            "E2",  # sign flip for LensMC convention
            mask="WHT",
        ),
    }
    
    cls = {}
    for i in range(1, nsims+1):
        print(f"Loading sim {i}", end='\r')
        if tomo == "True":
            file_path = path+f"cls/cls_maps_{i}_lmax_{lmax_partial}.fits"
        else:
            file_path = path+f"cls/cls0_maps_{i}_lmax_{lmax_partial}.fits"
        if os.path.exists(file_path) and recompute=="False":
            _cls = heracles.read(file_path)
        else:
            maps_path = f"/pscratch/sd/j/jaimerz/{mode}_sims/fullsky/maps/{mode}_sim_{i}_nside_{nside}/"
            if tomo == "True":
                maps = heracles.read_maps(maps_path+"SHE.fits")
            else:
                maps = heracles.read_maps(maps_path+"SHE0.fits")

            for key in list(maps.keys()):
                maps[key] = maps[key]*mask

            alms = heracles.transform(fields, maps)
            _cls = heracles.angular_power_spectra(alms)
            heracles.write(file_path, _cls)
        cls[i] = _cls
    print("Done")
if __name__ == "__main__":
    main()