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
        choices=["rr2", "dr1", "patch", "fullsky", "tr1"],
        help="mask type."
    )
    parser.add_argument(
        "--recompute",
        default="True",
        help="recompute cls."
    )
    args = parser.parse_args()
    print(f"Using method: {args.mask_type}")
    
    # Config
    config_path = "./sims_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    n = 500 #config['nsims']
    nside = config['nside']
    lmax_partial = config['lmax_partial']
    lmax_full = config['lmax_full']
    lmin = config['lmin']
    lmax_mask = config['lmax_mask']
    mode = args.mode  # "lognormal" or "gaussian"
    mask_type = args.mask_type  # Default to 'Patch' if not specified
    path = f"/pscratch/sd/j/jaimerz/{mode}_sims/{mask_type}/"
    recompute = args.recompute
    nbins=6
    
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
        "POS": Positions(mapper, mask="VIS"),
        "SHE": Shears(mapper, mask="WHT"),
        "VIS": Visibility(mapper),
        "WHT": Weights(mapper),
    }
    mask_mapper = HealpixMapper(nside=nside, lmax=lmax_mask, deconvolve=False)
    mask_fields = {
        "POS": Positions(mask_mapper, mask="VIS"),
        "SHE": Shears(mask_mapper, mask="WHT"),
        "VIS": Visibility(mask_mapper),
        "WHT": Weights(mask_mapper),
    }
    
    # mask cls
    vmaps = {}
    vmaps[("VIS", 1)] = mask
    vmaps[("WHT", 1)] = mask
    mask_alms = heracles.transform(mask_fields, vmaps)
    mask_cls = heracles.angular_power_spectra(mask_alms)
    mms = heracles.mixing_matrices(
        mask_fields,
        mask_cls,
        l1max=lmax_partial,
        l2max=lmax_full,
        l3max=lmax_mask,
    )
    heracles.write(path+f"/mixmat_l1max_{lmax_partial}_l2max_{lmax_mask}.fits", mms, clobber=True)
    heracles.write(path+f"cls/cls_mask_lmax_{lmax_mask}.fits", mask_cls, clobber=True)
    
    cls = {}
    for i in range(1, n+1):
        print(f"Loading sim {i}", end='\r')
        file_path = path+f"cls/cls_data_{i}_lmax_{lmax_partial}.fits"
        if os.path.exists(file_path) and recompute=="False":
            _cls = heracles.read(file_path)
        else:
            data_maps = {}
            sim_path = f"/pscratch/sd/j/jaimerz/{mode}_sims/fullsky/maps/{mode}_sim_{i}_nside_{nside}"
            data_maps = {}
            for j in range(1, nbins+1):
                #POS = heracles.read_maps(f"{sim_path}/POS_{j}.fits")[('POS', j)]
                SHE = heracles.read_maps(f"{sim_path}/SHE.fits")[('SHE', j)]
                if np.iscomplexobj(SHE):
                    SHE = np.array([SHE.real, SHE.imag])
                #if np.mean(POS)/np.std(POS) > 0.1:
                #    POS = (POS - np.mean(POS))/np.mean(POS)
                #POS *= mask
                SHE *= mask
                # spins
                #heracles.core.update_metadata(POS, spin=0)
                heracles.core.update_metadata(SHE, spin=2)
                # Full sky
                #data_maps[("POS", 1)] = POS1
                data_maps[("SHE", j)] = SHE
            # Compute Cls
            alms = transform(fields, data_maps)
            _cls = heracles.angular_power_spectra(alms, debias=False)
            # Save Cls
            heracles.write(file_path, _cls)
            # Transform to corrs
            _cls_pad = heracles.binned(_cls, np.arange(0, lmax_mask+2)) 
            _corrs = heracles.transforms.cl2corr(_cls_pad)
            # Save Corrs
            heracles.write(path+f"cls/wcls_data_{i}_lmax_{lmax_partial}.fits", _corrs)
        cls[i] = _cls
    print("Done")
    
    # Binning cls
    nlbins = config['nlbins']  # Default to 20 if
    ledges = np.logspace(np.log10(lmin), np.log10(lmax_full), nlbins + 1)
    lgrid = (ledges[1:] + ledges[:-1]) / 2
    cqs = heracles.binned(cls, ledges)
    
    # Covariance
    print("Computing covariances")
    cqs_cov = dices.jackknife_covariance(cqs, nd=0)
    
    # Save
    print("Saving covariances")
    heracles.write(path+f"/covs/cov_cqs_lmin_{lmin}_l1max_{lmax_full}.fits", cqs_cov)

if __name__ == "__main__":
    main()