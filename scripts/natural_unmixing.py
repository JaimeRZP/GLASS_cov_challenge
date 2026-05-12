import os
import yaml
import argparse
import os.path
import numpy as np
import heracles
import heracles.dices as dices
import time
from dataclasses import replace
from heracles.fields import Positions, Shears, Visibility, Weights
from heracles.healpy import HealpixMapper
from heracles.transforms import cl2corr, corr2cl
from heracles.unmixing import _naturalspice

# Config
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
        choices=["rr2", "dr1", "patch", "tr1"],
        help="mask type."
    )
    parser.add_argument(
        "--rtol",
        default=None,
        help="recompute cls."
    )
    parser.add_argument(
        "--recompute",
        default="False",
        help="recompute cls."
    )
    args = parser.parse_args()
    print(f"Using mask: {args.mask_type}")
    
    config_path = "./sims_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    n = 100 #config['nsims']
    nside = 2048 #config['nside']
    lmin = config['lmin']
    lmax_partial = config['lmax_partial']
    lmax_full = config['lmax_full']
    lmax_mask = config['lmax_mask']
    nlbins = config['nlbins'] 
    mode = args.mode
    mask_type = args.mask_type # Default to 'Patch' if not specified
    path = f"/pscratch/sd/j/jaimerz/{mode}_sims/{mask_type}/"
    mask_cls = heracles.read(f"{path}/cls/cls_mask_lmax_{lmax_mask}.fits")
    recompute = args.recompute

    #options
    options = {}
    if mask_type == "dr1":
        options[('VIS', 'VIS', 1, 1)] = 0.0005
        options[('VIS', 'WHT', 1, 1)] = 0.0005
        options[('WHT', 'WHT', 1, 1)] = 0.0005
    if mask_type == "tr1":
        options[('VIS', 'VIS', 1, 1)] = 0.001
        options[('VIS', 'WHT', 1, 1)] = 0.001
        options[('WHT', 'WHT', 1, 1)] = 0.001
    if mask_type == "patch":
        options[('VIS', 'VIS', 1, 1)] = 0.005 # 0.0335
        options[('VIS', 'WHT', 1, 1)] = 0.005 # 0.0020
        options[('WHT', 'WHT', 1, 1)] = 0.005 # 0.0335 
    
    
    mask_mapper = HealpixMapper(nside=nside, lmax=lmax_partial, deconvolve=False)
    mask_fields = {
        "POS": Positions(mask_mapper, mask="VIS"),
        "SHE": Shears(mask_mapper, mask="WHT"),
        "VIS": Visibility(mask_mapper),
        "WHT": Weights(mask_mapper),
    }

    # Correct the mask
    #for i in range(1, 6+1):
    #    for j in range(i, 6+1):
    #        mask_cls['WHT', 'WHT', i, j] = mask_cls['WHT', 'WHT', 1, 1]
    
    wmls = cl2corr(mask_cls)
    for m_key in list(wmls.keys()):
        rcond = 0.001 #options[m_key]
        wml = wmls[m_key].array
        wml = wml * heracles.unmixing.logistic(np.log10(abs(wml)), x0=np.log10(rcond * np.max(wml)))
        wmls[m_key] = replace(wmls[m_key], array=wml)
    
    cls = {}
    total_time = 0

    for i in range(1, n+1):
        start = time.perf_counter()

        file_path = f"{path}/cls_nu/cls0_data_nu_{i}_l1max_{lmax_partial}_l2max_{lmax_mask}.fits"
        if os.path.exists(file_path) and recompute=="False":
            nu_cls = heracles.read(file_path)
        else:
            # Load cls
            wcls_path = f"{path}/cls/wcls0_data_{i}_lmax_{lmax_partial}.fits"
            if os.path.exists(wcls_path):
                wcls = heracles.read(wcls_path)
            else:
                print(f"computing wcl {i}")
                data_cls = heracles.read(f"{path}/cls/cls0_data_{i}_lmax_{lmax_full}.fits")
                data_cls = heracles.binned(data_cls, np.arange(0, lmax_mask+2))
                wcls = cl2corr(data_cls)
                heracles.write(wcls_path, wcls)
            # PolSpice
            print(f"Unmixing sim {i}")
            nu_wcls = _naturalspice(wcls, wmls, mask_fields)
            nu_cls = corr2cl(nu_wcls)
            # Save cls
            heracles.write(file_path, nu_cls)

        cls[i] = nu_cls
        total_time += time.perf_counter() - start

    print("Done")
    avg_time = total_time / n
    print(f"Average time per iteration: {avg_time:.3f} seconds")
    
    # Binning cls
    ledges = np.logspace(np.log10(lmin), np.log10(lmax_full), nlbins + 1)
    lgrid = (ledges[1:] + ledges[:-1]) / 2
    print(f"Using {len(lgrid)} bins for the covariance matrix.")
    nu_cqs = heracles.binned(cls, ledges)
    
    # compute covariances
    print("Computing covariances")
    #nu_cls_cov = dices.jackknife_covariance(cls, nd=0)
    nu_cqs_cov = dices.jackknife_covariance(nu_cqs, nd=0)
    
    # Save
    print("Saving covariances")
    #heracles.write(path+f"/covs/cov_nu_cls_l1max_{lmax}_l2max_{lmax_mask}.fits", nu_cls_cov)
    heracles.write(path+f"/covs/cov_nu_cqs0_l1max_{lmax_full}_l2max_{lmax_mask}.fits", nu_cqs_cov)

if __name__ == "__main__":
    main()