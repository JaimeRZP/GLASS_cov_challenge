import os
import yaml
import numpy as np
import healpy as hp
import argparse
import heracles
import skysegmentor
import heracles.dices as dices
from heracles.io import read, write
from itertools import combinations

from heracles.healpy import HealpixMapper
from heracles.fields import Shears, Visibility, Weights
from heracles.dices.jackknife import _get_region_maps, _sum_alms_except
from heracles.transforms import cl2corr, corr2cl


def main():
    # Config from command line
    parser = argparse.ArgumentParser(description="Mask type")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["gaussian", "lognormal", "gatti"],
        help="simulation mode.",
    )
    parser.add_argument(
        "--mask_type",
        type=str,
        required=True,
        choices=["fullsky", "tr1"],
        help="mask type.",
    )
    parser.add_argument(
        "--recompute",
        default="False",
        help="recompute cls.",
    )
    args = parser.parse_args()
    print(f"Using method: {args.mask_type}")

    # Config
    config_path = "./sims_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    nside = config["nside"]
    lmax_full = config["lmax_full"]
    lmax_partial = config["lmax_partial"]
    lmax_mask = config["lmax_mask"]
    Njk = config["Njk"]
    mode = args.mode
    mask_type = args.mask_type
    recompute = args.recompute
    print(f"Recompute: {recompute}")

    # Simulation
    i = 4
    print(f"Loading sim: {i}")
    sim_path = f"/pscratch/sd/j/jaimerz/{mode}_sims/fullsky/maps/{mode}_sim_{i}_nside_{nside}/"

    # Mask
    if mask_type == "fullsky":
        vmap = np.ones(hp.nside2npix(nside))
    else:
        vmap = hp.read_map(f"/pscratch/sd/j/jaimerz/masks/{mask_type}_mask_nside_{nside}.fits")
    heracles.core.update_metadata(vmap, spin=0)

    # Data & Vis maps — load all SHE bins present in the sim directory
    data_maps = heracles.read_maps(f"{sim_path}/SHE.fits")
    for key, SHE in data_maps.items():
        if np.iscomplexobj(SHE):
            data_maps[key] = np.array([SHE.real, SHE.imag])
        data_maps[key] *= vmap
        heracles.core.update_metadata(data_maps[key], spin=2)

    nbins = len(data_maps)
    vis_maps = {}
    for _, b in data_maps.keys():
        vis_maps[("VIS", b)] = vmap
        vis_maps[("WHT", b)] = vmap

    # JK maps (one entry per vis_maps key)
    jkmap_path = (
        f"/pscratch/sd/j/jaimerz/{mode}_dices/masks/"
        f"{mask_type}_jkmap_nside_{nside}_njk_{Njk}.fits"
    )
    if os.path.isfile(jkmap_path):
        print("Found segmentation map")
        jk_map = hp.read_map(jkmap_path)
    else:
        print("Segmenting map")
        jk_map = skysegmentor.segmentmapN(vmap, Njk)
        hp.write_map(jkmap_path, jk_map)
    jk_maps = {key: jk_map for key in vis_maps}

    # Fields (SHE only)
    mapper = HealpixMapper(nside=nside, lmax=lmax_partial, deconvolve=False)
    fields = {
        "SHE": Shears(mapper, mask="WHT"),
        "VIS": Visibility(mapper),
        "WHT": Weights(mapper),
    }
    mask_mapper = HealpixMapper(nside=nside, lmax=lmax_mask, deconvolve=False)
    mask_fields = {
        "SHE": Shears(mask_mapper, mask="WHT"),
        "VIS": Visibility(mask_mapper),
        "WHT": Weights(mask_mapper),
    }

    output_path = f"/pscratch/sd/j/jaimerz/{mode}_dices/{mask_type}/"
    alms_path = output_path + "alms/"
    os.makedirs(alms_path, exist_ok=True)

    # Cls0 — full sky, independent of jackknife regions
    alms = heracles.transform(fields, data_maps)
    cls0 = heracles.angular_power_spectra(alms)
    write(f"{output_path}/cls/cls_0.fits", cls0)

    # Load or compute one ALM per jackknife region for data and (mask) vis maps.
    # All jackknife combinations are then derived by _sum_alms_except, avoiding
    # a full SHT per combination (N instead of N + N*(N-1)/2 transforms).
    print("Loading/computing region ALMs...")
    data_alms_regions = {}
    vis_alms_regions = {}
    for k in range(1, Njk + 1):
        data_alms_fname = alms_path + f"data_alms_{k}.fits"
        vis_alms_fname = alms_path + f"vis_alms_{k}.fits"
        if os.path.exists(data_alms_fname) and os.path.exists(vis_alms_fname):
            print(f" - Loading ALMs for region {k}", end="\r", flush=True)
            data_alms_regions[k] = heracles.read_alms(data_alms_fname)
            vis_alms_regions[k] = heracles.read_alms(vis_alms_fname)
        else:
            print(f" - Computing ALMs for region {k}", end="\r", flush=True)
            data_alms_regions[k] = heracles.transform(
                fields, _get_region_maps(data_maps, jk_maps, k)
            )
            vis_alms_regions[k] = heracles.transform(
                mask_fields, _get_region_maps(vis_maps, jk_maps, k)
            )
            heracles.write_alms(data_alms_fname, data_alms_regions[k])
            heracles.write_alms(vis_alms_fname, vis_alms_regions[k])
    print()

    # Full-sky mask Cls from summed region ALMs (no extra transform needed)
    mls0 = heracles.angular_power_spectra(_sum_alms_except(vis_alms_regions, ()))
    write(f"{output_path}/cls/mls_0.fits", mls0)

    # Cls1 — delete-one ensemble
    cls1 = {}
    for regions in combinations(range(1, Njk + 1), 1):
        (jk1,) = regions
        data_fname = output_path + f"cls1/cls_{jk1}.fits"
        print(f" - cls1 regions={regions}", end="\r", flush=True)
        if os.path.exists(data_fname) and recompute == "False":
            _cls = read(data_fname)
        else:
            alms_jk = _sum_alms_except(data_alms_regions, regions)
            _cls = heracles.angular_power_spectra(alms_jk)
            # bias correction
            _cls = dices.correct_bias(_cls, jk_maps, fields, *regions)
            # Mask correction
            vis_alms_jk = _sum_alms_except(vis_alms_regions, regions)
            _cls_mm = heracles.angular_power_spectra(vis_alms_jk)
            alphas = dices.get_mask_correlation_ratio(_cls_mm, mls0, unmixed=False)
            _wcls = cl2corr(_cls)
            _wcls = heracles.unmixing._naturalspice(_wcls, alphas, fields)
            _cls = corr2cl(_wcls)
            write(data_fname, _cls)
        cls1[regions] = _cls
    print("Cls1 done")

    # Binning
    nlbins = config.get("nlbins", 20)
    ledges = np.logspace(np.log10(100), np.log10(lmax_full), nlbins + 1)
    cqs0 = heracles.binned(cls0, ledges)
    cqs1 = heracles.binned(cls1, ledges)

    # Jackknife covariance (delete-one)
    cov_jk_path = output_path + f"/covs/cov_jk1_cqs_l1max_{lmax_full}.fits"
    print("Computing jackknife covariance")
    cov_jk = dices.jackknife_covariance(cqs1)
    write(cov_jk_path, cov_jk)

    # Cls2 — delete-two ensemble
    cls2 = {}
    for regions in combinations(range(1, Njk + 1), 2):
        (jk1, jk2) = regions
        data_fname = output_path + f"cls2/cls_{jk1}_{jk2}.fits"
        print(f" - cls2 regions={regions}", end="\r", flush=True)
        if os.path.exists(data_fname) and recompute == "False":
            _cls = read(data_fname)
        else:
            alms_jk = _sum_alms_except(data_alms_regions, regions)
            _cls = heracles.angular_power_spectra(alms_jk)
            # bias correction
            _cls = dices.correct_bias(_cls, jk_maps, fields, *regions)
            # Mask correction (disabled)
            vis_alms_jk = _sum_alms_except(vis_alms_regions, regions)
            _cls_mm = heracles.angular_power_spectra(vis_alms_jk)
            alphas = dices.get_mask_correlation_ratio(_cls_mm, mls0)
            _wcls = cl2corr(_cls)
            _wcls = heracles.unmixing._naturalspice(_wcls, alphas, fields)
            _cls = corr2cl(_wcls)
            write(data_fname, _cls)
        cls2[regions] = _cls
    print("Cls2 done")

    # Binning
    cqs2 = heracles.binned(cls2, ledges)

    # Debiasing
    debiased_path = output_path + f"/covs/cov_jk_debiased_cqs_l1max_{lmax}.fits"
    if os.path.isfile(debiased_path) and recompute == "False":
        debiased_cov_jk = read(debiased_path)
    else:
        print("Debiasing covariance")
        debiased_cov_jk = dices.debias_covariance(cov_jk, cqs0, cqs1, cqs2)
        write(debiased_path, debiased_cov_jk)

    # Shrinkage
    shrunk_path = output_path + f"/covs/cov_jk_shrunk_cqs_l1max_{lmax}.fits"
    if os.path.isfile(shrunk_path) and recompute == "False":
        shrunk_cov_jk = read(shrunk_path)
    else:
        print("Shrinking covariance")
        gauss_cov = dices.gaussian_covariance(cqs0)
        shrinkage_factor = dices.shrinkage_factor(cqs1, gauss_cov)
        print(f"Shrinkage factor: {shrinkage_factor}")
        shrunk_cov_jk = dices.shrink(cov_jk, gauss_cov, shrinkage_factor)
        write(shrunk_path, shrunk_cov_jk)

    # DICES
    dices_path = output_path + f"/covs/cov_jk_dices_cqs_l1max_{lmax}.fits"
    if os.path.isfile(dices_path) and recompute == "False":
        print("Done")
    else:
        print("Computing DICES covariance")
        dices_cov = dices.impose_correlation(debiased_cov_jk, shrunk_cov_jk)
        write(dices_path, dices_cov)


if __name__ == "__main__":
    main()
