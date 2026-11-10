import argparse
import glob
from collections import defaultdict
import os
import json

import torch
import yaml

import sys 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

from mast3r.model import AsymmetricMASt3R
from mast3r.retrieval.graph import make_pairs_fps
from mast3r.retrieval.processor import Retriever


def fps_pairs_to_dict(fps_pairs):
    pairs_dict = defaultdict(list)
    for i, j in fps_pairs:
        pairs_dict[str(i)].append(int(j))
        pairs_dict[str(j)].append(int(i))

    return dict(pairs_dict)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_root", type=str, required=True)
    parser.add_argument(
        "--retrieval_model",
        type=str,
        default="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
    )
    parser.add_argument(
        "--dataset_config_yaml",
        type=str,
        default="dataset/dataset_config/finetuning_real_datasets_1023.yml",
    )
    parser.add_argument("--downsample_factor", type=str, default="16")
    parser.add_argument(
        "--winsize_ratio",
        type=float,
        default=0.05,
        help="Determines winsize (N_a) when generating these pairs. winsize = total_images_in_scene * winsize_ratio",
    )
    parser.add_argument(
        "--refid_ratio",
        type=float,
        default=0.05,
        help="Determines refid (k) when generating these pairs. refid = total_images_in_scene * refid_ratio",
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    if os.path.isfile(args.dataset_config_yaml):
        with open(args.dataset_config_yaml, "r") as f:
            dataset_config = yaml.safe_load(f)
        training_folders = []
        for scene in dataset_config.keys():
            clean_iso_et = dataset_config[scene]["clean_iso_et"]
            training_folders += sorted(
                glob.glob(
                    f"{args.train_dataset_root}/{scene}/{clean_iso_et}/downsampled_{args.downsample_factor.zfill(3)}/images_npy_undistorted"
                )
            )

    else:
        training_folders = sorted(
            glob.glob(
                f"{args.train_dataset_root}/clean_npy/*/downsampled_{args.downsample_factor.zfill(3)}/images_undistorted"
            )
        )

    model = AsymmetricMASt3R.from_pretrained(
        "naver/" + "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    ).to(device)

    retriever = Retriever(args.retrieval_model, backbone=model, device=device)

    for folder in training_folders:
        print(f"Processing folder: {folder}")
        filelist = sorted(glob.glob(f"{folder}/*.npy"))

        refid = round(len(filelist) * args.refid_ratio)
        winsize = round(len(filelist) * args.winsize_ratio)
        pairs_output_path = f"{os.path.dirname(folder)}/pairs_rf{str(int(args.refid_ratio * 1000)).zfill(4)}_ws{str(int(args.winsize_ratio * 1000)).zfill(4)}.json"

        if os.path.exists(pairs_output_path):
            print(f"Pairs file already exists: {pairs_output_path}, skipping...")
            continue

        sim_matrix = retriever(filelist)
        fps_pairs, anchor_idxs = make_pairs_fps(
            sim_matrix, Na=winsize, tokK=refid, dist_thresh=None
        )
        pairs_dict = fps_pairs_to_dict(fps_pairs)

        print(f"Saving pairs to: {pairs_output_path}")
        with open(pairs_output_path, "w") as f:
            json.dump(pairs_dict, f, indent=4)

    del retriever
    del model
    torch.cuda.empty_cache()
