import csv
import argparse
import importlib.util
import json
import math
import os
import random
import sys
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataset import get_dataloaders
from model.autoencoder import Autoencoder
from model.opticalSimulation import MaterialLayer, SensorLayer, ONN
# from model.restormer250724 import Restormer
from model.RestormerOfficial import Restormer  #! important
# from model.Restormer261021 import Restormer  #! important


from testing_analysis import (
    compute_psnr, get_latent_cmap, get_latent_map, get_model_core, percentile, project_latent_for_viz, save_frequency_comparison, save_psnr_distribution_table, ssim_pt, visualize_results, visualize_results_by_mse,
    onn_output_debug as _onn_output_debug,
)


def onn_output_debug(model):
    return _onn_output_debug(model, TESTING_CONFIG, test_dataset)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test the model with a selectable config file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a config .py file, e.g. config_lists/Baseline_4F_Restormer8_v1_0809_config.py",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Config name inside config_lists, e.g. Baseline_4F_Restormer8_v1_0809",
    )
    return parser.parse_known_args()[0]


def load_config_module_from_path(config_path):
    path = Path(config_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    module_name = f"_runtime_config_{path.stem}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config module from: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_runtime_config():
    args = parse_args()

    config_path = args.config
    if args.config_name:
        config_path = os.path.join(
            "config_lists",
            f"{args.config_name}_config.py",
        )

    if config_path is None:
        import config as default_config

        return default_config, "config.py"

    module = load_config_module_from_path(config_path)
    return module, config_path


CONFIG_MODULE, CONFIG_SOURCE = load_runtime_config()

AUTOENCODER_CONFIG = CONFIG_MODULE.AUTOENCODER_CONFIG
DATASET_CONFIG = CONFIG_MODULE.DATASET_CONFIG
ENCODER_CONFIG = CONFIG_MODULE.ENCODER_CONFIG
RESTORMER_CONFIG = CONFIG_MODULE.RESTORMER_CONFIG
TESTING_CONFIG = CONFIG_MODULE.TESTING_CONFIG

print(f"[Config] Loaded from: {CONFIG_SOURCE}")

def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def is_main():
    return get_rank() == 0


distributed = TESTING_CONFIG.get(
    "distributed",
    ("LOCAL_RANK" in os.environ or "RANK" in os.environ),
)
local_rank = int(os.environ.get("LOCAL_RANK", 0))

if distributed:
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed testing requires CUDA.")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(backend=TESTING_CONFIG.get("backend", "nccl"))
else:
    local_rank = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device} | Distributed: {distributed}, Local Rank: {local_rank}")


def build_model():
    encoder = ONN(ENCODER_CONFIG).to(device)
    decoder = Restormer(RESTORMER_CONFIG).to(device)
    model = Autoencoder(
        encoder=encoder,
        decoder=decoder,
        config=AUTOENCODER_CONFIG,
    ).to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    return model


def prepare_test_loader():
    per_gpu_batch = TESTING_CONFIG.get("batch_size", 64)
    num_workers = TESTING_CONFIG.get("num_workers", 4)

    _, _, base_test_loader = get_dataloaders(
        DATASET_CONFIG,
        per_gpu_batch,
        num_workers=num_workers,
        distributed=distributed,
    )

    test_dataset = base_test_loader.dataset
    if not distributed:
        return base_test_loader, test_dataset

    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=per_gpu_batch,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return test_loader, test_dataset


test_loader, test_dataset = prepare_test_loader()


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )

    if hasattr(model, "module"):
        missing, unexpected = model.module.load_state_dict(state_dict, strict=True)  #!
    else:
        missing, unexpected = model.load_state_dict(state_dict, strict=True)  #!

    model.eval()
    if is_main():
        print(f"Model loaded from {model_path}")
        #! [HF-RECOVERY] Display injection strengths loaded from this checkpoint
        model_core = model.module if hasattr(model, "module") else model
        decoder = getattr(model_core, "decoder", None)
        gammas = getattr(decoder, "refinement_recovery_gammas", None)
        if gammas is not None:
            gamma_summary = " ".join(
                f"gamma_{idx + 1}={gamma.detach().item():.6f}"
                for idx, gamma in enumerate(gammas)
            )
            print(f"[HF-RECOVERY] {gamma_summary}")
        if missing:
            print(f"[load_model] Missing keys ignored: {missing}")
        if unexpected:
            print(f"[load_model] Unexpected keys ignored: {unexpected}")
    return model




#! [INPUT SHIFT] Begin: Input shift test helper
def shift_tensor_zero_fill(x, dy, dx):
    shifted = torch.zeros_like(x)
    height, width = x.shape[-2:]

    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    dst_y0 = max(0, dy)
    dst_x0 = max(0, dx)

    if src_y1 > src_y0 and src_x1 > src_x0:
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        shifted[..., dst_y0:dst_y1, dst_x0:dst_x1] = x[..., src_y0:src_y1, src_x0:src_x1]

    return shifted


def get_configured_input_shift():
    shift = TESTING_CONFIG.get("input_shift", (0, 0))
    return int(shift[0]), int(shift[1])
#! [INPUT SHIFT] End: Input shift test helper




def select_reconstruction(outputs):
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs










def test_model(model):
    model.eval()
    if is_main():
        dy, dx = get_configured_input_shift()
        print(f"[INPUT SHIFT] Model input shift: dy={dy}, dx={dx} pixels")
    local_imgs, local_recons = [], []
    local_mses = []

    local_sse = 0.0
    local_pixel_count = 0
    local_ssim_sum = 0.0
    local_ssim_count = 0

    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc="Testing", disable=not is_main()):
            imgs = imgs.to(device)
            #! [HF-RECOVERY] Shift the ONN input and use the equally shifted GT
            dy, dx = get_configured_input_shift()
            target_imgs = shift_tensor_zero_fill(imgs, dy, dx)
            outputs = model(target_imgs)
            recons = select_reconstruction(outputs)

            if torch.is_complex(recons):
                recons = torch.abs(recons)

            # Optional binary-output evaluation.  The model output is kept
            # unchanged by default; when enabled, all metrics and saved
            # reconstructions use a black/white image.
            if TESTING_CONFIG.get("binary_output_for_metrics", False):
                threshold = float(TESTING_CONFIG.get("binary_threshold", 0.5))
                recons = (recons >= threshold).to(recons.dtype)

            batch_mse_sum = F.mse_loss(recons, target_imgs, reduction="none").flatten(1).mean(dim=1)
            local_mses.extend(batch_mse_sum.detach().cpu().tolist())

            local_sse += F.mse_loss(recons, target_imgs, reduction="sum").item()
            local_pixel_count += target_imgs.numel()

            for i in range(imgs.size(0)):
                local_ssim_sum += ssim_pt(target_imgs[i : i + 1], recons[i : i + 1])
                local_ssim_count += 1

            local_imgs.append(target_imgs.cpu())
            local_recons.append(recons.cpu())

    if distributed:
        stats = torch.tensor(
            [local_sse, float(local_pixel_count), local_ssim_sum, float(local_ssim_count)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        global_mse = (stats[0] / stats[1]).item()
        global_ssim = (stats[2] / stats[3]).item()
    else:
        global_mse = local_sse / local_pixel_count
        global_ssim = local_ssim_sum / max(1, local_ssim_count)

    global_psnr = compute_psnr(global_mse)

    if distributed:
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(
            gathered,
            {
                "imgs": local_imgs,
                "recons": local_recons,
                "mses": local_mses,
            },
        )
    else:
        gathered = [
            {
                "imgs": local_imgs,
                "recons": local_recons,
                "mses": local_mses,
            }
        ]

    if is_main():
        all_imgs = []
        all_recons = []
        all_mses = []
        for payload in gathered:
            all_imgs.extend(payload["imgs"])
            all_recons.extend(payload["recons"])
            all_mses.extend(payload["mses"])

        if all_imgs:
            all_imgs = torch.cat(all_imgs, dim=0)
            all_recons = torch.cat(all_recons, dim=0)
        else:
            all_imgs = None
            all_recons = None
            all_mses = []

        print(f"Test MSE: {global_mse:.6f}, PSNR: {global_psnr:.4f}, SSIM: {global_ssim:.4f}")
        return all_imgs, all_recons, all_mses, global_mse, global_psnr, global_ssim

    return None, None, None, global_mse, global_psnr, global_ssim


#! [HF-RECOVERY] Begin: Frequency-domain panel export
#! [HF-RECOVERY] End: Frequency-domain panel export












if __name__ == "__main__":
    model_path = os.path.join(TESTING_CONFIG["weight_save_dir"], TESTING_CONFIG["weight_save_name"])
    model_name = os.path.basename(model_path)

    model = build_model()
    model = load_model(model, model_path)

    if TESTING_CONFIG.get("onn_debug", False) and is_main():
        onn_output_debug(model)

    all_imgs, all_recons, all_mses, mse, psnr, ssim = test_model(model)

    if is_main():
        if all_imgs is not None and all_recons is not None:
            visualize_results(
                model,
                all_imgs,
                all_recons,
                model_name,
                num_image=TESTING_CONFIG.get("save_first_n_reconstructions", 10),
                config=TESTING_CONFIG,
                all_mses=all_mses,
            )

            #! [HF-RECOVERY] Export the same fixed comparison samples across models
            if TESTING_CONFIG.get("save_frequency_domain_panels", True):
                fixed_seed = int(TESTING_CONFIG.get("fixed_comparison_seed", 42))
                fixed_count = min(
                    int(TESTING_CONFIG.get("fixed_comparison_count", 10)),
                    len(all_imgs),
                )
                fixed_rng = random.Random(fixed_seed)
                fixed_indices = sorted(fixed_rng.sample(range(len(all_imgs)), fixed_count))
                fixed_folder = f"fixed_comparison_seed_{fixed_seed}"
                fixed_latent_maps = get_latent_map(
                    model,
                    all_imgs[fixed_indices].to(device),
                    TESTING_CONFIG,
                    target_size=all_recons[fixed_indices].shape[-2:]
                    if TESTING_CONFIG.get("latent_resize_to_output", True) else None,
                )
                save_frequency_comparison(
                    all_imgs,
                    all_recons,
                    fixed_indices,
                    model_name,
                    TESTING_CONFIG,
                    "fixed",
                    latent_images={
                        idx: fixed_latent_maps[position]
                        for position, idx in enumerate(fixed_indices)
                    },
                    model=model,
                    output_subdir=fixed_folder,
                )

            if TESTING_CONFIG.get("save_mse_ranked_panels", True):
                visualize_results_by_mse(
                    all_imgs,
                    all_recons,
                    all_mses,
                    model_name,
                    TESTING_CONFIG,
                    model=model,
                )

            if TESTING_CONFIG.get("save_psnr_distribution_table", True):
                save_psnr_distribution_table(all_mses, model_name, TESTING_CONFIG)

        os.makedirs(TESTING_CONFIG["results_save_dir"], exist_ok=True)
        metrics_path = os.path.join(
            TESTING_CONFIG["results_save_dir"],
            f"{model_name}{TESTING_CONFIG['results_save_name_suffix']}",
        )
        with open(metrics_path, "w") as f:
            json.dump({"MSE": mse, "PSNR": psnr, "SSIM": ssim}, f, indent=2)
        print(f"Metrics saved at {metrics_path}")

    if distributed and is_distributed():
        dist.barrier()
        dist.destroy_process_group()

# Running command example:
# python test.py --config config_lists/Restormer_ONN8LBlock_v2_config.py    
