"""Shared, model-agnostic training helpers for baseline and 261106 runs."""

import os
import random

import torch


def capture_rng_state():
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_checkpoint(path, state):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    checkpoint = dict(state)
    checkpoint["rng_state"] = capture_rng_state()
    torch.save(checkpoint, path)


def load_training_checkpoint(path, device="cuda"):
    return torch.load(path, map_location=device)


def restore_training_state(checkpoint, model=None, optimizer=None, scheduler=None):
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    restore_rng_state(checkpoint.get("rng_state"))
    return (
        checkpoint.get("epoch", -1) + 1,
        checkpoint.get("global_step", 0),
        checkpoint.get("best_loss", float("inf")),
    )
