# -*- coding: utf-8 -*-
import json
import shutil
from os.path import join

import torch

from pcode.utils.op_files import is_jsonable
from pcode.utils.op_paths import build_dirs


def get_checkpoint_folder_name(conf):
    # get optimizer info.
    optim_info = "{}".format(conf.optimizer)

    # get n_participated
    # conf.n_participated = int(conf.n_clients * conf.participation_ratio + 0.5)

    # concat them together.
    return "_l2-{}_lr-{}_n_comm_rounds-{}_local_n_epochs-{}_batchsize-{}_n_clients_{}_n_participated-{}_optim-{}_agg_scheme-{}".format(
        conf.weight_decay,
        conf.lr,
        conf.n_comm_rounds,
        conf.local_n_epochs,
        conf.batch_size,
        conf.n_clients,
        conf.n_participated,
        optim_info,
        conf.fl_aggregate_scheme,
    )


def init_checkpoint(conf, rank=None):
    # init checkpoint_root for the main process.
    # If resume is specified, use the checkpoint directory from resume path
    if conf.resume is not None and conf.resume != "":
        # Extract checkpoint root from resume path
        # resume path could be a checkpoint file or directory
        import os
        if os.path.isfile(conf.resume):
            # If it's a file, use its directory
            conf.checkpoint_root = os.path.dirname(conf.resume)
        elif os.path.isdir(conf.resume):
            # If it's a directory, use it directly
            conf.checkpoint_root = conf.resume
        else:
            # Try to find checkpoint in the directory
            conf.checkpoint_root = conf.resume
        # Use print if logger is not initialized yet
        if hasattr(conf, 'logger') and conf.logger is not None:
            conf.logger.log(f"Resuming from checkpoint: {conf.checkpoint_root}")
        else:
            print(f"Resuming from checkpoint: {conf.checkpoint_root}")
    else:
        # Create new checkpoint directory
        conf.checkpoint_root = join(
            conf.checkpoint,
            conf.data,
            conf.arch,
            conf.experiment,
            conf.timestamp + get_checkpoint_folder_name(conf),
        )
    
    if conf.save_some_models is not None:
        conf.save_some_models = conf.save_some_models.split(",")

    if rank is None:
        # if the directory does not exists, create them.
        build_dirs(conf.checkpoint_root)
    else:
        conf.checkpoint_dir = join(conf.checkpoint_root, rank)
        build_dirs(conf.checkpoint_dir)


def _save_to_checkpoint(state, dirname, filename):
    checkpoint_path = join(dirname, filename)
    torch.save(state, checkpoint_path)
    return checkpoint_path


def save_arguments(conf):
    # save the configure file to the checkpoint.
    # write_pickle(conf, path=join(conf.checkpoint_root, "arguments.pickle"))
    with open(join(conf.checkpoint_root, "arguments.json"), "w") as fp:
        json.dump(
            dict(
                [
                    (k, v)
                    for k, v in conf.__dict__.items()
                    if is_jsonable(v) and type(v) is not torch.Tensor
                ]
            ),
            fp,
            indent=" ",
        )


def save_to_checkpoint(conf, state, is_best, dirname, filename, save_all=False):
    # save full state.
    checkpoint_path = _save_to_checkpoint(state, dirname, filename)
    best_model_path = join(dirname, "model_best.pth.tar")
    if is_best:
        shutil.copyfile(checkpoint_path, best_model_path)
    if save_all:
        shutil.copyfile(
            checkpoint_path,
            join(
                dirname, "checkpoint_c_round_%s.pth.tar" % state["current_comm_round"]
            ),
        )
    elif conf.save_some_models is not None:
        if str(state["current_comm_round"]) in conf.save_some_models:
            shutil.copyfile(
                checkpoint_path,
                join(
                    dirname,
                    "checkpoint_c_round_%s.pth.tar" % state["current_comm_round"],
                ),
            )
    # Always save the latest checkpoint for resume
    latest_checkpoint_path = join(dirname, "checkpoint_latest.pth.tar")
    shutil.copyfile(checkpoint_path, latest_checkpoint_path)


def load_checkpoint(checkpoint_path, map_location=None):
    """
    Load checkpoint from file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        map_location: Device to load checkpoint to (default: same as saved)
    
    Returns:
        checkpoint: Dictionary containing checkpoint state
    """
    if map_location is None:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    return checkpoint


def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in the directory.
    Priority: model_best.pth.tar > checkpoint_latest.pth.tar > checkpoint_c_round_*.pth.tar (latest)
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
    
    Returns:
        checkpoint_path: Path to the latest checkpoint, or None if not found
    """
    import os
    import glob
    
    # Priority 1: model_best.pth.tar
    best_path = join(checkpoint_dir, "model_best.pth.tar")
    if os.path.exists(best_path):
        return best_path
    
    # Priority 2: checkpoint_latest.pth.tar
    latest_path = join(checkpoint_dir, "checkpoint_latest.pth.tar")
    if os.path.exists(latest_path):
        return latest_path
    
    # Priority 3: checkpoint_c_round_*.pth.tar (find the latest one)
    pattern = join(checkpoint_dir, "checkpoint_c_round_*.pth.tar")
    checkpoints = glob.glob(pattern)
    if checkpoints:
        # Extract round numbers and find the maximum
        def get_round(filename):
            import re
            match = re.search(r'checkpoint_c_round_(\d+)\.pth\.tar', filename)
            return int(match.group(1)) if match else 0
        
        latest = max(checkpoints, key=get_round)
        return latest
    
    return None
