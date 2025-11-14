# -*- coding: UTF-8 -*-
import argparse
import os
from typing import Dict

import swanlab

# swanlab是一个用于机器学习的可视化工具，它可以帮助研究人员和开发人员跟踪、可视化和比较机器学习实验。
# swanlab提供了一套强大的功能，包括实时更新的可视化仪表板、实验管理、模型检查点存储和版本控制等。


def init_swanlab(conf: dict, project_name: str, experiment: str, metrics: Dict[str, str], step_metric: str):
    swanlab.login(api_key="rAsmX5uR6pd9bHR0tV3Ca")
    if metrics is None:
        metrics = {"loss": "min",
                   "accuracy": "max"}
    if 'project_name' in conf and conf.project_name is not None:
        project_name = conf.project_name
    if 'experiment' in conf and conf.experiment is not None:
        experiment = conf.experiment

    # Create a serializable config dict, excluding non-serializable objects
    serializable_config = {}
    excluded_keys = ['graph', 'logger', 'random_state', 'arch_info', 'checkpoint_root', 
                     'checkpoint_dir', '_fl_aggregate', 'fl_aggregate_scheme']
    for key, value in conf.__dict__.items():
        if key not in excluded_keys:
            try:
                # Try to serialize to check if it's serializable
                import yaml
                yaml.dump({key: value})
                serializable_config[key] = value
            except:
                # Skip non-serializable values
                pass
    
    swanlab.init(project=project_name, name=experiment, config=serializable_config)
    # swanlab.run.log_code(".")  # This API is not available in current swanlab version (0.5.0)
    # The log_code functionality is not available in the current version of swanlab
    # swanlab.define_metric is not available in swanlab 0.5.0
    # Try to use define_metric if available, otherwise skip it
    try:
        if hasattr(swanlab, 'define_metric'):
            swanlab.define_metric(step_metric, hidden=True)
            for metric, summary in metrics.items():
                swanlab.define_metric(metric, summary=summary, step_metric=step_metric)
    except AttributeError:
        # define_metric is not available in this version, skip it
        pass