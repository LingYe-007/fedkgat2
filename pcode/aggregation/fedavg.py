# -*- coding: utf-8 -*-
import copy

import torch
import torch.nn.functional as F

import pcode.aggregation.utils as agg_utils
import pcode.master_utils as master_utils
from pcode.utils.module_state import ModuleState
from pcode.utils.tensor_buffer import TensorBuffer


def compute_relation_similarity(relation_dist1, relation_dist2, similarity_method='kl_divergence'):
    '''
    计算两个关系分布之间的相似度
    
    Args:
        relation_dist1: torch.Tensor of shape [num_rel]，归一化的关系分布
        relation_dist2: torch.Tensor of shape [num_rel]，归一化的关系分布
        similarity_method: str, 'kl_divergence' 或 'cosine'
    
    Returns:
        similarity: float, 相似度分数（0-1之间，1表示完全相同）
    '''
    if similarity_method == 'kl_divergence':
        # 使用 KL 散度计算相似度
        # 添加小值避免 log(0)
        eps = 1e-10
        p = relation_dist1 + eps
        q = relation_dist2 + eps
        
        # 计算对称 KL 散度: (KL(p||q) + KL(q||p)) / 2
        kl_pq = (p * torch.log(p / q)).sum()
        kl_qp = (q * torch.log(q / p)).sum()
        kl_symmetric = (kl_pq + kl_qp) / 2
        
        # 将 KL 散度转换为相似度: similarity = 1 / (1 + kl)
        # 使用指数衰减: similarity = exp(-kl) 更平滑
        similarity = torch.exp(-kl_symmetric).item()
        
    elif similarity_method == 'cosine':
        # 使用余弦相似度
        similarity = F.cosine_similarity(
            relation_dist1.unsqueeze(0),
            relation_dist2.unsqueeze(0),
            dim=1
        ).item()
        
        # 确保相似度在 [0, 1] 范围内（余弦相似度范围是 [-1, 1]）
        similarity = (similarity + 1) / 2
        
    else:
        raise ValueError(f"Unsupported similarity_method: {similarity_method}. Use 'kl_divergence' or 'cosine'.")
    
    return similarity


def _fedavg(clientid2arch, n_selected_clients, flatten_local_models, client_models):
    weights = [
        torch.FloatTensor([1.0 / n_selected_clients]) for _ in range(n_selected_clients)
    ]

    # NOTE: the arch for different local models needs to be the same as the master model.
    # retrieve the local models.
    local_models = {}
    for client_idx, flatten_local_model in flatten_local_models.items():
        _arch = clientid2arch[client_idx]
        _model = copy.deepcopy(client_models[_arch])
        _model_state_dict = client_models[_arch].state_dict()
        flatten_local_model.unpack(_model_state_dict.values())
        _model.load_state_dict(_model_state_dict)
        local_models[client_idx] = _model

    # uniformly average the local models.
    # assume we use the runtime stat from the last model.
    _model = copy.deepcopy(_model)
    local_states = [
        ModuleState(copy.deepcopy(local_model.state_dict()))
        for _, local_model in local_models.items()
    ]
    model_state = local_states[0] * weights[0]
    for idx in range(1, len(local_states)):
        model_state += local_states[idx] * weights[idx]
    model_state.copy_to_module(_model)
    return _model


def _fedavg_kgcn(clientid2arch, n_selected_clients, flatten_local_models, client_models, 
                 conf=None, use_relation_adaptive=True, similarity_method='kl_divergence', 
                 temperature=1.0):
    '''
    联邦平均聚合函数，支持基于关系分布相似度的自适应聚合
    
    Args:
        clientid2arch: 客户端ID到架构的映射
        n_selected_clients: 选中的客户端数量
        flatten_local_models: 展平的本地模型
        client_models: 客户端模型字典
        conf: 配置对象（用于日志输出）
        use_relation_adaptive: 是否使用关系自适应聚合
        similarity_method: 相似度计算方法 ('kl_divergence' 或 'cosine')
        temperature: softmax 温度参数
    
    Returns:
        聚合后的模型
    '''
    # NOTE: the arch for different local models needs to be the same as the master model.
    # retrieve the local models.
    local_models = {}
    for client_idx, flatten_local_model in flatten_local_models.items():
        _arch = clientid2arch[client_idx]
        _model = copy.deepcopy(client_models[_arch])
        _model_state_dict = client_models[_arch].state_dict()
        
        # 处理两种格式：字典或 TensorBuffer
        if isinstance(flatten_local_model, dict) and 'model_grad' in flatten_local_model:
            # 如果是字典格式，从 model_grad 创建 TensorBuffer
            model_grads = flatten_local_model['model_grad']
            tensor_buffer = TensorBuffer(model_grads, device=torch.device('cpu'))
            tensor_buffer.unpack(_model_state_dict.values())
        else:
            # 如果是 TensorBuffer 格式，直接使用
            flatten_local_model.unpack(_model_state_dict.values())
        
        _model.load_state_dict(_model_state_dict)
        local_models[client_idx] = _model

    # 计算聚合权重
    if use_relation_adaptive:
        # 尝试从每个客户端获取关系分布
        # 优先使用从通信中接收的关系分布，否则从模型中提取
        relation_dists = {}
        client_ids = list(local_models.keys())
        
        for client_idx in client_ids:
            relation_dist = None
            
            # 方法1: 从 flatten_local_models 字典中获取关系分布（如果存在）
            if client_idx in flatten_local_models:
                flatten_local_model = flatten_local_models[client_idx]
                if isinstance(flatten_local_model, dict) and 'relation_distribution' in flatten_local_model:
                    relation_dist = flatten_local_model['relation_distribution']
                    if relation_dist is not None:
                        relation_dists[client_idx] = relation_dist
                        if conf and hasattr(conf, 'logger'):
                            non_zero_indices = (relation_dist > 1e-6).nonzero(as_tuple=True)[0]
                            if len(non_zero_indices) <= 20:
                                dist_summary = {idx.item(): relation_dist[idx].item() 
                                              for idx in non_zero_indices}
                                conf.logger.log(f"  Client {client_idx} relation distribution (from communication, top relations): {dist_summary}")
            
            # 方法2: 如果方法1失败，尝试从模型中提取
            if client_idx not in relation_dists or relation_dists[client_idx] is None:
                model = local_models[client_idx]
                if hasattr(model, 'get_normalized_relation_distribution'):
                    try:
                        relation_dist = model.get_normalized_relation_distribution()
                        relation_dists[client_idx] = relation_dist
                        # 记录客户端的关系分布（可选，仅记录非零值）
                        if conf and hasattr(conf, 'logger'):
                            non_zero_indices = (relation_dist > 1e-6).nonzero(as_tuple=True)[0]
                            if len(non_zero_indices) <= 20:  # 只记录前20个非零关系
                                dist_summary = {idx.item(): relation_dist[idx].item() 
                                              for idx in non_zero_indices}
                                conf.logger.log(f"  Client {client_idx} relation distribution (from model, top relations): {dist_summary}")
                    except Exception as e:
                        if conf and hasattr(conf, 'logger'):
                            conf.logger.log(f"Warning: Failed to get relation distribution for client {client_idx}: {e}")
                        relation_dists[client_idx] = None
                else:
                    relation_dists[client_idx] = None
        
        # 检查是否所有客户端都有关系分布
        valid_dists = {k: v for k, v in relation_dists.items() if v is not None}
        
        if len(valid_dists) >= 2:
            # 计算关系分布相似度矩阵
            n_clients = len(client_ids)
            similarity_matrix = torch.zeros(n_clients, n_clients)
            
            for i, client_i in enumerate(client_ids):
                if client_i in valid_dists:
                    dist_i = valid_dists[client_i]
                    for j, client_j in enumerate(client_ids):
                        if client_j in valid_dists:
                            dist_j = valid_dists[client_j]
                            similarity = compute_relation_similarity(dist_i, dist_j, similarity_method)
                            similarity_matrix[i, j] = similarity
                        else:
                            # 如果客户端没有关系分布，使用默认相似度
                            similarity_matrix[i, j] = 0.5
                else:
                    # 如果客户端没有关系分布，使用默认相似度
                    for j in range(n_clients):
                        similarity_matrix[i, j] = 0.5
            
            # 计算每个客户端的平均相似度（与其他所有客户端的平均相似度）
            client_similarities = similarity_matrix.mean(dim=1)  # [n_clients]
            
            # 使用 softmax 将相似度转换为权重
            weights_tensor = F.softmax(client_similarities / temperature, dim=0)
            weights = [weights_tensor[i].item() for i in range(n_clients)]
            
            # 日志输出
            if conf and hasattr(conf, 'logger'):
                conf.logger.log(f"Relation-adaptive aggregation enabled (method={similarity_method}):")
                conf.logger.log(f"  Client similarities: {client_similarities.tolist()}")
                conf.logger.log(f"  Aggregation weights: {weights}")
                # 可选：输出相似度矩阵（仅前几个客户端）
                if n_clients <= 10:
                    conf.logger.log(f"  Similarity matrix:\n{similarity_matrix.numpy()}")
        else:
            # 如果有效关系分布不足，回退到均匀权重
            if conf and hasattr(conf, 'logger'):
                conf.logger.log(f"Warning: Only {len(valid_dists)} clients have relation distributions. "
                              f"Falling back to uniform weights.")
            weights = [
                torch.FloatTensor([1.0 / n_selected_clients]) for _ in range(n_selected_clients)
            ]
    else:
        # 使用均匀权重（原始方法）
        weights = [
            torch.FloatTensor([1.0 / n_selected_clients]) for _ in range(n_selected_clients)
        ]

    # uniformly average the local models.
    # assume we use the runtime stat from the last model.
    _model = copy.deepcopy(_model)
    local_states = [
        ModuleState(copy.deepcopy(local_model.state_dict()))
        for _, local_model in local_models.items()
    ]
    model_state = local_states[0] * weights[0]
    for idx in range(1, len(local_states)):
        model_state += local_states[idx] * weights[idx]
    model_state.copy_to_module(_model)
    return _model


def fedavg(
        conf,
        clientid2arch,
        n_selected_clients,
        flatten_local_models,
        client_models,
        criterion,
        metrics,
        val_data_loader,
):
    if (
            "server_teaching_scheme" not in conf.fl_aggregate
            or "drop" not in conf.fl_aggregate["server_teaching_scheme"]
    ):
        # directly averaging.
        conf.logger.log(f"No indices to be removed.")
        return _fedavg(
            clientid2arch, n_selected_clients, flatten_local_models, client_models
        )
    else:
        # we will first perform the evaluation.
        # recover the models on the computation device.
        _, local_models = agg_utils.recover_models(
            conf, client_models, flatten_local_models
        )

        # get the weights from the validation performance.
        weights = []
        relationship = {}
        indices_to_remove = []
        random_guess_perf = agg_utils.get_random_guess_perf(conf)
        for idx, (client_id, local_model) in enumerate(local_models.items()):
            relationship[idx] = client_id
            validated_perfs = validate(
                conf,
                model=local_model,
                criterion=criterion,
                metrics=metrics,
                data_loader=val_data_loader,
            )
            perf = validated_perfs["top1"]
            weights.append(perf)

            # check the perf.
            if perf < random_guess_perf:
                indices_to_remove.append(idx)

        # update client_teacher.
        conf.logger.log(
            f"Indices to be removed for FedAvg: {indices_to_remove}; the original performance is: {weights}."
        )
        for index in indices_to_remove[::-1]:
            flatten_local_models.pop(relationship[index])
        return _fedavg(
            clientid2arch,
            n_selected_clients - len(indices_to_remove),
            flatten_local_models,
            client_models,
        )


def fedavg_kgcn(
        conf,
        clientid2arch,
        n_selected_clients,
        flatten_local_models,
        client_models,
        criterion,
        metrics,
        val_data_loader,
):
    # 从配置中读取关系自适应聚合参数
    use_relation_adaptive = getattr(conf, 'use_relation_adaptive_aggregation', True)
    if isinstance(use_relation_adaptive, dict):
        use_relation_adaptive = use_relation_adaptive.get('enabled', True)
    
    similarity_method = getattr(conf, 'relation_adaptive_similarity_method', 'kl_divergence')
    if isinstance(similarity_method, dict):
        similarity_method = similarity_method.get('similarity_method', 'kl_divergence')
    
    temperature = getattr(conf, 'relation_adaptive_temperature', 1.0)
    if isinstance(temperature, dict):
        temperature = temperature.get('temperature', 1.0)
    
    # 获取架构信息，用于返回字典格式
    # 假设所有客户端使用相同的架构（same_arch=True）
    arch = list(clientid2arch.values())[0] if clientid2arch else list(client_models.keys())[0]
    
    if (
            "server_teaching_scheme" not in conf.fl_aggregate
            or "drop" not in conf.fl_aggregate["server_teaching_scheme"]
    ):
        # directly averaging.
        conf.logger.log(f"No indices to be removed.")
        aggregated_model = _fedavg_kgcn(
            clientid2arch, 
            n_selected_clients, 
            flatten_local_models, 
            client_models,
            conf=conf,
            use_relation_adaptive=use_relation_adaptive,
            similarity_method=similarity_method,
            temperature=temperature
        )
        # 返回字典格式，与 master.py 期望的格式一致
        return {arch: aggregated_model}
    else:
        # we will first perform the evaluation.
        # recover the models on the computation device.
        _, local_models = agg_utils.recover_models(
            conf, client_models, flatten_local_models
        )

        # get the weights from the validation performance.
        weights = []
        relationship = {}
        indices_to_remove = []
        random_guess_perf = agg_utils.get_random_guess_perf(conf)
        for idx, (client_id, local_model) in enumerate(local_models.items()):
            relationship[idx] = client_id
            validated_perfs = validate(
                conf,
                model=local_model,
                criterion=criterion,
                metrics=metrics,
                data_loader=val_data_loader,
            )
            perf = validated_perfs["top1"]
            weights.append(perf)

            # check the perf.
            if perf < random_guess_perf:
                indices_to_remove.append(idx)

        # update client_teacher.
        conf.logger.log(
            f"Indices to be removed for FedAvg: {indices_to_remove}; the original performance is: {weights}."
        )
        for index in indices_to_remove[::-1]:
            flatten_local_models.pop(relationship[index])
        aggregated_model = _fedavg_kgcn(
            clientid2arch,
            n_selected_clients - len(indices_to_remove),
            flatten_local_models,
            client_models,
            conf=conf,
            use_relation_adaptive=use_relation_adaptive,
            similarity_method=similarity_method,
            temperature=temperature
        )
        # 返回字典格式，与 master.py 期望的格式一致
        return {arch: aggregated_model}


def validate(conf, model, data_loader, criterion, metrics):
    val_perf = master_utils.validate(
        conf=conf,
        coordinator=None,
        model=model,
        criterion=criterion,
        metrics=metrics,
        data_loader=data_loader,
        label=None,
        display=False,
    )
    del model
    return val_perf
