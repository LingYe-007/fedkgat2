#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验对比脚本：对比原始 KGCN 和改进后的 KGCN 的性能
支持关系注意力机制和关系自适应聚合的消融实验
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pcode.models.kgcn import KGCN
from pcode.aggregation.fedavg import compute_relation_similarity


def load_config(config_path):
    """加载配置文件"""
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_model_variants(num_usr, num_ent, num_rel, kg, args, device):
    """创建不同的模型变体"""
    models = {}
    
    # 原始模型（无关系注意力，无关系自适应聚合）
    args_original = argparse.Namespace(**vars(args))
    args_original.use_relation_attention = False
    args_original.track_relation_distribution = False
    models['original'] = KGCN(num_usr, num_ent, num_rel, kg, args_original, device)
    
    # 仅关系注意力
    args_attention = argparse.Namespace(**vars(args))
    args_attention.use_relation_attention = True
    args_attention.relation_attention_alpha = 0.5
    args_attention.relation_attention_type = 'mlp'
    args_attention.track_relation_distribution = False
    models['attention_only'] = KGCN(num_usr, num_ent, num_rel, kg, args_attention, device)
    
    # 仅关系自适应聚合（需要关系分布统计）
    args_adaptive = argparse.Namespace(**vars(args))
    args_adaptive.use_relation_attention = False
    args_adaptive.track_relation_distribution = True
    models['adaptive_only'] = KGCN(num_usr, num_ent, num_rel, kg, args_adaptive, device)
    
    # 完整改进（关系注意力 + 关系自适应聚合）
    args_full = argparse.Namespace(**vars(args))
    args_full.use_relation_attention = True
    args_full.relation_attention_alpha = 0.5
    args_full.relation_attention_type = 'mlp'
    args_full.track_relation_distribution = True
    models['full_improved'] = KGCN(num_usr, num_ent, num_rel, kg, args_full, device)
    
    return models


def evaluate_model(model, test_data, device):
    """评估模型性能"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in test_data:
            u, v, target = batch
            u = u.to(device)
            v = v.to(device)
            target = target.to(device)
            
            output = model((u, v))
            predictions.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # 计算指标
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    predictions_binary = (predictions > 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(targets, predictions_binary),
        'precision': precision_score(targets, predictions_binary, zero_division=0),
        'recall': recall_score(targets, predictions_binary, zero_division=0),
        'f1': f1_score(targets, predictions_binary, zero_division=0),
        'auc': roc_auc_score(targets, predictions) if len(np.unique(targets)) > 1 else 0.0,
    }
    
    return metrics


def get_relation_distribution_differences(models, client_ids):
    """计算不同客户端的关系分布差异"""
    distributions = {}
    for model_name, model in models.items():
        if hasattr(model, 'get_normalized_relation_distribution'):
            dists = {}
            for client_id in client_ids:
                # 模拟不同客户端的关系分布
                # 在实际应用中，这应该从训练数据中获取
                dist = model.get_normalized_relation_distribution()
                dists[client_id] = dist
            distributions[model_name] = dists
    
    # 计算客户端间的分布差异
    differences = {}
    for model_name, dists in distributions.items():
        if len(dists) >= 2:
            client_ids_list = list(dists.keys())
            pairwise_diffs = []
            for i in range(len(client_ids_list)):
                for j in range(i + 1, len(client_ids_list)):
                    dist1 = dists[client_ids_list[i]]
                    dist2 = dists[client_ids_list[j]]
                    similarity = compute_relation_similarity(dist1, dist2, 'kl_divergence')
                    pairwise_diffs.append(1 - similarity)  # 转换为差异度
            differences[model_name] = {
                'mean_diff': np.mean(pairwise_diffs),
                'std_diff': np.std(pairwise_diffs),
                'pairwise_diffs': pairwise_diffs
            }
    
    return differences


def visualize_results(results, output_dir):
    """可视化实验结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 性能指标对比
    model_names = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4))
    for idx, metric in enumerate(metrics):
        values = [results[model][metric] for model in model_names]
        axes[idx].bar(model_names, values)
        axes[idx].set_title(metric.upper())
        axes[idx].set_ylabel('Score')
        axes[idx].set_ylim([0, 1])
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 关系分布差异对比
    if 'relation_dist_differences' in results:
        fig, ax = plt.subplots(figsize=(10, 6))
        model_names = []
        mean_diffs = []
        std_diffs = []
        
        for model_name, diff_data in results['relation_dist_differences'].items():
            model_names.append(model_name)
            mean_diffs.append(diff_data['mean_diff'])
            std_diffs.append(diff_data['std_diff'])
        
        x = np.arange(len(model_names))
        ax.bar(x, mean_diffs, yerr=std_diffs, capsize=5)
        ax.set_xlabel('Model Variant')
        ax.set_ylabel('Mean Relation Distribution Difference')
        ax.set_title('Relation Distribution Heterogeneity Across Clients')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'relation_distribution_differences.png'), dpi=300, bbox_inches='tight')
        plt.close()


def run_ablation_study(config_path, output_dir='./experiment_results'):
    """运行消融实验"""
    print("=" * 60)
    print("开始消融实验")
    print("=" * 60)
    
    # 加载配置
    config = load_config(config_path)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 模拟参数（实际应用中应该从配置和数据中获取）
    num_usr = 100
    num_ent = 1000
    num_rel = 50
    kg = defaultdict(list)  # 模拟知识图谱
    for e in range(num_ent):
        for r in range(num_rel):
            kg[e].append((r, (e + 1) % num_ent))
    
    # 创建参数对象
    class Args:
        def __init__(self, config):
            self.n_iter = config.get('n_iter', 1)
            self.batch_size = config.get('batch_size', 32)
            self.dim = config.get('dim', 16)
            self.neighbor_sample_size = config.get('neighbor_sample_size', 5)
            self.aggregator = config.get('aggregator', 'sum')
            self.use_relation_attention = False
            self.relation_attention_alpha = 0.5
            self.relation_attention_type = 'mlp'
            self.track_relation_distribution = False
    
    args = Args(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型变体
    print("\n创建模型变体...")
    models = create_model_variants(num_usr, num_ent, num_rel, kg, args, device)
    print(f"创建了 {len(models)} 个模型变体: {list(models.keys())}")
    
    # 评估每个模型（这里使用模拟数据，实际应用中应该使用真实测试数据）
    print("\n评估模型性能...")
    results = {}
    
    # 模拟测试数据
    test_data = [
        (torch.randint(0, num_usr, (32,)), 
         torch.randint(0, num_ent, (32,)), 
         torch.randint(0, 2, (32,)).float())
        for _ in range(10)
    ]
    
    for model_name, model in models.items():
        print(f"  评估 {model_name}...")
        # 模拟训练（实际应用中应该进行真实训练）
        model.train()
        for _ in range(5):  # 模拟5个batch的训练
            u = torch.randint(0, num_usr, (32,)).to(device)
            v = torch.randint(0, num_ent, (32,)).to(device)
            _ = model((u, v))
        
        # 评估
        metrics = evaluate_model(model, test_data, device)
        results[model_name] = metrics
        print(f"    Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
    
    # 计算关系分布差异
    print("\n计算关系分布差异...")
    client_ids = list(range(5))  # 模拟5个客户端
    relation_dist_differences = get_relation_distribution_differences(models, client_ids)
    results['relation_dist_differences'] = relation_dist_differences
    
    # 可视化结果
    print("\n生成可视化图表...")
    visualize_results(results, output_dir)
    
    # 保存结果
    results_json = {}
    for key, value in results.items():
        if key == 'relation_dist_differences':
            results_json[key] = {
                k: {
                    'mean_diff': float(v['mean_diff']),
                    'std_diff': float(v['std_diff'])
                }
                for k, v in value.items()
            }
        else:
            results_json[key] = {k: float(v) for k, v in value.items()}
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)
    
    # 生成对比报告
    print("\n生成对比报告...")
    report_path = os.path.join(output_dir, 'comparison_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("模型对比报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("性能指标对比:\n")
        f.write("-" * 60 + "\n")
        for model_name, metrics in results.items():
            if model_name != 'relation_dist_differences':
                f.write(f"\n{model_name}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
        
        f.write("\n\n关系分布差异对比:\n")
        f.write("-" * 60 + "\n")
        if 'relation_dist_differences' in results:
            for model_name, diff_data in results['relation_dist_differences'].items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Mean Difference: {diff_data['mean_diff']:.4f}\n")
                f.write(f"  Std Difference: {diff_data['std_diff']:.4f}\n")
    
    print(f"\n实验完成！结果保存在: {output_dir}")
    print(f"  - 性能对比图: {output_dir}/performance_comparison.png")
    print(f"  - 关系分布差异图: {output_dir}/relation_distribution_differences.png")
    print(f"  - 结果JSON: {output_dir}/results.json")
    print(f"  - 对比报告: {output_dir}/comparison_report.txt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='运行模型对比实验')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default='./experiment_results', help='输出目录')
    
    args = parser.parse_args()
    
    run_ablation_study(args.config, args.output_dir)

