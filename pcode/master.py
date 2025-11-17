# -*- coding: utf-8 -*-
import collections
import copy
import os
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import swanlab

import pcode.create_aggregator as create_aggregator
import pcode.create_coordinator as create_coordinator
import pcode.create_dataset as create_dataset
import pcode.create_metrics as create_metrics
import pcode.create_model as create_model
import pcode.master_utils as master_utils
import pcode.utils.checkpoint as checkpoint
from pcode.utils.auto_distributed import recv_list, scatter_objects
from pcode.utils.early_stopping import EarlyStoppingTracker
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.utils.topk_eval import TopkEval
from swanlab_utils import init_swanlab


class Master(object):
    def __init__(self, conf):
        # 结果可视化
        init_swanlab(conf, 'graph_recommendation', conf.experiment,
                   {"loss": "min", "accuracy": "max", "auc": "max", "precision": "max", "recall": "max", "ndcg": "max"},
                   "comm_round")

        # Default training progress values (may be overridden when resuming)
        self.start_comm_round = 1  # Default start round
        self._saved_best_perf = None  # Will store best_perf from checkpoint

        self.init_parameters(conf)

        # dist.barrier()
        # 加载数据
        self.init_dataloader(conf)

        # 初始化模型
        self.init_model(conf)

        # 用于初始化损失函数（criterion）和评估指标（metrics）
        self.init_criterion_and_metric(conf)

        # define the aggregators.聚合器
        self.aggregator = create_aggregator.Aggregator(
            conf,
            model=self.master_model,
            criterion=self.criterion,
            metrics=self.metrics,
            dataset=self.dataset,
            test_loaders=self.test_loaders,
            clientid2arch=self.clientid2arch,
        )

        # define early_stopping_tracker.
        # conf.early_stopping_rounds ===0
        # conf.early_stopping_rounds 耐心轮次（例如：如果设置为 5，则表示模型在验证集上连续 5 次性能没有提升，训练就会提前结束）
        self.early_stopping_tracker = EarlyStoppingTracker(
            patience=conf.early_stopping_rounds
        )

        # save arguments to disk.
        conf.is_finished = False

        # Note: Checkpoint loading will be done after coordinator is initialized
        # to properly restore best_perf. Save arguments for new training.
        if conf.resume is None or conf.resume == "":
            checkpoint.save_arguments(conf)

        # simulation parameter
        self.worker_archs = collections.defaultdict(str) #默认值为 str（即空字符串）
        self.last_comm_round = self.start_comm_round - 1 #用于跟踪训练过程中的最后一个通信轮次

    def init_criterion_and_metric(self, conf):

        # define the criterion and metrics.
        # 用于初始化损失函数（criterion）和评估指标（metrics）
        # self.criterion = cross_entropy.CrossEntropyLoss(reduction="mean")
        # 定义评估指标
        # TopkEval 的构造函数中传入了数据配置 (self.conf.data)、训练数据集 (self.dataset['train'])、测试数据集 (self.dataset['test'])，以及 k_list（表示不同的 k 值）。
        self.topk_eval = TopkEval(
            self.conf.data,
            self.dataset['train'],
            self.dataset['test'],
            k_list=self.conf.k_list,
            logger=self.conf.logger,
        )
        # 定义损失函数，一个二元交叉熵损失函数，常用于二分类任务。
        self.criterion = torch.nn.BCELoss()
        # 定义评估指标
        self.metrics = create_metrics.Metrics(self.master_model, task="recommondation")
        # 记录一条日志，表示模型、数据集、损失函数和评估指标初始化完成
        conf.logger.log(f"Master initialized model/dataset/criterion/metrics.")
        # 初始化协调器
        self.coordinator = create_coordinator.Coordinator(conf, self.metrics)
        # 记录日志，表示聚合器或协调器已初始化完成
        conf.logger.log(f"Master initialized the aggregator/coordinator.\n")
        
        # Load checkpoint if resume is specified (after coordinator is initialized)
        if conf.resume is not None and conf.resume != "":
            self._load_checkpoint(conf)

    def init_dataloader(self, conf):
        # 初始化训练、验证和测试数据的加载器，并支持按用户分区的加载器设置，用于不同的客户端数据拆分
        # 该函数会返回一个字典格式的数据集对象 self.dataset，可能包含训练、验证和测试数据集
        self.dataset = create_dataset.define_dataset(conf, data=conf.data)
        # _, self.data_partitioner = create_dataset.define_data_loader(
        #     self.conf,
        #     dataset=self.dataset["train"],
        #     localdata_id=0,  # random id here.
        #     is_train=True,
        #     data_partitioner=None,
        # )
        # 记录日志，表示主进程已初始化本地训练数据。
        conf.logger.log(f"Master initialized the local training data with workers.")
        # create val loader.初始化验证集加载器
        # right now we just ignore the case of partitioned_by_user.
        # 检查是否存在验证集 self.dataset["val"]
        if self.dataset["val"] is not None:
            assert not conf.partitioned_by_user
            self.val_loader, _ = create_dataset.define_data_loader(
                conf, self.dataset["val"], is_train=False
            )
            conf.logger.log(f"Master initialized val data.")
        else:
            self.val_loader = None
        # create test loaders.初始化测试集加载器
        # localdata_id start from 0 to the # of clients - 1. client_id starts from 1 to the # of clients.
        # 根据配置 conf.partitioned_by_user 判断是否按用户分区
        if conf.partitioned_by_user:
            # 如果按用户分区，将每个客户端的数据集创建成一个单独的测试加载器，存入 self.test_loaders 列表中
            self.test_loaders = []
            for localdata_id in self.client_ids:
                test_loader, _ = create_dataset.define_data_loader(
                    conf,
                    self.dataset["test"],
                    localdata_id=localdata_id,
                    is_train=False,
                    shuffle=False,
                )
                self.test_loaders.append(copy.deepcopy(test_loader))
        else:
            # 直接创建一个单一的测试加载器 test_loader
            test_loader, _ = create_dataset.define_data_loader(
                conf, self.dataset["test"], is_train=False
            )
            print('self.dataset["test"]',self.dataset["test"])
            self.test_loaders = [test_loader]

    def init_parameters(self, conf):
        # some initializations.
        # 对参数进行初始化，包括配置对象、客户端和参与节点的 ID 列表，以及设备选择
        self.conf = conf
        # 初始化客户端 ID 列表:使用conf.n_clients（客户端数量）创建一个从 0 到 n_clients - 1 的 ID 列表，
        self.client_ids = list(range(conf.n_clients))
        # 初始化（表示每回合参与的客户端数目）参与节点 ID 列表:使用 conf.n_participated（参与节点数量）创建一个从 1 到 n_participated 的 ID 列表，赋值给 self.world_ids
        self.world_ids = list(range(1, 1 + conf.n_participated))
        # 选择计算设备
        self.conf.device = self.device = torch.device("cuda" if conf.on_cuda else "cpu")

    def init_model(self, conf):
        # 用于初始化模型及其状态字典，包括主模型、客户端模型架构，以及客户端 ID 与架构的映射关系
        # create model as well as their corresponding state_dicts.
        # 获取知识图
        self.conf.kg = self.dataset["train"].get_kg()
        print('def init_model(self, conf)---------------------------------------')
        print(self.dataset["train"])
        self.conf.n_clients = self.conf.kg[1]
        print('self.conf.n_clients-----------------------------------------------')
        print(self.conf.n_clients)
        # 更新 client_ids 以匹配实际的客户端数量
        self.client_ids = list(range(self.conf.n_clients))
        # 确保 n_participated 不超过实际的客户端数量
        if self.conf.n_participated > self.conf.n_clients:
            self.conf.logger.log(
                f"Warning: n_participated ({self.conf.n_participated}) exceeds n_clients ({self.conf.n_clients}). "
                f"Setting n_participated to {self.conf.n_clients}."
            )
            self.conf.n_participated = self.conf.n_clients
        # 更新 world_ids 以匹配更新后的 n_participated
        self.world_ids = list(range(1, 1 + self.conf.n_participated))
        # 初始化主模型
        _, self.master_model = create_model.define_model(
            conf, to_consistent_model=False
        )
        # 确定客户端架构:所有客户端的架构被放入一个集合中。如果多个客户端使用相同的架构，则该架构只会在集合中出现一次。
        self.used_client_archs = set(
            [
                create_model.determine_arch(conf, client_id, use_complex_arch=True)
                for client_id in range(1, 1 + conf.n_clients)
            ]
        )
        self.conf.used_client_archs = self.used_client_archs
        conf.logger.log(f"The client will use archs={self.used_client_archs}.")
        conf.logger.log("Master created model templates for client models.")


        # 初始化客户端模型
        self.client_models = dict(
            create_model.define_model(conf, to_consistent_model=False, arch=arch)
            for arch in self.used_client_archs
        )
        # 将第一个客户端模型的状态字典加载到主模型的聚合器 aggregator 中，以确保模型参数的一致性
        # state_dict() 是 PyTorch 中的一个方法，用于返回模型的状态字典，其中包含了模型的所有参数（权重和偏置）。
        # load_state_dict() 是 PyTorch 中的方法，用于将一个字典中的参数加载到模型中。它会更新模型中所有与字典中键匹配的参数。
        self.master_model.aggregator.load_state_dict(list(self.client_models.values())[0].state_dict())
        # 初始化主模型的梯度为零，避免梯度的累积
        # .parameters() 是 PyTorch 模型中的一个方法，它返回模型的所有可训练参数（如权重、偏置等）。每个参数都是一个 torch.Tensor 对象
        for param in self.master_model.parameters():
            param.grad=torch.zeros_like(param) #torch.zeros_like(param) 返回一个与 param 同形状的零张量

        # 创建客户端 ID 与架构的映射
        self.clientid2arch = dict(
            (
                client_id,
                create_model.determine_arch(
                    conf, client_id=client_id, use_complex_arch=True
                ),
            )
            for client_id in range(conf.n_clients)
        )
        print('conf.n_clients',conf.n_clients)
        self.conf.clientid2arch = self.clientid2arch
        conf.logger.log(
            f"Master initialize the clientid2arch mapping relations: {self.clientid2arch}."
        )

    def run(self):
        # run 方法实现了一个用于联邦学习的循环过程，主要包括通信轮次的管理、客户端选择、模型传递、聚合、早停检测等操作。
        # Start from the resumed round if checkpoint was loaded
        start_round = self.start_comm_round
        self.conf.logger.log(f"Starting training from comm_round={start_round} (total rounds={self.conf.n_comm_rounds})")
        for comm_round in range(start_round, 1 + self.conf.n_comm_rounds):
            # 循环遍历 comm_round（通信轮次）
            self.conf.graph.comm_round = comm_round
            self.conf.logger.log(
                f"Master starting one round of federated learning: (comm_round={comm_round})."
            )

            # get random n_local_epochs.获取随机的本地训练轮次数
            list_of_local_n_epochs = self._get_n_local_epoch(
                conf=self.conf, n_participated=self.conf.n_participated
            )
            # random select clients from a pool.客户端池中随机选择一些客户端参与当前通信轮次
            selected_client_ids = self._random_select_clients()
            selected_client_ids, list_of_local_n_epochs = self._select_clients_per_round(selected_client_ids,
                                                                                         list_of_local_n_epochs)

            # start one comm_round, parallel worker number of threads
            # 用于存储客户端上传的本地模型
            flatten_local_models={}
            for i, (client_ids, local_n_epoch) in enumerate(zip(selected_client_ids, list_of_local_n_epochs)):

                # detect early stopping.检测是否满足早停条件，如果满足则提前结束训练
                # self._check_early_stopping()
                # get current workers' client id.

                # init the activation tensor and broadcast to all clients (either start or stop).激活当前选定的客户端
                self._activate_selected_clients(
                    client_ids, self.conf.graph.comm_round, local_n_epoch
                )

                # will decide to send the model or stop the training.
                if not self.conf.is_finished:
                    # broadcast the model to activated clients.
                    self._send_model_to_selected_clients(client_ids)
                else:
                    # dist.barrier()
                    self.conf.logger.log(
                        f"Master finished the federated learning by early-stopping: (current comm_rounds={comm_round}, total_comm_rounds={self.conf.n_comm_rounds})"
                    )
                    return

                # wait to receive the local models.
                flatten_local_models.update(self._receive_models_from_selected_clients(
                    client_ids
                ))

            # aggregate the local models and evaluate on the validation dataset.
            self._aggregate(flatten_local_models)
            self._evaluate()

            # evaluate the aggregated model.
            self.conf.logger.log(
                f"Master finished comm_round={comm_round} of federated learning."
            )

        # formally stop the training (the master has finished all communication rounds).
        # dist.barrier()
        self._finishing()

    def _select_clients_per_round(self, selected_client_ids, list_of_local_n_epochs):
        # 将参与训练的客户端分配给
        client_ids = []
        local_n_epochs = []
        #  self.conf.workers进程总数，确保是整数类型
        workers = int(self.conf.workers) if isinstance(self.conf.workers, (str, int)) else self.conf.workers
        if self.conf.n_participated % workers:
            selected_client_ids += [-1] * (workers - self.conf.n_participated % workers)
            list_of_local_n_epochs += [-1] * (workers - self.conf.n_participated % workers)
        for i in range(0, self.conf.n_participated, workers):
            client_ids.append(selected_client_ids[i:i + workers])
            local_n_epochs.append(list_of_local_n_epochs[i:i + workers])

        return client_ids, local_n_epochs

    def _random_select_clients(self):
        ''' 从n_clients中随机选择n_participated个客户端参与本轮次训练'''
        selected_client_ids = self.conf.random_state.choice(
            self.client_ids, self.conf.n_participated, replace=False
        ).tolist()
        selected_client_ids.sort()
        self.conf.logger.log(
            f"Master selected {self.conf.n_participated} from {self.conf.n_clients} clients: {selected_client_ids}."
        )
        # if len(selected_client_ids)% self.conf.workers!=0:
        #     selected_client_ids+= [-1]*(self.conf.workers-len(selected_client_ids) % self.conf.workers)
        return selected_client_ids

    def _activate_selected_clients(
            self, selected_client_ids, comm_round, list_of_local_n_epochs
    ):
        # Activate the selected clients:
        # the first row indicates the client id,
        # the second row indicates the current_comm_round,
        # the third row indicates the expected local_n_epochs
        scatter_list=[]
        for selected_client_id, local_n_epoch in zip( selected_client_ids ,list_of_local_n_epochs):
            activation_msg={}
            activation_msg['client_id']=selected_client_id
            activation_msg['comm_round'] =comm_round
            activation_msg['local_epoch'] =local_n_epoch
            # client_arch= self.clientid2arch[selected_client_id]
            # rank=dist.get_rank()
            # if selected_client_id != -1 and  self.worker_archs[rank]!= client_arch :
            #     model = copy.deepcopy(self.client_models[client_arch])
            #     activation_msg.append(model)
            # else:
            #     activation_msg.append(None)
            scatter_list.append(activation_msg)
        scatter_objects(scatter_list)
        self.conf.logger.log(f"Master activated the selected clients.")
        # dist.barrier()

    def _send_model_to_selected_clients(self, selected_client_ids):
        # the master_model can be large; the client_models can be small and different.
        self.conf.logger.log(f"Master send the models to workers.")
        scatter_list = []
        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            # transfer parameters if new comm_round and client arch not changed.
            distribut_dict = {}
            if selected_client_id != -1:
                client_arch = self.clientid2arch[selected_client_id]
                # send the model if the worker_arch is different from the last comm_round.
                if self.last_comm_round != self.conf.graph.comm_round or self.worker_archs[worker_rank] !=client_arch:
                    self.worker_archs[worker_rank] = client_arch
                    distribut_dict['model']=self.client_models[client_arch]
                else:
                    distribut_dict['model']=None

                # 获取selected_client_id客户端（用户）的嵌入信息：包含：user_embeddings, entities, entity_embeddings,
                #relations, relation_embeddings, target(该user所有交互项的标签列表）
                distribut_dict['embeddings'] =self.master_model._get_embeddings(selected_client_id, self.dataset["train"], self.conf.local_batch_size)
                scatter_list.append(distribut_dict)
            else:
                scatter_list.append(None)
            # self.conf.logger.log (f"\tMaster send the current model={client_arch} to process_id={worker_rank}")
        scatter_objects(scatter_list)

        self.last_comm_round = self.conf.graph.comm_round
        self.conf.logger.log(
            f"\tMaster send the current model={client_arch} to process_id={worker_rank}."
        )
        # dist.monitored_barrier()


    def _receive_models_from_selected_clients(self, selected_client_ids):
        self.conf.logger.log(f"Master waits to receive the local models.")

        flatten_local_models = {}
        for worker_rank, expected_client_id in enumerate(selected_client_ids, start=1):
            tensors = recv_list(worker_rank)
            if len(tensors) == 0:
                continue

            metadata = tensors[0].to(torch.long).cpu()
            recv_client_id = int(metadata[0].item())
            grad_count = int(metadata[1].item())
            has_relation_dist = int(metadata[2].item()) if len(metadata) > 2 else 0

            if recv_client_id == -1:
                continue

            model_grads = tensors[1:1 + grad_count]
            embeddings_grad = tensors[1 + grad_count:1 + grad_count + 3]  # 3个embedding梯度
            
            # 提取关系分布（如果存在）
            relation_distribution = None
            if has_relation_dist == 1 and len(tensors) > 1 + grad_count + 3:
                relation_distribution = tensors[1 + grad_count + 3]

            flatten_local_models[recv_client_id] = {
                'model_grad': model_grads,
                'embeddings_grad': embeddings_grad,
                'relation_distribution': relation_distribution,  # 存储关系分布
            }

        self.conf.logger.log(f"Master received all local models.")
        return flatten_local_models


    def _avg_over_archs(self, flatten_local_models):
        # get unique arch from this comm. round.
        archs = set(
            [
                self.clientid2arch[client_idx]
                for client_idx in flatten_local_models.keys()
            ]
        )

        # average for each arch.
        archs_fedavg_models = {}
        for arch in archs:
            # extract local_models from flatten_local_models.
            _flatten_local_models = {}
            for client_idx, flatten_local_model in flatten_local_models.items():
                if self.clientid2arch[client_idx] == arch:
                    _flatten_local_models[client_idx] = flatten_local_model

            # average corresponding local models.
            self.conf.logger.log(
                f"Master uniformly average over {len(_flatten_local_models)} received models ({arch})."
            )
            fedavg_model = self.aggregator.aggregate(
                master_model=self.master_model,
                client_models=self.client_models,
                flatten_local_models=_flatten_local_models,
                aggregate_fn_name="_s1_federated_average",
            )
            archs_fedavg_models[arch] = fedavg_model
        return archs_fedavg_models

    def _aggregate(self, flatten_local_models):
        # uniformly averaged the model before the potential aggregation scheme.
        # same_arch = (
        #         len(self.client_models) == 1
        #         and self.conf.arch_info["master"] == self.conf.arch_info["worker"][0]
        # )
        # conf.same_arch=True
        same_arch = self.conf.same_arch
        # uniformly average local models with the same architecture.
        # fedavg_models = self._avg_over_archs(flatten_local_models)

        # 主模型更新梯度
        self.master_model.recode_grad(flatten_local_models)

        if not hasattr(self, 'optimizer'):
            self.optimizer = torch.optim.Adam(self.master_model.parameters(), lr=self.conf.lr,
                                              weight_decay=self.conf.weight_decay)
        eval_model_for_perf = None
        if same_arch:
            # TODO: 如何处理grad
            # 打印参数更新前的状态
            # for i, param in enumerate (self.master_model.parameters ()):
            #     print (
            #         f"Before optimizer step - Param {i}: Value mean: {param.data.mean ()} Grad mean: {param.grad.mean ()}")
            self.optimizer.step() # 根据累加后的梯度，更新主模型的参数
            # for i, param in enumerate (self.master_model.parameters ()):
            #     print (f"After optimizer step - Param {i}: Value mean: {param.data.mean ()}")
            self.optimizer.zero_grad(set_to_none=False)# 清空梯度
            fedavg_model = copy.deepcopy(self.master_model.aggregator) #深拷贝主模型的聚合器
            # 评估时需要完整的 master_model
            eval_model_for_perf = copy.deepcopy(self.master_model)
            fedavg_models = {'kgcn_aggregate': fedavg_model}
            # fedavg_model = list(fedavg_models.values())[0]
        else:
            fedavg_model = None

        # (smarter) aggregate the model from clients.
        # note that: if conf.fl_aggregate["scheme"] == "federated_average",
        #            then self.aggregator.aggregate_fn = None.
        if self.aggregator.aggregate_fn is not None:
            # evaluate the uniformly averaged model.
            if fedavg_model is not None:
                # 使用完整的 master_model 进行验证，避免直接调用 aggregator 导致 forward 参数不匹配
                eval_model = copy.deepcopy(eval_model_for_perf) if eval_model_for_perf is not None else copy.deepcopy(self.master_model)
                performance = master_utils.get_avg_perf_on_dataloaders(
                    self.conf,
                    self.coordinator,
                    eval_model,
                    self.criterion,
                    self.metrics,
                    self.test_loaders,
                    label=f"fedag_test_loader",
                )
                del eval_model
            else:
                assert "knowledge_transfer" in self.conf.fl_aggregate["scheme"]

                performance = None
                for _arch, _fedavg_model in fedavg_models.items():
                    master_utils.get_avg_perf_on_dataloaders(
                        self.conf,
                        self.coordinator,
                        _fedavg_model,
                        self.criterion,
                        self.metrics,
                        self.test_loaders,
                        label=f"fedag_test_loader_{_arch}",
                    )

            # aggregate the local models.
            client_models = self.aggregator.aggregate(
                master_model=self.master_model,
                client_models=self.client_models,
                fedavg_model=fedavg_model,
                fedavg_models=fedavg_models,
                flatten_local_models=flatten_local_models,
                performance=performance,
            )
            # here the 'client_models' are updated in-place.
            if same_arch:
                # here the 'master_model' is updated in-place only for 'same_arch is True'.
                self.master_model.load_state_dict(
                    list(client_models.values())[0].state_dict()
                )
            for arch, _client_model in client_models.items():
                self.client_models[arch].load_state_dict(_client_model.state_dict())
        else:
            # update self.master_model in place.
            # if same_arch:
            #     self.master_model.load_state_dict(fedavg_model.state_dict())
            # update self.client_models in place.
            # 更新客户端模型参数
            for arch, _fedavg_model in fedavg_models.items():
                # 表示将 _fedavg_model 的权重和偏置参数加载到 self.client_models[arch] 中的对应模型
                self.client_models[arch].load_state_dict(_fedavg_model.state_dict())
        self.conf.logger.log (
            f"\tMaster finish aggregate the models."
        )


    def _evaluate(self):
        # 每隔validation_interval轮，验证一次模型性能，保存最佳性能
        # 确保 validation_interval 是整数类型
        validation_interval = int(self.conf.validation_interval) if isinstance(self.conf.validation_interval, (str, int)) else self.conf.validation_interval
        if self.conf.graph.comm_round % validation_interval == 0:
            self._validation()
        # 每隔topk_eval_interval轮，评估一次性能
        # 确保 topk_eval_interval 是整数类型
        topk_eval_interval = int(self.conf.topk_eval_interval) if isinstance(self.conf.topk_eval_interval, (str, int)) else self.conf.topk_eval_interval
        if self.conf.graph.comm_round % topk_eval_interval == 0:
            try:
                self.topk_eval.eval(self.master_model, self.last_comm_round)
            except Exception as exc:
                self.conf.logger.log(
                    f"Top-K evaluation skipped at comm_round={self.conf.graph.comm_round}: {exc}"
                )


    def _validation(self):
        try:
            if self.conf.same_arch:
                master_utils.do_validation(
                    self.conf,
                    self.coordinator,
                    self.master_model,
                    self.criterion,
                    self.metrics,
                    self.test_loaders,
                    label=f"aggregated_test_loader",
                )
            else:
                for arch, _client_model in self.client_models.items():
                    master_utils.do_validation(
                        self.conf,
                        self.coordinator,
                        _client_model,
                        self.criterion,
                        self.metrics,
                        self.test_loaders,
                        label=f"aggregated_test_loader_{arch}",
                    )
        except Exception as exc:
            self.conf.logger.log(
                f"Validation failed at comm_round={self.conf.graph.comm_round}: {exc}. Continuing training..."
            )
            import traceback
            self.conf.logger.log(traceback.format_exc())

    def _check_early_stopping(self):
        meet_flag = False

        # consider both of target_perf and early_stopping
        if self.conf.target_perf is not None:
            assert 100 >= self.conf.target_perf > 0

            # meet the target perf.
            if (
                    self.coordinator.key_metric.cur_perf is not None
                    and self.coordinator.key_metric.cur_perf > self.conf.target_perf
            ):
                self.conf.logger.log("Master early stopping: meet target perf.")
                self.conf.meet_target = True
                meet_flag = True
            # or has no progress and early stop it.
            elif self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
                self.conf.logger.log(
                    "Master early stopping: not meet target perf but has no patience."
                )
                meet_flag = True
        # only consider the early stopping.
        else:
            if self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
                meet_flag = True

        if meet_flag:
            # we perform the early-stopping check:
            # (1) before the local training and (2) after the update of the comm_round.
            _comm_round = self.conf.graph.comm_round - 1
            self.conf.graph.comm_round = -1
            self._finishing(_comm_round)

    def _finishing(self, _comm_round=None):
        self.conf.logger.save_json()
        self.conf.logger.log(f"Master finished the federated learning.")
        self.conf.is_finished = True
        self.conf.finished_comm = _comm_round
        checkpoint.save_arguments(self.conf)
        os.system(f"echo {self.conf.checkpoint_root} >> {self.conf.job_id}")


    def _load_checkpoint(self, conf):
        """
        Load checkpoint and restore model state and training progress.
        
        Args:
            conf: Configuration object
        """
        import os
        from os.path import join
        
        # Find checkpoint file
        checkpoint_file = None
        if os.path.isfile(conf.resume):
            checkpoint_file = conf.resume
        elif os.path.isdir(conf.resume):
            # Try to find checkpoint in the directory
            checkpoint_file = checkpoint.find_latest_checkpoint(conf.resume)
            if checkpoint_file is None:
                # Try to find in rank 0 subdirectory
                rank_0_dir = join(conf.resume, "0")
                if os.path.isdir(rank_0_dir):
                    checkpoint_file = checkpoint.find_latest_checkpoint(rank_0_dir)
        else:
            # Treat as directory path
            checkpoint_file = checkpoint.find_latest_checkpoint(conf.resume)
        
        if checkpoint_file is None or not os.path.exists(checkpoint_file):
            conf.logger.log(f"Warning: Checkpoint file not found at {conf.resume}. Starting from scratch.")
            checkpoint.save_arguments(conf)
            return
        
        # Load checkpoint
        conf.logger.log(f"Loading checkpoint from: {checkpoint_file}")
        try:
            ckpt = checkpoint.load_checkpoint(checkpoint_file)
            
            # Restore model state
            if "state_dict" in ckpt:
                try:
                    self.master_model.load_state_dict(ckpt["state_dict"])
                    conf.logger.log("Model state loaded successfully.")
                except Exception as e:
                    conf.logger.log(f"Warning: Failed to load model state: {e}. Continuing with new model.")
            
            # Restore training progress
            if "current_comm_round" in ckpt:
                self.start_comm_round = ckpt["current_comm_round"] + 1  # Start from next round
                conf.logger.log(f"Resuming from comm_round {self.start_comm_round} (was at {ckpt['current_comm_round']})")
            else:
                conf.logger.log("Warning: current_comm_round not found in checkpoint. Starting from round 1.")
            
            # Restore best performance to coordinator
            best_perf = ckpt.get("best_perf")
            if best_perf is not None and hasattr(self, 'coordinator') and self.coordinator is not None:
                if isinstance(best_perf, dict):
                    # Restore best performance to coordinator's best trackers
                    for metric_name, best_value in best_perf.items():
                        if metric_name in self.coordinator.best_trackers:
                            self.coordinator.best_trackers[metric_name].best_perf = best_value
                    conf.logger.log(f"Best performance restored: {best_perf}")
                else:
                    # Some checkpoints store a single scalar best_perf (e.g., best AUC)
                    self._saved_best_perf = float(best_perf)
                    conf.logger.log(f"Best performance restored (scalar): {self._saved_best_perf}")
            
            conf.logger.log(f"Checkpoint loaded successfully. Will resume from comm_round {self.start_comm_round}")
            # Save arguments after loading checkpoint (to update checkpoint directory info)
            checkpoint.save_arguments(conf)
            
        except Exception as e:
            conf.logger.log(f"Error loading checkpoint: {e}. Starting from scratch.")
            checkpoint.save_arguments(conf)

    def _get_n_local_epoch(self, conf, n_participated):
        #  conf.min_local_epochs ===None
        if conf.min_local_epochs is None:
            # local_n_epochs == 1, n_participated===32
            return [conf.local_n_epochs] * n_participated
        else:
            # here we only consider to (uniformly) randomly sample the local epochs.
            assert conf.min_local_epochs > 1.0
            random_local_n_epochs = conf.random_state.uniform(
                low=conf.min_local_epochs, high=conf.local_n_epochs, size=n_participated
            )
            return random_local_n_epochs
