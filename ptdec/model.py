import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data.dataloader import default_collate
from typing import Tuple, Callable, Optional, Union
from tqdm import tqdm

from ptdec.utils import target_distribution, cluster_accuracy

from datetime import datetime
from .visualProfiler import VisualProfilerConfig

# ---- [20211203] Added by YangHP, 导入需要的包 ------------------
# from opennre.framework.data_loader import SentenceREDataset
import time
# from cuml import KMeans
# from cuml.cluster import KMeans
# ==============================================================

def train(
    # dataset: torch.utils.data.Dataset,
    static_dataloader: torch.utils.data.DataLoader,
    dataloader: torch.utils.data.DataLoader,
    re_model: torch.nn.Module,
    model: torch.nn.Module,
    epochs: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    stopping_delta: Optional[float] = None,
    collate_fn=default_collate,
    cuda: bool = True,
    ckpt = "ckpt/dec_model.pth.tar",
    sampler: Optional[torch.utils.data.sampler.Sampler] = None,
    silent: bool = False,
    update_freq: int = 10,
    evaluate_batch_size: int = 1024,
    update_callback: Optional[Callable[[float, float], None]] = None,
    epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None,
    tensorboardConfig:VisualProfilerConfig=None,
    log_path=None
) -> None:
    """
    Train the DEC model given a dataset, a model instance and various configuration parameters.

    :param dataset: instance of Dataset to use for training
    :param model: instance of DEC model to train
    :param epochs: number of training epochs
    :param batch_size: size of the batch to train with
    :param optimizer: instance of optimizer to use
    :param stopping_delta: label delta as a proportion to use for stopping, None to disable, default None
    :param collate_fn: function to merge a list of samples into mini-batch
    :param cuda: whether to use CUDA, defaults to True
    :param sampler: optional sampler to use in the DataLoader, defaults to None
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :param update_freq: frequency of batches with which to update counter, None disables, default 10
    :param evaluate_batch_size: batch size for evaluation stage, default 1024
    :param update_callback: optional function of accuracy and loss to update, default None
    :param epoch_callback: optional function of epoch and model, default None
    :return: None
    """

    # 暂时将static_dataloader和train_dataloader改为统一dataloader
    # TODO：区分static_dataloader和train_dataloader，可尝试在framework类中添加static_dataloader属性

    # static_dataloader = DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     # ---- [20211203] Added by YangHP, 添加参数collate_fn ----
    #     # collate_fn=collate_fn,
    #     collate_fn=SentenceREDataset.collate_fn,
    #     # =======================================================
    #     # ---- [20211203] Edited by YangHP, 改为True ----
    #     # pin_memory=False,
    #     pin_memory=True,
    #     # ==============================================
    #     sampler=sampler,
    #     shuffle=False,
    # )
    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     # ---- [20211203] Added by YangHP, 添加参数collate_fn ----
    #     # collate_fn=collate_fn,
    #     collate_fn=SentenceREDataset.collate_fn,
    #     # =======================================================
    #     # ---- [20211203] Edited by YangHP, 改为True ----
    #     pin_memory=True,
    #     # ==============================================
    #     sampler=sampler,
    #     shuffle=True,
    # )
    data_iterator = tqdm(
        static_dataloader,
        leave=True,
        unit="batch",
        postfix={
            "epo": -1,
            "acc": "%.4f" % 0.0,
            "lss": "%.8f" % 0.0,
            "dlb": "%.4f" % -1,
        },
        disable=silent,
    )

    kmeans = KMeans(n_clusters=model.cluster_number, n_init=20)
    model.train()
    features = []
    actual = []
    # form initial cluster centres
    for index, batch in enumerate(data_iterator):
        # if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
        #     batch, value = batch  # if we have a prediction label, separate it to actual
        #     actual.append(value)
        # elif len(batch) == 5:
        #     # ---- [20211202] Added by YangHP, 舍弃`batch[0]`(标签)，并将`batch`转为`tensor`类型 --------
        #     # TODO：  修改拆分batch的方法，原来1-4是特征；改为BagREDataset
        #     actual.append(batch[0])
        #     batch = torch.stack(batch[1:])
        #     # =======================================================================================
        if len(batch) == 7:
            if torch.cuda.is_available():               # 【判断】PyTorch能否【调用GPU】
                for i in range(len(batch)):
                    try:
                        batch[i] = batch[i].cuda()      # 【？】
                    except:
                        pass
            label = batch[0]                            # 获取标签，label.shape = [B]，所以是一个句袋对应一个标签
            # bag_name = batch[1]                         # 袋名
            scope = batch[2]                            # 作用域【YangHP：作用域指的是什么？】
            all_labels = torch.zeros(batch[2][-1][1])
            all_labels = all_labels.cuda()
            bag_index = 0
            for bag in scope:
                all_labels[bag[0]:bag[1]] = all_labels[bag[0]:bag[1]] + label[bag_index]
                bag_index += 1
            actual.append(all_labels)                     # 将标签加入所有标签的list中
            batch = batch[3:]                           # 【YangHP：“args”指的是什么？】
            token = batch[0].view(-1, batch[0].size(-1))
            pos1 = batch[1].view(-1, batch[1].size(-1))
            pos2 = batch[2].view(-1, batch[2].size(-1))
            if batch[3] is not None:
                mask = batch[3].view(-1, batch[3].size(-1))
            if mask is not None:
                rep = model.encoder(token, pos1, pos2, mask) # (nsum, H) 
            else:
                rep = model.encoder(token, pos1, pos2) # (nsum, H) 
            query = torch.zeros((rep.size(0))).long()
            if torch.cuda.is_available():
                query = query.cuda()
            for i in range(len(scope)):
                query[scope[i][0]:scope[i][1]] = label[i]
            att_mat = re_model.module.fc.weight[query] # (nsum, H)
            if re_model.module.use_diag:
                att_mat = att_mat * re_model.module.diag.unsqueeze(0)
            # att_score = (rep * att_mat).sum(-1) # (nsum)        rep.shape = [354, 690], att_mat.shape = [354, 690]
            att_score = rep.mul(att_mat)    # (nsum)        rep.shape = [354, 690], att_mat.shape = [354, 690]
            features.append(att_score.detach().cpu())

        # if cuda:
        #     actual[index] = actual[index].cuda(non_blocking=True)
        #     batch = batch.cuda(non_blocking=True)

        # feature = model.encoder(batch[0],batch[1],batch[2],batch[3]).t().mul(actual[index]).t()
    actual = torch.cat(actual).long()
    print("KMeans started at {}".format(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))))
    predicted = kmeans.fit_predict(torch.cat(features).numpy())
    predicted_previous = torch.tensor(np.copy(predicted), dtype=torch.long)
    _, accuracy = cluster_accuracy(predicted, actual.cpu().numpy())
    cluster_centers = torch.tensor(
        kmeans.cluster_centers_, dtype=torch.float, requires_grad=True
    )
    print("KMeans ended at {}".format(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))))
    if cuda:
        cluster_centers = cluster_centers.cuda(non_blocking=True)
    with torch.no_grad():
        # initialise the cluster centers
        model.state_dict()["assignment.cluster_centers"].copy_(cluster_centers)
    loss_function = nn.KLDivLoss(size_average=False)                    # KL散度损失函数
    delta_label = None
    print("DEC training started at {}".format(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))))
    best_metric = 0
    if tensorboardConfig is not None:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir=tensorboardConfig.writer_dir+'/'+str(datetime.now()))
        for epoch in range(epochs):
            torch.cuda.empty_cache()                        # 清理GPU缓存
            # features = []
            actual = []
            data_iterator = tqdm(
                dataloader,
                leave=True,
                unit="batch",
                postfix={
                    "epo": epoch,
                    "acc": "%.4f" % (accuracy or 0.0),
                    "lss": "%.8f" % 0.0,
                    "dlb": "%.4f" % (delta_label or 0.0),
                },
                disable=silent,
            )
            model.train()
            for index, batch in enumerate(data_iterator):
                torch.cuda.empty_cache()                        # 清理GPU缓存
                # if (isinstance(batch, tuple) or isinstance(batch, list)) and len(
                #     batch
                # ) == 2:
                #     batch, _ = batch  # if we have a prediction label, strip it away
                # elif len(batch) == 5:
                #     # ---- [20211202] Added by YangHP, 舍弃`batch[0]`(标签)，并将`batch`转为`tensor`类型 --------
                #     actual.append(batch[0])
                #     batch = torch.stack(batch[1:])
                #     # =======================================================================================
                if len(batch) == 7:
                    if torch.cuda.is_available():   # 【判断】PyTorch能否【调用GPU】
                        for i in range(len(batch)):
                            try:
                                batch[i] = batch[i].cuda()    # 【？】
                            except:
                                pass
                    label = batch[0]
                    # bag_name = batch[1]              # 袋名
                    scope = batch[2]                 # 作用域【YangHP：作用域指的是什么？】
                    all_labels = torch.zeros(batch[2][-1][1])
                    all_labels = all_labels.cuda()
                    bag_index = 0
                    for bag in scope:
                        all_labels[bag[0]:bag[1]] = all_labels[bag[0]:bag[1]] + label[bag_index]
                        bag_index += 1
                    actual.append(all_labels)                     # 将标签加入所有标签的list中
                    batch = batch[3:]                 # 【YangHP：“args”指的是什么？】
                    token = batch[0].view(-1, batch[0].size(-1))
                    pos1 = batch[1].view(-1, batch[1].size(-1))
                    pos2 = batch[2].view(-1, batch[2].size(-1))
                    if batch[3] is not None:
                        mask = batch[3].view(-1, batch[3].size(-1))
                    if mask is not None:
                        rep = model.encoder(token, pos1, pos2, mask) # (nsum, H) 
                    else:
                        rep = model.encoder(token, pos1, pos2) # (nsum, H) 
                    query = torch.zeros((rep.size(0))).long()
                    if torch.cuda.is_available():
                        query = query.cuda()
                    for i in range(len(scope)):
                        query[scope[i][0]:scope[i][1]] = label[i]
                    att_mat = re_model.module.fc.weight[query] # (nsum, H)
                    if re_model.module.use_diag:
                        att_mat = att_mat * re_model.module.diag.unsqueeze(0)
                    # att_score = (rep * att_mat).sum(-1) # (nsum)        rep.shape = [354, 690], att_mat.shape = [354, 690]
                    feature = rep.mul(att_mat)    # (nsum)        rep.shape = [354, 690], att_mat.shape = [354, 690]

                # if cuda:
                #     actual[index] = actual[index].cuda(non_blocking=True)
                #     batch = batch.cuda(non_blocking=True)
                
                # feature = model.encoder(batch[0],batch[1],batch[2],batch[3]).t().mul(actual[index]).t()

                output = model(feature)                         # (nsum, C) 模型的输出即为q_ij
                target = target_distribution(output).detach()   # (nsum, C) 计算目标分布，应该输入q_ij

                # 以e为底，对output中的所有元素求对数
                loss = loss_function(output.log(), target) / output.shape[0]    # output.shape[0] = nsum
                data_iterator.set_postfix(
                    epo=epoch,
                    acc="%.4f" % (accuracy or 0.0),
                    lss="%.8f" % float(loss.item()),
                    dlb="%.4f" % (delta_label or 0.0),
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step(closure=None)
                # features.append(feature.detach().cpu())
                if update_freq is not None and index % update_freq == 0:
                    loss_value = float(loss.item())
                    data_iterator.set_postfix(
                        epo=epoch,
                        acc="%.4f" % (accuracy or 0.0),
                        lss="%.8f" % loss_value,
                        dlb="%.4f" % (delta_label or 0.0),
                    )
                    writer.add_scalars("train/%s"%("acc"), {
                        "acc": accuracy or 0.0
                    }, epoch)
                    writer.add_scalars("train/%s"%("loss"), {
                        "loss": loss_value
                    }, epoch)
                    writer.add_scalars("train/%s"%("delta_label"), {
                        "delta_label": delta_label or 0.0
                    }, epoch)
                    if update_callback is not None:
                        update_callback(epoch, accuracy, loss_value, delta_label)
            torch.cuda.empty_cache()
            # 这里调用predict，需要传入dataset，所以predict的参数也需要相应修改
            print("Validation {} started at {}".format(epoch, time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))))
            predicted, actual = predict(
                dataloader,
                re_model,
                model,
                batch_size=evaluate_batch_size,
                collate_fn=collate_fn,
                silent=True,
                return_actual=True,
                cuda=cuda,
            )
            print("Validation {} ended at {}".format(epoch, time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))))
            delta_label = (
                float((predicted != predicted_previous).float().sum().item())
                / predicted_previous.shape[0]
            )
            if stopping_delta is not None and delta_label < stopping_delta:
                print(
                    'Early stopping as label delta "%1.5f" less than "%1.5f".'
                    % (delta_label, stopping_delta)
                )
                break
            predicted_previous = predicted
            _, accuracy = cluster_accuracy(predicted.cpu().numpy(), actual.cpu().numpy())
            data_iterator.set_postfix(
                epo=epoch,
                acc="%.4f" % (accuracy or 0.0),
                lss="%.8f" % 0.0,
                dlb="%.4f" % (delta_label or 0.0),
            )
            data=open(log_path,'a+')
            print("迭代次数, 损失, ACC, delta_label")
            print("迭代次数, 损失, ACC, delta_label", file=data)
            print("%d, %.4f, %.4f, %.4f" % (epoch, loss_value, accuracy or 0.0, delta_label or 0.0))
            print("%d, %.4f, %.4f, %.4f" % (epoch, loss_value, accuracy or 0.0, delta_label or 0.0), file=data)
            writer.add_scalars("validation/%s"%("acc"), {
                "acc": accuracy or 0.0
            }, epoch)
            writer.add_scalars("validation/%s"%("loss"), {
                "loss": loss_value
            }, epoch)
            writer.add_scalars("validation/%s"%("delta_label"), {
                "delta_label": delta_label or 0.0
            }, epoch)

            if epoch_callback is not None:
                epoch_callback(epoch, model)
            if accuracy > best_metric:
                print("Best ckpt and saved.")
                print("Best ckpt and saved.", file=data)
                torch.save({'state_dict': model.state_dict()}, ckpt)  # 存储文件
                best_metric = accuracy      # 最好的指标
            data.close()

def predict(
    # dataset: torch.utils.data.Dataset,
    dataloader: torch.utils.data.DataLoader,
    re_model: torch.nn.Module,
    model: torch.nn.Module,
    batch_size: int = 1024,
    collate_fn=default_collate,
    cuda: bool = True,
    silent: bool = False,
    return_actual: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Predict clusters for a dataset given a DEC model instance and various configuration parameters.

    :param dataset: instance of Dataset to use for training
    :param model: instance of DEC model to predict
    :param batch_size: size of the batch to predict with, default 1024
    :param collate_fn: function to merge a list of samples into mini-batch
    :param cuda: whether CUDA is used, defaults to True
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :param return_actual: return actual values, if present in the Dataset
    :return: tuple of prediction and actual if return_actual is True otherwise prediction
    """
    # dataloader = DataLoader(
    #     # ---- [20211203] Edited by YangHP, pin_memory改为True ----
    #     dataset, pin_memory=True, batch_size=batch_size, shuffle=False,
    #     # ---- [20211203] Added by YangHP, 添加参数collate_fn ----
    #     # collate_fn=collate_fn,
    #     collate_fn=SentenceREDataset.collate_fn,
    #     # =======================================================
    # )
    data_iterator = tqdm(dataloader, leave=True, unit="batch", disable=silent,)
    result = []
    # features = []
    actual = []
    model.eval()
    for batch in data_iterator:
        # if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
        #     batch, value = batch  # unpack if we have a prediction label
        #     if return_actual:
        #         actual.append(value)
        # elif len(batch) == 5:
        #     if return_actual:
        #         # ---- [20211202] Added by YangHP, 舍弃`batch[0]`(标签)，并将`batch`转为`tensor`类型 --------
        #         actual.append(batch[0])
        #         batch = torch.stack(batch[1:])
        #         # =======================================================================================
        if len(batch) == 7:
            if torch.cuda.is_available():   # 【判断】PyTorch能否【调用GPU】
                for i in range(len(batch)):
                    try:
                        batch[i] = batch[i].cuda()    # 【？】
                    except:
                        pass
            label = batch[0]
            # bag_name = batch[1]              # 袋名
            scope = batch[2]                 # 作用域【YangHP：作用域指的是什么？】
            all_labels = torch.zeros(batch[2][-1][1])
            all_labels = all_labels.cuda()
            bag_index = 0
            for bag in scope:
                all_labels[bag[0]:bag[1]] = all_labels[bag[0]:bag[1]] + label[bag_index]
                bag_index += 1
            actual.append(all_labels)                     # 将标签加入所有标签的list中
            batch = batch[3:]                 # 【YangHP：“args”指的是什么？】
            token = batch[0].view(-1, batch[0].size(-1))
            pos1 = batch[1].view(-1, batch[1].size(-1))
            pos2 = batch[2].view(-1, batch[2].size(-1))
            if batch[3] is not None:
                mask = batch[3].view(-1, batch[3].size(-1))
            if mask is not None:
                rep = model.encoder(token, pos1, pos2, mask) # (nsum, H) 
            else:
                rep = model.encoder(token, pos1, pos2) # (nsum, H) 
            query = torch.zeros((rep.size(0))).long()
            if torch.cuda.is_available():
                query = query.cuda()
            for i in range(len(scope)):
                query[scope[i][0]:scope[i][1]] = label[i]
            att_mat = re_model.module.fc.weight[query] # (nsum, H)
            if re_model.module.use_diag:
                att_mat = att_mat * re_model.module.diag.unsqueeze(0)
            # att_score = (rep * att_mat).sum(-1) # (nsum)        rep.shape = [354, 690], att_mat.shape = [354, 690]
            feature = rep.mul(att_mat)    # (nsum)        rep.shape = [354, 690], att_mat.shape = [354, 690]
        elif return_actual:
            raise ValueError(
                "Dataset has no actual value to unpack, but return_actual is set."
            )
        # if cuda:
        #     actual[-1] = actual[-1].cuda(non_blocking=True)
        #     batch = batch.cuda(non_blocking=True)
        # feature = model.encoder(batch[0],batch[1],batch[2],batch[3]).t().mul(actual[-1]).t()
        # features.append(feature)
        result.append(
            model(feature).detach().cpu()
        )  # move to the CPU to prevent out of memory on the GPU
    if return_actual:
        # return torch.cat(result).max(1)[1], torch.cat(actual).long(), features
        return torch.cat(result).max(1)[1], torch.cat(actual).long()
    else:
        return torch.cat(result).max(1)[1]
