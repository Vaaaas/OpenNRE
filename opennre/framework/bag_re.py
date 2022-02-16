import torch
from torch import nn, optim
import json
from .data_loader import SentenceRELoader, BagRELoader
from .utils import AverageMeter
from .visualProfiler import VisualProfilerConfig
from datetime import datetime
from tqdm import tqdm
import logging

class BagRE(nn.Module):

    def __init__(self, 
                 model,
                 train_path, 
                 val_path, 
                 test_path,
                 ckpt, 
                 batch_size=32, 
                 max_epoch=100, 
                 lr=0.1, 
                 weight_decay=1e-5, 
                 opt='sgd',
                 bag_size=0,
                 loss_weight=False):
    
        super().__init__()
        self.max_epoch = max_epoch
        self.bag_size = bag_size
        # Load data
        if train_path != None:
            self.train_loader = BagRELoader(
                train_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                True,
                bag_size=bag_size,
                entpair_as_bag=False)
            self.static_loader = BagRELoader(
                train_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False,
                bag_size=bag_size,
                entpair_as_bag=True)

        if val_path != None:
            self.val_loader = BagRELoader(
                val_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False,
                bag_size=bag_size,
                entpair_as_bag=True)
        
        if test_path != None:
            self.test_loader = BagRELoader(
                test_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False,
                bag_size=bag_size,
                entpair_as_bag=True
            )
        # Model
        self.model = nn.DataParallel(model)
        # Criterion
        if loss_weight:
            self.criterion = nn.CrossEntropyLoss(weight=self.train_loader.dataset.weight)
        else:
            self.criterion = nn.CrossEntropyLoss()
        # Params and optimizer
        params = self.model.parameters()
        self.lr = lr
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == 'adamw':
            from transformers import AdamW
            params = list(self.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.01,
                    'lr': lr,
                    'ori_lr': lr
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.0,
                    'lr': lr,
                    'ori_lr': lr
                }
            ]
            self.optimizer = AdamW(grouped_params, correct_bias=False)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'bert_adam'.")
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt

    def train_model(self, metric='auc', dec_model = None, tensorboardConfig:VisualProfilerConfig=None, log_path=None, sample_numbers_log_path= None, labels_log_path=None, valid_t=0.5, noise_t=-0.5, lambda_param=0.6):
        best_metric = 0      # 初始化指标的最优值
        # tensorboardX
        if tensorboardConfig is not None:
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(log_dir=tensorboardConfig.writer_dir+'/'+str(datetime.now()))
            for epoch in range(self.max_epoch):
                # Train
                global_step = 0
                self.train()
                print("=== Epoch %d train ===" % epoch)
                avg_loss = AverageMeter()           # 初始化平均损失值
                avg_acc = AverageMeter()            # 初始化平均准确度
                avg_pos_acc = AverageMeter()        # 初始化平均位置准确度
                t = tqdm(self.train_loader)         # Tqdm 是一个快速，可扩展的Python进度条
                for iter, data in enumerate(t):     # enumerate()枚举函数
                    if torch.cuda.is_available():   # 判断PyTorch能否调用GPU
                        for i in range(len(data)):
                            try:
                                data[i] = data[i].cuda()
                            except:
                                pass
                    label = data[0]                 # (B) 标签      Tensor
                    bag_name = data[1]              # (B) 袋名      list    (bag_name[0] = ["<头实体代码>", "<尾实体代码>", "<联系名>"])
                    scope = data[2]                 # (B, 2) 作用域 Tensor  ，指每个句袋的范围，句袋的起始句索引和末尾句索引
                    args = data[3:]                 # (4) 句表征    list    args[0].shape = [1, nsum, L], args = [token, pos1, pos2, mask]
                    # 这里计算损失，仍然是以袋为单位计算的
                    logits, noise_loss = self.model(label, scope, *args, bag_size=self.bag_size, dec_model=dec_model, valid_t=valid_t, noise_t=noise_t, lambda_param=lambda_param, sample_numbers_log_path=sample_numbers_log_path, labels_log_path=labels_log_path)    # (B, C) 最后一个全连接层的输出
                    loss = self.criterion(logits, label) + noise_loss                   # (B)
                    score, pred = logits.max(-1)                                        # score(B)为每个样本的最大的分，pred(B)为每个样本的预测结果（相应索引）
                    acc = float((pred == label).long().sum()) / label.size(0)
                    pos_total = (label != 0).long().sum()                               # 将所有元素转为整数并求和，即为非“NA”联系的样本总数
                    pos_correct = ((pred == label).long() * (label != 0).long()).sum()  # 非“NA”联系的样本的正确总数
                    if pos_total > 0:
                        pos_acc = float(pos_correct) / float(pos_total)
                    else:
                        pos_acc = 0

                    # Log   日志更新
                    avg_loss.update(loss.item(), 1)       # 更新平均损失值
                    avg_acc.update(acc, 1)                # 更新准确度
                    avg_pos_acc.update(pos_acc, 1)        # 更新位置准确度
                    t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg, pos_acc=avg_pos_acc.avg)
                    
                    # # TensorboardX
                    # writer.add_scalar("train/loss", loss.item(), global_step)
                    # writer.add_scalar("train/accuracy", acc, global_step)
                    
                    # Optimize   优化
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                # Val 
                print("=== Epoch %d val ===" % epoch)
                result = self.eval_model(self.val_loader)    # 评估结果
                print("AUC: %.4f" % result['auc'])
                print("Micro F1: %.4f" % (result['max_micro_f1']))

                data=open(log_path,'a+')

                print("迭代次数,平均损失, 平均pos_acc, AUC, max_micro_f1")
                print("迭代次数,平均损失, 平均pos_acc, AUC, max_micro_f1", file=data)
                print("%d, %.4f, %.4f, %.4f, %.4f" % (epoch, avg_loss.avg, avg_pos_acc.avg, result['auc'], result['max_micro_f1']))
                print("%d, %.4f, %.4f, %.4f, %.4f" % (epoch, avg_loss.avg, avg_pos_acc.avg, result['auc'], result['max_micro_f1']), file=data)
                # torch.save({'state_dict': self.model.module.state_dict()}, "ckpt/epoch-%d.pth.tar"%(epoch))  # 存储文件
                if result[metric] > best_metric:
                    print("Best ckpt and saved.")
                    print("Best ckpt and saved.", file=data)
                    torch.save({'state_dict': self.model.module.state_dict()}, self.ckpt)  # 存储文件
                    best_metric = result[metric]      # 最好的指标
                # tensorboard
                writer.add_scalars("train/%s"%("loss_avg"), {
                    "loss_avg": avg_loss.avg
                }, epoch)
                writer.add_scalars("train/%s"%("pos_acc_avg"), {
                    "pos_acc_avg": avg_pos_acc.avg
                }, epoch)
                writer.add_scalars("validation/%s"%(metric), {
                    "%s"%(metric): result[metric],
                    "best_%s"%(metric): best_metric
                }, epoch)
                writer.add_scalars("validation/F1", {
                    "micro_f1": result['micro_f1']
                }, epoch)
                # 若当前的result[metric]与best_metric的差距大于等于0.005，则停止训练
                # if (result[metric] - best_metric) <= -0.005:
                    # print("Early stopping.")
                    # print("Early stopping.", file=data)
                    # break
                data.close()
                
            data=open(log_path,'a+') 
            print("Best %s on val set: %f" % (metric, best_metric))
            print("Best %s on val set: %f" % (metric, best_metric), file=data)
            data.close()
    
    def eval_model(self, eval_loader):
        self.model.eval()
        with torch.no_grad():
            t = tqdm(eval_loader)
            pred_result = []
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                bag_name = data[1]
                scope = data[2]
                args = data[3:]
                logits = self.model(label, scope, *args, bag_size=self.bag_size)
                loss = self.criterion(logits, label)
                score, pred = logits.max(-1) # (B)
                acc = float((pred == label).long().sum()) / label.size(0)
                pos_total = (label != 0).long().sum()
                pos_correct = ((pred == label).long() * (label != 0).long()).sum()
                if pos_total > 0:
                    pos_acc = float(pos_correct) / float(pos_total)
                else:
                    pos_acc = 0

                # Log
                avg_loss.update(loss.item(), 1)
                avg_acc.update(acc, 1)
                avg_pos_acc.update(pos_acc, 1)
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg, pos_acc=avg_pos_acc.avg)
                
                # Optimize
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Val 
            print("=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.val_loader)
            print("AUC: %.4f" % result['auc'])
            print("Micro F1: %.4f" % (result['max_micro_f1']))
            if result[metric] > best_metric:
                print("Best ckpt and saved.")
                torch.save({'state_dict': self.model.module.state_dict()}, self.ckpt)
                best_metric = result[metric]
        print("Best %s on val set: %f" % (metric, best_metric))

    def eval_model(self, eval_loader):
        self.model.eval()
        with torch.no_grad():
            t = tqdm(eval_loader)
            pred_result = []
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                bag_name = data[1]
                scope = data[2]
                args = data[3:]
                logits = self.model(None, scope, *args, train=False, bag_size=self.bag_size) # results after softmax
                logits = logits.cpu().numpy()
                for i in range(len(logits)):
                    for relid in range(self.model.module.num_class):
                        if self.model.module.id2rel[relid] != 'NA':
                            pred_result.append({
                                'entpair': bag_name[i][:2], 
                                'relation': self.model.module.id2rel[relid], 
                                'score': logits[i][relid]
                            })
            result = eval_loader.dataset.eval(pred_result)
        return result

    def load_state_dict(self, state_dict):
        self.model.module.load_state_dict(state_dict)
