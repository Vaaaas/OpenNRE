import torch
import time
from torch import nn, optim
from .base_model import BagRE

class BagDenoise(BagRE):
    """
    Instance attention for bag-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, rel2id, use_diag=True):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.fc = nn.Linear(self.sentence_encoder.hidden_size, num_class)
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        self.drop = nn.Dropout()
        for rel, id in rel2id.items():
            self.id2rel[id] = rel
        if use_diag:                                                                # diag: diagonal matrix，对角矩阵
            self.use_diag = True
            self.diag = nn.Parameter(torch.ones(self.sentence_encoder.hidden_size))
        else:
            self.use_diag = False

    def infer(self, bag):
        """
        Args:
            bag: bag of sentences with the same entity pair
                [{
                  'text' or 'token': ..., 
                  'h': {'pos': [start, end], ...}, 
                  't': {'pos': [start, end], ...}
                }]
        Return:
            (relation, score)
        """
        self.eval()
        tokens = []
        pos1s = []
        pos2s = []
        masks = []
        for item in bag:
            token, pos1, pos2, mask = self.sentence_encoder.tokenize(item)
            tokens.append(token)
            pos1s.append(pos1)
            pos2s.append(pos2)
            masks.append(mask)
        tokens = torch.cat(tokens, 0).unsqueeze(0)                                                      # (n, L)
        pos1s = torch.cat(pos1s, 0).unsqueeze(0)
        pos2s = torch.cat(pos2s, 0).unsqueeze(0)
        masks = torch.cat(masks, 0).unsqueeze(0)
        scope = torch.tensor([[0, len(bag)]]).long()                                                    # (1, 2)
        bag_logits = self.forward(None, scope, tokens, pos1s, pos2s, masks, train=False).squeeze(0)     # (N) after softmax
        score, pred = bag_logits.max(0)
        score = score.item()
        pred = pred.item()
        rel = self.id2rel[pred]
        return (rel, score)
    
    def forward(self, label, scope, token, pos1, pos2, mask=None, train=True, bag_size=0, dec_model=None, valid_t=0.5, noise_t=-0.5, lambda_param=0.6, sample_numbers_log_path=None, labels_log_path=None):
        """
        Args:
            label: (B), label of the bag
            scope: (B), scope for each bag
            token: (nsum, L), index of tokens
            pos1: (nsum, L), relative position to head entity
            pos2: (nsum, L), relative position to tail entity
            mask: (nsum, L), used for piece-wise CNN
        Return:
            logits, (B, N)

        Dirty hack:
            When the encoder is BERT, the input is actually token, att_mask, pos1, pos2, but
            since the arguments are then fed into BERT encoder with the original order,
            the encoder can actually work out correclty.

        nsum: number of sentences in the bag
        B: batch_size
        L: max_length
        """
        # [001] ---- [20211211] Commented by YangHP, 对词嵌入做变换 -----------
        """
            token.shape = [1, nsum, L]
            pos1.shape = [1, nsum, L]
            pos2.shape = [1, nsum, L]
            mask.shape = [1, nsum, L]
        """
        if bag_size > 0:
            token = token.view(-1, token.size(-1))
            pos1 = pos1.view(-1, pos1.size(-1))
            pos2 = pos2.view(-1, pos2.size(-1))
            if mask is not None:
                mask = mask.view(-1, mask.size(-1))
        else:                                                                   # bag_size = 0
            begin, end = scope[0][0], scope[-1][1]
            token = token[:, begin:end, :].view(-1, token.size(-1))
            pos1 = pos1[:, begin:end, :].view(-1, pos1.size(-1))
            pos2 = pos2[:, begin:end, :].view(-1, pos2.size(-1))
            if mask is not None:
                mask = mask[:, begin:end, :].view(-1, mask.size(-1))
            scope = torch.sub(scope, torch.zeros_like(scope).fill_(begin))
        """
            token.shape = [nsum, L]
            pos1.shape = [nsum, L]
            pos2.shape = [nsum, L]
            mask.shape = [nsum, L]
            len(scope) = B, scope[-1][1] = nsum
        """
        # ===================================================================

        # [002] ---- [20211211] Commented by YangHP, 注意力 -----------------------------------------------------------------
        # Attention
        if train:
            # Train
            if mask is not None:
                rep = self.sentence_encoder(token, pos1, pos2, mask)                        # (nsum, H)
            else:
                rep = self.sentence_encoder(token, pos1, pos2)                              # (nsum, H) 

            sample_numbers_data=open(sample_numbers_log_path,'a+')
            labels_data=open(labels_log_path,'a+')

            if bag_size == 0:
                bag_rep = []
                bag_noise_loss = []

                query = torch.zeros((rep.size(0))).long()                                   # (nsum)
                if torch.cuda.is_available():
                    query = query.cuda()
                for i in range(len(scope)):
                    query[scope[i][0]:scope[i][1]] = label[i]
                
                att_mat = self.fc.weight[query]                                             # (nsum, H), self.fc.weight.shape = (58, H), 注意力矩阵
                if self.use_diag:
                    att_mat = att_mat * self.diag.unsqueeze(0)                              # att_mat.shape = [nsum, H]，att_mat即为文献[156]"Noise Detector"中的L（标签表征）
                att_score = (rep * att_mat).sum(-1)                                         # (nsum) ———— 其中的元素即文献[156]"Noise Detector"中的a_i

                for i in range(len(scope)):                                                 # 遍历每个句袋
                    valid_in_bag = 0
                    noise_in_bag_all = 0
                    noise_in_bag_not_zero = 0


                    bag_mat = rep[scope[i][0]:scope[i][1]]                                  # (n, H) 句袋表征矩阵，句子个数 * H
                    this_att_score = att_score[scope[i][0]:scope[i][1]]

                    # 若句袋中只有一个样本，则直接取改句的表征作为袋表征
                    if len(bag_mat) == 1:
                        bag_rep.append(bag_mat)
                        bag_noise_loss.append(0)
                    else:
                        # 计算匹配度的Z-值
                        z_att_score = (this_att_score - this_att_score.mean()) / this_att_score.std()

                        # 从z_att_score中找出大于有效阈值的所有值的索引
                        valid_index = torch.nonzero(z_att_score > valid_t).squeeze(1)

                        # 从z_att_score中找出小于噪声阈值的所有值的索引
                        noise_index = torch.nonzero(z_att_score < noise_t).squeeze(1)
                        # 计算出所有句子属于各类别的概率
                        instance_logit = self.softmax(self.fc(bag_mat))                         # (n, N) 

                        # 若无有效样本，从z_att_score中找出最大值的索引
                        # 判断valid_index是否为空
                        if valid_index.size(0) == 0:
                            # 从z_att_score中找出最大值的索引
                            valid_index = torch.LongTensor([torch.argmax(z_att_score)]).cuda()
                            # 从noise_index中删除valid_index中出现的元素    TODO: 有待改进，提高计算效率
                            noise_index = [i for i in noise_index if i not in valid_index]
                            noise_index = torch.LongTensor(noise_index)
                            # noise_index = torch.Tensor([i for i in noise_index if i not in valid_index])

                        if valid_index.size(0) == 1:
                            # 只有一个有效样本
                            bag_rep.append(bag_mat[valid_index])
                            valid_in_bag = 1
                        else:
                            # 计算匹配度的softmax标准化值(在所有有效样本的范围内计算softmax标准化)
                            softmax_att_score = self.softmax(this_att_score[valid_index])    # (n)  softmax标准化后的匹配度，每个句子有一个匹配度
                            # 由有效样本计算出袋表征
                            bag_rep.append(torch.matmul(softmax_att_score, bag_mat[valid_index]).unsqueeze(0))
                            valid_in_bag = len(valid_index)

                        # 若存在噪声样本，且原签不为“NA”
                        if len(noise_index) > 0 and label[i] != 0:
                            noise_in_bag_all = len(noise_index)
                            # 判断bag_mat[noise_index]是否为1维张量
                            if len(bag_mat[noise_index].unsqueeze(0).shape) == 1:
                                noise_mat = bag_mat[noise_index].unsqueeze(0).cuda()                        # (n, H)
                            else:
                                noise_mat = bag_mat[noise_index].cuda()                                     # (n, H)
                            
                            cluster_result = dec_model(noise_mat)                                           # (n, N) 模型的输出即为q_ij
                            new_labels = cluster_result.max(1)[1]                                           # (n) 样本相应的聚类签

                            # 找出new_labels中非0的值的索引
                            non_zero_index = torch.nonzero(new_labels).squeeze(1)                           # 被聚为“NA”的样本须排除掉
                            noise_in_bag_not_zero = len(non_zero_index)

                            noise_mat = noise_mat[non_zero_index]                                           # (n', H)
                            new_labels = new_labels[non_zero_index]                                         # (n')
                            cluster_result = cluster_result[non_zero_index, new_labels]                     # (n', N)

                            noise_y_j_logit = instance_logit[non_zero_index, new_labels]                   # (n')
                            print("{},{}".format(label[i],new_labels))
                            print("{},{}".format(label[i],new_labels), file=labels_data)

                            # 计算损失函数中与噪声样本相关的项
                            # 计算noise_y_j_logist与cluster_result的乘积
                            noise_logits = - lambda_param * torch.matmul(torch.log(noise_y_j_logit), cluster_result).sum()

                            bag_noise_loss.append(noise_logits)
                        else:
                            bag_noise_loss.append(0)
                        print("{},{},{},{},{}".format(len(bag_mat), valid_in_bag, noise_in_bag_all, noise_in_bag_not_zero, len(bag_mat)-valid_in_bag-noise_in_bag_all))
                        print("{},{},{},{},{}".format(len(bag_mat), valid_in_bag, noise_in_bag_all, noise_in_bag_not_zero, len(bag_mat)-valid_in_bag-noise_in_bag_all), file=sample_numbers_data)
                bag_rep = torch.stack(bag_rep, 0).squeeze(1)                                                # (B, H)
                bag_noise_loss = torch.tensor(bag_noise_loss).sum() / len(bag_noise_loss)
            else:
                batch_size = label.size(0)
                query = label.unsqueeze(1)                                                                  # (B, 1)
                att_mat = self.fc.weight[query]                                                             # (B, 1, H)
                if self.use_diag:
                    att_mat = att_mat * self.diag.unsqueeze(0)
                rep = rep.view(batch_size, bag_size, -1)
                att_score = (rep * att_mat).sum(-1)                                                         # (B, bag)
                softmax_att_score = self.softmax(att_score)                                                 # (B, bag)
                # 将句袋中所有加权后的句表征相加，得到一个句表征
                bag_rep = (softmax_att_score.unsqueeze(-1) * rep).sum(1)                                    # (n, 1) * (n, H) -> (n, H) -> (H)  (rep_h)
            sample_numbers_data.close()
            labels_data.close()

            bag_rep = self.drop(bag_rep)                                                                    # (B, H)
            bag_logits = self.fc(bag_rep)                                                                   # (B, N)
        else:
            # Evaluate
            bag_noise_loss = []
            if bag_size == 0:
                rep = []
                bs = 256
                total_bs = len(token) // bs + (1 if len(token) % bs != 0 else 0)

                # 将token分成total_bs个小的batch来计算句表征
                for b in range(total_bs):
                    with torch.no_grad():
                        left = bs * b
                        right = min(bs * (b + 1), len(token))
                        if mask is not None:        
                            rep.append(self.sentence_encoder(token[left:right], pos1[left:right], pos2[left:right], mask[left:right]).detach()) # (nsum, H) 
                        else:
                            rep.append(self.sentence_encoder(token[left:right], pos1[left:right], pos2[left:right]).detach())                   # (nsum, H) 
                rep = torch.cat(rep, 0)                                                                                                         # (nSent, H) 全部句表征

                bag_logits = []
                att_mat = self.fc.weight.transpose(0, 1)                                                        # (H, N) 获取标签表征后，将维度从(N, H)变为(H, N)，为了后面的点积计算方便
                if self.use_diag:
                    att_mat = att_mat * self.diag.unsqueeze(1)
                att_score = torch.matmul(rep, att_mat)                                                          # (nsum, H) * (H, N) -> (nsum, N)
                
                for i in range(len(scope)):
                    bag_mat = rep[scope[i][0]:scope[i][1]]                                                      # (n, H)

                    # TODO: 是否需要修改测试部分的袋表征逻辑？
                    softmax_att_score = self.softmax(att_score[scope[i][0]:scope[i][1]].transpose(0, 1))        # (N, (softmax)n) 
                    rep_for_each_rel = torch.matmul(softmax_att_score, bag_mat)                                 # (N, n) * (n, H) -> (N, H)
                    logit_for_each_rel = self.softmax(self.fc(rep_for_each_rel))                                # ((each rel)N, (logit)N)
                    
                    logit_for_each_rel = logit_for_each_rel.diag()                                              # (N)
                    bag_logits.append(logit_for_each_rel)
                bag_logits = torch.stack(bag_logits, 0) # after **softmax**
            else:
                if mask is not None:
                    rep = self.sentence_encoder(token, pos1, pos2, mask)                            # (nsum, H) 
                else:
                    rep = self.sentence_encoder(token, pos1, pos2)                                  # (nsum, H) 

                batch_size = rep.size(0) // bag_size
                att_mat = self.fc.weight.transpose(0, 1)
                if self.use_diag:
                    att_mat = att_mat * self.diag.unsqueeze(1) 
                att_score = torch.matmul(rep, att_mat)                                              # (nsum, H) * (H, N) -> (nsum, N)
                att_score = att_score.view(batch_size, bag_size, -1)                                # (B, bag, N)
                rep = rep.view(batch_size, bag_size, -1)                                            # (B, bag, H)
                softmax_att_score = self.softmax(att_score.transpose(1, 2))                         # (B, N, (softmax)bag)
                rep_for_each_rel = torch.matmul(softmax_att_score, rep)                             # (B, N, bag) * (B, bag, H) -> (B, N, H)
                bag_logits = self.softmax(self.fc(rep_for_each_rel)).diagonal(dim1=1, dim2=2)       # (B, (each rel)N)
        return bag_logits, bag_noise_loss   # (B, H)
