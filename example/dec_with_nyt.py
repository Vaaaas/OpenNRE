import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.optim import SGD
import torch
import uuid

from ptdec.dec import DEC
from ptdec.model import train, predict
from ptdec.utils import cluster_accuracy

import sys, os, json
import logging
import opennre
import time

# ---- [20211120] Added by YangHP, 设置命令行参数 ------------------------------------------------------
import argparse
def parse_args():
    parser = argparse.ArgumentParser()

    # DEC
    parser.add_argument(
        "--cuda", help="whether to use CUDA (default False).", type=bool, default=True
    )
    parser.add_argument(
        "--testing_mode",
        help="whether to run in testing mode (default False).",
        type=bool,
        default=False
    )

    # RE
    parser.add_argument('--ckpt', default='nyt10_pcnn_att', 
            help='Checkpoint name')
    parser.add_argument('--dec_ckpt', default='dec_model', 
            help='DEC Checkpoint name')
    parser.add_argument('--result', default='', 
            help='Save result name')

    # Data
    parser.add_argument('--metric', default='auc', choices=['micro_f1', 'auc'],
            help='Metric for picking up best checkpoint')
    parser.add_argument('--dataset', default='none', choices=['none', 'wiki_distant', 'nyt10', 'nyt10m', 'wiki20m'],
            help='Dataset. If not none, the following args can be ignored')
    parser.add_argument('--train_file', default='', type=str,
            help='Training data file')
    parser.add_argument('--val_file', default='', type=str,
            help='Validation data file')
    parser.add_argument('--test_file', default='', type=str,
            help='Test data file')
    parser.add_argument('--rel2id_file', default='', type=str,
            help='Relation to ID file')

    # Bag related
    parser.add_argument('--bag_size', type=int, default=0,
            help='Fixed bag size. If set to 0, use original bag sizes')

    # Hyper-parameters
    parser.add_argument('--batch_size', default=64, type=int,
            help='Batch size')
    parser.add_argument('--lr', default=0.1, type=float,
            help='Learning rate')
    parser.add_argument('--optim', default='sgd', type=str,
            help='Optimizer')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
            help='Weight decay')
    parser.add_argument('--max_length', default=128, type=int,
            help='Maximum sentence length')
    parser.add_argument('--max_epoch', default=70, type=int,
            help='Max number of training epochs')

    # Others
    parser.add_argument('--seed', default=42, type=int,
            help='Random seed')

    # Exp
    parser.add_argument('--encoder', default='pcnn', choices=['pcnn', 'cnn'])
    parser.add_argument('--aggr', default='att', choices=['one', 'att', 'avg'])

    args = parser.parse_args()

    return args

def main():
    # [001] ---- [20211128] Commented by YangHP, 设置并格式化命令行参数，设置TensorBoard等 ------------
    print("DEC with NYT_10 started at {}".format(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))))
    # 设置并格式化命令行参数
    args = parse_args()
    
    # Some basic settings 设定一些基本的设置
    root_path = '.'                                             # 设定根目录的路径
    sys.path.append(root_path)
    if not os.path.exists('ckpt'):                              # 如果不存在ckpt(checkpoint)文件夹，则创建
        os.mkdir('ckpt')
    if len(args.ckpt) == 0:                                     # 如果没有设定ckpt(checkpoint)文件名，则以当前数据集和网络结构为名称
        args.ckpt = '{}_{}'.format(args.dataset, 'pcnn_att')
    ckpt = 'ckpt/{}.pth.tar'.format(args.ckpt)                  # 设定ckpt(checkpoint)路径

    if len(args.dec_ckpt) == 0:                                     # 如果没有设定ckpt(checkpoint)文件名，则以当前数据集和网络结构为名称
        args.dec_ckpt = '{}_{}'.format(args.dataset, 'dec_model')
    dec_ckpt = 'ckpt/{}.pth.tar'.format(args.dec_ckpt)                  # 设定ckpt(checkpoint)路径
    # [End of 001] ================================================================================

    # [002] ---- [20211204] Added by YangHP, 准备数据 --------------------------------------------------------------------------------------------------------------------

    # [002-1] ---- [20211204] Added by YangHP, 下载并设置数据集路径 --------------------------------------------------------------------------------------------------
    if args.dataset != 'none':
        opennre.download(args.dataset, root_path=root_path)                                                         # 自动下载数据集
        args.train_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_train.txt'.format(args.dataset))   # 设定训练数据集路径
        args.val_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_val.txt'.format(args.dataset))       # 设定验证数据集路径
        if not os.path.exists(args.val_file):                                                                       # 如果不存在验证数据集，则使用测试数据集作为验证数据集
            logging.info("Cannot find the validation file. Use the test file instead.")
            args.val_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_test.txt'.format(args.dataset))
        args.test_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_test.txt'.format(args.dataset))     # 设定测试数据集路径
        args.rel2id_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_rel2id.json'.format(args.dataset))# 设定“联系到ID”映射文件的路径
    else:
        if not (os.path.exists(args.train_file) and os.path.exists(args.val_file) and os.path.exists(args.test_file) and os.path.exists(args.rel2id_file)):
            raise Exception('--train_file, --val_file, --test_file and --rel2id_file are not specified or files do not exist. Or specify --dataset')
    # [End of 002-1] ==============================================================================================================================================

    # [002-2] ---- [20211204] Commented by YangHP, 输出所有参数 -------------
    logging.info('Arguments:')
    for arg in vars(args):
        logging.info('    {}: {}'.format(arg, getattr(args, arg)))
    # [End of 002-2] ======================================================

    # [002-3] ---- [20211204] Commented by YangHP, 读取“联系到ID”的映射文件 -----
    rel2id = json.load(open(args.rel2id_file))
    # [End of 002-3] =========================================================

    # [002-4] ---- [20211204] Commented by YangHP, Download and load glove 下载并读取glove词向量模型 ----------
    opennre.download('glove', root_path=root_path)
    word2id = json.load(open(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_word2id.json')))
    word2vec = np.load(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_mat.npy'))
    # [End of 002-4] =======================================================================================

    # [End of 002] =======================================================================================================================================================

    # [003] ---- [20211204] Commented by YangHP, 模型定义与实例化 ------------------------------------------------------------------------------

    # [003-1] ---- [20211204] Commented by YangHP, Define and instantiate the sentence encoder 定义并实例化句编码器 ---------------
    if args.encoder == 'pcnn':
        # 初始化PCNN句编码器
        sentence_encoder = opennre.encoder.PCNNEncoder(
            token2id=word2id,
            max_length=args.max_length,
            word_size=50,
            position_size=5,
            hidden_size=230,
            blank_padding=True,
            kernel_size=3,
            padding_size=1,
            word2vec=word2vec,
            dropout=0.5
        )
    elif args.encoder == 'cnn':
        sentence_encoder = opennre.encoder.CNNEncoder(
            token2id=word2id,
            max_length=args.max_length,
            word_size=50,
            position_size=5,
            hidden_size=230,
            blank_padding=True,
            kernel_size=3,
            padding_size=1,
            word2vec=word2vec,
            dropout=0.5
        )
    else:
        raise NotImplementedError
    # [End of 003-1] =============================================================================================================

    # [003-2] ---- [20211204] Commented by YangHP, Define and instantiate the model 定义并实例化模型 -----------
    if args.aggr == 'att':
        model = opennre.model.BagAttention(sentence_encoder, len(rel2id), rel2id)
    elif args.aggr == 'avg':
        model = opennre.model.BagAverage(sentence_encoder, len(rel2id), rel2id)
    elif args.aggr == 'one':
        model = opennre.model.BagOne(sentence_encoder, len(rel2id), rel2id)
    else:
        raise NotImplementedError
    # [End of 003-2] =========================================================================================

    # [003-3] ---- [20211204] Commented by YangHP, Define and instantiate the whole training framework 定义并实例化整个训练框架 ----
    framework = opennre.framework.BagRE(
        train_path=args.train_file,
        val_path=args.val_file,
        test_path=args.test_file,
        model=model,
        ckpt=ckpt,
        batch_size=args.batch_size,
        max_epoch=args.max_epoch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        opt=args.optim,
        bag_size=args.bag_size)
    # [End of 003-3] ============================================================================================================
    # [End of 003] =====================================================================================================================

    # [004] ---- [20211204] Commented by YangHP, 读入预训练出的模型的参数 -----
    framework.load_state_dict(torch.load(ckpt)['state_dict'])   # Copies parameters and buffers from :attr:`state_dict` into this module and its descendants.将预训练的参数权重加载到模型中
    # [End of 004] =========================================================
    
    # [005] ---- [20211204] Commented by YangHP, DEC 阶段 ----------------------------------------
    print("DEC stage.")

    # 实例化DEC对象    
    # ---- [20211203] Commented by YangHP, hidden_dimension要等于编码器输出的句子表征的维数 -------------------------
    dec_model = DEC(cluster_number=47, hidden_dimension=690, encoder=framework.model.module.sentence_encoder)
    # ==========================================================================================================

    if args.cuda:
        dec_model.cuda()

    # 设置DEC的优化器
    dec_optimizer = SGD(dec_model.parameters(), lr=0.004, momentum=0.9)

    # 训练DEC
    if not args.testing_mode:
        train(
            static_dataloader=framework.static_loader,
            dataloader=framework.train_loader,
            re_model=framework.model,
            # dataset=test_dataset,
            model=dec_model,
            epochs=args.max_epoch,
            batch_size=args.batch_size,
            optimizer=dec_optimizer,
            stopping_delta=0.000001,
            cuda=args.cuda,
            ckpt = dec_ckpt,
            # update_callback=training_callback
        )
        print("DEC training ended at {}".format(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))))

    dec_model.load_state_dict(torch.load(dec_ckpt)['state_dict'])   # Copies parameters and buffers from :attr:`state_dict` into this module and its descendants.将预训练的参数权重加载到模型中

    # 测试DEC，并获取结果
    print("DEC testing started at {}".format(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))))
    predicted, actual,features = predict(
        framework.test_loader, framework.model, dec_model, 1024, silent=True, return_actual=True, cuda=args.cuda
    )
    print("DEC testing ended at {}".format(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))))

    print("=====================")
    print(features)
    print("=====================")

    # [End of 005] ==============================================================================

    # [006] ---- [20211128] Commented by YangHP, 计算并输出准确率 ----------------------------------
    actual = actual.cpu().numpy()
    predicted = predicted.cpu().numpy()
    # 计算准确率
    reassignment, accuracy = cluster_accuracy(actual, predicted)
    print("Final DEC accuracy: %s" % accuracy)
    if not args.testing_mode:
        predicted_reassigned = [
            reassignment[item] for item in predicted
        ]  # TODO numpify
        confusion = confusion_matrix(actual, predicted_reassigned)
        normalised_confusion = (
            confusion.astype("float") / confusion.sum(axis=1)[:, np.newaxis]
        )
        confusion_id = uuid.uuid4().hex
        sns.heatmap(normalised_confusion).get_figure().savefig(
            "confusion_%s.png" % confusion_id
        )
        print("Writing out confusion diagram with UUID: %s" % confusion_id)
        # writer.close()
    # visualize clusters using t-SNE
    # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300).fit_transform(predicted)

    print("DEC with NYT_10 ended at {}".format(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))))
    # [End of 006] ==============================================================================

if __name__ == "__main__":
    main()
