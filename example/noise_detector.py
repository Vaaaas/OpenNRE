import numpy as np
import torch

import sys, os, json
import logging
import opennre

# ---- [20211203] Added by YangHP, 导入需要的包 ------------------
from ptdec.dec import DEC
import time
import random
import matplotlib.pyplot as plt
# ==============================================================

# ---- [20211120] Added by YangHP, 设置命令行参数 ------------------------------------------------------
import argparse

# ---- [20211113] Added by YangHP, 忽略警告(warning)信息 ------------------------------------------
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
# ===============================================================================================

# ---- [20211231] Added by YangHP, 添加可视化训练过程的包 ------------------------------------------
from opennre.framework.visualProfiler import VisualProfilerConfig
# ===============================================================================================

def draw_PR_curve(precision, recall, label, t0):
    plt.plot(recall, precision, label=label)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 0.4])
    plt.ylim([0.3, 1.0])
    plt.grid(True)
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.savefig("result/P-R Curve-%s.png" %(t0), dpi=300)

# 设定随机数种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()

    # RE
    parser.add_argument('--ckpt', default='', 
            help='Checkpoint name')
    parser.add_argument('--dec_ckpt', default='dec_model', 
            help='DEC Checkpoint name')
    parser.add_argument('--result', default='result_denoise', 
            help='Save result name')
    parser.add_argument('--only_test', action='store_true', 
            help='Only run test')

    # Data
    parser.add_argument('--metric', default='auc', choices=['micro_f1', 'auc'],
            help='Metric for picking up best checkpoint')
    parser.add_argument('--dataset', default='nyt10', choices=['none', 'wiki_distant', 'nyt10', 'nyt10m', 'wiki20m'],
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
    parser.add_argument('--batch_size', default=160, type=int,
            help='Batch size')
    parser.add_argument('--lr', default=0.01, type=float,
            help='Learning rate')
    parser.add_argument('--optim', default='sgd', type=str,
            help='Optimizer')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
            help='Weight decay')
    parser.add_argument('--max_length', default=128, type=int,
            help='Maximum sentence length')
    parser.add_argument('--max_epoch', default=1, type=int,
            help='Max number of training epochs')
    parser.add_argument('--valid_t', default=0.5, type=float,
            help='Valid sample threshold')
    parser.add_argument('--noise_t', default=-0.5, type=float,
            help='Noise sample threshold')
    parser.add_argument('--lambda_param', default=0.6, type=float,
            help='Lambda parameter for noise Loss')

    # Others
    parser.add_argument('--seed', default=42, type=int,
            help='Random seed')

    # Exp
    parser.add_argument('--encoder', default='pcnn', choices=['pcnn', 'cnn'])
    parser.add_argument('--aggr', default='denoise', choices=['one', 'att', 'avg', 'denoise', 'denoiseC', 'DCRE'])

    args = parser.parse_args()

    return args

def main():
    # [001] ---- [20211128] Commented by YangHP, 设置并格式化命令行参数，设置TensorBoard等 ------------
    # 设置并格式化命令行参数
    args = parse_args()
	
    # Set random seed 设定随机数种子
    set_seed(args.seed)

    t0 = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = "log/{}_{}_{}_{}_{}_{}_{}.log".format(args.ckpt, args.batch_size, args.lr, args.valid_t, args.noise_t, args.lambda_param, t0)
    sample_numbers_log_path = "log/{}_{}_sample_numbers.log".format(args.ckpt, t0)
    labels_log_path = "log/{}_{}_labels.log".format(args.ckpt, t0)
    if not os.path.exists('log'):                              # 如果不存在log文件夹，则创建
        os.mkdir('log')
    
    log_file=open(log_path,'a+') 
    
    logging.basicConfig(filename=log_path, encoding='utf-8', level=logging.DEBUG)
    
    # Some basic settings 设定一些基本的设置
    root_path = '.'                                             # 设定根目录的路径
    sys.path.append(root_path)                                  # 将根目录添加到PATH中
    if not os.path.exists('ckpt'):                              # 如果不存在ckpt(checkpoint)文件夹，则创建
        os.mkdir('ckpt')
    if len(args.ckpt) == 0:                                     # 如果没有设定ckpt(checkpoint)文件名，则以当前数据集和网络结构为名称
        args.ckpt = '{}_{}_{}_{}_{}_{}_{}'.format(args.aggr, args.batch_size, args.lr, args.valid_t, args.noise_t, args.lambda_param, t0)
        # args.ckpt = '{}_{}'.format(args.dataset, 'pcnn_att')
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
    print('Arguments:', file=log_file)
    logging.info('    t0: {}'.format(t0))
    print('    t0: {}'.format(t0), file=log_file)
    for arg in vars(args):
        logging.info('    {}: {}'.format(arg, getattr(args, arg)))
        print('    {}: {}'.format(arg, getattr(args, arg)), file=log_file)
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
    elif args.aggr == 'denoise':
        model = opennre.model.BagDenoise(sentence_encoder, len(rel2id), rel2id)
    elif args.aggr == 'denoiseC':
        model = opennre.model.BagDenoiseClassifier(sentence_encoder, len(rel2id), rel2id)
    elif args.aggr == 'DCRE':
        model = opennre.model.BagDCRE(sentence_encoder, len(rel2id), rel2id)
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

    dec_model = None

    # 加载预训练的DEC模型
    if args.aggr == 'DCRE' or args.aggr == 'denoise':
        dec_model = DEC(cluster_number=47, hidden_dimension=690, encoder=framework.model.module.sentence_encoder)
        dec_model.load_state_dict(torch.load(dec_ckpt)['state_dict'])   # Copies parameters and buffers from :attr:`state_dict` into this module and its descendants.将预训练的参数权重加载到模型中
        dec_model = dec_model.cuda()
    log_file.close()

    # [004] ---- [20211212] Commented by YangHP, Train the model 训练模型 ----
    if not args.only_test:
        visualConfig = VisualProfilerConfig(os.path.join(root_path, "tensorboardXModuleTest"))
        framework.load_state_dict(torch.load("ckpt/nyt10_pcnn_denoise_160_0.01_0.5_-0.5_0.6_202202051910.pth.tar")['state_dict'])   # Copies parameters and buffers from :attr:`state_dict` into this module and its descendants.将预训练的参数权重加载到模型中    
        framework.train_model(args.metric, dec_model, tensorboardConfig=visualConfig, log_path=log_path, sample_numbers_log_path=sample_numbers_log_path, labels_log_path=labels_log_path, valid_t=args.valid_t, noise_t=args.noise_t, lambda_param=args.lambda_param)
    # [End of 004] ===========================================================

    # [005] ---- [20211204] Commented by YangHP, 读入预训练出的模型的参数 -----
    framework.load_state_dict(torch.load(ckpt)['state_dict'])   # Copies parameters and buffers from :attr:`state_dict` into this module and its descendants.将预训练的参数权重加载到模型中    
    result = framework.eval_model(framework.test_loader)
    # [End of 005] =========================================================

    # [006] ---- [20211128] Commented by YangHP, Print the result 输出结果 -------------------------------------------------------------------
    log_file=open(log_path,'a+') 
    logging.info('Test set results:')
    print('Test set results:', file=log_file)
    logging.info('AUC: %.5f' % (result['auc']))
    logging.info('Maximum micro F1: %.5f' % (result['max_micro_f1']))
    logging.info('Maximum macro F1: %.5f' % (result['max_macro_f1']))
    logging.info('Micro F1: %.5f' % (result['micro_f1']))
    logging.info('Macro F1: %.5f' % (result['macro_f1']))
    logging.info('P@100: %.5f' % (result['p@100']))
    logging.info('P@200: %.5f' % (result['p@200']))
    logging.info('P@300: %.5f' % (result['p@300']))
    print('AUC, Maximum micro F1, Maximum macro F1, Micro F1, Macro F1, P@100, P@200, P@300', file=log_file)
    print('%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f'%(result['auc'], result['max_micro_f1'], result['max_macro_f1'], 
        result['micro_f1'], result['macro_f1'], result['p@100'], result['p@200'], result['p@300']), file=log_file)
    log_file.close()
    # Save precision/recall points 保存“精度/召回”点
    # 若文件夹不存在，则创建文件夹
    if not os.path.exists('result'):
        os.makedirs('result')
    np.save('result/{}_p-{}.npy'.format(args.result, t0), result['np_prec'])
    np.save('result/{}_r-{}.npy'.format(args.result, t0), result['np_rec'])
    json.dump(result['max_micro_f1_each_relation'], open('result/{}_mmicrof1_rel-{}.json'.format(args.result, t0), 'w'), ensure_ascii=False)
    # [End of 006] =================================================================================================================================

    # [007] ---- [20211229] Commented by YangHP, S绘制“P-R”曲线 ----------
    precision = np.load('result/{}_p-{}.npy'.format(args.result, t0))
    recall = np.load('result/{}_r-{}.npy'.format(args.result, t0))

    draw_PR_curve(precision, recall, 'Denoise-%s'%t0, t0)
    # [End of 007] ======================================================

if __name__ == "__main__":
    main()
