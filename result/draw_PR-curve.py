import numpy as np
import os, sys
# os.chdir(sys.path[0])

root_path = "result/"

def read_npy(path):
    return np.load(path)

# YangHPs
# t01 = "202201040152"
# t01 = "202202051910"
t01 = "202202100125"
# PCNN+ATT
# t02 = "202201100258"
# t02 = "202202090034"
t02 = "202202100710"
# "DCRE"
# t03 = "202202070028"
t03 = "202202100111"

precision1 = np.load('{}result_denoise_p-{}.npy'.format(root_path, t01))
recall1 = np.load('{}result_denoise_r-{}.npy'.format(root_path, t01))
label1 = "LSCRE"

precision2 = np.load('{}result_pcnn_att_p-{}.npy'.format(root_path, t02))
recall2 = np.load('{}result_pcnn_att_r-{}.npy'.format(root_path, t02))
label2 = "PCNN+ATT"

precision3 = np.load('{}result_denoise_p-{}.npy'.format(root_path, t03))
recall3 = np.load('{}result_denoise_r-{}.npy'.format(root_path, t03))
label3 = "DCRE"

## 绘制PR曲线
import matplotlib.pyplot as plt
import time

def draw_PR_curve(precision1, recall1, precision2, recall2, precision3, recall3, label1, label2, label3):
    #用3次多项式拟合
    f1 = np.polyfit(recall1, precision1, 10)
    p1 = np.poly1d(f1)
    print(p1)#打印出拟合函数
    yvals1 = p1(recall1)  #拟合y值
    
    f2 = np.polyfit(recall2, precision2, 10)
    p2 = np.poly1d(f2)
    print(p2)
    #也可使用yvals=np.polyval(f1, x)
    yvals2 = p2(recall2)

    f3 = np.polyfit(recall3, precision3, 10)
    p3 = np.poly1d(f3)
    print(p3)
    #也可使用yvals=np.polyval(f1, x)
    yvals3 = p3(recall3)
    
    # 绘制虚线

    #绘图
    plt.plot(recall2, precision2, label=label2, color="#0343df", alpha=0.8)
    plt.plot(recall3, precision3, label=label3, color="#2dfffe", alpha=0.8)
    plt.plot(recall1, precision1, label=label1, color="#f000f0", alpha=0.8)
    # plt.plot(recall2, yvals2, '#0243d9', alpha=0.6, label='PCNN+ATT 10th polyfit', linestyle="--")
    # plt.plot(recall3, yvals3, '#66B266', alpha=0.6, label='DCRE 10th polyfit', linestyle="--")
    # plt.plot(recall1, yvals1, '#f900f9', alpha=0.6, label='YangHPs 10th polyfit', linestyle="--")
    
    # plt.rcParams["font.sans-serif"]=["Microsoft YaHei"] #设置字体
    # plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 0.4])
    plt.ylim([0.3, 1.0])
    plt.grid(True)
    # plt.title('Fig 4. P-R Curve')
    plt.legend(loc="upper right")
    # plt.show()
    plt.savefig("%sP-R Curve-vs-PCNN+ATT-DCRE-part-oldDEC.png"%(root_path), dpi=300)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.savefig("%sP-R Curve-vs-PCNN+ATT-DCRE-all-oldDEC.png"%(root_path), dpi=300)

draw_PR_curve(precision1, recall1, precision2, recall2, precision3, recall3, label1, label2, label3)



# result_tags = [
#     "YangHPs",
#     "PCNN+ATT",
#     "DCRE"
# ]

# t0s = [
#     "202201040152",
#     "202201100258",
#     "202201300413"
# ]

# precisions = []
# recalls = []
# labels = []
# colors = []

# for index in range(0, len(result_tags)):
#     if result_tags[index] == "YangHPs" or result_tags[index] == "DCRE":
#         temp_precision = np.load('{}result_{}_p-{}.npy'.format(root_path, "denoise", t0s[index]))
#         temp_recall = np.load('{}result_{}_r-{}.npy'.format(root_path, "denoise", t0s[index]))
#     else:
#         temp_precision = np.load('{}result_{}_p-{}.npy'.format(root_path, result_tags[index], t0s[index]))
#         temp_recall = np.load('{}result_{}_r-{}.npy'.format(root_path, result_tags[index], t0s[index]))
#     temp_label = result_tags[index]

# def draw_PR_curve(precisions, recalls, labels, colors):
#     #用3次多项式拟合
#     f1 = np.polyfit(recall1, precision1, 10)
#     p1 = np.poly1d(f1)
#     print(p1)#打印出拟合函数
#     yvals1 = p1(recall1)  #拟合y值
    
#     f2 = np.polyfit(recall2, precision2, 10)
#     p2 = np.poly1d(f2)
#     print(p2)
#     #也可使用yvals=np.polyval(f1, x)
#     yvals2 = p2(recall2)
    
#     # 绘制虚线

#     #绘图
#     plt.plot(recall2, precision2, label=label2, color="#0343df", alpha=0.8)
#     plt.plot(recall1, precision1, label=label1, color="#f000f0", alpha=0.8)
#     plot4 = plt.plot(recall2, yvals2, '#0243d9', alpha=0.6, label='PCNN+ATT 10th polyfit', linestyle="--")
#     plot2 = plt.plot(recall1, yvals1, '#f900f9', alpha=0.6, label='YangHPs 10th polyfit', linestyle="--")
    

#     # plt.rcParams["font.sans-serif"]=["Microsoft YaHei"] #设置字体
#     # plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.0])
#     # plt.xlim([0.0, 0.4])
#     # plt.ylim([0.3, 1.0])
#     plt.grid(True)
#     plt.title('Fig 4. P-R Curve')
#     plt.legend(loc="upper right")
#     # plt.show()
#     plt.savefig("%sP-R Curve-vs-PCNN+ATT-all.png"%(root_path), dpi=300)


