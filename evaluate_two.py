'''
import cv2
import numpy as np
import os

def main(input_path,label_path):
    files = os.listdir(input_path).copy()
    n = 0
    OA = 0
    Miou = 0
    for z in range(len(files)):
        img = cv2.imread(input_path+str(files[z]))
        label = cv2.imread(label_path+str(files[z]))

        pre_mat = (img / 255).flatten().astype(int)
        pre_mu = pre_mat.ravel()[np.flatnonzero(pre_mat)]

        gt_mat = (label / 255).flatten().astype(int)

        img = gt_mat ^ pre_mat
        im = 1 - img

        new = gt_mat & pre_mat
        pre_new = new.ravel()[np.flatnonzero(new)]

        accuracy = im.sum() / pre_mat.shape[0]
        iou = pre_new.sum() / pre_mu.shape[0]

        n = n+1
        OA = OA+accuracy
        Miou = Miou+iou
    print(OA/n)
    print(Miou/n)


input_path = 'E:\GRSL\experimental\segnet\\'
label_path = 'E:\GRSL\label\\'
main(input_path,label_path)

'''

import numpy as np
import os
import cv2
from scipy.io import loadmat,savemat
from PIL import Image
from sklearn.metrics import confusion_matrix


def transfer(x):
    d1 = {}
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] in d1:
                d1[x[i][j]]+=1
            else:
                d1.update({x[i][j]: 0})
    print(d1)

    for key,value in d1.items():
        if(value == max(d1.values())):
            z = key

    for m in range(x.shape[0]):
        for n in range(x.shape[1]):
            if x[m][n] == z:
                x[m][n]=255
            else:
                x[m][n]=0
    return x

def whiteblack(img):
    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            if img[m][n] == 255:
                img[m][n] = 0
            else:
                img[m][n] = 1
    return img

def inv_whiteblack(img):

    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            if img[m][n] == 255:
                img[m][n] = 1
            else:
                img[m][n] = 0
    return img

def kappa(confusion_matrix):
    """计算kappa值系数"""
    pe_rows = np.sum(confusion_matrix, axis=0)
    pe_cols = np.sum(confusion_matrix, axis=1)
    sum_total = sum(pe_cols)
    pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
    po = np.trace(confusion_matrix) / float(sum_total)
    return (po - pe) / (1 - pe)

def f1_score(confusion_matrix):
    """计算F1 score"""
    tp = confusion_matrix[1, 1]
    fp = confusion_matrix[0, 1]
    fn = confusion_matrix[1, 0]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1

def eval(x,y,name):
    gt_mat = x.flatten()

    pre_mat = y.flatten()
    con = confusion_matrix(gt_mat, pre_mat)

    if con.shape == (1,1):
        print('--------------%s-----------------' % name)
        print(con)
        recall = 1
        kappa1 = 1
        iou_1 = 1
        acc = 1
        precision = 1
        f1 = 1
    elif con.shape == (2,2):
        if con[1,1] + con[1,0] == 0:
            recall = 0
            precision = 0
            print('error')
        else:
            recall = con[1, 1] / (con[1, 1] + con[1, 0])
            precision = con[1, 1] / (con[1, 1] + con[0, 1])
        kappa1 = kappa(con)
        iou_1 = iou(con)
        f1 = f1_score(con)
        print('--------------%s-----------------' % name)
        print(con)
        a = con[0,0]+con[1,1]
        b = con[1,0]+con[0,0]+con[1,1]+con[0,1]
        acc = a/b
    elif con.shape == (3,3):
        print('--------------%s-----------------' % name)
        print(con)
        A = con[0, 0] + con[0, 1] + con[0, 2]
        B = con[1, 1] + con[1, 0] + con[1, 2]
        if A == 0:
            precision = 0
            recall = 0
            print('error')
        elif B == 0:
            precision = 0
            recall = 0
            print('error')
        else:
            precision_land = con[0, 0] / (con[0, 0] + con[1, 0] + con[2, 0])
            precision_aqua = con[1, 1] / (con[1, 1] + con[0, 1] + con[2, 1])
            precision_sea = con[2, 2] / (con[2, 2] + con[0, 2] + con[1, 2])
            precision = (precision_sea+precision_aqua+precision_land)/3

            recall_land = con[0, 0] / (con[0, 0] + con[0, 1] + con[0, 2])
            recall_aqua = con[1, 1] / (con[1, 1] + con[1, 0] + con[1, 2])
            recall_sea = con[2, 2] / (con[2, 2] + con[2, 0] + con[2, 1])
            recall = (recall_land + recall_aqua + recall_sea)/3

        kappa1 = kappa(con)
        iou_1 = iou3(con)
        f1 = f1_score(con)
        a = con[0,0]+con[1,1]+con[2,2]
        b = con[1,0]+con[0,0]+con[1,1]+con[0,1]+con[0,2]+con[2,0]+con[1,2]+con[2,1]+con[2,2]
        acc = a/b
    else:
        print('!!!!!!!!!!!!error!!!!!!!!!!!!')
        import time
        time.sleep(1000)
    print('acc=',acc)
    print('kappa=',kappa1)
    print('iou=',iou_1)
    print('precision=',precision)
    print('recall=',recall)
    print('F1 score:', f1)
    return acc,kappa1,iou_1,precision,recall,f1

def iou(con):

    pii_0 = con[0,0]
    pij_0 = con[0,1] + con [1,0] + pii_0
    iou_0 = pii_0/pij_0

    pii_1 = con[1,1]
    pij_1 = con[1,0] + con[0,1] + pii_1
    iou_1 = pii_1/pij_1

    miou = (iou_0 + iou_1)/2
    return miou

def iou3(con):

    pii_0 = con[0,0]
    pij_0 = con[0,1] + con[1,0] + + con[0,2] + con[2,0] + pii_0
    iou_0 = pii_0/pij_0

    pii_1 = con[1,1]
    pij_1 = con[1,0] + con[0,1] + con[1,2] + con[2,1] + pii_1
    iou_1 = pii_1/pij_1

    pii_2 = con[2,2]
    pij_2 = con[2,0] + con[0,2] + con[1,2] + con[2,1] + pii_2
    iou_2 = pii_2/pij_2

    miou = (iou_0 + iou_1 + iou_2)/3
    return miou


def main(input_path_pre,input_path_gt):
    files = os.listdir(input_path_pre)

    S = 0
    K = 0
    P = 0
    Z = 0
    R = 0
    F = 0
    for i in range(len(files)):
        name = str(files[i])
        namep = name[:-4]
        gt_mat = cv2.imread(input_path_gt + namep + '.png', 0)
        # gt_mat = whiteblack(gt_mat)
        pre_png = cv2.imread(input_path_pre + namep + '.png', 0) #.png
        # pre_png = whiteblack(pre_png)
        acc, kap, iou_1, precision, recall, f1 = eval(gt_mat, pre_png, name)

        if np.isnan(precision):
            precision = 0

        R = R + recall
        S = S + acc
        K = K + kap
        P = P + iou_1
        Z = Z + precision
        F = F + f1
        mean_recall = R / (i + 1)
        mean_acc = S / (i + 1)
        mean_kap = K / (i + 1)
        mean_iou = P / (i + 1)
        mean_precision = Z / (i + 1)
        mean_f1 = F / (i + 1)
        print('mF1 score:', mean_f1)
        print('miou=', mean_iou)
        print('mkap=',mean_kap)
        print('macc=',mean_acc)
        print('mprecision= ',mean_precision)
        print('mrecall= ',mean_recall)


#
inpath_pre = r'/root/autodl-tmp/mssgu/results%5Cpre%5Cindian_0.6346617.png'
#
inpath_gt = r'/root/autodl-tmp/mssgu/data/ff1024/ff1024_gt.png'


main(inpath_pre, inpath_gt)

