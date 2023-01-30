# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np
from scipy.io import loadmat
import scipy

def intra_distance(x0, y0=None, eu=True, rm_0=False):
    x = x0
    if y0 is None:
        y = x0
    else:
        y = y0
    """
    get the Euclidean Distance between to matrix
    (x-y)^2 = x^2 + y^2 - 2xy
    :param x:
    :param y:
    :return:
    """
    (rowx, colx) = x.shape
    (rowy, coly) = y.shape
    if colx != coly:
        raise RuntimeError('colx must be equal with coly')
    if eu == True:
        xy = np.dot(x, y.T)
        x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (rowx, 1)), repeats=rowy, axis=1)
        y2 = np.repeat(np.reshape(np.sum(np.multiply(y, y), axis=1), (rowy, 1)), repeats=rowx, axis=1).T
        dist = x2 + y2 - 2 * xy
        if y0 is None:
            if dist.shape[0] == 1:
                return 0
            dist = dist[~np.eye(dist.shape[0], dtype=bool)].reshape(dist.shape[0], -1)
        if rm_0 == True:
            np.sqrt(dist + 1e-6).mean() * dist.shape[0] / (dist.shape[0]-1)
        return np.sqrt(dist+1e-6).mean()
    else:
        dist = np.dot(x, y.T)
        if y0 is None:
            if dist.shape[0] == 1:
                return 0
            dist = dist[~np.eye(dist.shape[0], dtype=bool)].reshape(dist.shape[0], -1)
        return dist.mean()



def calculate_correction_for_all():
    result = scipy.io.loadmat('cam_result.mat')
    train_feature = result['train_f']
    train_cam = result['train_cam'][0]
    train_label = result['train_label'][0]
    sum_same = 0
    sum_diff = 0
    effective_id_all = 0

    for lable in set(train_label):
        index = np.where(lable == train_label)[0]
        feature = train_feature[index]
        cam = train_cam[index]
        cam_no_re = set(cam)
        same = 0.0
        diff = 0.0
        cnt = 0
        for c in cam_no_re:
            same += intra_distance(feature[np.where(c == cam)[0]])
            cnt += 1
        same /= (cnt + 1e-6)
        cnt = 0
        for c1 in cam_no_re:
            for c2 in cam_no_re:
                if c1 != c2:
                    diff += intra_distance(feature[np.where(c1 == cam)[0]], feature[np.where(c2 == cam)[0]])
                    cnt += 1
        diff /= (cnt + 1e-6)
        if same < 1e-6 or diff < 1e-6:
            continue
        else:
            sum_same += same
            sum_diff += diff
            effective_id_all += 1
        # print('effective_id = %d  diff = %.5f   same = %.5f' % (effective_id, diff, same))
    avg_same_all = sum_same / (effective_id_all + 1e-6)
    avg_diff_all = sum_diff / (effective_id_all + 1e-6)
    print('effective_id = %d  avg_diff_all = %.5f   avg_same_all = %.5f  correction_all = %.5f' % (effective_id_all, avg_diff_all, avg_same_all, (avg_diff_all - avg_same_all)))
    return avg_diff_all-avg_same_all

def get_cluster_distances(tgt_path, domain=0, flag='all'):
    print('Calculating feature distances...')
    m = loadmat(str(domain) + '_' + flag + '_' + tgt_path + '_pytorch_target_result.mat')
    train_feature = m['train_f']
    train_cam = m['train_cam'][0]
    train_label = m['train_label'][0]
    intra_d = 0
    inter_d = 0
    center_cnt = 0
    center = np.zeros((len(set(train_label)), train_feature.shape[1]))
    for lable in set(train_label):
        index = np.where(lable == train_label)[0]
        feature = train_feature[index]
        center[center_cnt] = np.average(feature, 0)
        intra_d += intra_distance(np.expand_dims(center[center_cnt], 0), feature)
        center_cnt += 1
    intra_d /= center_cnt
    inter_d = intra_distance(center, center, rm_0=True)
    print('intra_d: %.4f' % intra_d)
    print('inter_d: %.4f' % inter_d)
    return intra_d, inter_d

if __name__ == '__main__':
    get_cluster_distances('market')