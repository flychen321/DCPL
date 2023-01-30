# -*- coding: utf-8 -*-
from __future__ import print_function, division

import shutil

import numpy as np
from scipy.io import loadmat
import scipy
from get_multi_target_features import get_features
from rerank_for_cluster import re_ranking
from cal_cam import get_info_cam, calculate_correction, calculate_cam_flag
import os


def intra_distance(x0, y0=None, eu=True, rm_0=False, reduce=True):
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
        if reduce == False:
            return np.sqrt(dist + 1e-6)
        if rm_0 == True:
            dist = dist[~np.eye(dist.shape[0], dtype=bool)].reshape(dist.shape[0], -1)
            return np.sqrt(dist + 1e-6).mean()
        return np.sqrt(dist + 1e-6).mean()
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
    print('effective_id = %d  avg_diff_all = %.5f   avg_same_all = %.5f  correction_all = %.5f' % (
    effective_id_all, avg_diff_all, avg_same_all, (avg_diff_all - avg_same_all)))
    return avg_diff_all - avg_same_all


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


def get_margin(source, target):
    data_dir_stage1 = os.path.join('data', source, 'pytorch')
    data_dir_stage2 = os.path.join('data', target, 'pytorch')
    process_num = 1000
    # for i in np.arange(10):
    ## shutil.copyfile(os.path.join('./model', 'net_%s.pth' % ('last_1_'+str(i))), os.path.join('./model', 'net_last_1.pth'))
    # get_features(flag='all', multi_domain=False, order=i, data_dir=data_dir_stage1,
    #              net_loss_model=1,
    #              domain_num=2,
    #              which_epoch='last')
    # get_features(flag='all', multi_domain=False, order=i, data_dir=data_dir_stage2,
    #              net_loss_model=1,
    #              domain_num=2,
    #              which_epoch='last')
    m = loadmat(str(0) + '_' + 'all' + '_' + source + '_pytorch_target_result.mat')
    source_features = m['train_f']
    m = loadmat(str(0) + '_' + 'all' + '_' + target + '_pytorch_target_result.mat')
    target_features = m['train_f']
    target_cams = m['train_cam'][0]
    target_label = m['train_label'][0]
    rerank_features = re_ranking(source_features, target_features, lambda_value=0.1)
    rerank_dist_original = intra_distance(rerank_features, reduce=False)

    # c_all = 0.255
    get_info_cam(test_dir=target, which_epoch='last', stage=1)
    c_all, c_0, c_1 = calculate_correction()
    print('correction: %.4f    %.4f   %.4f' % (c_all, c_0, c_1))
    cam_flag = calculate_cam_flag(target_cams)
    correction = (1.0 - cam_flag) * c_all
    rerank_dist_refined = rerank_dist_original + correction
    intra_d = 0
    inter_d = 0
    for j in np.arange(rerank_dist_original.shape[0]):
        index0 = np.where(target_label[j] == target_label)[0]
        intra_d += np.average(rerank_dist_original[j][index0]) * len(index0) / (len(index0) - 1)
        index1 = np.where(target_label[j] != target_label)[0]
        index2 = np.argsort(rerank_dist_original[j])
        index3 = np.setdiff1d(index2, index0, assume_unique=True)[:10]
        inter_d += np.average(rerank_dist_original[j][index3])
        # inter_d += np.average(np.sort(rerank_dist_original[j][index1])[:20])
    intra_d /= j
    inter_d /= j
    print('original intra_d: %.4f' % intra_d)
    print('original inter_d: %.4f' % inter_d)
    print('original diff: %.4f' % (inter_d - intra_d))

    intra_d = 0
    inter_d = 0
    for j in np.arange(rerank_dist_refined.shape[0]):
        index0 = np.where(target_label[j] == target_label)[0]
        intra_d += np.average(rerank_dist_refined[j][index0]) * len(index0) / (len(index0) - 1)
        index1 = np.where(target_label[j] != target_label)[0]
        index2 = np.argsort(rerank_dist_refined[j])
        index3 = np.setdiff1d(index2, index0, assume_unique=True)[:10]
        inter_d += np.average(rerank_dist_refined[j][index3])
    intra_d /= j
    inter_d /= j
    print('refined intra_d: %.4f' % intra_d)
    print('refined inter_d: %.4f' % inter_d)
    print('refined diff: %.4f' % (inter_d - intra_d))


def get_margin_delta_ratio(source, target, delta=0.95):
    data_dir_stage1 = os.path.join('data', source, 'pytorch')
    data_dir_stage2 = os.path.join('data', target, 'pytorch')
    process_num = 1000
    # for i in np.arange(10):
    ## shutil.copyfile(os.path.join('./model', 'net_%s.pth' % ('last_1_'+str(i))), os.path.join('./model', 'net_last_1.pth'))
    # get_features(flag='all', multi_domain=False, order=0, data_dir=data_dir_stage1,
    #              net_loss_model=1,
    #              domain_num=2,
    #              which_epoch='last')
    # get_features(flag='all', multi_domain=False, order=0, data_dir=data_dir_stage2,
    #              net_loss_model=1,
    #              domain_num=2,
    #              which_epoch='last')
    m = loadmat(str(0) + '_' + 'all' + '_' + source + '_pytorch_target_result.mat')
    source_features = m['train_f']
    m = loadmat(str(0) + '_' + 'all' + '_' + target + '_pytorch_target_result.mat')
    target_features = m['train_f']
    target_cams = m['train_cam'][0]
    target_label = m['train_label'][0]
    rerank_features = re_ranking(source_features, target_features, lambda_value=0.1)
    rerank_dist_original = intra_distance(rerank_features, reduce=False)

    # c_all = 0.255
    get_info_cam(test_dir=target, which_epoch='last', stage=1)
    c_all, c_0, c_1 = calculate_correction()
    print('correction: %.4f    %.4f   %.4f' % (c_all, c_0, c_1))
    cam_flag = calculate_cam_flag(target_cams)
    correction = (1.0 - cam_flag) * c_all
    rerank_dist_refined = rerank_dist_original + correction
    acc = 0
    cnt = 0
    for j in np.arange(rerank_dist_original.shape[0]):
        index0 = np.where(rerank_dist_original[j] < delta)
        index1 = np.where(target_label[index0] == target_label[j])
        index2 = np.where(target_label[index0] != target_label[j])
        if len(index0[0]) > 8:
            acc += len(index1[0]) / len(index0[0])
            cnt += 1
    acc /= cnt + 1e-6
    print('original  cnt : %d    acc: %.4f' % (cnt, acc))

    acc = 0
    cnt = 0
    for j in np.arange(rerank_dist_refined.shape[0]):
        index0 = np.where(rerank_dist_refined[j] < delta)
        index1 = np.where(target_label[index0] == target_label[j])
        index2 = np.where(target_label[index0] != target_label[j])
        if len(index0[0]) > 8:
            acc += len(index1[0]) / len(index0[0])
            cnt += 1
    acc /= cnt + 1e-6
    print('refined  cnt : %d    acc: %.4f' % (cnt, acc))


def get_margin_delta(source, target, delta=0.95):
    data_dir_stage1 = os.path.join('data', source, 'pytorch')
    data_dir_stage2 = os.path.join('data', target, 'pytorch')
    process_num = 1000
    i = 0
    for i in np.arange(1, 10):
        if i == -1:
            shutil.copyfile(os.path.join('./model', 'net_pretrain.pth'), os.path.join('./model', 'net_last_1.pth'))
        else:
            shutil.copyfile(os.path.join('./model', 'net_%s.pth' % ('last_1_'+str(i))), os.path.join('./model', 'net_last_1.pth'))
        get_features(flag='all', multi_domain=False, order=i, data_dir=data_dir_stage1,
                     net_loss_model=1,
                     domain_num=2,
                     which_epoch='last')
        get_features(flag='all', multi_domain=False, order=i, data_dir=data_dir_stage2,
                     net_loss_model=1,
                     domain_num=2,
                     which_epoch='last')
        m = loadmat(str(0) + '_' + 'all' + '_' + source + '_pytorch_target_result.mat')
        source_features = m['train_f']
        m = loadmat(str(0) + '_' + 'all' + '_' + target + '_pytorch_target_result.mat')
        target_features = m['train_f']
        target_cams = m['train_cam'][0]
        target_names = m['train_name'][0]
        target_label = m['train_label'][0]
        rerank_features = re_ranking(source_features, target_features, lambda_value=0.1)
        result = {'rerank_features': rerank_features, 'names': target_names}
        scipy.io.savemat('rerank_features.mat', result)
        # rerank_features = loadmat('rerank_features.mat')['rerank_features']
        rerank_dist_original = intra_distance(rerank_features, reduce=False)

        # c_all = 0.863
        c_all = 1.21
        get_info_cam(test_dir=target, which_epoch='last', stage=1)
        c_all, c_0, c_1 = calculate_correction()
        print('i = %d   correction: %.4f    %.4f   %.4f' % (i, c_all, c_0, c_1))
        cam_flag = calculate_cam_flag(target_cams)
        correction = (1.0 - cam_flag) * c_all
        rerank_dist_refined = rerank_dist_original + correction
        acc = 0
        cnt = 0
        margin = 0
        margin_pn = 0
        cnt_pn = 0
        for j in np.arange(rerank_dist_original.shape[0]):
            index0 = np.where(rerank_dist_original[j] < delta)[0]
            index1 = index0[np.where(target_label[index0] == target_label[j])]
            index2 = index0[np.where(target_label[index0] != target_label[j])]
            index3 = np.where(target_label == target_label[j])[0]
            index4 = np.where(target_label != target_label[j])[0]
            pos_dist = np.sort(rerank_dist_original[j][index3])[-1]
            neg_dist = np.sort(rerank_dist_original[j][index4])[0]

            if len(index0) > 8:
                if len(index1) > 0 and len(index2) > 0:
                    margin += np.average(rerank_dist_original[j][index2]) - np.average(rerank_dist_original[j][index1])
                    cnt += 1
                elif len(index1) > 0:
                    margin += neg_dist - np.average(rerank_dist_original[j][index1])
                    cnt += 1
                margin_pn += neg_dist - pos_dist
                cnt_pn += 1
            # if j % 1000 == 0:
            #     print('j = %4d  len(index) = %d  %d  %d  %d  %d' % (
            #     j, len(index0), len(index1), len(index2), len(index3), len(index4)))
            #     print('j = %4d  pos_dist = %.4f  neg_dist = %.4f  ' % (j, pos_dist, neg_dist))
            #     print('j = %4d  margin = %.4f  margin_pn = %.4f  ' % (j, (margin / (cnt + 1e-6)), (margin_pn / (cnt_pn + 1e-6))))
        margin /= cnt + 1e-6
        margin_pn /= cnt_pn + 1e-6
        print('i = %d   original  cnt    : %d    margin   : %.4f' % (i, cnt, margin))
        print('i = %d   original  cnt_pn : %d    margin_pn: %.4f' % (i, cnt_pn, margin_pn))

        cnt = 0
        margin = 0
        margin_pn = 0
        cnt_pn = 0
        for j in np.arange(rerank_dist_refined.shape[0]):
            index0 = np.where(rerank_dist_refined[j] < delta)[0]
            index1 = index0[np.where(target_label[index0] == target_label[j])]
            index2 = index0[np.where(target_label[index0] != target_label[j])]
            index3 = np.where(target_label == target_label[j])[0]
            index4 = np.where(target_label != target_label[j])[0]
            pos_dist = np.sort(rerank_dist_refined[j][index3])[-1]
            neg_dist = np.sort(rerank_dist_refined[j][index4])[0]

            if len(index0) > 8:
                if len(index1) > 0 and len(index2) > 0:
                    margin += np.average(rerank_dist_refined[j][index2]) - np.average(rerank_dist_refined[j][index1])
                    cnt += 1
                elif len(index1) > 0:
                    margin += neg_dist - np.average(rerank_dist_refined[j][index1])
                    cnt += 1
                margin_pn += neg_dist - pos_dist
                cnt_pn += 1
            # if j % 1000 == 0:
            #     print('j = %4d  len(index) = %d  %d  %d  %d  %d' % (
            #         j, len(index0), len(index1), len(index2), len(index3), len(index4)))
            #     print('j = %4d  pos_dist = %.4f  neg_dist = %.4f  ' % (j, pos_dist, neg_dist))
            #     print('j = %4d  margin = %.4f  margin_pn = %.4f  ' % (j, (margin / (cnt + 1e-6)), (margin_pn / (cnt_pn + 1e-6))))
        margin /= cnt + 1e-6
        margin_pn /= cnt_pn + 1e-6
        print('i = %d   refined  cnt    : %d    margin   : %.4f' % (i, cnt, margin))
        print('i = %d   refined  cnt_pn : %d    margin_pn: %.4f' % (i, cnt_pn, margin_pn))

def get_margin_delta2(source, target, delta=0.95):
    data_dir_stage1 = os.path.join('data', source, 'pytorch')
    data_dir_stage2 = os.path.join('data', target, 'pytorch')
    process_num = 1000
    i = 0
    # for i in np.arange(-1, 10):
    #     if i == -1:
    #         shutil.copyfile(os.path.join('./model', 'net_pretrain.pth'), os.path.join('./model', 'net_last_1.pth'))
    #     else:
    #         shutil.copyfile(os.path.join('./model', 'net_%s.pth' % ('last_1_'+str(i))), os.path.join('./model', 'net_last_1.pth'))
    get_features(flag='all', multi_domain=False, order=i, data_dir=data_dir_stage1,
                 net_loss_model=1,
                 domain_num=2,
                 which_epoch='last')
    get_features(flag='all', multi_domain=False, order=i, data_dir=data_dir_stage2,
                 net_loss_model=1,
                 domain_num=2,
                 which_epoch='last')
    m = loadmat(str(0) + '_' + 'all' + '_' + source + '_pytorch_target_result.mat')
    source_features = m['train_f']
    m = loadmat(str(0) + '_' + 'all' + '_' + target + '_pytorch_target_result.mat')
    target_features = m['train_f']
    target_cams = m['train_cam'][0]
    target_label = m['train_label'][0]
    rerank_features = re_ranking(source_features, target_features, lambda_value=0.1)
    result = {'rerank_features': rerank_features}
    scipy.io.savemat('rerank_features.mat', result)
    rerank_features = loadmat('rerank_features.mat')['rerank_features']
    rerank_dist_original = intra_distance(rerank_features, reduce=False)

    # c_all = 0.229
    c_all = 0.86
    # get_info_cam(test_dir=target, which_epoch='last', stage=1)
    # c_all, c_0, c_1 = calculate_correction()
    # print('correction: %.4f    %.4f   %.4f' % (c_all, c_0, c_1))
    cam_flag = calculate_cam_flag(target_cams)
    correction = (1.0 - cam_flag) * c_all
    rerank_dist_refined = rerank_dist_original + correction

    intra = 0
    inter = 0
    for j in np.arange(rerank_dist_original.shape[0]):
        index0 = np.where(target_label == target_label[j])[0]
        index1 = np.where(target_label != target_label[j])[0]
        intra += np.average(rerank_dist_original[j][index0])
        inter += np.average(np.sort(rerank_dist_original[j][index1])[:20])
    margin = (inter - intra) / j
    print('original  cnt    : %d    margin   : %.4f' % (j, margin))
    print('original  inter : %.4f    intra: %.4f' % (inter, intra))

    intra = 0
    inter = 0
    for j in np.arange(rerank_dist_refined.shape[0]):
        index0 = np.where(target_label == target_label[j])[0]
        index1 = np.where(target_label != target_label[j])[0]
        intra += np.average(rerank_dist_refined[j][index0])
        inter += np.average(np.sort(rerank_dist_refined[j][index1])[:20])
    margin = (inter - intra) / j
    print('refined  cnt    : %d    margin   : %.4f' % (j, margin))
    print('refined  inter : %.4f    intra: %.4f' % (inter, intra))


if __name__ == '__main__':
    # get_cluster_distances('market')
    get_margin_delta(source='market', target='duke')
    # get_margin_delta2(source='market', target='duke')
