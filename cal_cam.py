# -*- coding: utf-8 -*-
from __future__ import print_function, division

import argparse
import os

import numpy as np
import scipy.io
import torch
from torchvision import datasets, transforms

from model import ft_net, DisentangleNet
from model import load_whole_network

######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Cal_cam')
parser.add_argument('--which_epoch', default='best', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir', default='market', type=str, help='./test_data')
parser.add_argument('--name', default='', type=str, help='save model path')
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--net_loss_model', default=1, type=int, help='net_loss_model')
parser.add_argument('--domain_num', default=5, type=int, help='domain_num')
parser.add_argument('--stage', default=1, type=int, help='stage')
parser.add_argument('--gpu', type=str, default='0', help='GPU id to use.')
opt = parser.parse_args()
print('opt = %s' % opt)


def get_info_cam(test_dir=None, net_loss_model=None, domain_num=None, which_epoch=None, stage=2):
    if test_dir != None:
        opt.test_dir = test_dir
    if net_loss_model != None:
        opt.net_loss_model = net_loss_model
    if domain_num != None:
        opt.domain_num = domain_num
    if which_epoch != None:
        opt.which_epoch = which_epoch

    print('opt.which_epoch = %s' % opt.which_epoch)
    print('opt.test_dir = %s' % opt.test_dir)
    print('opt.name = %s' % opt.name)
    print('opt.batchsize = %s' % opt.batchsize)
    name = opt.name
    data_dir = os.path.join('data', opt.test_dir, 'pytorch')
    print('data_dir = %s' % data_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    ######################################################################
    # Load Data
    # ---------
    data_transforms = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if stage == 1:
        dataset_list = ['train_all']
    else:
        dataset_list = ['train_all_cluster']
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in dataset_list}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=False, num_workers=0) for x in dataset_list}
    use_gpu = torch.cuda.is_available()

    ######################################################################
    # Extract feature
    # ----------------------
    #
    # Extract feature from  a trained model.
    #
    def fliplr(img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    def extract_feature_original(model, dataloaders):
        features = torch.FloatTensor()
        for data in dataloaders:
            img, label = data
            n, c, h, w = img.size()
            ff_d = torch.FloatTensor(n, 512).zero_().cuda()
            for i in range(2):
                if (i == 1):
                    img = fliplr(img)
                input_img = img.cuda()
                ret = model(input_img)
                did_outputs = ret[1]
                # did_outputs = ret[3]
                ff_d = ff_d + did_outputs
            # norm feature
            fnorm = torch.norm(ff_d, p=2, dim=1, keepdim=True)
            ff_d = ff_d.div(fnorm.expand_as(ff_d))
            ff_d = ff_d.detach().cpu().float()
            features = torch.cat((features, ff_d), 0)
        return features

    def extract_feature(model, dataloaders):
        features = torch.FloatTensor()
        for data in dataloaders:
            img, label = data
            n, c, h, w = img.size()
            ff_d = torch.FloatTensor(n, 512).zero_().cuda()
            ff_s = torch.FloatTensor(n, 512).zero_().cuda()
            for i in range(2):
                if (i == 1):
                    img = fliplr(img)
                input_img = img.cuda()
                did_outputs = model(input_img)[1]
                sid_outputs = model(input_img)[3]
                ff_d = ff_d + did_outputs
                ff_s = ff_s + sid_outputs
            # norm feature
            fnorm = torch.norm(ff_d, p=2, dim=1, keepdim=True)
            ff_d = ff_d.div(fnorm.expand_as(ff_d))
            ff_d = ff_d.detach().cpu().float()
            fnorm = torch.norm(ff_s, p=2, dim=1, keepdim=True)
            ff_s = ff_s.div(fnorm.expand_as(ff_s))
            ff_s = ff_s.detach().cpu().float()
            features = torch.cat((features, torch.cat((ff_d, ff_s), 1)), 0)
        return features

    def get_id(img_path):
        camera_id = []
        labels = []
        names = []
        for path, v in img_path:
            # filename = path.split('/')[-1]
            filename = os.path.basename(path)
            label = filename[0:4]
            if 'msmt' in opt.test_dir:
                camera = filename[9:11]
            else:
                camera = filename.split('c')[1][0]
            if label[0:2] == '-1':
                labels.append(-1)
            else:
                labels.append(int(label))
            camera_id.append(int(camera))
            names.append(filename)
        return camera_id, labels, names

    dataset_path = []
    for i in range(len(dataset_list)):
        dataset_path.append(image_datasets[dataset_list[i]].imgs)

    dataset_cam = []
    dataset_label = []
    dataset_name = []
    for i in range(len(dataset_list)):
        cam, label, n = get_id(dataset_path[i])
        dataset_cam.append(cam)
        dataset_label.append(label)
        dataset_name.append(n)

    ######################################################################
    # Load Collected data Trained model
    print('---------test-----------')
    class_num = len(os.listdir(os.path.join(data_dir, 'train_all_new')))
    sid_num = class_num
    did_num = class_num * opt.domain_num
    did_embedding_net = ft_net(id_num=did_num)
    sid_embedding_net = ft_net(id_num=sid_num)
    model = DisentangleNet(did_embedding_net, sid_embedding_net)
    if use_gpu:
        model.cuda()
    if 'best' in opt.which_epoch or 'last' in opt.which_epoch:
        model = load_whole_network(model, name, opt.which_epoch + '_' + str(opt.net_loss_model))
    else:
        model = load_whole_network(model, name, opt.which_epoch)
    model = model.eval()
    if use_gpu:
        model = model.cuda()

    # Extract feature
    dataset_feature = []
    with torch.no_grad():
        for i in range(len(dataset_list)):
            dataset_feature.append(extract_feature(model, dataloaders[dataset_list[i]]))

    result = {'train_f': dataset_feature[0].numpy(), 'train_label': dataset_label[0], 'train_cam': dataset_cam[0],
              'train_name': dataset_name[0]}
    scipy.io.savemat('cam_result.mat', result)


def intra_distance(x0, y0=None, eu=True):
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
        return np.sqrt(dist + 1e-6).mean()
    else:
        dist = np.dot(x, y.T)
        if y0 is None:
            if dist.shape[0] == 1:
                return 0
            dist = dist[~np.eye(dist.shape[0], dtype=bool)].reshape(dist.shape[0], -1)
        return dist.mean()


def calculate_correction(stage=1):
    result = scipy.io.loadmat('cam_result.mat')
    train_feature_original = result['train_f']
    train_name_original = result['train_name'][0]
    train_cam = result['train_cam'][0]
    train_label = result['train_label'][0]
    train_feature = scipy.io.loadmat('rerank_features.mat')['rerank_features']
    train_name = scipy.io.loadmat('rerank_features.mat')['names']

    if stage != 1:
        train_feature2 = np.zeros((train_feature_original.shape[0], train_feature.shape[1]))
        for i in np.arange(train_feature_original.shape[0]):
            train_feature2[i] = train_feature[np.where(train_name_original[i] == train_name)[0]]
        train_feature = train_feature2
        # train_feature = train_feature[np.where(train_name_original == train_name)[0]]

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
    sum_same = 0
    sum_diff = 0
    effective_id_0 = 0
    for lable in set(train_label):
        index = np.where(lable == train_label)[0]
        feature = train_feature[index][:, :int(train_feature.shape[1] / 2)]
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
            effective_id_0 += 1
        # print('effective_id = %d  diff = %.5f   same = %.5f' % (effective_id, diff, same))
    avg_same_0 = sum_same / (effective_id_0 + 1e-6)
    avg_diff_0 = sum_diff / (effective_id_0 + 1e-6)
    sum_same = 0
    sum_diff = 0
    effective_id_1 = 0
    for lable in set(train_label):
        index = np.where(lable == train_label)[0]
        feature = train_feature[index][:, int(train_feature.shape[1] / 2):]
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
            effective_id_1 += 1
        # print('effective_id = %d  diff = %.5f   same = %.5f' % (effective_id, diff, same))
    avg_same_1 = sum_same / (effective_id_1 + 1e-6)
    avg_diff_1 = sum_diff / (effective_id_1 + 1e-6)
    print('effective_id = %d  avg_diff_all = %.5f   avg_same_all = %.5f  correction_all = %.5f' % (
    effective_id_all, avg_diff_all, avg_same_all, (avg_diff_all - avg_same_all)))
    print('effective_id = %d  avg_diff_0 = %.5f   avg_same_0 = %.5f  correction_0 = %.5f' % (
    effective_id_0, avg_diff_0, avg_same_0, (avg_diff_0 - avg_same_0)))
    print('effective_id = %d  avg_diff_1 = %.5f   avg_same_1 = %.5f  correction_1 = %.5f' % (
    effective_id_1, avg_diff_1, avg_same_1, (avg_diff_1 - avg_same_1)))
    return avg_diff_all - avg_same_all, avg_diff_0 - avg_same_0, avg_diff_1 - avg_same_1


def calculate_cam_flag(target_cams):
    cam_flag = np.zeros((target_cams.shape[0], target_cams.shape[0]))
    for i0 in np.arange(target_cams.shape[0]):
        for j0 in np.arange(i0, target_cams.shape[0]):
            if target_cams[i0] != target_cams[j0]:
                cam_flag[i0, j0] = 1.0
                cam_flag[j0, i0] = 1.0
    return cam_flag


if __name__ == '__main__':
    # get_info_cam(test_dir=opt.test_dir, net_loss_model=opt.net_loss_model, domain_num=opt.domain_num,
    #               which_epoch=opt.which_epoch, stage=opt.stage)
    calculate_correction()
