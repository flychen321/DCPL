import numpy as np
import os
import shutil
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering, MeanShift
from sklearn import metrics
from collections import Counter
from rerank_for_cluster import re_ranking
import statistics


def intra_distance(x, y=None):
    if y is None:
        y = x
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
    xy = np.dot(x, y.T)
    x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (rowx, 1)), repeats=rowy, axis=1)
    y2 = np.repeat(np.reshape(np.sum(np.multiply(y, y), axis=1), (rowy, 1)), repeats=rowx, axis=1).T
    dist = x2 + y2 - 2 * xy
    dist = np.clip(dist, 0, None)
    return np.sqrt(dist + 1e-6)


def analysis_features():
    m = loadmat('duke' + '_pytorch_target_result.mat')
    print('type(m) = %s' % type(m))
    print(m.keys())
    target_features = m['train_f']
    data_num = len(target_features)

    np.random.seed(100)

    print('len(target_features) = %s' % len(target_features))
    print('target_features[0].size = %d' % target_features[0].size)
    # savemat(os.path.join(dst_path, 'target_features.mat'), {'features': target_features})

    target_labels = m['train_label'][0]
    # print('target_labels = %s' % (target_labels))
    # print('unique lable = %s' % np.unique(np.sort(target_labels)))
    print('real class_num = %s' % len(np.unique(np.sort(target_labels))))
    print('len(target_labels) = %s' % len(target_labels))

    target_names = m['train_name']
    print(sorted(Counter(target_labels).values())[:10])
    print(sorted(Counter(target_labels).values())[-10:])

    indices = np.random.permutation(len(target_labels))
    target_labels = target_labels[indices]
    target_features = target_features[indices]

    same_dist = []
    diff_dist = []
    same_max = []
    diff_min = []
    same_avg = []
    diff_avg = []

    for i in np.arange(len(target_features)):
        dist = ((target_features[i] - target_features) * (target_features[i] - target_features)).sum(1)
        same_sub = [dist[j] for j in np.arange(len(target_features)) if target_labels[j] == target_labels[i]]
        diff_sub = [dist[j] for j in np.arange(len(target_features)) if target_labels[j] != target_labels[i]]
        same_dist.append(same_sub)
        diff_dist.append(diff_sub[: len(same_sub)])
        same_max.append(np.max(same_sub))
        diff_min.append(np.min(diff_sub))
        same_avg.append(np.sum(same_sub) / (len(same_sub) - 1 + 1e-6))
        diff_avg.append(np.mean(diff_sub))
        if i % 200 == 0:
            print('i = %3d' % i)
        if i > 1000:
            break

    cnt = np.sum(np.array(same_max) < np.array(diff_min))
    ratio = cnt / len(same_max)
    print(ratio)
    print('avg same_max = %.3f    avg diff_min = %.3f' % (np.mean(same_max), np.mean(diff_min)))
    print('avg same = %.3f    avg diff = %.3f' % (np.mean(same_avg), np.mean(diff_avg)))


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
    return intra_d, inter_d, intra_d - intra_d



def generate_cluster(cluster_result_path, dist=None, eps=1.2, min_samples=10, src_path=None, tgt_path=None, flag='did',
                      epoch=0):
    m = loadmat(str(0) + '_' + flag + '_' + src_path + '_pytorch_target_result.mat')
    source_features = m['train_f']
    m = loadmat(str(0) + '_' + flag + '_' + tgt_path + '_pytorch_target_result.mat')
    target_features = m['train_f']
    print('len(target_features) = %s' % len(target_features))
    print('target_features[0].size = %d' % target_features[0].size)
    target_labels = m['train_label'][0]
    print('real class_num = %s' % len(np.unique(np.sort(target_labels))))
    print('len(target_labels) = %s' % len(target_labels))
    target_names = m['train_name']
    rerank_features = loadmat('rerank_features.mat')['rerank_features']
    if os.path.exists(cluster_result_path):
        shutil.rmtree(cluster_result_path)
    os.mkdir(cluster_result_path)
    process_num = m['train_label'][0].shape[0]
    if dist is None:
        X = target_features[:process_num]
    else:
        X = dist[:process_num]
    labels_true = target_labels[:process_num]
    names = target_names[:process_num]
    print('DBSCAN starting ......')
    print('min_samples = %d' % min_samples)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels_db = db.labels_
    sorted_v = sorted(Counter(labels_db).values())
    print(sorted_v[:min(20, len(set(labels_db)))])
    print(sorted_v[-min(20, len(set(labels_db))):])
    print(sorted_v)
    print("DBSCAN")
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels_db))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels_db))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels_db))
    print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels_db))
    print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels_db))
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels_db)) - (1 if -1 in labels_db else 0)
    cluster_num = n_clusters_
    print('Estimated number of clusters: %d' % n_clusters_)
    n_node = np.sum(labels_db != -1)
    print('Estimated number of nodes: %d' % n_node)
    aver = n_node / n_clusters_
    dir_cnt = 0
    image_cnt = 0
    labels2 = labels_db
    # if 0:
    if 'duke' in tgt_path:
        rerank_kmeans = True
        if rerank_kmeans:
            did_rerank_dist = re_ranking(source_features[:, :512], target_features[:, :512])
            X = did_rerank_dist[:process_num]
        else:
            X = target_features[:process_num][:, :512]
        cluster_center = np.zeros((cluster_num, len(X[0])))
        for i in np.arange(cluster_num):
            cluster_center[i] = np.average(X[np.where(labels_db == i)[0]], 0)
        db = KMeans(n_clusters=cluster_num, init=cluster_center).fit(X)
        labels_did = db.labels_
        if rerank_kmeans:
            sid_rerank_dist = re_ranking(source_features[:, 512:], target_features[:, 512:])
            X = sid_rerank_dist[:process_num]
        else:
            X = target_features[:process_num][:, 512:]
        cluster_center = np.zeros((cluster_num, len(X[0])))
        for i in np.arange(cluster_num):
            cluster_center[i] = np.average(X[np.where(labels_db == i)[0]], 0)
        db = KMeans(n_clusters=cluster_num, init=cluster_center).fit(X)
        labels_sid = db.labels_

        cnt_k1 = 0
        cnt_k2 = 0
        iou_th = 0.45
        km_index = np.array([], dtype=np.int32)
        km_lable = np.array([], dtype=np.int32)
        print('iou_th = %.3f' % iou_th)
        for j in np.arange(cluster_num):
            db_nodes = np.where(labels_db == j)[0]
            did_nodes = np.where(labels_did == j)[0]
            did_iou_max = len(np.intersect1d(db_nodes, did_nodes)) / len(
                np.union1d(db_nodes, did_nodes))
            sid_nodes = np.where(labels_sid == j)[0]
            sid_iou_max = len(np.intersect1d(db_nodes, sid_nodes)) / len(
                np.union1d(db_nodes, sid_nodes))
            if did_iou_max > iou_th and sid_iou_max > iou_th:
                did_noise_nodes = np.setdiff1d(did_nodes, db_nodes)
                sid_noise_nodes = np.setdiff1d(sid_nodes, db_nodes)
                k_effect = np.intersect1d(did_noise_nodes, sid_noise_nodes)
                if len(k_effect) > 0:
                    # labels_db[k_effect] = j
                    km_index = np.concatenate((km_index, k_effect))
                    km_lable = np.concatenate((km_lable, np.repeat(j, len(k_effect))))
                    cnt_k1 += len(k_effect)

        effect_nodes = np.where(labels_db != -1)[0]
        outlier_nodes = np.where(labels_db == -1)[0]
        if epoch >= 5:
            print('epoch = %d len(effect_nodes) = %d  len(outlier_nodes) = %d' % (
            epoch, len(effect_nodes), len(outlier_nodes)))
            effect_features = target_features[effect_nodes]
            outlier_features = target_features[outlier_nodes]
            labels_true_outlier = labels_true[outlier_nodes]
            X = re_ranking(effect_features, outlier_features)
            eps = np.sort(X, axis=1)[:, :int(0.002 * len(X[0]))].mean()
            # print('second eps = %.3f' % eps)
            out_db = DBSCAN(eps=eps, min_samples=6, metric='precomputed').fit(X)
            labels_db_outlier = out_db.labels_
            cluster_num_outlier = len(set(labels_db_outlier)) - (1 if -1 in labels_db_outlier else 0)
            print('Estimated number of clusters in outliers: %d' % cluster_num_outlier)
            out_n_node = np.sum(labels_db_outlier != -1)
            print('Estimated number of nodes in outliers: %d' % out_n_node)
            X = target_features[:process_num]
            cnt_out = len(set(labels_db_outlier[np.where(labels_db_outlier != -1)]))
            cnt_eff = cluster_num
            dis_oe = np.ones((cnt_out, cnt_eff)) * 1000
            for i in np.arange(cnt_out):
                for j in np.arange(cnt_eff):
                    dis_oe[i][j] = intra_distance(X[outlier_nodes[np.where(labels_db_outlier == i)[0]]],
                                                  X[np.where(labels_db == j)[0]]).min()
            min_value_1 = np.min(dis_oe, axis=1)
            min_index_1 = np.argmin(dis_oe, axis=1)
            sort_index_1 = np.argsort(dis_oe, axis=1)
            min_value_0 = np.min(dis_oe, axis=0)
            min_index_0 = np.argmin(dis_oe, axis=0)
            sort_index_0 = np.argsort(dis_oe, axis=0)
            db2_nodes = np.array([], dtype=np.int32)
            db2_lables = np.array([], dtype=np.int32)
            for i in sorted(set(labels_db_outlier[np.where(labels_db_outlier != -1)])):
                index_i = outlier_nodes[np.where(labels_db_outlier == i)[0]]
                if min_index_0[min_index_1[i]] == i:
                    cnt_k2 += len(index_i)
                    db2_nodes = np.concatenate((db2_nodes, index_i))
                    db2_lables = np.concatenate((db2_lables, np.repeat(min_index_1[i], len(index_i))))
            if len(db2_nodes) == len(db2_lables):
                kk = np.arange(len(db2_nodes))
                print('len(kk) = %d' % len(kk))
                for k in kk:
                    labels_db[db2_nodes[k]] = db2_lables[k]
            else:
                print('len(db2_nodes) = %d  len(db2_lables) = %d  not equal !!!' % (len(db2_nodes), len(db2_lables)))

        print('len(km_index) = %d  len(km_lable) = %d  len(set(km_index)) = %d  len(set(km_lable)) = %d' % (
            len(km_index), len(km_lable), len(set(km_index)), len(set(km_lable))))
        print('len(set(km_lable)) = %d' % len(set(km_lable)))
        print('len(set(labels_true[km_index])) = %d' % len(set(labels_true[km_index])))
        if len(km_index) == len(km_lable):
            for k in np.arange(len(km_index)):
                labels_db[km_index[k]] = km_lable[k]
        else:
            print('len(km_index) = %d  len(km_lable) = %d  not equal !!!' % (len(km_index), len(km_lable)))
        print('cnt_k1 = %d  cnt_k2 = %d' % (cnt_k1, cnt_k2))

        if os.path.exists(cluster_result_path):
            shutil.rmtree(cluster_result_path)
        os.mkdir(cluster_result_path)
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels_db)) - (1 if -1 in labels_db else 0)
        print('After Estimated number of clusters: %d' % n_clusters_)
        n_node = np.sum(labels_db != -1)
        print('After Estimated number of nodes: %d' % n_node)
        aver = n_node / n_clusters_
        dir_cnt = 0
        image_cnt = 0

        dir_dict = {}
        for i in sorted(set(labels_db[np.where(labels_db != -1)])):
            files = names[np.where(labels_db == i)]
            if (len(files) >= 2) and (len(files) < aver * 3.0):
                dir_path = os.path.join(cluster_result_path, str(dir_cnt).zfill(4))
                os.mkdir(dir_path)
                dir_dict[str(dir_cnt)] = str(i)
                for file in files:
                    file = file.strip()
                    shutil.copy(
                        os.path.join(os.path.split(os.path.split(cluster_result_path)[0])[0], 'bounding_box_train',
                                     file),
                        os.path.join(dir_path, file))
                dir_cnt += 1
                image_cnt += len(files)
            else:
                labels_db[np.where(labels_db == i)] = -1
                # print('i = %d  aver = %.3f  len(files) = %d' % (i, aver, len(files)))
        dir_cnt1 = dir_cnt
        image_cnt1 = image_cnt
        print('cluster number: %d file number: %d ' % (dir_cnt1, image_cnt1))
    else:
        for i in np.arange(n_clusters_):
            files = names[np.where(labels_db == i)]
            if (len(files) >= np.max((2, int(min_samples * 0.5)))) and (len(files) < aver * 3.0):
                dir_path = os.path.join(cluster_result_path, str(dir_cnt).zfill(4))
                os.mkdir(dir_path)
                for file in files:
                    file = file.strip()
                    shutil.copy(
                        os.path.join(os.path.split(os.path.split(cluster_result_path)[0])[0], 'bounding_box_train',
                                     file),
                        os.path.join(dir_path, file))
                dir_cnt += 1
                image_cnt += len(files)
            else:
                labels2[np.where(labels_db == i)] = -1
        dir_cnt1 = dir_cnt
        image_cnt1 = image_cnt
        print(
            'first cluster number: %d file number:%d ' % (dir_cnt1, image_cnt1))
    cluster_outlier_path = os.path.join(os.path.split(cluster_result_path)[0], 'train_all_cluster_outlier')
    if os.path.exists(cluster_outlier_path):
        shutil.rmtree(cluster_outlier_path)
    os.mkdir(cluster_outlier_path)
    files = names[np.where(labels2 == -1)]
    for file in files:
        file = file.strip()
        shutil.copy(
            os.path.join(os.path.split(os.path.split(cluster_outlier_path)[0])[0], 'bounding_box_train',
                         file),
            os.path.join(cluster_outlier_path, file))
    print('outlier file num = %d' % (len(files)))

    print('valid cluster number: %d    file number:%d    aver = %.2f' % (
        dir_cnt, image_cnt, image_cnt / (dir_cnt + 1e-3)))
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels2))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels2))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels2))
    print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels2))
    print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels2))


if __name__ == '__main__':
    # analysis_features()
    dir_dict = {}
    for i in np.arange(5):
        dir_dict[str(i)] = i * 2
    print(dir_dict)
    # generate_cluster('data/duke/pytorch/train_all_cluster')
