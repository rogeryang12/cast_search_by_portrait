import os
import numpy as np
import pickle
from tqdm import tqdm
from argparse import ArgumentParser

from eval import eval


parser = ArgumentParser(description='face model test')
parser.add_argument('--mode', default='val')
parser.add_argument('--root', default='data')
parser.add_argument('--gt', type=str, default='val_label.json')
parser.add_argument('--submission', type=str, default='result.txt')

parser.add_argument('--face_rerank', default=False, action='store_true')
parser.add_argument('--body_rerank', default=False, action='store_true')

parser.add_argument('--thres', default=[0.5, 0.4, 0.3, 0.25], type=float, nargs='+')
args = parser.parse_args()

args.submission = '{}_{}'.format(args.mode, args.submission)
args.gt = '{}/{}'.format(args.root, args.gt)
args.runs = len(args.thres) - 1


def load_pkl(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
        return data


def unique_list(lst):
    unique_lst = []
    val_set = set()
    for x in lst:
        if x not in val_set:
            unique_lst.append(x)
            val_set.add(x)
    return unique_lst


def reciprocal(original_dist, query_num, k1=20, k2=6, lambda_value=0.3):
    """
    Created on Mon Jun 26 14:46:56 2017

    @author: luohao
    Modified by Houjing Huang, 2017-12-22.
    Modified by Xingze Li 2019-8-1.
    """

    """
    CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
    url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
    Matlab version: https://github.com/zhunzhong07/person-re-ranking
    """

    gallery_num = original_dist.shape[0]
    all_num = gallery_num
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2.)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2.)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = 1. * weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist

    return final_dist[:query_num, query_num:]


def decide():

    body_dict = load_pkl('{}/retina_{}.pkl'.format(args.root, args.mode))
    face_dict = load_pkl('{}/face_{}.pkl'.format(args.root, args.mode))

    reid_dicts = [

        load_pkl('{}/triplet_resnet50_{}.pkl'.format(args.root, args.mode)),
        # load_pkl('{}/mgn_resnet50_{}.pkl'.format(args.root, args.mode))

        # load_pkl('{}/mgn_densenet169_{}.pkl'.format(args.root, args.mode)),
        # load_pkl('{}/mgn_densenet201_{}.pkl'.format(args.root, args.mode)),
        # load_pkl('{}/mgn_resnet152_{}.pkl'.format(args.root, args.mode)),
        # load_pkl('{}/mgn_resnext50_32x4d_{}.pkl'.format(args.root, args.mode)),
        # load_pkl('{}/mgn_resnext101_32x8d_{}.pkl'.format(args.root, args.mode)),
    ]

    print('load data done')

    result = {}
    for movie in tqdm(sorted(body_dict.keys())):
        cast = body_dict[movie]['cast']
        candi = body_dict[movie]['candidates']
        num_cast = len(cast)
        num_candi = len(candi)

        cast_ids = np.array([item['id'] for item in cast])
        candi_ids = np.array([item['id'] for item in candi])

        candi_files = np.array([os.path.basename(item['file']).split('_')[0] for item in candi])
        file_cnt = {}
        for file in candi_files:
            if file not in file_cnt:
                file_cnt[file] = 1
            else:
                file_cnt[file] += 1
        candi_file_cnt = np.array([file_cnt[file] for file in candi_files])
        no_face = np.array([item['faces'] is None for item in candi])

        result.update({cast_id: [] for cast_id in cast_ids})

        feats = face_dict[movie]
        feats = feats / np.linalg.norm(feats, axis=-1, keepdims=True)
        if args.face_rerank:
            sim = np.dot(feats, feats.T)
            dists = np.sqrt(2 - 2 * sim + 1e-4)
            dists = reciprocal(dists, num_cast)
            cast_candi_sim = 1 - dists
        else:
            cast_candi_sim = np.dot(feats[:num_cast], feats[num_cast:].T)
        candi_candi_sim = np.dot(feats[num_cast:], feats[num_cast:].T)
        
        cast_candi_sim[:, no_face] = -2
        candi_candi_sim[:, no_face] = -2
        candi_candi_sim[no_face, :] = -2

        sort_index = np.argsort(1 - cast_candi_sim, axis=-1)
        match = cast_candi_sim > args.thres[0]
        match_num = match.sum(1)
        total_match = match
        idx = np.where(match_num < 5)[0]
        if len(idx) > 0:
            match[idx[:, np.newaxis], sort_index[idx, :5]] = True
            match_num[idx] = 5

        for i in range(num_cast):
            result[cast_ids[i]].extend(sort_index[i][:match_num[i]])

        for run in range(args.runs):
            sims = []
            for i in range(num_cast):
                query = np.where(total_match[i])[0]
                sim = candi_candi_sim[query].mean(0)
                sims.append(sim)

            cast_candi_sim = np.stack(sims)
            sort_index = np.argsort(1 - cast_candi_sim, axis=-1)
            match = cast_candi_sim > args.thres[run + 1]
            match_num = match.sum(1)
            total_match = np.logical_or(match, total_match)

            for i in range(num_cast):
                prev = result[cast_ids[i]]
                cur = unique_list(prev + list(sort_index[i][:match_num[i]]))
                result[cast_ids[i]] = cur

        match = total_match

        decided = np.where(match.sum(0))[0]
        decided = list(decided)
        decided_set = set(decided)
        unknown = [i for i in range(num_candi) if i not in decided_set]

        embs = [reid_dict[movie] for reid_dict in reid_dicts]
        embs = [emb / np.linalg.norm(emb, axis=-1, keepdims=True) for emb in embs]
        embs = np.concatenate(embs, axis=-1)
        embs /= np.linalg.norm(embs, axis=-1, keepdims=True)
        for reid_dict in reid_dicts:
            del reid_dict[movie]

        qembs = embs
        gembs = embs[np.array(unknown)]

        if args.body_rerank:
            emb = np.concatenate([qembs, gembs], axis=0)
            body_sim = np.dot(emb, emb.T)
            dists = np.sqrt(2 - 2 * body_sim + 1e-4)
            dists = reciprocal(dists, len(qembs))
            body_sim = 1 - dists
        else:
            body_sim = np.dot(qembs, gembs.T)

        origin_sim = np.dot(embs, embs.T)
        knn = np.argpartition(1 - origin_sim, 9, axis=-1)[:, :9]

        knn_sim = origin_sim[np.arange(len(embs))[:, np.newaxis], knn]
        exp_knn_sim = np.exp(knn_sim)
        weight = exp_knn_sim / exp_knn_sim.sum(axis=-1, keepdims=True)

        knn_embs = embs[knn.reshape(-1)].reshape(len(embs), 9, -1)
        knn_embs = (knn_embs * weight[:, :, np.newaxis]).sum(axis=1)
        knn_embs /= np.linalg.norm(knn_embs, axis=-1, keepdims=True)

        knn_qembs = knn_embs
        knn_gembs = knn_embs[np.array(unknown)]
        knn_body_sim = np.dot(knn_qembs, knn_gembs.T)

        body_sim = (body_sim + knn_body_sim) / 2

        decide_sim = np.dot(embs, embs[np.array(decided)].T)

        for i in range(num_cast):
            ind = np.unique(result[cast_ids[i]])
            sim = body_sim[ind]
            if len(ind) > 3:
                sim = sim[np.argpartition(1 - sim, 3, axis=0), np.arange(len(unknown))][:3]
            exp_sim = np.exp(sim)
            weight = exp_sim / exp_sim.sum(0)
            sim = (sim * weight).sum(0)

            end = []
            idx = np.where(candi_file_cnt[ind] > 1)[0]
            for index in ind[idx]:
                jj = np.where(candi_files == candi_files[index])[0]
                end.extend(jj)
            end = set(end)

            for j in np.argsort(1 - sim):
                jj = unknown[j]
                if jj not in end:
                    result[cast_ids[i]].append(jj)

            for j in end:
                result[cast_ids[i]].append(j)

            sim = decide_sim[ind].max(0)
            for j in np.argsort(1 - sim):
                result[cast_ids[i]].append(decided[j])

        for i, cast_id in enumerate(cast_ids):
            vals = result[cast_id]
            vals = unique_list(vals)
            result[cast_id] = [candi_ids[j] for j in vals]

    with open(args.submission, 'w') as f:
        for key, vals in result.items():
            f.writelines('{} {}\n'.format(key, ','.join(vals)))


if __name__ == '__main__':
    decide()
    if args.mode == 'val':
        eval(args.submission, args.gt)
