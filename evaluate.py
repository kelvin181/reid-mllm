import os
import numpy as np
from scipy.spatial.distance import cdist

def get_id_and_cam(fname):
    """Extracts person ID and camera ID from filename. Returns (None, None) if invalid."""
    try:
        pid = int(fname[:4])
        cam = int(fname.split('_c')[1][0])
        return pid, cam
    except Exception as e:
        print(f"Warning: Could not parse id/cam from filename '{fname}': {e}")
        return None, None

def compute_cmc_map(distmat, query_ids, gallery_ids, query_cams, gallery_cams, cmc_topk=(1, 5, 10, 1000)):
    """Compute CMC and mAP. Simple implementation for Market1501-style evaluation."""
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[None, :] == query_ids[:, None])
    cmc_scores = np.zeros(len(cmc_topk))
    APs = []
    for i in range(num_q):
        # Remove gallery samples with same pid and same cam as query
        valid = ~((gallery_ids == query_ids[i]) & (gallery_cams == query_cams[i]))
        y_true = matches[i, valid]
        y_score = -distmat[i][valid]
        if not np.any(y_true):
            continue
        # CMC
        index = np.where(y_true)[0]
        first_index = index[0]
        for j, k in enumerate(cmc_topk):
            if first_index < k:
                cmc_scores[j:] += 1
                break
        # mAP
        # Compute precision at each correct match
        num_rel = y_true.sum()
        tmp_cmc = y_true.cumsum()
        precision = tmp_cmc / (np.arange(len(y_true)) + 1)
        AP = (precision * y_true).sum() / num_rel
        APs.append(AP)
    cmc_scores = cmc_scores / num_q
    mAP = np.mean(APs) if APs else 0.0
    return cmc_scores, mAP


def evaluate_all_queries(query_emb_path, gallery_emb_path, cmc_topk=(1, 5, 10, 1000)):
    print(f"Using query embeddings from: {query_emb_path}")
    print(f"Using gallery embeddings from: {gallery_emb_path}")
    query_embeddings = np.load(query_emb_path, allow_pickle=True).item()
    gallery_embeddings = np.load(gallery_emb_path, allow_pickle=True).item()

    query_filenames = sorted(query_embeddings.keys())
    gallery_filenames = sorted(gallery_embeddings.keys())


    # Only keep filenames with valid id/cam
    def valid_id_cam(f):
        pid, cam = get_id_and_cam(f)
        return pid is not None and cam is not None

    query_valid = [f for f in query_filenames if valid_id_cam(f)]
    gallery_valid = [f for f in gallery_filenames if valid_id_cam(f)]

    query_features = np.array([query_embeddings[f] for f in query_valid])
    gallery_features = np.array([gallery_embeddings[f] for f in gallery_valid])

    # Normalize each query and gallery feature to unit length
    query_norms = np.linalg.norm(query_features, axis=1, keepdims=True)
    query_features = query_features / np.clip(query_norms, a_min=1e-12, a_max=None)
    gallery_norms = np.linalg.norm(gallery_features, axis=1, keepdims=True)
    gallery_features = gallery_features / np.clip(gallery_norms, a_min=1e-12, a_max=None)

    query_ids = np.array([get_id_and_cam(f)[0] for f in query_valid])
    query_cams = np.array([get_id_and_cam(f)[1] for f in query_valid])
    gallery_ids = np.array([get_id_and_cam(f)[0] for f in gallery_valid])
    gallery_cams = np.array([get_id_and_cam(f)[1] for f in gallery_valid])

    # Compute full distance matrix

    distmat = cdist(query_features, gallery_features, metric="euclidean")

    # Compute CMC and mAP

    cmc_scores, mAP = compute_cmc_map(distmat, query_ids, gallery_ids, query_cams, gallery_cams, cmc_topk)

    print(f"Mean AP: {mAP:.2%}")
    for k, score in zip(cmc_topk, cmc_scores):
        print(f"CMC top-{k}: {score:.2%}")


    # For each query, print the top 5 images and the first correct match
    for idx, query_img_name in enumerate(query_valid):
        ranked_indices = np.argsort(distmat[idx])
        print(f"Query: {query_img_name}")
        # Print top 5 ranked gallery images
        # Score is the euclidean distance between embeddings
        print("  Top 5 gallery images:")
        for rank, i in enumerate(ranked_indices[:5], 1):
            print(f"    Rank {rank}: {gallery_valid[i]} | Score: {distmat[idx, i]:.4f}")
        # Find and print the first correct match
        found = False
        for rank, i in enumerate(ranked_indices, 1):
            g_pid = gallery_ids[i]
            g_cam = gallery_cams[i]
            if g_pid == query_ids[idx] and g_cam != query_cams[idx]:
                print(f"  First correct match at rank {rank}: {gallery_valid[i]}")
                found = True
                break
        if not found:
            print("  No correct match found in gallery.")

    return {
        "mAP": mAP,
        "cmc_scores": cmc_scores,
        "distmat": distmat,
        "query_filenames": query_valid,
        "gallery_filenames": gallery_valid
    }

if __name__ == "__main__":
    query_emb_path = os.path.join(os.getcwd(), "query_im_end_embeddings.npy")
    gallery_emb_path = os.path.join(os.getcwd(), "gallery_im_end_embeddings.npy")
    results = evaluate_all_queries(query_emb_path, gallery_emb_path)
    print(results["mAP"], results["cmc_scores"], results["distmat"].shape)
