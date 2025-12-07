import numpy as np
from scipy.spatial.distance import cdist

def get_id_and_cam(fname):
    pid = int(fname[:4])
    cam = int(fname.split('_c')[1][0])
    return pid, cam

def evaluate_all_queries(query_emb_path, gallery_emb_path, cmc_topk=(1, 5, 10)):
    # Load embeddings
    query_embeddings = np.load(query_emb_path, allow_pickle=True).item()
    gallery_embeddings = np.load(gallery_emb_path, allow_pickle=True).item()

    query_filenames = sorted(query_embeddings.keys())
    gallery_filenames = sorted(gallery_embeddings.keys())

    gallery_features = np.array([gallery_embeddings[f] for f in gallery_filenames])
    gallery_ids = [None for _ in gallery_filenames]
    gallery_cams = [None for _ in gallery_filenames]

    all_results = []
    for query_img_name in query_filenames:
        query_feature = query_embeddings[query_img_name]
        query_id, query_cam = get_id_and_cam(query_img_name)
        # Compute distances
        distmat = cdist(query_feature.reshape(1, -1), gallery_features, metric="euclidean")[0]
        # Rank gallery images by distance (lower is better)
        ranked_indices = np.argsort(distmat)
        ranked_gallery = [(gallery_filenames[i], distmat[i], gallery_ids[i], gallery_cams[i]) for i in ranked_indices]
        # Store results for this query
        all_results.append({
            "query": query_img_name,
            "query_id": query_id,
            "query_cam": query_cam,
            "ranked_gallery": ranked_gallery
        })
        print(f"Query: {query_img_name}")
        for rank, (gname, score, gid, gcam) in enumerate(ranked_gallery[:10], 1):
            print(f"  Rank {rank}: {gname} | Score: {score:.4f}")
    return all_results

if __name__ == "__main__":
    query_emb_path = r"e:/Projects/Honours/query_im_end_embeddings.npy"
    gallery_emb_path = r"e:/Projects/Honours/gallery_im_end_embeddings.npy"
    evaluate_all_queries(query_emb_path, gallery_emb_path)
