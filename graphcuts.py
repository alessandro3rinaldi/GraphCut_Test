import numpy as np
from sklearn.neighbors import KDTree


def build_graph_weights(points, k=6, inlier_bias=1.0):
    """
    Given a point cloud builds graph weights 
    based on each node's distance to its k neighbors
    """
    tree = KDTree(points)                           # build spatial index for fast neighbor lookup
    distances, indices = tree.query(points, k=k+1)  # query k+1 because the first neighbor is the point itself
    distances = distances[:, 1:]                    # remove self (distance = 0)
    indices   = indices[:, 1:]                      # remove self index

    mean_dist = distances.mean(axis=1)              # average distance to k neighbors, one value per point
    mu        = mean_dist.mean()                    # global mean distance, used as normalization scale

    raw      = np.exp(-mean_dist / mu)              # gaussian decay: dense neighborhoods implies high value (close to 1)
    source_w = raw * inlier_bias                    # inlier attraction: boosted by bias, always non-negative
    sink_w   = 1.0 - raw                            # outlier attraction: complement of raw, always in [0, 1]

    return indices, distances, mu, source_w, sink_w


def graphcut_pymaxflow(points, k=6, inlier_bias=1.0):
    """
    Uses the Boykov-Kolmogorov max-flow algorithm via PyMaxflow.
    Fastest approach.
    """
    import maxflow

    N = len(points)
    indices, distances, mu, source_w, sink_w = build_graph_weights(points, k, inlier_bias)

    g = maxflow.Graph[float](N, N * k)  # preallocate graph with N nodes and N*k edges
    g.add_nodes(N)                      # add one node per point

    for i in range(N):
        g.add_tedge(i, source_w[i], sink_w[i])  # connect each node to source and sink with unary weights

    for i in range(N):
        for j_idx, j in enumerate(indices[i]):
            if j > i:                                        # avoid adding each edge twice
                w = np.exp(-distances[i, j_idx] / mu)        # pairwise weight: closer points → stronger connection
                g.add_edge(i, j, w, w)                       # undirected edge: same weight in both directions

    g.maxflow()                                              # run BK max-flow → finds the min cut

    return np.array([g.get_segment(i) for i in range(N)])   # 0 = source side (inlier), 1 = sink side (outlier)


def graphcut_networkx(points, k=6, inlier_bias=1.0):
    """
    Uses NetworkX's minimum cut on a general directed graph.
    Adds a virtual source node S and sink node T connected to all points.
    Slower than PyMaxflow but uses no special library.
    """
    import networkx as nx

    N = len(points)
    indices, distances, mu, source_w, sink_w = build_graph_weights(points, k, inlier_bias)

    G = nx.DiGraph()  # directed graph (required by nx.minimum_cut)

    # Adding terminal nodes and unary weighted edges
    S = N
    T = N + 1

    for i in range(N):
        G.add_edge(S, i, capacity=source_w[i])
        G.add_edge(i, T, capacity=sink_w[i])

    for i in range(N):
        for j_idx, j in enumerate(indices[i]):
            if j > i:                                        # same as above
                w = np.exp(-distances[i, j_idx] / mu)
                G.add_edge(i, j, capacity=w)
                G.add_edge(j, i, capacity=w)

    _, (source_side, _) = nx.minimum_cut(G, S, T)  # returns cut value and the two partitions

    labels = np.ones(N, dtype=int)   # initialize everyone as outlier (1)
    for node in source_side:
        if node != S:                # skip the virtual source node
            labels[node] = 0         # source side = inlier

    return labels


def ransac_circle(points, epsilon=0.3, n_iter=500):
    """
    RANSAC for circle fitting.
    Returns labels, best center, best radius.
    """
    pts2d          = points[:, :2]                          # use only x, y
    best_labels    = np.ones(len(points), dtype=int)        # initialize everyone as outlier
    best_n_inliers = 0                                      # track best inlier count so far
    best_center    = None                                   # track best circle center
    best_radius    = None                                   # track best circle radius

    for _ in range(n_iter):

        idx        = np.random.choice(len(pts2d), 3, replace=False)  # sample 3 random point indices
        p1, p2, p3 = pts2d[idx[0]], pts2d[idx[1]], pts2d[idx[2]]     # extract the 3 points

        # Fit circle through 3 points (closed form solution)
        ax, ay = p1
        bx, by = p2
        cx, cy = p3
        D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))  # determinant of the linear system
        if abs(D) < 1e-6:
            continue  # degenerate case: points are collinear, no circle exists

        # Circumcenter formula: intersection of perpendicular bisectors
        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / D
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / D
        center = np.array([ux, uy])
        radius = np.linalg.norm(p1 - center)                # radius = distance from center to any of the 3 points

        # Count inliers: points within epsilon of the circle
        residuals = np.abs(np.linalg.norm(pts2d - center, axis=1) - radius)  # geometric distance from circle
        labels    = (residuals > epsilon).astype(int)                         # 0 = inlier, 1 = outlier
        n_inliers = (labels == 0).sum()                                       # count inliers

        if n_inliers > best_n_inliers:   # keep the best model found so far
            best_n_inliers = n_inliers
            best_labels    = labels
            best_center    = center      # save center of best circle
            best_radius    = radius      # save radius of best circle

    print(f"RANSAC: best inliers = {best_n_inliers}")
    return best_labels, best_center, best_radius


def ransac_with_graphcut(points, epsilon=0.3, n_iter=500, k=6):
    """
    RANSAC with Graph Cut inside every iteration.
    At each iteration:
      1. Sample 3 points, fit a circle
      2. Run Graph Cut to refine inliers using:
         - Unary term:    normalized residual from the circle
         - Pairwise term: residual similarity between neighbors
    Keep the iteration with the most inliers after Graph Cut.
    Returns labels, best center, best radius.
    """
    import maxflow

    pts2d = points[:, :2]  # use only x, y
    N     = len(points)

    tree = KDTree(points)
    distances, indices = tree.query(points, k=k + 1)
    distances = distances[:, 1:]   # remove self
    indices   = indices[:, 1:]     # remove self
    mu        = distances.mean()   # global mean distance for normalization

    best_labels    = np.ones(N, dtype=int)  # initialize everyone as outlier
    best_n_inliers = 0                      # track best inlier count so far
    best_center    = None                   # track best circle center
    best_radius    = None                   # track best circle radius

    for _ in range(n_iter):

        # --- Step 1: RANSAC: fit circle ---
        idx        = np.random.choice(N, 3, replace=False)
        p1, p2, p3 = pts2d[idx[0]], pts2d[idx[1]], pts2d[idx[2]]

        ax, ay = p1
        bx, by = p2
        cx, cy = p3
        D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(D) < 1e-6:
            continue

        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / D
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / D
        center = np.array([ux, uy])
        radius = np.linalg.norm(p1 - center)

        # --- Step 2: unary term — normalized residual from circle ---
        residuals = np.abs(np.linalg.norm(pts2d - center, axis=1) - radius)  # geometric distance from circle
        res_norm  = np.clip(residuals / epsilon, 0, 1)  # normalize to [0, 1]
        source_w  = 1 - res_norm  # high weight toward inlier if close to circle
        sink_w    = res_norm      # high weight toward outlier if far from circle

        # --- Step 3: Graph Cut ---
        g = maxflow.Graph[float](N, N * k)
        g.add_nodes(N)

        for i in range(N):
            g.add_tedge(i, source_w[i], sink_w[i])

        for i in range(N):
            for j_idx, j in enumerate(indices[i]):
                if j > i:
                    res_i      = res_norm[i]
                    res_j      = res_norm[j]
                    similarity = 1 - abs(res_i - res_j)  # high if both points have similar residuals
                    g.add_edge(i, j, similarity, similarity)

        g.maxflow()
        labels    = np.array([g.get_segment(i) for i in range(N)])
        n_inliers = (labels == 0).sum()

        if n_inliers > best_n_inliers:   # keep the best model found so far
            best_n_inliers = n_inliers
            best_labels    = labels
            best_center    = center      # save center of best circle
            best_radius    = radius      # save radius of best circle

    print(f"RANSAC+GC: best inliers = {best_n_inliers}")
    return best_labels, best_center, best_radius