import numpy as np
from scipy.spatial import cKDTree

class SegmentIndex:
    """Exact K nearest supplied straight segments in the local projected CRS.

    Densified samples provide a spatial index, not final distance estimates.
    Every segment includes its endpoints, with sample gaps <= spacing metres.
    Any segment within distance U has a sample within U + spacing/2. Querying
    this ball and reranking by exact point-to-segment distance therefore
    includes every segment that can beat the current Kth distance bound U.
    Arc row IDs are preserved, including geometrically identical reverse arcs.
    """
    def __init__(self, xy_nodes, arcs, spacing=30.0):
        if spacing <= 0:
            raise ValueError("spacing must be positive")
        self.arcs = arcs
        self.start = xy_nodes[arcs[:, 0]]
        self.vector = xy_nodes[arcs[:, 1]] - self.start
        self.length2 = (self.vector ** 2).sum(axis=1)
        self.length = np.sqrt(self.length2)
        self.spacing = spacing
        intervals = np.maximum(1, np.ceil(self.length / spacing).astype(int))
        counts = intervals + 1
        self.sample_arc = np.repeat(np.arange(len(arcs)), counts)
        beginnings = np.repeat(np.cumsum(counts) - counts, counts)
        fraction = (np.arange(counts.sum()) - beginnings) / np.repeat(intervals, counts)
        samples = self.start[self.sample_arc] + fraction[:, None] * self.vector[self.sample_arc]
        self.tree = cKDTree(samples)

    def distances(self, point, ids):
        ids = np.asarray(ids, dtype=int)
        delta = point - self.start[ids]
        fraction = np.divide((delta * self.vector[ids]).sum(axis=1), self.length2[ids],
                             out=np.zeros(len(ids)), where=self.length2[ids] > 0)
        fraction = np.clip(fraction, 0, 1)
        snapped = self.start[ids] + fraction[:, None] * self.vector[ids]
        return np.linalg.norm(point - snapped, axis=1), fraction, snapped

    def nearest(self, point, k=5):
        k = min(k, len(self.arcs))
        if k < 1:
            raise ValueError("k and number of arcs must be positive")
        n = min(max(k * 8, 32), len(self.sample_arc))
        while True:
            _, sample_ids = self.tree.query(point, k=n)
            ids = np.unique(self.sample_arc[np.atleast_1d(sample_ids)])
            if len(ids) >= k:
                break
            n = min(n * 2, len(self.sample_arc))
        d, _, _ = self.distances(point, ids)
        upper = np.partition(d, k - 1)[k - 1]
        sample_ids = self.tree.query_ball_point(point, upper + self.spacing / 2 + 1e-7)
        ids = np.unique(self.sample_arc[sample_ids])
        d, fraction, snapped = self.distances(point, ids)
        # Round sub-micrometre floating noise so reverse geometry ties use arc ID.
        order = np.lexsort((ids, np.round(d, 7)))[:k]
        return ids[order], d[order], fraction[order], snapped[order]


