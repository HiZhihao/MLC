from typing import List
import random
import math


class KMeans:
    def __init__(self, k: int, max_iters: int = 100, tol: float = 1e-4, seed: int = 42):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.seed = seed
        self.centroids = []

    def fit(self, X: List[List[float]]) -> List[int]:
        if not X:
            return []

        random.seed(self.seed)
        n = len(X)
        d = len(X[0])

        # 1. 随机初始化 k 个中心
        indices = random.sample(range(n), self.k)
        self.centroids = [X[i][:] for i in indices]

        labels = [0] * n

        for _ in range(self.max_iters):
            # 2. assignment: 每个点分配到最近的中心
            for i in range(n):
                labels[i] = self._closest_centroid(X[i])

            # 3. update: 重新计算每个簇的中心
            new_centroids = []
            for cluster_id in range(self.k):
                cluster_points = [X[i] for i in range(n) if labels[i] == cluster_id]

                if not cluster_points:
                    # empty cluster: 随机重新选一个点
                    new_centroids.append(X[random.randint(0, n - 1)][:])
                else:
                    centroid = []
                    for j in range(d):
                        avg = sum(point[j] for point in cluster_points) / len(cluster_points)
                        centroid.append(avg)
                    new_centroids.append(centroid)

            # 4. 判断是否收敛
            max_shift = 0
            for i in range(self.k):
                shift = self._distance(self.centroids[i], new_centroids[i])
                max_shift = max(max_shift, shift)

            self.centroids = new_centroids

            if max_shift < self.tol:
                break

        return labels

    def predict(self, X: List[List[float]]) -> List[int]:
        return [self._closest_centroid(x) for x in X]

    def _closest_centroid(self, point: List[float]) -> int:
        best_idx = 0
        best_dist = float("inf")

        for i, centroid in enumerate(self.centroids):
            dist = self._distance(point, centroid)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        return best_idx

    def _distance(self, a: List[float], b: List[float]) -> float:
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))