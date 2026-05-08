from typing import List
import random
import math
# typing 是干嘛的
# 感觉难度还是对数组维度的掌握，还是挺难的
class KMeans:
    def __init__(self, k: int, max_iters: int = 100, tol: float = 1e-4, seed: int = 42):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol # 中心移动距离阈值
        self.seed = seed
        self.centroids = []

    def fit(self, X: List[List[float]]) -> List[int]:
        # 输入训练数据，【【多个维度，一个数据】，输出label[[int]]每个数据的label
        if not X:
            return []

        random.seed(self.seed)
        n = len(X) # 样本数
        d = len(X[0]) # 维度数

        # 1. 随机初始化 k 个中心
        indices = random.sample(range(n), self.k)
        #它会从 0 ~ n-1 这些下标里，随机选出 k 个不同的下标。组成一个list
        self.centroids = [X[i][:] for i in indices]
        #根据刚才选中的下标，从数据里取出对应的点，作为初始中心。 X 中的第i个元素
        labels = [0] * n
        # 初试时都是0
        for _ in range(self.max_iters): # 外循环iteration
            # 2. assignment: 每个点分配到最近的中心
            for i in range(n):
                labels[i] = self._closest_centroid(X[i])

            # 3. update: 重新计算每个簇的中心
            new_centroids = [] # 存放新的中心
            # 内循环clusters 
            for cluster_id in range(self.k):# 先处理一个cluster 再处理下一个 
                # 所有的属于这个clusterid的所有的点，组成一个这个cluster点的集合
                cluster_points = [X[i] for i in range(n) if labels[i] == cluster_id]
                if not cluster_points: #都划分到别的cluster id下面了
                    # empty cluster: 随机重新选一个点
                    new_centroids.append(X[random.randint(0, n - 1)][:])
                else:
                    centroid = [] # 用来放簇的中心，为什么是个列表是因为【x,y]
                    for j in range(d): # 对每个维度算average
                        avg = sum(point[j] for point in cluster_points) / len(cluster_points)
                        centroid.append(avg)
                    new_centroids.append(centroid)

            # 4. 判断是否收敛，如果最大的位移小于tol说明收敛了
            max_shift = 0
            for i in range(self.k):
                shift = self._distance(self.centroids[i], new_centroids[i])
                max_shift = max(max_shift, shift)

            self.centroids = new_centroids

            if max_shift < self.tol:
                break

        return labels

    def predict(self, X: List[List[float]]) -> List[int]: # 这行代码是输出的时候用的
        return [self._closest_centroid(x) for x in X]

    def _closest_centroid(self, point: List[float]) -> int: # 距离这个点最近的中心点是哪个？
        best_idx = 0
        best_dist = float("inf")

        for i, centroid in enumerate(self.centroids):
            dist = self._distance(point, centroid)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        return best_idx

    def _distance(self, a: List[float], b: List[float]) -> float: # 输入时两个list这点要注意
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))