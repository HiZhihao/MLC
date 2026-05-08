import numpy as np


class SoftmaxCrossEntropy:
    def __init__(self):
        self.probs = None   # shape: (N, C)存储之前softmax 算的概率
        self.y = None       # shape: (N,)

    def forward(self, logits, y):
        """
        logits: shape (N, C)
        y:      shape (N,)   each value in [0, C-1]
        return:
            loss: float
            probs: shape (N, C)
            # 行是样本，列是真实类别、例如：
            # logits = np.array([
            [2.0, 1.0, 0.1], 样本1
            [0.5, 2.5, 0.3] 样本2 
])
np.max(logits, axis=1, keepdims=True)axis=1 的意思是：沿着“列方向”找每一行的最大值。如果 keepdims=True，结果形状是 (2, 1)
沿着“列方向”找每一行的最大值。
        """
        # 1) numerical stability就是每一行都减去这一行自己的最大值。为什么要减最大值，为了防止 exp() 爆掉。你直接 np.exp(1001)，可能就溢出了。
        #这叫 数值稳定性处理。
        shifted = logits - np.max(logits, axis=1, keepdims=True)

        # 2) softmax 每一行都除以自己的行和。
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # 3) cross entropy
        N = logits.shape[0]#拿 batch size如果 logits shape 是 (2, 3)，那么：意思是 batch 里有 2 个样本。
        correct_class_probs = probs[np.arange(N), y] # np.arange(2) = np.array([0, 1])，y = np.array([0, 1])
        """probs[np.arange(N), y]

        就是：

        probs[[0, 1], [0, 1]]

        意思是：

        取 probs[0, 0]
        取 probs[1, 1]

        也就是：[i for i in range(2)]

        第 0 个样本的真实类别概率
        第 1 个样本的真实类别概率 我们现在只关心：模型给真实类别分了多少概率。
        。"""
        loss = -np.mean(np.log(correct_class_probs + 1e-12))
        """correct_class_probs + 1e-12
        这是为了防止概率极端接近 0 时，log(0) 出问题。
        比如如果概率是 0，log(0) 是不合法的。为什么前面要加负号因为正确类别概率越大，log(p) 越接近 0。
         -log(真实类别的概率)log(0.01) ≈ -4.605 log(1)    = 0 概率 p 一定在 0 到 1 之间。真实类别概率越大，log(p) 越接近 0。
         真实类别概率越小，log(p) 就越负。但是我们想要的是预测越对，loss 越小。
        log(0.9) 是一个接近 0 的负数,log(0.01) 是一个很小的负数，绝对值很大预测对的时候 loss 小预测错的时候 loss 大"""

        # cache for backward
        self.probs = probs
        self.y = y

        return loss, probs

    def backward(self):
        """
        return:
            dlogits: shape (N, C)
        """
        N = self.probs.shape[0]

        dlogits = self.probs.copy()#如果不 copy()，后面减 1 会直接把缓存里的概率改掉。
        dlogits[np.arange(N), self.y] -= 1 #这句就是把正确类别的位置减 1。
        #为什么是减 1 因为 softmax 和 cross entropy 合并后的梯度公式是：
         #dlogits = probs - one_hot(y)
        dlogits /= N

        return dlogits


class LinearClassifier:
    def __init__(self, in_dim, num_classes):
        # small random init
        self.W = 0.01 * np.random.randn(in_dim, num_classes)   # shape: (D, C)
        self.b = np.zeros((1, num_classes))                    # shape: (1, C)

    def forward(self, X):
        """
        X: shape (N, D)
        return logits: shape (N, C)
        """
        return X @ self.W + self.b

    def backward(self, X, dlogits):
        """
        X: shape (N, D)
        dlogits: shape (N, C)

        return:
            dX: shape (N, D)
            dW: shape (D, C)
            db: shape (1, C)
        """
        dW = X.T @ dlogits
        db = np.sum(dlogits, axis=0, keepdims=True)
        dX = dlogits @ self.W.T
        return dX, dW, db

    def step(self, dW, db, lr):
        self.W -= lr * dW
        self.b -= lr * db


def accuracy(logits, y):
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == y)


def train(X, y, in_dim, num_classes, epochs=200, lr=0.1, batch_size=32):
    """
    X: shape (N, D)
    y: shape (N,)
    """
    model = LinearClassifier(in_dim, num_classes)
    criterion = SoftmaxCrossEntropy()

    N = X.shape[0]
    losses = []

    for epoch in range(epochs):
        # shuffle each epoch
        indices = np.random.permutation(N)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, N, batch_size):
            end = start + batch_size
            xb = X_shuffled[start:end]   # shape: (B, D)
            yb = y_shuffled[start:end]   # shape: (B,)

            # forward
            logits = model.forward(xb)               # (B, C)
            loss, probs = criterion.forward(logits, yb)

            # backward
            dlogits = criterion.backward()           # (B, C)
            _, dW, db = model.backward(xb, dlogits)

            # update
            model.step(dW, db, lr)

            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        if epoch % 20 == 0 or epoch == epochs - 1:
            train_logits = model.forward(X)
            train_acc = accuracy(train_logits, y)
            print(f"epoch={epoch:03d} loss={avg_loss:.4f} acc={train_acc:.4f}")

    return model, losses