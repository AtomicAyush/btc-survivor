"""A one hidden layer tanh network in numpy, with a hand written backward pass, Huber loss and AdamW."""

from __future__ import annotations

import numpy as np


class MLP:
    """All parameters live in one flat array, with views for W1, b1, W2 and b2."""

    def __init__(self, n_in: int, hidden: int, params: np.ndarray | None = None) -> None:
        self.n_in, self.hidden = n_in, hidden
        self.size = n_in * hidden + hidden + hidden + 1
        self.params = np.zeros(self.size) if params is None else np.array(params, dtype=float)
        if self.params.shape != (self.size,):
            raise ValueError("parameter vector has the wrong size")
        a = n_in * hidden
        self.W1 = self.params[:a].reshape(hidden, n_in)
        self.b1 = self.params[a:a + hidden]
        self.W2 = self.params[a + hidden:a + 2 * hidden]
        self.b2 = self.params[a + 2 * hidden:]
        self.decay_mask = np.zeros(self.size)
        self.decay_mask[:a] = 1.0
        self.decay_mask[a + hidden:a + 2 * hidden] = 1.0

    @classmethod
    def random(cls, n_in: int, hidden: int, rng: np.random.Generator, output_scale: float) -> "MLP":
        """Xavier uniform weights, zero biases, and the output layer shrunk so early predictions are small."""
        model = cls(n_in, hidden)
        limit1 = np.sqrt(6.0 / (n_in + hidden))
        limit2 = np.sqrt(6.0 / (hidden + 1))
        model.W1[:] = rng.uniform(-limit1, limit1, (hidden, n_in))
        model.W2[:] = rng.uniform(-limit2, limit2, hidden) * output_scale
        return model

    def predict(self, x: np.ndarray) -> float:
        return float(np.tanh(self.W1 @ x + self.b1) @ self.W2 + self.b2[0])

    def loss_and_grad(self, X: np.ndarray, y: np.ndarray, delta: float) -> tuple[float, np.ndarray]:
        """Mean Huber loss over the batch and its gradient with respect to the flat parameters."""
        H = np.tanh(X @ self.W1.T + self.b1)
        out = H @ self.W2 + self.b2[0]
        r = out - y
        small = np.abs(r) <= delta
        loss = float(np.mean(np.where(small, 0.5 * r * r, delta * (np.abs(r) - 0.5 * delta))))
        g = np.clip(r, -delta, delta) / len(y)
        dA = np.outer(g, self.W2) * (1.0 - H * H)
        grad = np.concatenate([(dA.T @ X).ravel(), dA.sum(axis=0), H.T @ g, [g.sum()]])
        return loss, grad


class AdamW:
    def __init__(self, size: int, beta1: float, beta2: float, eps: float = 1e-8) -> None:
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = np.zeros(size)
        self.v = np.zeros(size)
        self.t = 0

    def step(self, model: MLP, grad: np.ndarray, lr: float, weight_decay: float, grad_clip: float) -> None:
        """Clip the gradient norm, take one Adam step, and decay the weight matrices but not the biases."""
        norm = float(np.linalg.norm(grad))
        if norm > grad_clip:
            grad = grad * (grad_clip / norm)
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad * grad
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        model.params -= lr * (m_hat / (np.sqrt(v_hat) + self.eps) + weight_decay * model.decay_mask * model.params)
