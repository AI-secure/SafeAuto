import os
import numpy as np
import pickle
from scipy.special import softmax
from pgm.config import BDDX, DriveLM

def compute_satisfaction(data, formulas):
    # Formulas are pure arithmetic, so evaluate them on whole columns at once.
    args = np.asarray(data).T
    return np.array([f(args) for f in formulas], dtype=float).T

def pad_llm_predicates(data, config):
    """Pad predicate vectors that were extracted without the MLLM action block.

    Vectors generated before MLLM training only contain the observed
    predicates (actions + environment + control signals). The MLLM action
    predicates are unknown at PGM-training time, so they are set to 0, which
    makes every MLLM-related rule trivially satisfied.
    """
    data = np.asarray(data, dtype=float)
    full_dim = config.action_num + config.condition_num
    llm_dim = sum(1 for name in config.predicate if name.endswith('_LLM'))
    if data.shape[1] == full_dim:
        return data
    if data.shape[1] == full_dim - llm_dim:
        return np.concatenate([data, np.zeros((data.shape[0], llm_dim))], axis=1)
    raise ValueError(
        f"Predicate vectors have {data.shape[1]} dims, expected {full_dim} "
        f"(or {full_dim - llm_dim} without the MLLM action block)."
    )

def log_sum_exp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))

def compute_log_likelihood(sat, w, reg):
    ws = sat @ w
    return np.sum(ws) - log_sum_exp(ws) - 0.5 * reg * np.sum(w ** 2)

def update_weights(w, sat, lr, reg, trainable=None):
    ws = sat @ w
    exp_ws = np.exp(ws - log_sum_exp(ws))
    grad = np.sum(sat, 0) - np.sum(exp_ws[:, None] * sat, 0) - reg * w
    if trainable is not None:
        grad = grad * trainable
    return w + lr * grad

def generate_possible_instances(cond, action_num, cond_num):
    return np.array([
        np.concatenate([np.eye(action_num)[i], cond])
        for i in range(action_num)
    ])

def compute_accuracy(true, pred):
    return np.mean(true == pred)

class PGM:
    def __init__(self, config, weights=None, learning_rate=1e-5, max_iter=10000, tol=1e-6, regularization=0.01):
        self.formulas = config.formulas
        self.action_num = config.action_num
        self.condition_num = config.condition_num
        self.config = config
        self.weights = np.array(weights) if weights is not None else np.ones(len(self.formulas))
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.regularization = regularization

    def train_mln(self, data, save_path):
        data = pad_llm_predicates(data, self.config)
        w, prev_ll, prev_acc = self.weights, -np.inf, -np.inf
        true_labels = np.argmax(data[:, :self.action_num], 1)
        # The data never changes during training, so both satisfaction tables
        # are computed once outside the loop.
        sat = compute_satisfaction(data, self.formulas)
        conds = np.repeat(data[:, self.action_num:], self.action_num, axis=0)
        acts = np.tile(np.eye(self.action_num), (len(data), 1))
        cand_sat = compute_satisfaction(np.concatenate([acts, conds], axis=1), self.formulas)
        cand_sat = cand_sat.reshape(len(data), self.action_num, -1)
        # Formulas whose satisfaction never varies on the training data (e.g.
        # rules over the zero-padded MLLM predicates) carry no gradient
        # information; the dataset-level softmax would otherwise inflate their
        # weights without bound, so they stay at their initial value.
        trainable = sat.std(0) > 0
        for it in range(self.max_iter):
            ll = compute_log_likelihood(sat, w, self.regularization)
            probs = softmax(cand_sat @ w, axis=1)
            avg_prob = probs[np.arange(len(data)), true_labels].mean()
            print(f"[INFO] Iter {it}, Avg GT Prob: {avg_prob}, LogLik: {ll}")
            if abs(ll - prev_ll) < self.tol or abs(avg_prob - prev_acc) < self.tol:
                np.save(save_path, w)
                print(f"[INFO] Converged. Saving weights to {save_path}.")
                break
            if avg_prob > prev_acc:
                prev_acc = avg_prob
                np.save(save_path, w)
                print(f"[INFO] Saving weights at iter {it} to {save_path}.")
            prev_ll = ll
            w = update_weights(w, sat, self.learning_rate, self.regularization, trainable)
            self.weights = w
        return w

    def eval(self, test_data):
        test_data = pad_llm_predicates(test_data, self.config)
        true_labels = np.argmax(test_data[:, :self.action_num], 1)
        preds = [self.infer_action_probability(x[self.action_num:])[1] for x in test_data]
        return compute_accuracy(true_labels, np.array(preds))

    def infer_action_probability(self, cond):
        instances = generate_possible_instances(cond, self.action_num, self.condition_num)
        sat = compute_satisfaction(instances, self.formulas)
        probs = softmax(sat @ self.weights)
        return probs, np.argmax(probs)

    def validate_instance(self, instance):
        return [i for i, f in enumerate(self.formulas[:self.config.hardrule_num]) if f(instance) != 1]

    def compute_instance_probability(self, instance):
        cond = instance[self.action_num:]
        probs, _ = self.infer_action_probability(cond)
        return probs[instance[:self.action_num].argmax()]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PGM Training Script")
    parser.add_argument('--dataset', type=str, default='bddx', help='Training dataset')
    parser.add_argument('--weights', type=str, default=None, help='Path to initial weights (optional)')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for training')
    parser.add_argument('--max_iter', type=int, default=10000, help='Maximum number of training iterations')
    parser.add_argument('--tol', type=float, default=1e-6, help='Tolerance for convergence')
    parser.add_argument('--regularization', type=float, default=0.01, help='Regularization strength')
    parser.add_argument('--output_dir', type=str, default='pgm/ckpts/pgm', help='Path to save trained weights')
    args = parser.parse_args()

    config = {'bddx': BDDX, 'drivelm': DriveLM}.get(args.dataset, None)
    if config is None:
        raise ValueError("Unsupported dataset.")
    config = config()
    os.makedirs(args.output_dir, exist_ok=True)
    train_vector_path = f"pgm/predicates/{args.dataset}/train_vectors.pkl"
    with open(train_vector_path, 'rb') as f:
        train_data = np.array(pickle.load(f))
    weights = np.load(args.weights) if args.weights else None
    pgm = PGM(config, weights=weights, learning_rate=args.learning_rate, max_iter=args.max_iter, tol=args.tol, regularization=args.regularization)
    pgm.train_mln(train_data, save_path=f"{args.output_dir}/{args.dataset}_weights.npy")
