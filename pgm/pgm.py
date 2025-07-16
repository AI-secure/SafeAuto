import os
import numpy as np
import pickle
from scipy.special import softmax
from pgm.config import BDDX, DriveLM

def compute_satisfaction(data, formulas):
    return np.array([[f(x) for f in formulas] for x in data])

def log_sum_exp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))

def compute_log_likelihood(sat, w, reg):
    ws = sat @ w
    return np.sum(ws) - log_sum_exp(ws) - 0.5 * reg * np.sum(w ** 2)

def update_weights(w, sat, lr, reg):
    ws = sat @ w
    exp_ws = np.exp(ws - log_sum_exp(ws))
    grad = np.sum(sat, 0) - np.sum(exp_ws[:, None] * sat, 0) - reg * w
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
        w, prev_ll, prev_acc = self.weights, -np.inf, -np.inf
        true_labels = np.argmax(data[:, :self.action_num], 1)
        for it in range(self.max_iter):
            sat = compute_satisfaction(data, self.formulas)
            ll = compute_log_likelihood(sat, w, self.regularization)
            avg_prob = np.mean([
                self.infer_action_probability(x[self.action_num:])[0][y]
                for x, y in zip(data, true_labels)
            ])
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
            w = update_weights(w, sat, self.learning_rate, self.regularization)
            self.weights = w
        return w

    def eval(self, test_data):
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
