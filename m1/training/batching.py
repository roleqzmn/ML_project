import math
import random


class RatioBatchSampler:
    def __init__(self, labels, batch_size=64, seed=42, allowed_per_batch=None):
        if allowed_per_batch is None:
            if batch_size < 4 or batch_size % 4:
                raise ValueError("Default 1:3 ratio requires batch_size divisible by 4")
            allowed_per_batch = batch_size // 4
        if not 1 <= allowed_per_batch < batch_size:
            raise ValueError("allowed_per_batch must be >= 1 and < batch_size")
        indices = {label: [] for label in (0, 1)}
        for index, label in enumerate(labels):
            if label not in indices:
                raise ValueError("Only labels 0 and 1 are supported")
            indices[label].append(index)
        if not all(indices.values()):
            raise ValueError("Both classes are required for ratio batches")
        self.denied, self.allowed = indices[0], indices[1]
        self.allowed_per_batch = allowed_per_batch
        self.denied_per_batch = batch_size - allowed_per_batch
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return math.ceil(len(self.denied) / self.denied_per_batch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        order = self.denied.copy()
        rng.shuffle(order)
        for start in range(0, len(order), self.denied_per_batch):
            batch = order[start:start + self.denied_per_batch]
            n_allowed = math.ceil(len(batch) * self.allowed_per_batch / self.denied_per_batch)
            batch += rng.choices(self.allowed, k=n_allowed)
            rng.shuffle(batch)
            yield batch
