import gc
import torch
import numpy as np


def convert_storage(storage):
    obses = {
        k: np.array([item[0][k] for item in storage], dtype=np.int8)
        for k in ["board_and_hand", "stage", "first_to_act_next_stage"]
    }

    obses["bets_and_stacks"] = np.array(
        [item[0]["bets_and_stacks"] for item in storage], dtype=np.float32
    )

    ts = np.array([item[1] for item in storage], dtype=np.float32)
    values = np.array([item[2] for item in storage], dtype=np.float32)

    return obses, ts, values


class GPUBoundedStorage:
    def __init__(self, max_size, target_size=4):
        self.max_size = max_size
        self.current_len = 0
        self.current_idx = 0

        self.obs = {
            "board_and_hand": torch.zeros(
                (max_size, 21), device="cuda", dtype=torch.int8, requires_grad=False
            ),
            "stage": torch.zeros(
                max_size, device="cuda", dtype=torch.int8, requires_grad=False
            ),
            "first_to_act_next_stage": torch.zeros(
                max_size, device="cuda", dtype=torch.int8, requires_grad=False
            ),
            "bets_and_stacks": torch.zeros(
                (max_size, 8), device="cuda", requires_grad=False
            ),
        }

        self.ts = torch.zeros((max_size, 1), device="cuda", requires_grad=False)
        self.values = torch.zeros(
            (max_size, target_size), device="cuda", requires_grad=False
        )

    def get_storage(self):
        if self.current_len == self.max_size:
            return self.obs, self.ts, self.values
        return (
            {k: v[: self.current_len] for k, v in self.obs.items()},
            self.ts[: self.current_len],
            self.values[: self.current_len],
        )

    def __len__(self):
        return self.current_len

    def save(self, filename):
        torch.save(
            {
                "obs": {k: v.cpu() for k, v in self.obs.items()},
                "ts": self.ts.cpu(),
                "values": self.values.cpu(),
                "current_len": self.current_len,
                "current_idx": self.current_idx,
            },
            filename,
        )

    def load(self, filename):
        data = torch.load(filename, weights_only=True)

        del self.obs
        del self.ts
        del self.values
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        self.obs = {k: v.cuda() for k, v in data["obs"].items()}
        self.ts = data["ts"].cuda()
        self.values = data["values"].cuda()
        self.current_len = data["current_len"]
        self.current_idx = data["current_idx"]

    def add_all(self, items):
        obses, ts, values = items

        if not len(ts):
            return

        obses = {k: torch.tensor(v, device="cuda") for k, v in obses.items()}
        ts = torch.tensor(ts, device="cuda", dtype=torch.float32)
        values = torch.tensor(values, device="cuda", dtype=torch.float32)

        num_items = len(ts)

        if self.current_len + num_items <= self.max_size:
            start_idx = self.current_len
            end_idx = self.current_len + num_items
            self.current_len += num_items
            for k, v in obses.items():
                self.obs[k][start_idx:end_idx] = v
            self.ts[start_idx:end_idx] = ts[..., None]
            self.values[start_idx:end_idx] = values
            return

        if self.current_len < self.max_size:
            first_part = self.max_size - self.current_len
            for k, v in obses.items():
                self.obs[k][self.current_len :] = v[:first_part]
            self.ts[self.current_len :] = ts[:first_part][..., None]
            self.values[self.current_len :] = values[:first_part]
            self.current_len = self.max_size

            for k, v in obses.items():
                self.obs[k][: num_items - first_part] = v[first_part:]
            self.ts[: num_items - first_part] = ts[first_part:][..., None]
            self.values[: num_items - first_part] = values[first_part:]
            self.current_idx = num_items - first_part
            return

        if self.current_idx + num_items <= self.max_size:
            for k, v in obses.items():
                self.obs[k][self.current_idx : self.current_idx + num_items] = v
            self.ts[self.current_idx : self.current_idx + num_items] = ts[..., None]
            self.values[self.current_idx : self.current_idx + num_items] = values
            self.current_idx = (self.current_idx + num_items) % self.max_size
            return

        first_part = self.max_size - self.current_idx
        for k, v in obses.items():
            self.obs[k][self.current_idx :] = v[:first_part]
        self.ts[self.current_idx :] = ts[:first_part][..., None]
        self.values[self.current_idx :] = values[:first_part]
        self.current_idx = 0

        for k, v in obses.items():
            self.obs[k][: num_items - first_part] = v[first_part:]
        self.ts[: num_items - first_part] = ts[first_part:][..., None]
        self.values[: num_items - first_part] = values[first_part:]
        self.current_idx = num_items - first_part
