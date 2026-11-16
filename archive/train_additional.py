
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

HOUR_MS = 3_600_000
SEQUENCE_LENGTH = 24


class SatelliteGRU(nn.Module):
    def __init__(self, n_circuits=33, hidden_size=64):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_circuits,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.output = nn.Linear(hidden_size, n_circuits)

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.output(hidden[-1])


class PowerSequences(Dataset):
    def __init__(self, values, sequence_length=24):
        self.values = torch.tensor(values, dtype=torch.float32)
        self.sequence_length = sequence_length

    def __len__(self):
        return max(0, len(self.values) - self.sequence_length)

    def __getitem__(self, index):
        x = self.values[index:index + self.sequence_length]
        y = self.values[index + self.sequence_length]
        return x, y


def load_hourly_power(path):
    print(f"Loading {path}", flush=True)

    chunks = []

    for chunk in pd.read_csv(path, chunksize=100_000):
        chunk["ut_ms"] = (
            pd.to_numeric(chunk["ut_ms"], errors="coerce")
            .floordiv(HOUR_MS)
            .mul(HOUR_MS)
        )

        chunks.append(
            chunk.groupby("ut_ms").agg(["sum", "count"])
        )

    if not chunks:
        raise ValueError("No training data found.")

    # Combine hourly sums and counts from every chunk.
    combined = pd.concat(chunks).groupby(level=0).sum()

    sums = combined.xs("sum", axis=1, level=1)
    counts = combined.xs("count", axis=1, level=1)

    hourly = sums.div(counts.replace(0, np.nan))
    hourly = hourly.sort_index()

    columns = [c for c in hourly.columns if c.startswith("NPWD")]

    if len(columns) != 33:
        raise ValueError(f"Expected 33 circuits, found {len(columns)}")

    hourly = hourly[columns]
    hourly = hourly.replace([np.inf, -np.inf], np.nan)
    hourly = hourly.ffill().bfill().fillna(0)

    print(f"Loaded {len(hourly)} hourly observations.", flush=True)

    return hourly, columns


def train(args):
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)

    hourly, columns = load_hourly_power(
        Path(args.data_dir) / "power.csv"
    )

    values = hourly.to_numpy(dtype=np.float32)

    # Fit normalization using training data only.
    split = int(len(values) * 0.9)
    training_values = values[:split]

    mean = training_values.mean(axis=0)
    std = training_values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    normalized = (training_values - mean) / std
    normalized = np.clip(normalized, -10, 10)

    dataset = PowerSequences(normalized, SEQUENCE_LENGTH)

    if len(dataset) == 0:
        raise ValueError("Not enough observations to train.")

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0
    )

    model = SatelliteGRU(n_circuits=len(columns))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("Starting GRU training...", flush=True)

    model.train()

    for epoch in range(args.epochs):
        total_loss = 0.0

        for x, y in loader:
            optimizer.zero_grad()
            predictions = model(x)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(x)

        average_loss = total_loss / len(dataset)

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"- MSE: {average_loss:.6f}",
            flush=True
        )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "columns": columns,
            "mean": mean,
            "std": std,
            "sequence_length": SEQUENCE_LENGTH,
            "hidden_size": 64
        },
        output_path
    )

    print(f"Model saved to {output_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default="data/annee1"
    )

    parser.add_argument(
        "--out",
        default="models/satellite_gru.pt"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3
    )

    args = parser.parse_args()
    train(args)
