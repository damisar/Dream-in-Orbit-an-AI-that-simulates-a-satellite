
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train import SatelliteGRU


HOUR_MS = 3_600_000


def predict(test_dir, model_path, output_path):
    test_dir = Path(test_dir)

    # Load the trained neural network and preprocessing metadata.
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    columns = checkpoint["columns"]
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    std = np.where(std == 0, 1.0, std)

    sequence_length = checkpoint["sequence_length"]

    model = SatelliteGRU(
        n_circuits=len(columns),
        hidden_size=checkpoint["hidden_size"]
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Read the first seven days of real satellite measurements.
    initial = pd.read_csv(test_dir / "power_init.csv")

    initial["ut_ms"] = (
        pd.to_numeric(initial["ut_ms"], errors="coerce")
        .floordiv(HOUR_MS)
        .mul(HOUR_MS)
    )

    initial = initial.dropna(subset=["ut_ms"])

    history = (
        initial.groupby("ut_ms")[columns]
        .mean()
        .sort_index()
        .ffill()
        .bfill()
    )

    # Read the exact timestamps requested by the evaluator.
    hours = pd.read_csv(test_dir / "heures_a_predire.csv")

    timestamps = hours["ut_ms"].to_numpy(dtype=np.int64)

    if len(timestamps) == 0:
        raise ValueError("No prediction hours were provided.")

    if len(columns) != 33:
        raise ValueError("The model must predict 33 circuits.")

    if len(history) < sequence_length:
        raise ValueError("Insufficient initial history.")

    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Prediction timestamps must be increasing.")

    # Normalize the real initial observations.
    sequence = (
        (history.to_numpy(dtype=np.float32) - mean) / std
    )[-sequence_length:]

    predictions = []
    current_time = int(history.index[-1])

    # Generate predictions without using future real consumption.
    with torch.no_grad():
        for timestamp in timestamps:
            timestamp = int(timestamp)

            if timestamp <= current_time:
                raise ValueError(
                    "Prediction hours overlap the initial history."
                )

            if (timestamp - current_time) % HOUR_MS != 0:
                raise ValueError(
                    "Prediction timestamps are not hourly aligned."
                )

            while current_time < timestamp:
                x = torch.tensor(
                    sequence[np.newaxis, :, :],
                    dtype=torch.float32
                )

                next_normalized = (
                    model(x).cpu().numpy()[0]
                )

                sequence = np.vstack(
                    [sequence[1:], next_normalized]
                )

                current_time += HOUR_MS

            next_prediction = sequence[-1] * std + mean
            predictions.append(next_prediction.copy())

    result = pd.DataFrame(
        predictions,
        columns=columns
    )

    result.insert(0, "ut_ms", timestamps)

    result = result.replace(
        [np.inf, -np.inf], np.nan
    )

    if result.isna().any().any():
        raise ValueError("Predictions contain missing values.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")
    print(f"Rows: {len(result)}")
    print(f"Electrical circuits: {len(columns)}")
    print(f"Missing values: {result.isna().sum().sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_dir",
        required=True
    )

    parser.add_argument(
        "--out",
        required=True
    )

    parser.add_argument(
        "--model",
        default="models/satellite_gru.pt"
    )

    args = parser.parse_args()

    predict(
        args.test_dir,
        args.model,
        args.out
    )
