import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from get_training_data import get_wind_data, get_energy_data


class SimpleNN(nn.Module):

    def __init__(self, X, y):
        super(SimpleNN, self).__init__()

        self.define_size(X, y)
        self.setup(X, y)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

    def define_size(self, X, y):
        input_size = 3
        hidden_size = 10
        output_size = 1

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    
    def setup(self, X, y):
        # Keep observations in chronological order
        split_index = int(len(X) * 0.8)

        X_train = X[:split_index]
        X_test = X[split_index:]

        y_train = y[:split_index]
        y_test = y[split_index:]

        split_index = int(len(matched) * 0.8)

        print("Training dates:",
              matched.iloc[0]["Year"], matched.iloc[0]["Month"],
              "to",
              matched.iloc[split_index - 1]["Year"],
              matched.iloc[split_index - 1]["Month"])

        print("Testing dates:",
              matched.iloc[split_index]["Year"],
              matched.iloc[split_index]["Month"],
              "to",
              matched.iloc[-1]["Year"],
              matched.iloc[-1]["Month"])

        # Standardise the input features
        self.scaler = StandardScaler()

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Standardise the target values too
        self.y_scaler = StandardScaler()

        y_train_scaled = self.y_scaler.fit_transform(
            y_train.reshape(-1, 1)
        )

        torch.manual_seed(42)

        criterion = nn.MSELoss()

        optimizer = optim.Adam(
            self.parameters(),
            lr=0.01
        )

        X_train_tensor = torch.tensor(
            X_train, dtype=torch.float32
        )

        y_train_tensor = torch.tensor(
            y_train_scaled, dtype=torch.float32
        )

        # Train on standardised targets
        for epoch in range(1000):

            outputs = self(X_train_tensor)

            loss = criterion(outputs, y_train_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}, "
                    f"Loss: {loss.item():.4f}"
                )

        # Evaluate in the original energy units
        with torch.no_grad():
            X_test_tensor = torch.tensor(
                X_test, dtype=torch.float32
            )

            predicted_scaled = self(X_test_tensor).numpy()

            # Convert predictions back to original energy units
            predicted_np = self.y_scaler.inverse_transform(
                predicted_scaled
            ).flatten()

            y_test_np = y_test.flatten()

            mae = np.mean(np.abs(predicted_np - y_test_np))
            rmse = np.sqrt(np.mean((predicted_np - y_test_np) ** 2))

            print(f"MAE: {mae:.2f}")
            print(f"RMSE: {rmse:.2f}")

            print("\nTest predictions:")
            for actual, prediction in zip(y_test_np, predicted_np):
                print(f"Actual: {actual:.0f} | Predicted: {prediction:.0f}")
        
        baseline_prediction = np.full_like(
            y_test_np,
            fill_value=np.mean(y_train)
        )

        baseline_mae = np.mean(
            np.abs(baseline_prediction - y_test_np)
        )

        print(f"Baseline MAE: {baseline_mae:.2f}")
        print(f"Neural network MAE: {mae:.2f}")



def prepare_data():
    wind_data = np.asarray(get_wind_data())
    energy_data = np.asarray(get_energy_data())

    wind = pd.DataFrame(
        wind_data,
        columns=["Year", "Month", "Speed"]
    )

    energy = pd.DataFrame(
        energy_data,
        columns=["Year", "Month", "Total"]
    )

    matched = (
        pd.merge(
            wind,
            energy,
            on=["Year", "Month"],
            how="inner",
            validate="one_to_one"
        )
        .sort_values(["Year", "Month"])
        .reset_index(drop=True)
    )

    month = matched["Month"].to_numpy(dtype=np.float32)

    # Cyclical month features
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    X = np.column_stack([
        month_sin,
        month_cos,
        matched["Speed"].to_numpy(dtype=np.float32)
    ])

    y = matched["Total"].to_numpy(dtype=np.float32)

    return matched, X, y


# Prepare data and train only once when this module is imported.
matched, X, y = prepare_data()

model = SimpleNN(X, y)
model.eval()


def get_prediction(month, speed):
    # Construct features in exactly the same order as during training.
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    features = np.array(
        [[month_sin, month_cos, speed]],
        dtype=np.float32
    )

    # Use the input scaler fitted during training.
    features_scaled = model.scaler.transform(features)

    features_tensor = torch.tensor(
        features_scaled,
        dtype=torch.float32
    )

    with torch.no_grad():
        prediction_scaled = model(features_tensor).numpy()

    # Return energy generation in the original units.
    prediction = model.y_scaler.inverse_transform(
        prediction_scaled
    )[0, 0]

    return float(prediction)


if __name__ == "__main__":
    print("Training complete.")
    print("Observations:", len(matched))
    print("Example prediction:", get_prediction(1, 8.0))

# el gpt is cracked bro