import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
from Dataset import Dataset_Class
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pdb
from sklearn.metrics import mean_absolute_error, r2_score
import random
import os


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(feature_dim, 1)

    def forward(self, x):
        weights = F.softmax(self.attention(x), dim=1)
        weighted = torch.sum(weights * x, dim=1)
        return weighted


class ResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(ResidualLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.residual_proj = nn.Linear(input_size, hidden_size * 2)

    def forward(self, x):
        residual = self.residual_proj(x)
        lstm_out, _ = self.lstm(x)
        return lstm_out + residual


class LSTMModel(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=3,
        dropout=0.1,
    ):
        super(LSTMModel, self).__init__()
        self.lstm = ResidualLSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.attention = Attention(hidden_size * 2)

        self.fc1 = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out = self.lstm(x)
        context = self.attention(lstm_out)

        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout(out)
        output = self.fc2(out)
        return output


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class StockPriceForecaster:
    def __init__(
        self,
        company: str,
        start_date=None,
        end_date=None,
        max_encoder_length=100,
        max_prediction_length=1,
        batch_size=68,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Model_Path=f"models/best_model_{company}.pt"
        set_seed(42)
        self.company = company
        self.encoder_len = max_encoder_length
        self.prediction_len = max_prediction_length
        self.batch_size = batch_size
        # Assuming Dataset_Class is defined elsewhere
        if start_date is not None and end_date:
            self.dataset_class = Dataset_Class(
                company, start_date, end_date, load_dataset=True
            )
        else:
            self.dataset_class = Dataset_Class(company, load_dataset=True)
        self.stock_df = self.dataset_class.stock_data
        self.full_df = self.dataset_class.data
        self.prepare_data()
        self.model = None

    def prepare_data(self):
        self.full_df["date"] = pd.to_datetime(self.full_df["date"])

        if all(
            col in self.full_df.columns
            for col in ["positive_score", "negative_score", "neutral_score"]
        ):
            self.full_df["avg_sentiment_score"] = (
                self.full_df["positive_score"] - self.full_df["negative_score"]
            )
            self.full_df["sentiment_balance"] = self.full_df["positive_score"] / (
                self.full_df["negative_score"] + 0.001
            )
        elif "score" in self.full_df.columns:
            self.full_df["avg_sentiment_score"] = self.full_df["score"]
            self.full_df["sentiment_balance"] = 1.0
        else:
            self.full_df["avg_sentiment_score"] = 0.0
            self.full_df["sentiment_balance"] = 1.0
            print("Warning: No sentiment columns found. Using default values.")

        self.full_df["avg_sentiment_score"] = (
            self.full_df["avg_sentiment_score"].ffill().fillna(0)
        )
        self.full_df["sentiment_balance"] = (
            self.full_df["sentiment_balance"].ffill().fillna(1)
        )

        self.full_df = self.full_df.sort_values("date").reset_index(drop=True)
        self.full_df["time_idx"] = range(len(self.full_df))
        self.full_df["stock_symbol"] = self.company

        if "Close" in self.full_df.columns and "close" not in self.full_df.columns:
            self.full_df.rename(columns={"Close": "close"}, inplace=True)

        self.close_mean = self.full_df["close"].mean()
        self.close_std = self.full_df["close"].std()
        self.full_df["close"] = (
            self.full_df["close"] - self.close_mean
        ) / self.close_std
        self.full_df["avg_sentiment_score"] = (
            self.full_df["avg_sentiment_score"]
            - self.full_df["avg_sentiment_score"].mean()
        ) / self.full_df["avg_sentiment_score"].std()
        self.full_df["sentiment_balance"] = (
            self.full_df["sentiment_balance"] - self.full_df["sentiment_balance"].mean()
        ) / self.full_df["sentiment_balance"].std()

    def build_model(self):
        input_size = 3
        hidden_size = 200
        self.model = LSTMModel(input_size, hidden_size, self.prediction_len)
        self.model.to(self.device)

    def train_model(self, max_epochs=200, patience=30):
        device = self.device
        df = self.full_df.copy()
        features = ["close", "avg_sentiment_score", "sentiment_balance"]
        X, y, dates = [], [], []
        for i in range(len(df) - self.encoder_len - self.prediction_len):
            X.append(df[features].iloc[i : i + self.encoder_len].values)
            y.append(
                df["close"]
                .iloc[i + self.encoder_len : i + self.encoder_len + self.prediction_len]
                .values
            )
            dates.append(
                df["date"].iloc[i + self.encoder_len + self.prediction_len - 1]
            )
        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32)
        dataset = CustomDataset(X, y)
        total_len = len(dataset)
        val_len = int(0.2 * total_len)
        test_len = int(0.1 * total_len)
        train_len = total_len - val_len - test_len
        X_train, X_val, X_test = (
            X[:train_len],
            X[train_len : train_len + val_len],
            X[train_len + val_len :],
        )
        y_train, y_val, y_test = (
            y[:train_len],
            y[train_len : train_len + val_len],
            y[train_len + val_len :],
        )
        train_ds = CustomDataset(X_train, y_train)
        val_ds = CustomDataset(X_val, y_val)
        test_ds = CustomDataset(X_test, y_test)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)
        self.test_loader = DataLoader(test_ds, batch_size=self.batch_size)
        self.dates_test = dates[train_len + val_len :]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs
        )
        best_val_loss = float("inf")
        patience_counter = 0
        epoch_bar = tqdm(range(max_epochs), desc="Training")
        for epoch in epoch_bar:
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    output = self.model(batch_X)
                    loss = criterion(output, batch_y)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            scheduler.step()
            epoch_bar.set_postfix({"Train Loss": train_loss, "Val Loss": val_loss})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.Model_Path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    epoch_bar.set_postfix(
                        {
                            "Train Loss": train_loss,
                            "Val Loss": val_loss,
                            "Early Stopping": "Triggered",
                        }
                    )
                    break
        self.model.load_state_dict(
            torch.load(self.Model_Path, map_location=self.device)
        )
    def predict(self):
        
        self.model.eval()

        preds, trues = [], []
        with torch.no_grad():
            for batch_X, batch_y in self.test_loader:
                batch_X = batch_X.to(self.device)
                pred = self.model(batch_X).cpu().numpy()
                true = batch_y.numpy()
                preds.append(pred)
                trues.append(true)

        predictions = np.concatenate(preds, axis=0).flatten()
        targets = np.concatenate(trues, axis=0).flatten()
        return (
            predictions * self.close_std + self.close_mean,
            targets * self.close_std + self.close_mean,
        )

    def plot_prediction(self):
        predictions, targets = self.predict()

        dates = pd.to_datetime(self.dates_test)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dates, targets, label="True Future Price", color="green")
        ax.plot(
            dates,
            predictions,
            label="Predicted Price",
            color="red",
            linestyle="--",
        )

        ax.set_title(f"{self.company} Stock Forecast (Test Set)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(f"./graphs/{self.company}_forecast.png")
        plt.close(fig)
        buf.seek(0)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        print(f"MAE: {mae}, R² Score: {r2}")
        return buf


if __name__ == "__main__":
    forecaster = StockPriceForecaster("AMZN")
    forecaster.build_model()
    forecaster.train_model(max_epochs=500)

    img_buf = forecaster.plot_prediction()

    with open("aapl_forecast.png", "wb") as f:
        f.write(img_buf.getvalue())
