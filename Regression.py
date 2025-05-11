import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
from Dataset import Dataset_Class
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class SelfAttention(nn.Module):
    """
    A Multi-Head Self-Attention module based on the Transformer architecture.
    """

    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        # Ensure embed_dim is divisible by num_heads for even distribution
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        # Linear layers for Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output linear layer after attention calculation
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, _ = x.size()

        # Project and reshape Q, K, V for multi-head attention
        # (batch_size, seq_len, embed_dim) -> (batch_size, num_heads, seq_len, head_dim)
        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Scaled Dot-Product Attention: (Q @ K^T) / sqrt(d_k)
        # scores shape: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Multiply attention weights with V
        # attn_output shape: (batch_size, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)

        # Concatenate heads and project back to original embedding dimension
        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, embed_dim)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

        # Final linear projection
        output = self.out_proj(attn_output)
        return output


class LSTMModel(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=2,
        dropout=0.1,
        num_heads=8,
    ):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        # Add the SelfAttention layer
        # The input dimension to self-attention is hidden_size * 2 because of bidirectional LSTM
        self.self_attention = SelfAttention(
            embed_dim=hidden_size * 2, num_heads=num_heads
        )

        # The original attention layer, now applied *after* self-attention
        # This layer is used to produce a single context vector from the sequence representation
        # enhanced by self-attention.
        self.attention = torch.nn.Linear(hidden_size * 2, 1)

        self.fc1 = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)

        # Pass input through LSTM
        # lstm_out shape: (batch_size, seq_len, hidden_size * 2) due to bidirectional
        lstm_out, _ = self.lstm(x)

        # Apply self-attention to the LSTM output
        # self_attn_out shape: (batch_size, seq_len, hidden_size * 2)
        self_attn_out = self.self_attention(lstm_out)

        # Apply the original attention mechanism on the self-attention output
        # This learns a weighted sum of the self-attention enhanced features
        # attn_weights shape: (batch_size, seq_len, 1)
        attn_weights = torch.softmax(self.attention(self_attn_out), dim=1)

        # Compute the context vector as a weighted sum of the self-attention outputs
        # context shape: (batch_size, hidden_size * 2)
        context = torch.sum(attn_weights * self_attn_out, dim=1)

        # Pass the context vector through the feed-forward layers
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
        max_encoder_length=365,
        max_prediction_length=60,
        batch_size=32,
    ):
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

    def build_model(self):
        input_size = 3
        hidden_size = 32
        self.model = LSTMModel(input_size, hidden_size, self.prediction_len)
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def train_model(self, max_epochs=20):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        df = self.full_df.copy()
        features = ["close", "avg_sentiment_score", "sentiment_balance"]

        X, y = [], []
        for i in range(len(df) - self.encoder_len - self.prediction_len):
            X.append(df[features].iloc[i : i + self.encoder_len].values)
            y.append(
                df["close"]
                .iloc[i + self.encoder_len : i + self.encoder_len + self.prediction_len]
                .values
            )

        X = torch.tensor(np.array(X), dtype=torch.float32).to(device)
        y = torch.tensor(np.array(y), dtype=torch.float32).to(device)

        dataset = CustomDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        epoch_bar = tqdm(range(max_epochs), desc="Training")
        for epoch in epoch_bar:
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            epoch_bar.set_postfix({"epoch": epoch + 1, "loss": avg_loss})


    def predict(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

        df = self.full_df.copy()
        features = ["close", "avg_sentiment_score", "sentiment_balance"]
        recent_data = df[features].iloc[-self.encoder_len :].values
        X_input = torch.tensor(recent_data, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = self.model(X_input).cpu().numpy().flatten()
        return pred * self.close_std + self.close_mean

    def plot_prediction(self, index=0):
        predictions = self.predict()
        true_target = (
            self.full_df["close"].iloc[-self.prediction_len :].values * self.close_std
            + self.close_mean
        )
        encoder_target = (
            self.full_df["close"].iloc[-self.encoder_len : -self.prediction_len].values
            * self.close_std
            + self.close_mean
        )
        encoder_dates = (
            self.full_df["date"].iloc[-self.encoder_len : -self.prediction_len].tolist()
        )
        prediction_dates = self.full_df["date"].iloc[-self.prediction_len :].tolist()
        full_dates = encoder_dates + prediction_dates

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            full_dates[: len(encoder_target)], encoder_target, label="Historical Price"
        )
        ax.plot(
            full_dates[len(encoder_target) :],
            true_target,
            label="True Future Price",
            color="green",
        )
        ax.plot(
            full_dates[len(encoder_target) :],
            predictions,
            label="Predicted Price",
            color="red",
            linestyle="--",
        )

        ax.set_title(f"{self.company} Stock Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)
        fig.autofmt_xdate()

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return buf


if __name__ == "__main__":
    forecaster = StockPriceForecaster("AAPL")
    forecaster.build_model()
    forecaster.train_model(max_epochs=100)

    img_buf = forecaster.plot_prediction()

    with open("aapl_forecast.png", "wb") as f:
        f.write(img_buf.getvalue())
