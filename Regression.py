import pandas as pd
from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
    GroupNormalizer,
)
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from io import BytesIO
from Dataset import Dataset_Class
import pdb

class StockPriceForecaster:
    def __init__(
        self,
        company: str,
        start_date,
        end_date,
        max_encoder_length=30,
        max_prediction_length=7,
        batch_size=64,
    ):
        """
        Initialize and prepare the dataset.

        Args:
            company (str): Stock symbol (e.g., "AAPL", "TSLA").
            start_date (str): Start date for the data (YYYY-MM-DD).
            end_date (str): End date for the data (YYYY-MM-DD).
            max_encoder_length (int): Number of days to use as input.
            max_prediction_length (int): Number of days to predict.
        """
        self.company = company
        self.encoder_len = max_encoder_length
        self.prediction_len = max_prediction_length
        self.batch_size = batch_size
        # Load stock data using Dataset_Class
        self.dataset_class = Dataset_Class(
            company, start_date, end_date, load_dataset=True
        )
        self.stock_df = self.dataset_class.stock_data

        # Get the full dataset that includes sentiment data
        self.full_df = self.dataset_class.data

        # Ensure we have the correct columns for sentiment analysis
        self.prepare_data()

        self.model = None
        self.tft_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None

    def prepare_data(self):
        """
        Prepare the data for the temporal fusion transformer.
        Creates necessary format and calculates sentiment metrics.
        """
        # Ensure dates are in datetime format
        self.full_df["date"] = pd.to_datetime(self.full_df["date"])

        # Calculate sentiment metrics from the sentiment scores if they exist
        if all(
            col in self.full_df.columns
            for col in ["positive_score", "negative_score", "neutral_score"]
        ):
            # Calculate average sentiment score
            self.full_df["avg_sentiment_score"] = (
                self.full_df["positive_score"] - self.full_df["negative_score"]
            )

            # Calculate sentiment balance (positive vs negative)
            self.full_df["sentiment_balance"] = self.full_df["positive_score"] / (
                self.full_df["negative_score"] + 0.001
            )  # Add small constant to avoid division by zero
        elif "score" in self.full_df.columns:
            # If only a general score is available
            self.full_df["avg_sentiment_score"] = self.full_df["score"]
            self.full_df["sentiment_balance"] = 1.0  # Default value
        else:
            # Create placeholder columns if no sentiment data
            self.full_df["avg_sentiment_score"] = 0.0
            self.full_df["sentiment_balance"] = 1.0
            print("Warning: No sentiment columns found. Using default values.")

        # Fill missing values
        self.full_df["avg_sentiment_score"] = (
            self.full_df["avg_sentiment_score"].ffill().fillna(0)
        )
        self.full_df["sentiment_balance"] = (
            self.full_df["sentiment_balance"].ffill().fillna(1)
        )

        # Ensure required format for TFT
        self.full_df = self.full_df.sort_values("date").reset_index(drop=True)

        # Create the time_idx column
        self.full_df["time_idx"] = range(len(self.full_df))

        # Add a static categorical column for the stock symbol
        self.full_df["stock_symbol"] = self.company

        # Make sure 'Close' is renamed to 'close' for consistency
        if "Close" in self.full_df.columns and "close" not in self.full_df.columns:
            self.full_df.rename(columns={"Close": "close"}, inplace=True)

    def create_tft_dataset(self):
        """
        Create a TimeSeriesDataSet compatible with the TemporalFusionTransformer.
        """

        df = self.full_df.drop(columns=["title", "company_focused_summary"])
        close = torch.tensor(df["close"].values, dtype=torch.float32)
        df["close"] = close
        training_cutoff = int(df["time_idx"].max() * 0.8)
        pdb.set_trace()
        self.training = TimeSeriesDataSet(
            df[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="close",
            group_ids=["stock_symbol"],
            max_encoder_length=self.encoder_len,
            max_prediction_length=self.prediction_len,
            time_varying_unknown_reals=[
                "close",
                "avg_sentiment_score",
                "sentiment_balance",
            ],
            allow_missing_timesteps=True,
            target_normalizer=GroupNormalizer(groups=["stock_symbol"]),
        )

        validation = TimeSeriesDataSet.from_dataset(
            self.training, df, min_prediction_idx=training_cutoff + 1
        )
        # Use the recommended split method
        self.train_dataloader = self.training.to_dataloader(
            train=True, batch_size=self.batch_size, num_workers=0
        )
        pdb.set_trace()
        self.validation_dataloader = validation.to_dataloader(
            train=False, batch_size=self.batch_size, num_workers=0
        )

    def build_model(self):
        """
        Build the Temporal Fusion Transformer model.
        """
        self.model = TemporalFusionTransformer.from_dataset(
            self.training,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def train_model(self, max_epochs=20):
        """
        Train the model using a PyTorch training loop.

        Args:
            max_epochs (int): Maximum number of training epochs.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.03)
        criterion = torch.nn.MSELoss()
        for epoch in range(max_epochs):
            self.model.train()
            train_loss = 0.0
            try:
                i=0
                for batch in self.train_dataloader:
                    i+=1
                    batch = {k: v.to(device) for k, v in batch.items()}
                    output = self.model(batch)
                    loss = criterion(output["prediction"], batch["decoder_target"])
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
            except Exception as e:
                pdb.set_trace()

            train_loss /= len(self.train_dataloader)

            # Validation loop
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in self.val_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    output = self.model(batch)
                    loss = criterion(output["prediction"], batch["decoder_target"])
                    val_loss += loss.item()

            val_loss /= len(self.val_dataloader)

            print(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    def predict(self):
        """
        Make predictions using the trained model.

        Returns:
            tuple: Raw predictions and input data.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        predict_dataset = self.tft_dataset.slice(
            start=len(self.full_df) - self.encoder_len - self.prediction_len,
            stop=len(self.full_df),
        )
        dataloader = DataLoader(predict_dataset, batch_size=1)
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                output = self.model(batch)
                predictions.append(output["prediction"].cpu().numpy())
        return predictions

    def plot_prediction(self, index=0):
        """
        Plot the actual vs. predicted closing prices for one sample.

        Args:
            index (int): Index of the sample to plot.

        Returns:
            BytesIO: In-memory PNG image.
        """
        predictions = self.predict()
        # Extract actual and predicted
        pred_target = predictions[index]
        true_target = self.full_df["close"].iloc[-self.prediction_len:].values
        encoder_target = self.full_df["close"].iloc[-self.encoder_len:-self.prediction_len].values

        # Combine for time axis
        time_idx = list(range(len(encoder_target) + len(pred_target)))

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            time_idx[: len(encoder_target)], encoder_target, label="Historical Price"
        )
        ax.plot(
            time_idx[len(encoder_target) :],
            true_target,
            label="True Future Price",
            color="green",
        )
        ax.plot(
            time_idx[len(encoder_target) :],
            pred_target,
            label="Predicted Price",
            color="red",
            linestyle="--",
        )

        ax.set_title(f"{self.company} Stock Forecast")
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return buf


if __name__ == "__main__":
    # Example usage
    forecaster = StockPriceForecaster("AAPL", "2018-01-01", "2020-10-01")
    forecaster.create_tft_dataset()
    forecaster.build_model()
    forecaster.train_model(max_epochs=10)

    # Generate prediction plot
    img_buf = forecaster.plot_prediction()

    # You could save or display the plot
    with open("aapl_forecast.png", "wb") as f:
        f.write(img_buf.getvalue())
