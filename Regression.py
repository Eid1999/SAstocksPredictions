import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_lightning import Trainer
from Dataset import Dataset_Class
import matplotlib.pyplot as plt
from io import BytesIO
from SentimentAnalysis import FinBertSentimentAnalyzer

class StockPriceForecaster:
    def __init__(self, company: str, start_date, end_date, max_encoder_length=30, max_prediction_length=7):
        """
        Initialize and prepare the dataset.
        """
        self.company = company
        self.encoder_len = max_encoder_length
        self.prediction_len = max_prediction_length

        self.dataset_class = Dataset_Class(company, start_date, end_date)
        self.stock_df = self.dataset_class.stock_data
        self.sentiment_df = FinBertSentimentAnalyzer(dataset=self.dataset_class).analyze_sentiment()

        self.model = None
        self.tft_dataset = None
        self.train_dataloader = None

    def prepare_data(self):
        self.sentiment_df["Date"] = pd.to_datetime(self.sentiment_df["Date"])
        self.full_df = pd.merge(self.stock_df, self.sentiment_df, on="Date", how="left")

        # Fill missing values
        self.full_df["avg_sentiment_score"] = self.full_df["avg_sentiment_score"].fillna(method="ffill").fillna(0)
        self.full_df["sentiment_balance"] = self.full_df["sentiment_balance"].fillna(method="ffill").fillna(0)

        # Ensure required format
        self.full_df = self.full_df.sort_values("Date").reset_index(drop=True)
        self.full_df["time_idx"] = range(len(self.full_df))
        self.full_df["stock_symbol"] = self.company
        self.full_df.rename(columns={"Close": "close"}, inplace=True)

    def create_tft_dataset(self):
        """
        Create a TimeSeriesDataSet compatible with the TemporalFusionTransformer.
        """
        self.tft_dataset = TimeSeriesDataSet(
            self.full_df,
            time_idx="time_idx",
            target="close",
            group_ids=["stock_symbol"],
            max_encoder_length=self.encoder_len,
            max_prediction_length=self.prediction_len,
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=[
                "close", "avg_sentiment_score", "sentiment_balance"
            ],
            static_categoricals=["stock_symbol"]
        )
        # Split dataset before creating dataloaders
        training, validation = self.tft_dataset.split_before(0.8)

        self.train_dataloader = training.to_dataloader(train=True, batch_size=64)
        self.val_dataloader = validation.to_dataloader(train=False, batch_size=64)

    def build_model(self):
        """
        Build and partially freeze the Temporal Fusion Transformer model.
        """
        self.model = TemporalFusionTransformer.from_dataset(
            self.tft_dataset,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1
        )

        # Freeze all layers first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze only the output layer (head)
        for name, param in self.model.named_parameters():
            if "output_layer" in name:
                param.requires_grad = True

    def train_model(self, max_epochs=20):
        """
        Train the model using the PyTorch Lightning Trainer.
        """
        trainer = Trainer(
            max_epochs=max_epochs,
            gradient_clip_val=0.1,
            accelerator="auto"  # Use GPU if available
        )
        trainer.fit(self.model, train_dataloaders=self.train_dataloader, val_dataloaders=self.val_dataloader)

    def predict(self):
        # Get the most recent window
        predict_dataset = self.tft_dataset.slice(
        start=len(self.full_df) - self.encoder_len - self.prediction_len,
        stop=len(self.full_df)
        )
        return self.model.predict(predict_dataset, mode="raw", return_x=True)
    
    def plot_prediction(self, index=0):
        """
        Plot the actual vs. predicted closing prices for one sample.
        Returns:
            BytesIO: In-memory PNG image.
        """
        raw_preds, x = self.predict()

        # Extract actual and predicted
        pred_target = raw_preds["prediction"][index].detach().cpu().numpy()
        true_target = x["decoder_target"][index].detach().cpu().numpy()
        encoder_target = x["encoder_target"][index].detach().cpu().numpy()

        # Combine for time axis
        time_idx = list(range(len(encoder_target) + len(pred_target)))

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time_idx[:len(encoder_target)], encoder_target, label="Historical Price")
        ax.plot(time_idx[len(encoder_target):], true_target, label="True Future Price", color="green")
        ax.plot(time_idx[len(encoder_target):], pred_target, label="Predicted Price", color="red", linestyle="--")

        ax.set_title(f"{self.company} Stock Forecast")
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return buf