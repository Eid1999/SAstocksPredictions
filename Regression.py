import pandas as pd
from Dataset import Dataset_Class
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_lightning import Trainer

d = Dataset_Class("AAPL")

sentiment_df = d.news_data

sentiment_df["Date"] = pd.to_datetime(sentiment_df["Date"])

stock_df = d.stock_data

full_df = pd.merge(stock_df, sentiment_df, on="Date", how="left")
full_df = full_df.fillna(0)  # fill missing sentiment values with 0


max_encoder_length = 30
max_prediction_length = 7


# Create the dataset
tft_dataset = TimeSeriesDataSet(
    full_df,
    time_idx="time_idx",
    target="close",  # target variable (closing price)
    group_ids=["stock_symbol"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=[
        "close", "avg_sentiment_score", "sentiment_balance"
    ],
    static_categoricals=["stock_symbol"]
)

train_dataloader = tft_dataset.to_dataloader(train=True, batch_size=64)

# Create model from dataset
model = TemporalFusionTransformer.from_dataset(
    tft_dataset,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1
)

# Step 1: Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Step 2: Unfreeze the last layers — decoder and output head
for name, param in model.named_parameters():
    if "output_layer" in name:
        param.requires_grad = True


trainer = Trainer(max_epochs=20, gradient_clip_val=0.1)
trainer.fit(model, train_dataloaders=train_dataloader)

raw_predictions, x = model.predict(tft_dataset, mode="raw", return_x=True)