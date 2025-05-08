from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from transformers import pipeline
import pandas as pd
import torch
import torch.nn.functional as F
from Dataset import Dataset_Class
from tqdm import tqdm



def load_finbert_model():
    """
    Loads the FinBERT model and tokenizer, using GPU if available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    return model, tokenizer, device

def analyze_sentiment_with_finbert(df, model, tokenizer, device, text_column="Tweet", batch_size = 200):
    """
    Efficiently performs sentiment analysis using FinBERT with GPU batching.

    Args:
        df (pd.DataFrame): DataFrame with the text data.
        model: Hugging Face model (on GPU if available).
        tokenizer: Tokenizer for the model.
        device: torch.device("cuda") or torch.device("cpu")
        text_column (str): Column containing the text to analyze.
        batch_size (int): Batch size for processing.

    Returns:
        pd.DataFrame: Original DataFrame with sentiment labels and scores.
    """
    df = df.copy()
    texts = df[text_column].astype(str).tolist()
    sentiments = []
    scores = []

    model.eval()

    for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment Analysis"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            labels = torch.argmax(probs, dim=1)

        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        for j in range(len(batch_texts)):
            sentiments.append(label_map[labels[j].item()])
            scores.append(probs[j][labels[j]].item())

    df["sentiment"] = sentiments
    df["sentiment_score"] = scores
    return df



if __name__ == "__main__":
    # dt = Dataset_Class("AAPL").news_data
    dt = Dataset_Class("AAPL").news_data

    ner_pipeline = load_ner_pipeline()

    # Filter based on Apple-related terms
    filtered_dt = filter_news_by_ner_and_keywords(dt, ner_pipeline, text_column="News")

    model, tokenizer, device = load_finbert_model()
    sentiment_df = analyze_sentiment_with_finbert(filtered_dt, model, tokenizer, device, text_column="News", batch_size=200)
    print(sentiment_df)


    # Filter examples
    positive_df = sentiment_df[sentiment_df["sentiment"] == "positive"][["News", "sentiment", "sentiment_score"]].head(50)
    neutral_df  = sentiment_df[sentiment_df["sentiment"] == "neutral"][["News", "sentiment", "sentiment_score"]].head(50)
    negative_df = sentiment_df[sentiment_df["sentiment"] == "negative"][["News", "sentiment", "sentiment_score"]].head(50)

    # Save each to a separate CSV
    positive_df.to_csv("AAPL_positive_news.csv", index=False)
    neutral_df.to_csv("AAPL_neutral_news.csv", index=False)
    negative_df.to_csv("AAPL_negative_news.csv", index=False)

    print("CSVs saved as:")
    print("• AAPL_positive_news.csv")
    print("• AAPL_neutral_news.csv")
    print("• AAPL_negative_news.csv")
