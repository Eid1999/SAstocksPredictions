# finbert_sentiment_analyzer.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from Dataset import Dataset_Class

class FinBertSentimentAnalyzer:
    def __init__(self, dataset: Dataset_Class, model_name="ProsusAI/finbert"):
        """
        Initializes the FinBERT sentiment analyzer, loading the model and tokenizer.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}
        self.df = dataset.news_data


    def analyze_sentiment(self, text_column="News", batch_size=200):
        """
        Performs sentiment analysis on a DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing a text column.
            text_column (str): Column with text data.
            batch_size (int): Batch size for GPU processing.

        Returns:
            pd.DataFrame: DataFrame with sentiment labels and scores.
        """
        df = self.df.copy()
        texts = df[text_column].astype(str).tolist()
        sentiments = []
        scores = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment Analysis"):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                labels = torch.argmax(probs, dim=1)

            for j in range(len(batch_texts)):
                sentiment = self.label_map[labels[j].item()]
                score = probs[j][labels[j]].item()
                sentiments.append(sentiment)
                scores.append(score)

        df["sentiment"] = sentiments
        df["sentiment_score"] = scores
        return df

    def save_sentiment_files(self, df, output_prefix="output"):
        """
        Saves the positive, neutral, and negative sentiment entries to separate CSV files.

        Args:
            df (pd.DataFrame): DataFrame that includes sentiment analysis results.
            output_prefix (str): Prefix for the output file names.
        """
        for label in ["positive", "neutral", "negative"]:
            filtered_df = df[df["sentiment"] == label][["News", "sentiment", "sentiment_score"]].head(50)
            file_name = f"{output_prefix}_{label}_news.csv"
            filtered_df.to_csv(file_name, index=False)
            print(f"• {file_name} saved.")

