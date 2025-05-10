import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from Dataset import Dataset_Class
import pandas as pd
import pdb
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
class FinBertTargetSentimentAnalyzer:

    def __init__(
        self,
        dataset: Dataset_Class,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        text_column: str = "company_focused_summary",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()
        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}
        self.df = dataset.news_data
        self.text_column = text_column
        self.keywords = dataset.keywords
        self.stock_symbol = dataset.stock_symbol

    def analyze_target_sentiment(self, batch_size=16):
        """
        Analyze sentiment for specific aspects (keywords) in the text using the model.
        Save only the scores for each category. If no information is available, set neutral=1.0 and others=0.0.
        """
        df = self.df.copy()
        texts = df[self.text_column].fillna("No information").astype(str).tolist()
        dates = df["date"].tolist()  # Include the date column
        all_results = []

        for i in tqdm(
            range(0, len(texts), batch_size), desc="Targeted Sentiment Analysis"
        ):
            batch_texts = texts[i : i + batch_size]
            batch_dates = dates[i : i + batch_size]  # Corresponding dates for the batch

            for text, date in zip(batch_texts, batch_dates):
                result = {"date": date, self.text_column: text}  # Include the date

                if text.strip().lower() == "no information":
                    # Default scores when no information is available
                    result.update(
                        {
                            "negative_score": 0.0,
                            "neutral_score": 1.0,
                            "positive_score": 0.0,
                        }
                    )
                    all_results.append(result)
                    continue

                # Initialize scores
                negative_score = 0.0
                neutral_score = 0.0
                positive_score = 0.0

                for keyword in self.keywords:
                    if keyword.lower() in text.lower():
                        input_text = f"{keyword} [SEP] {text}"
                        inputs = self.tokenizer(
                            input_text,
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors="pt",
                        ).to(self.device)

                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            probs = F.softmax(outputs.logits, dim=1)
                            scores = probs[0].tolist()  # Get all scores for the sentiment categories

                        # Accumulate scores for each category
                        negative_score += scores[0]
                        neutral_score += scores[1]
                        positive_score += scores[2]

                # Normalize scores if multiple keywords are found
                total_keywords = len(self.keywords)
                if total_keywords > 0:
                    negative_score /= total_keywords
                    neutral_score /= total_keywords
                    positive_score /= total_keywords

                # Update result with scores
                result.update(
                    {
                        "negative_score": negative_score,
                        "neutral_score": neutral_score,
                        "positive_score": positive_score,
                    }
                )

                all_results.append(result)

        return pd.DataFrame(all_results)

    def save_sentiment_file(
        self, df
    ):
        output_file = f"sentiment_analysis_{self.stock_symbol}.csv"
        df.to_csv(output_file, index=False)
        print(f"• {output_file} saved.")


if __name__ == "__main__":
    dt = Dataset_Class("AAPL", load_dataset=False)
    analyzer = FinBertTargetSentimentAnalyzer(dt)
    result_df = analyzer.analyze_target_sentiment()
    analyzer.save_sentiment_file(result_df)
    result_df = dt.add_setiment(result_df)
    dt.save_csv()
