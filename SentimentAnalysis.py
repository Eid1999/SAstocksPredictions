import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from Dataset import Dataset_Class
import pandas as pd
import pdb

class FinBertTargetSentimentAnalyzer:

    def __init__(
        self,
        dataset: Dataset_Class,
        model_name: str = "yiyanghkust/finbert-tone",
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

    def analyze_target_sentiment(self, batch_size=64):
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
                    result.update(
                        {"keyword": None, "sentiment": "neutral", "score": 0.0}
                    )
                    all_results.append(result)
                    continue

                aspect_sentiments = []

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
                            label = torch.argmax(probs, dim=1).item()
                            sentiment = self.label_map[label]
                            score = probs[0][label].item()

                        aspect_sentiments.append((keyword, sentiment, score))

                if aspect_sentiments:
                    # Select keyword with highest score
                    best = max(aspect_sentiments, key=lambda x: x[2])
                    result.update(
                        {"keyword": best[0], "sentiment": best[1], "score": best[2]}
                    )
                else:
                    result.update(
                        {"keyword": None, "sentiment": "neutral", "score": 0.0}
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
    dt = Dataset_Class("MSFT", load_dataset=False)
    analyzer = FinBertTargetSentimentAnalyzer(dt)
    result_df = analyzer.analyze_target_sentiment()
    analyzer.save_sentiment_file(result_df)
    result_df = dt.add_setiment(result_df)
    dt.save_csv()
