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
        model_name: str = "yangheng/deberta-v3-base-absa-v1.1",
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
        Analyze aspect-based sentiment using yangheng/deberta-v3-base-absa.
        Uses [CLS] TEXT [SEP] ASPECT [SEP] format expected by the model.
        """
        df = self.df.copy()
        texts = df[self.text_column].fillna("No information").astype(str).tolist()
        dates = df["date"].tolist()
        all_results = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Targeted Sentiment Analysis"):
            batch_texts = texts[i:i + batch_size]
            batch_dates = dates[i:i + batch_size]

            for text, date in zip(batch_texts, batch_dates):
                result = {"date": date, self.text_column: text}

                if text.strip().lower() == "no information":
                    result.update({
                        "negative_score": 0.0,
                        "neutral_score": 1.0,
                        "positive_score": 0.0,
                    })
                    all_results.append(result)
                    continue

                negative_scores = []
                neutral_scores = []
                positive_scores = []

                for keyword in self.keywords:
                    if keyword.lower() in text.lower():
                        # Tokenize using two-part format: text + aspect
                        inputs = self.tokenizer(
                            text.strip(),
                            keyword.strip(),
                            padding=True,
                            truncation_strategy='only_first',  # truncate long sentences
                            max_length=512,
                            return_tensors="pt"
                        ).to(self.device)

                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            probs = F.softmax(outputs.logits, dim=1)
                            scores = probs[0].cpu().numpy()

                        negative_scores.append(scores[0])
                        neutral_scores.append(scores[1])
                        positive_scores.append(scores[2])

                # If no keywords matched
                if not negative_scores:
                    result.update({
                        "negative_score": 0.0,
                        "neutral_score": 1.0,
                        "positive_score": 0.0,
                    })
                else:
                    result.update({
                        "negative_score": float(sum(negative_scores) / len(negative_scores)),
                        "neutral_score": float(sum(neutral_scores) / len(neutral_scores)),
                        "positive_score": float(sum(positive_scores) / len(positive_scores)),
                    })

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
