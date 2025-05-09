import yfinance
import kagglehub
import pandas as pd
import os
import pdb

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
from tqdm import tqdm
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import nltk


from nltk.tokenize import sent_tokenize
nltk.download("punkt_tab")
nltk.download("punkt")


class Dataset_Class:

    def __init__(self, stock_symbol, start_date="2010-1-1", end_date="2020-1-1", load_dataset=False):
        self.keywords = dict()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_hug = 0 if torch.cuda.is_available() else -1
        self.stock_symbol = stock_symbol
        self.news_data = pd.DataFrame(columns=["title", "date"])
        self.start_date = start_date
        self.end_date = end_date
        if load_dataset:
            self.load_csvs()
        else:
            self.get_news()
            self.clean_data()
            self.retrive_stock_values()
            self.get_keywords()
            self.generate_company_focused_summary()

    def load_csv(self):
        self.news_data = pd.read_csv(f"./summary_{self.stock_symbol}_news.csv")
        self.data=pd.read_csv(f"./summary_{self.stock_symbol}_data.csv")

    def retrive_stock_values(self):

        df = yfinance.download(
            self.stock_symbol,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
        )
        if df is not None:
            df.index=pd.to_datetime(df.index)
            self.stock_data = df.resample("M").last()
        else:
            exit(0)

    def get_news(self):
        try:
            path = kagglehub.dataset_download(
                "miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests",
                "analyst_ratings_processed.csv",
                force_download=True,
            )

            self.extract_dataset(path)
            print(f"Dataset loaded successfully from: {path}")
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")

    def extract_dataset(self, path):

        try:
            # If path is a directory, look for CSV files
            if os.path.isdir(path):
                csv_files = []

                # Walk through directory
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.endswith(".csv"):
                            csv_files.append(os.path.join(root, file))

                if csv_files:
                    # Load the first CSV file found
                    df = pd.read_csv(csv_files[0])
                    print(f"Found CSV files: {csv_files}")
                    print(f"Loading: {csv_files[0]}")
                    print(f"Dataframe shape: {df.shape}")
                    print("\nPreview of the data:")
                    print(df.head())
                    self.news_data = df
                else:
                    print("No CSV files found in the directory")

            # If path is a direct CSV file
            elif path.endswith(".csv"):
                df = pd.read_csv(path)
                print(f"Loading CSV file: {path}")
                print(f"Dataframe shape: {df.shape}")
                print("\nPreview of the data:")
                print(df.head())
                self.news_data = df

            else:
                print(f"Unsupported file format: {path}")

        except Exception as e:
            print(f"Error extracting dataset: {str(e)}")

    def clean_data(self):
        # Clean the data
        try:
            news_column = "title"
            # Only work with necessary columns and drop NAs upfront
            news_data = self.news_data[[news_column, "date"]].dropna()

            # Vectorized date conversion for better performance
            print("Converting dates...")
            # Convert to datetime directly with errors='coerce'
            news_data["date"] = pd.to_datetime(news_data["date"], errors="coerce", utc=True).dt.normalize()
            news_data = news_data.dropna(subset=["date"])
            news_data["date"] = news_data["date"].dt.tz_localize(None)  # Remove timezone
            news_data["date"] = news_data["date"].dt.to_period("ME").dt.to_timestamp(how="end")

            # Filter by date range
            mask = (news_data["date"] >= self.start_date) & (news_data["date"] <= self.end_date)
            news_data = news_data[mask]

            # Group by month and concatenate text
            news_data = (
                news_data.groupby("date")["title"]
                .apply(lambda x: " ".join(x))
                .reset_index()
            )
            # Aggregate news by date
            print("Aggregating news by date...")
            aggregated_news = (
                news_data.groupby("date")[news_column]
                .agg(lambda x: " | ".join(x))
                .reset_index()
            )
            self.news_data = aggregated_news.sort_values(by="date")
            self.news_data.reset_index(drop=True, inplace=True)

            print(f"Cleaned data shape: {self.news_data.shape}")

        except Exception as e:
            print(f"Error in clean_data: {str(e)}")
            print("Preview of problematic data:")
            print(self.news_data[["date"]].head())
            raise

    def create_data(self):
        pass

    def get_keywords(
        self 
    ):

        if self.stock_symbol == "AAPL":
            keywords = {
                "company": ["apple", "aapl", "apple inc"],
                "products": [
                    "iphone",
                    "ipad",
                    "macbook",
                    "mac",
                    "ios",
                    "apple watch",
                    "airpods",
                    "imac",
                    "apple tv",
                    "m1 chip",
                    "vision pro",
                    "m2 chip",
                ],
                "people": ["tim cook", "steve jobs"],
            }
        elif self.stock_symbol == "MSFT":
            keywords = {
                "company": ["microsoft", "msft", "microsoft corp"],
                "products": ["windows", "office", "azure", "surface", "xbox"],
                "people": ["satya nadella", "bill gates"],
            }
        else:
            keywords = {
                "company": ["tesla", "tsla", "tesla inc"],
                "products": [
                    "model s",
                    "model 3",
                    "model x",
                    "model y",
                    "cybertruck",
                ],
                "people": ["elon musk"],
            }
        self.target_company = keywords["company"][0]
        keywords = [item for sublist in keywords.values() for item in sublist]

        self.keywords = keywords

    def generate_company_focused_summary(
        self,
        text_column="title",
        target_company="apple",  # Add target company here
        max_length=150,
        min_length=30,
        batch_size=8,
        save=True,
        ):


        if not hasattr(self, "summary_tokenizer"):
            self.summary_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
            self.summary_model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
            self.summary_model.to(self.device)

        all_summaries = []
        is_company_related = []
        texts = self.news_data[text_column].astype(str).tolist()

        # Lowercase keywords to simplify comparison
        lowercase_keywords = [k.lower() for k in self.keywords]
        target = target_company.lower()

        for i in tqdm(range(0, len(texts), batch_size), desc="Generating Company-Focused Summaries"):
            batch_texts = texts[i : i + batch_size]
            batch_summaries = []
            batch_is_related = []

            for text in batch_texts:
                sentences = sent_tokenize(text)
                
                # Step 1: Get sentences mentioning ONLY the target company
                related_sentences = [
                    s for s in sentences
                    if target in s.lower()
                    and all(k not in s.lower() for k in lowercase_keywords if k != target)
                ]

                # Step 2: Fallback if none found
                if not related_sentences:
                    related_sentences = [s for s in sentences if target in s.lower()]

                if related_sentences:
                    relevant_text = f"Summarize the following only with regard to {target.capitalize()}: " + " ".join(related_sentences)
                    inputs = self.summary_tokenizer(
                        relevant_text,
                        max_length=1024,
                        truncation=True,
                        return_tensors="pt",
                    ).to(self.device)

                    with torch.no_grad():
                        summary_ids = self.summary_model.generate(
                            inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            max_length=max_length,
                            min_length=min_length,
                            length_penalty=2.0,
                            num_beams=4,
                            early_stopping=True,
                        )
                    summary = self.summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    batch_summaries.append(summary)
                    batch_is_related.append(True)
                else:
                    batch_summaries.append("neutral")
                    batch_is_related.append(False)

            all_summaries.extend(batch_summaries)
            is_company_related.extend(batch_is_related)

        self.news_data["company_focused_summary"] = all_summaries
        self.news_data["is_company_related"] = is_company_related



        if save:
            related = self.news_data[
                (self.news_data["is_company_related"] == True)
                & (self.news_data["company_focused_summary"] != "neutral")
            ]
            self.news_data[["company_focused_summary", "is_company_related"]].to_csv(
                f"summary_{self.stock_symbol.lower()}_news", index=False
            )
            self.news_data.to_csv(
                f"summary_{self.stock_symbol.lower()}_data", index=False
            )
            print(f"Saved {len(related)} {self.stock_symbol} focused summaries")

        return self.news_data

    def __str__(self):
        return str(self.data)


# Test the implementation
if __name__ == "__main__":
    dt = Dataset_Class("AAPL")
