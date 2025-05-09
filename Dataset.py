import yfinance
import kagglehub
import pandas as pd
import os
import sys
import pdb
import spacy
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from tqdm import tqdm
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import nltk
from langchain_ollama import ChatOllama


from nltk.tokenize import sent_tokenize
nltk.download("punkt_tab")
nltk.download("punkt")
from spacy import require_gpu

class Dataset_Class:

    def __init__(
        self,
        stock_symbol,
        start_date="2009-02-14",
        end_date="2020-06-11",
        load_dataset=False,
    ):
        self.keywords = dict()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_hug = 0 if torch.cuda.is_available() else -1
        self.stock_symbol = stock_symbol
        self.news_data = pd.DataFrame(columns=["title", "date"])
        self.start_date = start_date
        self.end_date = end_date
        if load_dataset:
            self.load_csv()
        else:

            self.get_news()
            self.clean_data()
            self.retrive_stock_values()

            self.get_keywords()
            self.generate_company_focused_summary_keywords()
            self.merge_stock_and_news_data()

    def load_csv(self):

        self.data=pd.read_csv(f"./{self.stock_symbol.lower()}_data.csv")

        # Filter by date range
        mask = (self.data["date"] >= self.start_date) & (
            self.data["date"] <= self.end_date
        )
        self.data = self.data[mask]
        self.news_data=self.data[["date", "company_focused_summary"]]
        self.stock_data=self.data[["date", "Close"]]
        pdb.set_trace()

    def retrive_stock_values(self):

        self.stock_data = yfinance.download(
            self.stock_symbol,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
        )
        if isinstance(self.stock_data.columns, pd.MultiIndex):
            # Flatten the multi-index columns by selecting the first level
            self.stock_data.columns = self.stock_data.columns.get_level_values(0)
            self.stock_data.reset_index(inplace=True)
        self.stock_data=self.stock_data[["Date","Close"]]
        self.stock_data.rename(
            columns={"Date": "date"}, inplace=True
        )
        self.stock_data.reset_index(inplace=True)

    def merge_stock_and_news_data(self):
        """
        Perform a left join between self.stock_data and self.news_data on the 'date' column.
        """
        try:
            # Merge the two DataFrames on the 'date' column
            self.data = pd.merge(
                self.stock_data,
                self.news_data,
                on="date",
                how="left"
            )
            self.data.fillna("No Information", inplace=True)
            self.data.to_csv(
                f"{self.stock_symbol.lower()}_data.csv", index=False
            )

        except Exception as e:
            print(f"Error merging stock and news data: {str(e)}")

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

            # Filter by date range
            mask = (news_data["date"] >= self.start_date) & (
                news_data["date"] <= self.end_date
            )
            news_data = news_data[mask]

            # Group by exact date and concatenate text
            print("Aggregating news by date...")
            aggregated_news = (
                news_data.groupby("date")[news_column]
                .apply(lambda x: " | ".join(x))
                .reset_index()
            )

            self.news_data = aggregated_news.sort_values(by="date")
            self.news_data.reset_index(drop=True, inplace=True)

            print(f"Cleaned data shape: {self.news_data.shape}")
            self.news_data.to_csv(
                f"data_{self.stock_symbol.lower()}_news.csv", index=False
            )

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
            dict_keywords = {
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
            dict_keywords = {
                "company": ["microsoft", "msft", "microsoft corp"],
                "products": ["windows", "office", "azure", "surface", "xbox"],
                "people": ["satya nadella", "bill gates"],
            }
        else:
            dict_keywords = {
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
        self.target_company = dict_keywords["company"][0]
        keywords = [item for sublist in dict_keywords.values() for item in sublist]
        self.dict_keywords = dict_keywords
        self.keywords = keywords

    def generate_company_focused_summary_keywords(self, save=True):
        # Create a new column for the summaries while maintaining the original structure
        self.news_data["company_focused_summary"] = None  # Initialize the column

        print(f"Using keywords: {self.keywords}")  # Debug keyword check

        for index, row in self.news_data.iterrows():
            content = str(row.get("title", "")).strip()
            if not content:
                print(f"Empty or invalid content at row {index}")
                self.news_data.at[index, "company_focused_summary"] = "No Information"
                continue

            # Split by ., -, |, ; but avoid splitting decimal numbers
            sentences = re.split(r"(?<!\d)\.(?!\d)|[\-\|\;]", content)
            matched_sentences = []

            for sentence in sentences:
                sentence = sentence.strip()
                if any(
                    re.search(rf"\b{re.escape(keyword)}\b", sentence, re.IGNORECASE)
                    for keyword in self.keywords
                ):
                    matched_sentences.append(sentence)

            if matched_sentences:
                self.news_data.at[index, "company_focused_summary"] = ". ".join(matched_sentences)
            else:

                self.news_data.at[index, "company_focused_summary"] = "No Information"
        if save:

            self.news_data[["date","company_focused_summary"]].to_csv(
                f"summary_{self.stock_symbol.lower()}_news.csv", index=False
            )
            self.news_data.to_csv(
                f"summary_{self.stock_symbol.lower()}_data.csv", index=False
            )

    def generate_company_focused_summary_ner(self, save=False, use_gpu=True):
        """
        Generate summaries of news content focused on company entities using spaCy's NER.
        Uses GPU if `use_gpu=True` and a transformer model is available.
        """

        # Load spaCy model with GPU support if requested
        nlp = None
        if use_gpu:
            try:

                require_gpu()  # use default GPU
                nlp = spacy.load("en_core_web_trf")
                print("Using GPU with en_core_web_trf")
            except Exception as e:
                print(f"Failed to load GPU-based model: {e}")
                print("Falling back to CPU-based model...")

        if nlp is None:
            try:
                nlp = spacy.load("en_core_web_md")
                print("Using en_core_web_md (CPU)")
            except OSError:
                try:
                    nlp = spacy.load("en_core_web_sm")
                    print("Using en_core_web_sm (CPU)")
                except OSError:
                    print("No spaCy models found. Downloading en_core_web_sm...")
                    try:
                        import subprocess

                        subprocess.check_call(
                            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"]
                        )
                        nlp = spacy.load("en_core_web_sm")
                        print("Downloaded and loaded en_core_web_sm model")
                    except Exception as e:
                        print(f"Error installing spaCy model: {e}")
                        nlp = spacy.blank("en")  # very basic fallback

        self.news_data["company_focused_summary"] = None  # Initialize the column
        print("Starting NER-based filtering...")

        # Normalize company-related terms
        company_terms = set()
        for category in ["company", "products", "people"]:
            if hasattr(self, category):
                company_terms.update([term.lower() for term in getattr(self, category)])

        for index, row in tqdm(
            self.news_data.iterrows(), total=len(self.news_data), desc="Processing news"
        ):
            content = str(row.get("title", "")).strip()
            if not content:
                print(f"Empty or invalid content at row {index}")
                self.news_data.at[index, "company_focused_summary"] = "No Information"
                continue

            sentences = re.split(r"(?<!\d)\.(?!\d)|[\-\|\;]", content)
            matched_sentences = []

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                doc = nlp(sentence)
                entities_found = [
                    ent.text.lower()
                    for ent in doc.ents
                    if ent.label_ in ["ORG", "PRODUCT", "PERSON", "GPE"]
                ]

                if any(entity in company_terms for entity in entities_found):
                    matched_sentences.append(sentence)

            if matched_sentences:
                self.news_data.at[index, "company_focused_summary"] = ". ".join(
                    matched_sentences
                )
            else:
                print(
                    f"No match at row {index}:\n  Content: {content}\n  Entities: {entities_found}\n  Terms: {company_terms}"
                )
                self.news_data.at[index, "company_focused_summary"] = "No Information"

        if save:
            self.news_data[["date", "company_focused_summary"]].to_csv(
                f"summary_{self.stock_symbol.lower()}_news.csv", index=False
            )
            self.news_data.to_csv(
                f"summary_{self.stock_symbol.lower()}_data.csv", index=False
            )

    def __str__(self):
        return str(self.data)


# Test the implementation
if __name__ == "__main__":
    dt = Dataset_Class("AAPL", load_dataset=True)
