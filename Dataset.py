import yfinance
import kagglehub
import pandas as pd
import os
import zipfile
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


class Dataset_Class:

    def __init__(self, stock_symbol, start_date="2010-1-1", end_date="2020-1-1"):
        self.keywords = dict()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_hug = 0 if torch.cuda.is_available() else -1
        self.stock_symbol = stock_symbol
        self.news_data = pd.DataFrame(columns=["title", "date"])
        self.start_date = start_date
        self.end_date = end_date
        self.get_news()
        self.clean_data()
        self.retrive_stock_values()
        self.load_ner_pipeline()
        self.filter_news_by_ner_and_keywords(self.news_data)
        self.generate_company_focused_summary()

    def retrive_stock_values(self):
        self.stock_data = yfinance.download(
            self.stock_symbol,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
        )

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

            # If path is a zip file
            elif path.endswith(".zip"):
                with zipfile.ZipFile(path, "r") as zip_ref:
                    # Extract to a temporary directory
                    temp_dir = os.path.join(os.path.dirname(path), "temp_extracted")
                    os.makedirs(temp_dir, exist_ok=True)
                    zip_ref.extractall(temp_dir)

                    # Look for CSV files in extracted content
                    self.news_data = self.extract_dataset(temp_dir)

                    # Clean up
                    import shutil

                    shutil.rmtree(temp_dir)
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
            news_data["date"] = pd.to_datetime(
                news_data["date"], errors="coerce", utc=True
            )

            # Drop rows with invalid dates
            news_data = news_data.dropna(subset=["date"])

            # Format dates to YYYY-MM-DD (date only)
            news_data["date"] = news_data["date"].dt.strftime("%Y-%m-%d")
            news_data["date"] = pd.to_datetime(news_data["date"])

            # Filter by date range before aggregating (reduces processing)
            mask = (news_data["date"] >= self.start_date) & (
                news_data["date"] <= self.end_date
            )
            news_data = news_data[mask]

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

    def load_ner_pipeline(self):
        model_name = (
            "Jean-Baptiste/roberta-large-ner-english"  # Or FinBERT-NER if available
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner_pipe = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=self.device_hug,
        )

    def filter_news_by_ner_and_keywords(
        self, df, text_column="title", keywords=None, save=True
    ):

        if keywords is None:
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

        keywords = [k.lower() for k in keywords]

        texts = df[text_column].astype(str).tolist()
        results = []
        batch_size = 200

        for i in tqdm(range(0, len(texts), batch_size), desc="NER Filtering"):
            batch = texts[i : i + batch_size]
            batch_results = self.ner_pipe(batch)
            results.extend(batch_results)

        keep_indices = []
        for i, entities in enumerate(results):
            if any(
                any(kw in entity["word"].lower() for kw in keywords)
                for entity in entities
            ):
                keep_indices.append(i)

        self.news_data = df.iloc[keep_indices].reset_index(drop=True)
        self.keywords = keywords
        if save == True:
            self.news_data.to_csv(f"{self.stock_symbol}_news.csv", index=False)

    def generate_company_focused_summary(
        self,
        text_column="title",
        max_length=150,
        min_length=30,
        batch_size=8,
        save=True,
    ):
        """
        Generates summaries focused on the specific company/stock
        """
        # Initialize T5 model and tokenizer
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        model = T5ForConditionalGeneration.from_pretrained("t5-base")

        model.to(self.device)

        # Create company-specific prompt prefix
        company_names = {"AAPL": "Apple", "MSFT": "Microsoft", "TSLA": "Tesla"}
        company_name = company_names.get(self.stock_symbol, self.stock_symbol)
        flat_keywords = [kw for kws in self.keywords.values() for kw in kws]
        keywords_string = ", ".join(flat_keywords)

        prefix = (
            f"If there is any mention of {keywords_string} in the news in the following text, "
            "provide a brief summary. If not, reply with 'neutral'. Summarize: "
        )

        all_summaries = []  # List to store all summaries
        texts = self.news_data[text_column].astype(str).tolist()

        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="Generating Company-Focused Summaries",
        ):
            batch_texts = texts[i : i + batch_size]
            prefixed_texts = [prefix + text for text in batch_texts]

            # Prepare inputs
            inputs = tokenizer(
                prefixed_texts,
                max_length=512,
                truncation=True,
                padding="longest",
                return_tensors="pt",
            ).to(self.device)

            # Generate summaries
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                )

            # Decode summaries
            batch_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Clean up summaries for this batch - ALWAYS remove prefix first
            for summary in batch_summaries:
                all_summaries.append(summary)

        # Add summaries to the dataframe
        self.news_data["company_focused_summary"] = all_summaries

        if save:
            self.news_data[["company_focused_summary"]].to_csv(
                f"summary_{self.stock_symbol}_news.csv", index=False
            )

        return self.news_data

    def __str__(self):
        return str(self.data)


# Test the implementation
if __name__ == "__main__":
    dt = Dataset_Class("AAPL")
