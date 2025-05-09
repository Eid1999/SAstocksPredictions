import yfinance
import kagglehub
import pandas as pd
import os
import sys
import spacy
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import Optional


class Dataset_Class:
    def __init__(
        self,
        stock_symbol: str,
        start_date: str = "2009-04-27",
        end_date: str = "2020-06-11",
        load_dataset: bool = False,
    ):
        """
        Initialize the Dataset_Class with stock symbol, date range, and dataset loading options.

        Args:
            stock_symbol (str): Stock symbol (e.g., "AAPL", "TSLA").
            start_date (str): Start date for the data (YYYY-MM-DD).
            end_date (str): End date for the data (YYYY-MM-DD).
            load_dataset (bool): Whether to load an existing dataset or fetch new data.
        """
        self.stock_symbol = stock_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.news_data = pd.DataFrame(columns=["title", "date"])
        self.get_keywords()

        if load_dataset:
            self.load_csv()
        else:
            self.get_news()
            self.clean_data()
            self.retrive_stock_values()
            self.generate_company_focused_summary_keywords()
            self.merge_stock_and_news_data()

    def load_csv(self) -> None:
        """
        Load an existing dataset from a CSV file and filter it by the date range.
        """
        self.data = pd.read_csv(f"./{self.stock_symbol.lower()}_data.csv")
        self.data["date"] = pd.to_datetime(self.data["date"])

        # Filter data by date range
        mask = (self.data["date"] >= self.start_date) & (self.data["date"] <= self.end_date)
        self.data = self.data[mask]

        # Separate news and stock data
        self.news_data = self.data[["date", "company_focused_summary"]]
        self.stock_data = self.data[["date", "Close"]]

    def retrive_stock_values(self) -> None:
        """
        Retrieve stock values using the yfinance library and process the data.
        """
        self.stock_data = yfinance.download(
            self.stock_symbol,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
        )

        # Flatten multi-index columns if present
        if isinstance(self.stock_data.columns, pd.MultiIndex):
            self.stock_data.columns = self.stock_data.columns.get_level_values(0)

        self.stock_data.reset_index(inplace=True)
        self.stock_data = self.stock_data[["Date", "Close"]]
        self.stock_data.rename(columns={"Date": "date"}, inplace=True)

    def merge_stock_and_news_data(self, save: bool = True) -> None:
        """
        Merge stock data and news data on the 'date' column.

        Args:
            save (bool): Whether to save the merged data to a CSV file.
        """
        try:
            self.data = pd.merge(
                self.stock_data,
                self.news_data,
                on="date",
                how="left"
            )
            self.data.fillna("No Information", inplace=True)

            if save:
                self.data.to_csv(f"{self.stock_symbol.lower()}_data.csv", index=False)
        except Exception as e:
            print(f"Error merging stock and news data: {str(e)}")

    def get_news(self) -> None:
        """
        Download and extract news data from Kaggle.
        """
        try:
            path = kagglehub.dataset_download(
                "miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests",
                "analyst_ratings_processed.csv",
                force_download=True,
            )
            self.extract_dataset(path)
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")

    def extract_dataset(self, path: str) -> None:
        """
        Extract the dataset from a given path.

        Args:
            path (str): Path to the dataset (directory or CSV file).
        """
        try:
            if os.path.isdir(path):
                csv_files = [os.path.join(root, file)
                             for root, _, files in os.walk(path) for file in files if file.endswith(".csv")]

                if csv_files:
                    self.news_data = pd.read_csv(csv_files[0])
                else:
                    print("No CSV files found in the directory")
            elif path.endswith(".csv"):
                self.news_data = pd.read_csv(path)
            else:
                print(f"Unsupported file format: {path}")
        except Exception as e:
            print(f"Error extracting dataset: {str(e)}")

    def clean_data(self) -> None:
        """
        Clean and preprocess the news data.
        """
        try:
            self.news_data["date"] = pd.to_datetime(self.news_data["date"], errors="coerce").dt.normalize()
            self.news_data.dropna(subset=["date"], inplace=True)

            # Filter by date range
            mask = (self.news_data["date"] >= self.start_date) & (self.news_data["date"] <= self.end_date)
            self.news_data = self.news_data[mask]

            # Aggregate news by date
            self.news_data = self.news_data.groupby("date")["title"].apply(" | ".join).reset_index()
        except Exception as e:
            print(f"Error in clean_data: {str(e)}")

    def get_keywords(self) -> None:
        """
        Define keywords for the target company based on the stock symbol.
        """
        if self.stock_symbol == "AAPL":
            dict_keywords = {
                "company": ["Apple", "aapl", "apple inc"],
                "products": ["iphone", "ipad", "macbook", "ios", "airpods"],
                "people": ["tim cook", "steve jobs"],
            }
        elif self.stock_symbol == "MSFT":
            dict_keywords = {
                "company": ["Microsoft", "msft", "microsoft corp"],
                "products": ["windows", "azure", "xbox"],
                "people": ["satya nadella", "bill gates"],
            }
        else:
            dict_keywords = {
                "company": ["Tesla", "tsla", "tesla inc"],
                "products": ["model s", "model 3", "cybertruck"],
                "people": ["elon musk"],
            }

        self.target_company = dict_keywords["company"][0]
        self.keywords = [item for sublist in dict_keywords.values() for item in sublist]

    def generate_company_focused_summary_keywords(self, save=False):
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

    def sccater_plot(self, start_date: str, end_date: str) -> None:
        """
        Generate a scatter plot showing days with and without relevant news.

        Args:
            start_date (str): Start date for the plot (YYYY-MM-DD).
            end_date (str): End date for the plot (YYYY-MM-DD).
        """
        mask = (self.data["date"] >= start_date) & (self.data["date"] <= end_date)
        data = self.data[mask]
        data["has_news"] = (data["company_focused_summary"] != "No Information").astype(int)

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(12, 6))
        sns.scatterplot(
            x=data["date"],
            y=data["has_news"],
            hue=data["has_news"],
            palette={0: "red", 1: "blue"},
            style=data["has_news"],
            markers={0: "X", 1: "o"},
            s=100,
        )
        plt.title(f"News Timeline for {self.target_company}", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("News Indicator", fontsize=12)
        plt.yticks([0, 1], labels=["No News", "Relevant News"])
        plt.legend(title="News Status", loc="upper left", bbox_to_anchor=(1.05, 1))
        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"./graphs/{self.stock_symbol.lower()}_news_timeline_seaborn.png")
        plt.show()

    def pie_graph(self) -> None:
        """
        Generate a pie chart showing the proportion of days with and without relevant news.
        """
        self.data["has_news"] = (self.data["company_focused_summary"] != "No Information").astype(int)
        news_counts = self.data["has_news"].value_counts()
        labels = ["No Information", "Relevant News"]
        sizes = [news_counts.get(0, 0), news_counts.get(1, 0)]
        colors = sns.color_palette("pastel")[:2]

        plt.figure(figsize=(8, 8))
        plt.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            wedgeprops={"edgecolor": "black"},
            textprops={"fontsize": 12},
        )
        plt.title(f"Proportion of News for {self.target_company}", fontsize=16)
        plt.savefig(f"./graphs/{self.stock_symbol.lower()}_news_pie_chart_seaborn.png")
        plt.show()


# Test the implementation
if __name__ == "__main__":
    dt = Dataset_Class("TSLA", load_dataset=True)
    dt.sccater_plot("2018-01-01", "2018-02-01")
    dt.pie_graph()
