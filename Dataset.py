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
        load_dataset: bool = True,
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
        if self.stock_data is None:
            print(f"No data found for {self.stock_symbol} in the specified date range.")
            return
        # Flatten multi-index columns if present
        if isinstance(self.stock_data.columns, pd.MultiIndex):
            self.stock_data.columns = self.stock_data.columns.get_level_values(0)

        self.stock_data.reset_index(inplace=True)
        self.stock_data = self.stock_data[["Date", "Close"]]
        self.stock_data.rename(columns={"Date": "date"}, inplace=True)
        self.stock_data["date"] = pd.to_datetime(self.stock_data["date"], utc=True)

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
            self.news_data = self.data[["date", "company_focused_summary"]]
            self.stock_data = self.data[["date", "Close"]]

            if save:
                self.save_csv()
        except Exception as e:
            print(f"Error merging stock and news data: {str(e)}")

    def save_csv(self) -> None:
        self.data.to_csv(f"{self.stock_symbol.lower()}_data.csv", index=False)

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
            # Convert the 'date' column to datetime, coercing invalid values to NaT
            self.news_data["date"] = pd.to_datetime(self.news_data["date"], errors="coerce", utc=True)

            # Drop rows with invalid or missing dates
            self.news_data = self.news_data[self.news_data["date"].notna()]

            # Normalize the dates to remove time components
            self.news_data["date"] = self.news_data["date"].dt.normalize()

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
        elif self.stock_symbol == "AMZN":
            dict_keywords = {
                "company": ["Amazon", "amzn", "amazon.com"],
                "products": ["aws", "prime", "alexa", "echo", "kindle"],
                "people": ["jeff bezos", "andy jassy"],
            }
        elif self.stock_symbol == "GOOGL":
                dict_keywords = {
                    "company": ["Google", "GOOGL", "Alphabet Inc", "Alphabet"],
                    "products": [
                        "Search", "YouTube", "Android", "Chrome", "Gmail", 
                        "Google Maps", "Google Drive", "Google Cloud", "Pixel", 
                        "Google Play", "Google Assistant"
                    ],
                    "people": ["Sundar Pichai", "Larry Page", "Sergey Brin"]
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
        # self.news_data["company_focused_summary"] = self.news_data[
        #     "company_focused_summary"
        # ].apply(
        #     lambda x: (
        #         ". ".join(sorted(set(x.split(". ")), key=x.split(". ").index))
        #         if isinstance(x, str)
        #         else x
        #     )
        # )
        if save:

            self.news_data[["date","company_focused_summary"]].to_csv(
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

    def add_setiment(
        self, sentiment: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Add sentiment analysis results to the dataset.

        Args:
            sentiment (pd.DataFrame): DataFrame containing sentiment analysis results.
        """
        if sentiment is not None:
            # Exclude 'company_focused_summary' from the sentiment DataFrame
            sentiment = sentiment.drop(columns=["company_focused_summary"], errors="ignore")

            # Merge on the 'date' column
            self.data = pd.merge(
                self.data,
                sentiment,
                on="date",
                how="left"
            )

        else:
            print("No sentiment data provided.")
            return None
    def plot_sentiment_distribution(self, chart_type: str = "bar") -> None:
        """
        Plot the distribution of sentiments as a bar chart or pie chart, excluding "No Information".

        Args:
            chart_type (str): Type of chart to plot ("bar" or "pie").
        """
        if "sentiment" not in self.data.columns:
            print("The 'sentiment' column is missing in the dataset.")
            return

        # Exclude rows with "No Information"
        filtered_data = self.data[self.data["company_focused_summary"] != "No Information"]
        sentiment_counts = filtered_data["sentiment"].value_counts()

        if chart_type == "bar":
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                palette={"positive": "green", "neutral": "blue", "negative": "red"},
            )
            plt.title(f"Sentiment Distribution for {self.stock_symbol}", fontsize=16)
            plt.xlabel("Sentiment", fontsize=12)
            plt.ylabel("Count", fontsize=12)
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"./graphs/{self.stock_symbol.lower()}_sentiment_distribution_bar.png")
            plt.show()

        elif chart_type == "pie":
            plt.figure(figsize=(8, 8))
            plt.pie(
                sentiment_counts.values,
                labels=sentiment_counts.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=["green", "blue", "red"],
                wedgeprops={"edgecolor": "black"},
                textprops={"fontsize": 12},
            )
            plt.title(f"Sentiment Distribution for {self.stock_symbol}", fontsize=16)
            plt.savefig(f"./graphs/{self.stock_symbol.lower()}_sentiment_distribution_pie.png")
            plt.show()

        else:
            print("Invalid chart type. Please choose 'bar' or 'pie'.")

    def plot_sentiment_vs_stock_price(self) -> None:
        """
        Plot the correlation between sentiment categories (positive, neutral, negative) and stock prices.

        This scatter plot shows how sentiment categories relate to stock prices.
        """
        if "sentiment" not in self.data.columns or "Close" not in self.data.columns:
            print("Required columns ('sentiment' and 'Close') are missing in the dataset.")
            return

        # Exclude rows with "No Information"
        filtered_data = self.data[self.data["company_focused_summary"] != "No Information"]

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(12, 6))
        sns.stripplot(
            x=filtered_data["sentiment"],
            y=filtered_data["Close"],
            palette={"positive": "green", "neutral": "blue", "negative": "red"},
            jitter=True,
            alpha=0.7,
        )
        plt.title(f"Sentiment Categories vs Stock Price for {self.stock_symbol}", fontsize=16)
        plt.xlabel("Sentiment", fontsize=12)
        plt.ylabel("Stock Price (Close)", fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"./graphs/{self.stock_symbol.lower()}_sentiment_vs_stock_price_categories.png")
        plt.show()

    def plot_sa_distribution_by_movement(self, score_type="positive_score"):
        filtered = self.data[self.data["company_focused_summary"] != "No Information"].copy()
        filtered["movement"] = filtered["Close"].diff().apply(lambda x: "rise" if x > 0 else ("fall" if x < 0 else "flat"))
        filtered = filtered[filtered["movement"] != "flat"]

        plt.figure(figsize=(8, 6))
        sns.violinplot(x="movement", y=score_type, data=filtered, palette="muted", order=["fall", "rise"])
        plt.title(f"{score_type} Distribution by Stock Movement")
        plt.xlabel("Stock Movement")
        plt.ylabel("Sentiment Score")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"./graphs/{self.stock_symbol.lower()}_{score_type}_sa_distribution.png")
        plt.show()
    def plot_correlation_matrix(self):
        cols = ["positive_score", "neutral_score", "negative_score", "Close"]
        corr = self.data[cols].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix: Sentiment vs Stock Price")
        plt.tight_layout()
        plt.savefig(f"./graphs/{self.stock_symbol.lower()}_correlation_matrix.png")
        plt.show()
# Test the implementation
if __name__ == "__main__":
    dt = Dataset_Class("GOOGL", load_dataset=True)
    if True:
        dt.sccater_plot("2018-01-01", "2018-02-01")
        dt.pie_graph()
        dt.plot_sentiment_distribution("pie")
        dt.plot_sentiment_vs_stock_price()
        dt.plot_sa_distribution_by_movement()
        dt.plot_correlation_matrix()
