# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import string
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Ensure NLTK data is downloaded (run this once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# --- Configuration ---
# List of URLs to analyze
URLS = [
    # Add your target URLs here, for example:
    # "https://en.wikipedia.org/wiki/Python_(programming_language)",
    # "https://en.wikipedia.org/wiki/Natural_language_processing"
    "https://www.strategicmarketresearch.com/market-report/power-drill-market", # Placeholder - Replace with actual URLs
    "https://www.marketdataforecast.com/market-reports/north-america-power-tools-market" ,
    "https://www.researchandmarkets.com/report/power-drill",
    "https://www.grandviewresearch.com/industry-analysis/power-tools-market",
    "https://www.gminsights.com/industry-analysis/electric-power-tools-market",
    "https://www.maximizemarketresearch.com/market-report/global-power-drill-market/88127/",
    "https://www.grandviewresearch.com/industry-analysis/rotary-hammer-drill-market",
    "https://www.grandviewresearch.com/industry-analysis/us-power-tools-market-report" # Placeholder - Replace with actual URLs
]

# Output directory for results
OUTPUT_DIR = "word_analysis_results"

# Number of top words to display in heatmap and table
TOP_N_WORDS = 20

# --- Functions ---

def fetch_text_from_url(url):
    """Fetches and extracts visible text content from a given URL."""
    try:
        response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Get text, strip leading/trailing whitespace, and handle multiple lines
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def preprocess_text(text):
    """Cleans and preprocesses English text: lowercase, tokenize, remove punctuation and stopwords."""
    if not text:
        return []
    # Convert to lowercase
    text = text.lower()
    # Remove numbers and words containing numbers
    text = re.sub(r'\b\w*\d\w*\b', '', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove punctuation
    tokens = [word for word in tokens if word.isalpha()] # Keep only alphabetic tokens
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    # Add any custom stopwords if needed
    # custom_stopwords = {'example', 'custom'}
    # stop_words.update(custom_stopwords)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1] # Remove single-letter words too
    return tokens

def generate_word_cloud(word_counts, filename):
    """Generates and saves a word cloud from word counts."""
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(filename)
    plt.close()
    print(f"Word cloud saved to {filename}")

def generate_heatmap(word_counts, filename, top_n=20):
    """Generates and saves a heatmap of top N word frequencies."""
    top_words = word_counts.most_common(top_n)
    if not top_words:
        print("No words found to generate heatmap.")
        return
    df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
    df = df.set_index('Word')

    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt="d", cmap="viridis")
    plt.title(f'Top {top_n} Word Frequency Heatmap')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Heatmap saved to {filename}")

def generate_frequency_table(word_counts, filename, top_n=20):
    """Generates and saves a frequency table (CSV) of top N words."""
    top_words = word_counts.most_common(top_n)
    if not top_words:
        print("No words found to generate frequency table.")
        return
    df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
    df.to_csv(filename, index=False)
    print(f"Frequency table saved to {filename}")

# --- Main Execution ---
if __name__ == "__main__":
    if not URLS or all(url.startswith("https://example.com") for url in URLS):
        print("Error: Please update the 'URLS' list in the script with the actual web page URLs you want to analyze.")
    else:
        print(f"Analyzing {len(URLS)} URLs...")
        all_tokens = []
        for url in URLS:
            print(f"Fetching and processing {url}...")
            text = fetch_text_from_url(url)
            if text:
                tokens = preprocess_text(text)
                all_tokens.extend(tokens)
            else:
                print(f"Skipping {url} due to fetch error.")

        if not all_tokens:
            print("No text could be processed from the provided URLs.")
        else:
            print(f"Total words processed (after cleaning): {len(all_tokens)}")
            word_counts = Counter(all_tokens)
            print(f"Unique words found: {len(word_counts)}")

            # Create output directory if it doesn't exist
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
                print(f"Created output directory: {OUTPUT_DIR}")

            # Generate outputs
            wordcloud_file = os.path.join(OUTPUT_DIR, "word_cloud.png")
            heatmap_file = os.path.join(OUTPUT_DIR, "frequency_heatmap.png")
            table_file = os.path.join(OUTPUT_DIR, "frequency_table.csv")

            generate_word_cloud(word_counts, wordcloud_file)
            generate_heatmap(word_counts, heatmap_file, top_n=TOP_N_WORDS)
            generate_frequency_table(word_counts, table_file, top_n=TOP_N_WORDS)

            print("\nAnalysis complete. Results saved in:")
            print(f"- Word Cloud: {wordcloud_file}")
            print(f"- Heatmap: {heatmap_file}")
            print(f"- Frequency Table: {table_file}")
            print(f"\nTop {TOP_N_WORDS} words:")
            for word, count in word_counts.most_common(TOP_N_WORDS):
                print(f"- {word}: {count}")