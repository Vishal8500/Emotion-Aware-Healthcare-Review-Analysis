import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the medical reviews
df = pd.read_csv(r"D:\NLP\medical_reviews.csv")

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Apply VADER sentiment analysis
def get_sentiment_label(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

print("Analyzing sentiments...")
df['vader_sentiment'] = df['text'].astype(str).apply(get_sentiment_label)

# Map star ratings to general labels
def star_to_label(stars):
    if stars >= 4:
        return 'Positive'
    elif stars == 3:
        return 'Neutral'
    else:
        return 'Negative'

df['rating_sentiment'] = df['stars'].apply(star_to_label)

# Find mismatches
df['sentiment_mismatch'] = df['vader_sentiment'] != df['rating_sentiment']

# Save to a new CSV
output_path = r"D:\NLP\medical_sentiment_mismatch.csv"
df.to_csv(output_path, index=False)

print(f"Analysis complete. Results saved to: {output_path}")
print("Sample mismatches:")
print(df[df['sentiment_mismatch']].head())
