import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.cluster import KMeans
import numpy as np

# Style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Load the processed file
df = pd.read_csv("final_adjusted_ratings.csv")

# ===============================
# 📊 1. Boxplot: Original vs Adjusted Ratings
# ===============================
plt.figure()
sns.boxplot(data=df[['stars', 'adjusted_rating']])
plt.title("📊 Distribution: Original vs Adjusted Ratings")
plt.ylabel("Rating")
plt.xticks([0, 1], ['Original', 'Adjusted'])
plt.tight_layout()
plt.savefig("1_boxplot_original_vs_adjusted.png")
plt.show()

# ===============================
# 📊 2. Bar Chart: Top 10 Emotions in Mismatches
# ===============================
mismatch_df = df[df['sentiment_mismatch']]
emotion_series = mismatch_df['top_emotions'].dropna().str.split(', ')
flat_emotions = [e.strip().lower() for sublist in emotion_series for e in sublist]
emotion_counts = Counter(flat_emotions)
top_emotions_df = pd.DataFrame(emotion_counts.items(), columns=['emotion', 'count']).sort_values(by='count', ascending=False)

plt.figure()
sns.barplot(x='count', y='emotion', data=top_emotions_df.head(10), palette="viridis")
plt.title("💥 Top 10 Emotions in Sentiment Mismatches")
plt.xlabel("Frequency")
plt.ylabel("Emotion")
plt.tight_layout()
plt.savefig("2_top_emotions_bar.png")
plt.show()

# ===============================
# 🌡 3. Heatmap: VADER Sentiment vs Emotion (Mismatched)
# ===============================
exploded = mismatch_df.copy()
exploded['emotion'] = exploded['top_emotions'].str.split(', ')
exploded = exploded.explode('emotion')
exploded['emotion'] = exploded['emotion'].str.lower().str.strip()

heatmap_data = pd.crosstab(exploded['vader_sentiment'], exploded['emotion'])

plt.figure(figsize=(14, 6))
sns.heatmap(heatmap_data, cmap="YlGnBu", linewidths=0.5)
plt.title("🌡 Heatmap: Emotion vs Sentiment (Mismatched Only)")
plt.xlabel("Emotion")
plt.ylabel("VADER Sentiment")
plt.tight_layout()
plt.savefig("3_heatmap_emotion_sentiment.png")
plt.show()

# ===============================
# 🕸️ 4. Radar Chart: Rating Change Distribution (Mismatched Only)
# ===============================
from math import pi

# Compute rating change columns (if not already present)
if 'rating_change_status' not in df.columns:
    df['rating_change'] = df['adjusted_rating'] - df['stars']
    df['rating_change_status'] = df['rating_change'].apply(
        lambda x: 'Increased' if x > 0 else 'Decreased' if x < 0 else 'Same'
    )

# Prepare data
mismatch_subset = df[df['sentiment_mismatch']]
radar_data = mismatch_subset['rating_change_status'].value_counts().reset_index()
radar_data.columns = ['Change Type', 'Count']

# Ensure consistent order (so the radar always looks the same)
categories = ['Increased', 'Decreased', 'Same']
radar_data = radar_data.set_index('Change Type').reindex(categories).fillna(0).reset_index()

# Radar setup
N = len(categories)
values = radar_data['Count'].values.tolist()
values += values[:1]  # repeat first value to close circle
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Plot radar
plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)
plt.xticks(angles[:-1], categories, color='black', size=12)
ax.plot(angles, values, linewidth=2, linestyle='solid', color='teal')
ax.fill(angles, values, color='teal', alpha=0.25)
plt.title("🕸️ Rating Change Distribution (Mismatched Only)", size=15, y=1.08)
plt.tight_layout()
plt.savefig("4_rating_change_radar.png", bbox_inches='tight')
plt.show()


# ===============================
# ⚖️ 5. Scatter Plot: Original vs Adjusted Ratings
# ===============================
plt.figure()
sns.scatterplot(x='stars', y='adjusted_rating', data=df[df['sentiment_mismatch']], alpha=0.6, color="purple")
plt.plot([1, 5], [1, 5], 'r--', label='No Change Line')
plt.title("⚖️ Original vs Adjusted Ratings (Sentiment Mismatches)")
plt.xlabel("Original Rating")
plt.ylabel("Adjusted Rating")
plt.legend()
plt.tight_layout()
plt.savefig("5_scatter_original_vs_adjusted.png")
plt.show()

# ===============================
# ✅ 6A. Clustering Based on Sentiment–Rating Gap
# ===============================
df['normalized_stars'] = (df['stars'] - 1) / 4
sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
df['sentiment_score'] = df['vader_sentiment'].map(sentiment_map)
df['intensity_gap'] = df['sentiment_score'] - df['normalized_stars']

X = df[['sentiment_score', 'normalized_stars', 'intensity_gap']].fillna(0)
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
df['cluster'] = kmeans.labels_

# ===============================
# ✅ 6B. Scatter Plot: Cluster View
# ===============================
plt.figure()
sns.scatterplot(data=df, x='normalized_stars', y='sentiment_score', hue='cluster', palette="Set1")
plt.title("🔍 Star Rating vs Sentiment Score (KMeans Clusters)")
plt.xlabel("Normalized Star Rating")
plt.ylabel("Sentiment Score")
plt.tight_layout()
plt.savefig("6a_cluster_scatter.png")
plt.show()

# ===============================
# ✅ 6C. % of High Star Reviews with Strong Negative Emotions
# ===============================
high_star = df[df['stars'] >= 4].copy()
high_star['emotion_list'] = high_star['top_emotions'].str.lower().str.split(', ')
high_star['has_neg_emotion'] = high_star['emotion_list'].apply(
    lambda x: any(em.strip() in ['anger', 'sadness', 'fear', 'disgust'] for em in x)
)
neg_percent = (high_star['has_neg_emotion'].sum() / len(high_star)) * 100

plt.figure()
sns.barplot(x=['4–5 Star Reviews'], y=[neg_percent], palette="Reds")
plt.ylabel('% with Anger/Sadness/Fear')
plt.title("😡 % of 4–5 Star Reviews with Strong Negative Emotions")
plt.tight_layout()
plt.savefig("6b_high_star_neg_emotion_percent.png")
plt.show()

# ===============================
# ✅ 7. Insight Reporting
# ===============================
five_star = df[df['stars'] == 5]
five_star['has_negative'] = five_star['top_emotions'].str.lower().str.contains('anger|sadness|fear|disgust')
percent_5star_neg = five_star['has_negative'].mean() * 100
print(f"📌 {percent_5star_neg:.2f}% of 5-star reviews contain strong negative emotions.")

# If 'service_type' exists in the dataset
if 'service_type' in df.columns:
    exploded['service_type'] = df.set_index('text').loc[exploded['text'], 'service_type'].values
    service_emotion = exploded.groupby('service_type')['emotion'].apply(lambda x: Counter(x).most_common(1)[0][0])
    print("\n📊 Dominant emotion per service:")
    print(service_emotion)

top_mismatch_emotions = exploded[exploded['sentiment_mismatch']]['emotion'].value_counts().head(5)
print("\n🚨 Top 5 emotions in mismatched reviews:")
print(top_mismatch_emotions)
