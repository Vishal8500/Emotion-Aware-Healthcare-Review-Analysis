import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import joblib
# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
INPUT_ASPECT_FILE = r"D:\NLP\final_adjusted_ratings.csv"
INPUT_MISMATCH_FILE = "medical_sentiment_mismatch.csv"
OUTPUT_FILE = "final_absa_adjusted_ratings.csv"

# Same emotions used in training
EMOTION_LIST = [
    'anger', 'disgust', 'fear', 'sadness', 'disappointment',
    'annoyance', 'remorse', 'grief', 'embarrassment', 'nervousness',
    'joy', 'gratitude', 'love', 'admiration', 'approval', 'caring',
    'excitement', 'optimism', 'relief', 'pride'
]

# ------------------------------------------------------------
# NEURAL NETWORK MODEL
# ------------------------------------------------------------
class EmotionWeightModel(nn.Module):
    def __init__(self, input_size=len(EMOTION_LIST), hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Predict a single adjustment score
        )

    def forward(self, x):
        return self.net(x)

# Load trained model (or create dummy model if unavailable)
MODEL_PATH = r"D:\NLP\emotion_weight_model.pth"

try:
    model = EmotionWeightModel()
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"✅ Loaded trained model from {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Could not load model — using untrained random weights.\n{e}")
    model = EmotionWeightModel()

model.eval()


scaler = joblib.load("emotion_scaler.pkl")

# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------
print(f"Loading aspect data from {INPUT_ASPECT_FILE}...")
aspect_df = pd.read_csv(INPUT_ASPECT_FILE)

print(f"Loading original mismatch data from {INPUT_MISMATCH_FILE}...")
df = pd.read_csv(INPUT_MISMATCH_FILE)

# ------------------------------------------------------------
# 1. Compute model-based aspect emotion adjustments
# ------------------------------------------------------------
review_adjustments = {}
print("Calculating model-based emotional adjustments...")

for _, row in tqdm(aspect_df.iterrows(), total=aspect_df.shape[0]):
    review_id = row['review_id']
    emotions = str(row['sentence_top_emotions']).split(', ')

    # Convert to one-hot input vector for the model
    input_vector = torch.zeros(len(EMOTION_LIST))
    for em in emotions:
        em = em.strip().lower()
        if em in EMOTION_LIST:
            input_vector[EMOTION_LIST.index(em)] = 1.0

    # Predict adjustment using the model
    with torch.no_grad():
        adjustment_score = model(input_vector.unsqueeze(0)).item()

    # Accumulate adjustment per review
    review_adjustments[review_id] = review_adjustments.get(review_id, 0.0) + adjustment_score

df['aspect_emotion_adjustment'] = df.index.map(review_adjustments).fillna(0.0)

# ------------------------------------------------------------
# 2. Calculate ABSA-anchored adjusted rating
# ------------------------------------------------------------
def compute_anchored_ear(row):
    if row['sentiment_mismatch'] and row['aspect_emotion_adjustment'] != 0:
        original = row['stars']
        adjustment = row['aspect_emotion_adjustment']
        return round(min(5, max(1, original + adjustment)), 1)
    else:
        return row['stars']

df['absa_adjusted_rating'] = df.apply(compute_anchored_ear, axis=1)

# ------------------------------------------------------------
# 3. Compute Emotion-Only Rating
# ------------------------------------------------------------
def compute_emotion_only_rating(adjustment_score):
    if adjustment_score == 0.0:
        return None
    base_rating = 3.0
    new_rating = base_rating + adjustment_score
    return round(min(5, max(1, new_rating)), 1)

df['emotion_only_rating'] = df['aspect_emotion_adjustment'].apply(compute_emotion_only_rating)
df['emotion_only_rating'] = df['emotion_only_rating'].fillna(df['stars'])

# ------------------------------------------------------------
# 4. Save and Preview
# ------------------------------------------------------------
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Success! File saved to: {OUTPUT_FILE}")
print("\nPreview:")
print(df[['text', 'stars', 'aspect_emotion_adjustment', 'emotion_only_rating']].head())
