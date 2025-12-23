import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Configure output to handle Unicode characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv("dataset.csv")

df = df[[
    "track_name",
    "artists",
    "track_genre",
    "popularity",
    "danceability",
    "energy",
    "tempo",
    "valence"
]]

df.columns = [
    "song", "artist", "genre", "popularity",
    "danceability", "energy", "tempo", "valence"
]

df.dropna(inplace=True)
# df = df[df["popularity"] >= 50]  # Removed to allow searching whole dataset
df.reset_index(drop=True, inplace=True)
df["song_clean"] = df["song"].str.lower().str.strip()

features = df[["danceability", "energy", "tempo", "valence"]]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
mood_mapping = {
    "happy": {"energy": 0.8, "valence": 0.9},
    "sad": {"energy": 0.3, "valence": 0.2},
    "energetic": {"energy": 0.9, "valence": 0.7},
    "chill": {"energy": 0.4, "valence": 0.5}
}


def recommend_similar_songs(song_name, n=5):
    song_name = song_name.lower().strip()

    # Check existence safely
    if song_name not in df["song_clean"].values:
        print(" Song not found. Try exact song name from dataset.")
        return

    # Get first matching index safely
    song_index = df[df["song_clean"] == song_name].index[0]

    # Compute cosine similarity correctly
    similarity_scores = cosine_similarity(
        scaled_features[song_index].reshape(1, -1),
        scaled_features
    )[0]

    # Use temp dataframe (DO NOT MODIFY df)
    temp_df = df.copy()
    temp_df["similarity"] = similarity_scores

    recommendations = (
        temp_df
        .sort_values("similarity", ascending=False)
        .iloc[1:n+1]
    )

    return recommendations[["song", "artist", "genre", "popularity"]]

while True:
    print("\n--- Song-to-Song Recommendations ---")
    print("Some songs from the dataset to try:")
    print(df["song"].sample(10).values)

    song_input = input("\nEnter a song name (or type 'exit' to quit): ")
    if song_input.lower() == 'exit':
        print("Goodbye!")
        break

    recs = recommend_similar_songs(song_input, 5)
    if recs is not None:
        print(f"\n Recommendations for '{song_input}':")
        print(recs)
    



'''
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df["energy"], kde=True)
plt.title("Energy Distribution")
plt.show()
corr = df[["danceability", "energy", "tempo", "valence"]].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Audio Feature Correlation")
plt.show()
sns.scatterplot(
    x=df["energy"],
    y=df["valence"],
    hue=df["genre"],
    legend=False
)
plt.title("Energy vs Valence")
plt.show()
'''