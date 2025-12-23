import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv("dataset.csv")

df = df[[
    "track_name",
    "artists",
    "track_genre",
    "danceability",
    "energy",
    "tempo",
    "valence"
]]

df.columns = ["song", "artist", "genre", "danceability", "energy", "tempo", "valence"]

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

mood_mapping = {
    "happy": {"energy": 0.8, "valence": 0.9},
    "sad": {"energy": 0.3, "valence": 0.2},
    "energetic": {"energy": 0.9, "valence": 0.7},
    "chill": {"energy": 0.4, "valence": 0.5}
}   
user_genre = input("Enter genre: ")
user_mood = input("Enter mood (happy/sad/energetic/chill): ").lower()
genre_df = df[df["genre"].str.lower() == user_genre.lower()]

features = genre_df[["danceability", "energy", "tempo", "valence"]]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

user_vector = [
    0.5,
    mood_mapping[user_mood]["energy"],
    genre_df["tempo"].mean(),
    mood_mapping[user_mood]["valence"]
]
user_vector = scaler.transform([user_vector])
similarity_scores = cosine_similarity(user_vector, scaled_features)
genre_df["similarity"] = similarity_scores[0]

recommendations = genre_df.sort_values(
    by="similarity", ascending=False
).head(10)
print("\n Recommended Songs For You:\n")

for _, row in recommendations.iterrows():
    print(f"{row['song']} - {row['artist']}")

