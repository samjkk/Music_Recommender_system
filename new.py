import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Encoding Fix (Windows)
# ----------------------------
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ----------------------------
# LOAD DATASET
# ----------------------------
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
df.reset_index(drop=True, inplace=True)

# Clean text for matching
df["song_clean"] = df["song"].str.lower().str.strip()
df["artist_clean"] = df["artist"].str.lower().str.strip()

# ----------------------------
# FEATURE SCALING
# ----------------------------
features = df[["danceability", "energy", "tempo", "valence"]]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# ----------------------------
# COSINE SIMILARITY FUNCTION
# ----------------------------
def recommend_similar_songs(song_index, n=5):
    similarity_scores = cosine_similarity(
        scaled_features[song_index].reshape(1, -1),
        scaled_features
    )[0]

    temp_df = df.copy()
    temp_df["similarity"] = similarity_scores

    recommendations = (
        temp_df
        .sort_values("similarity", ascending=False)
        .iloc[1:n+1]
    )

    return recommendations[["song", "artist", "genre", "popularity"]]

# ----------------------------
# SPOTIFY API (SEARCH ONLY)
# ----------------------------
sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id="4c21298798eb4dd7b5378659442bb5d9",
        client_secret="e104e28009ca40d39f746471c81ac292"
    )
)

print("Connected to Spotify API")

# ----------------------------
# MAIN LOOP
# ----------------------------
while True:
    print("\n--- Spotify → Dataset Song Recommendation ---")
    user_input = input("Enter song name (or 'exit'): ")

    if user_input.lower() == "exit":
        print("Goodbye 👋")
        break

    # 🔍 Search song on Spotify
    results = sp.search(q=user_input, type="track", limit=1)["tracks"]["items"]

    if not results:
        print("❌ Song not found on Spotify.")
        continue

    track = results[0]
    spotify_song = track["name"].lower().strip()
    spotify_artist = track["artists"][0]["name"].lower().strip()

    print(f"\n🎧 Found on Spotify: {track['name']} – {track['artists'][0]['name']}")

    # 🔗 Match with dataset
    matches = df[
        (df["song_clean"] == spotify_song) &
        (df["artist_clean"].str.contains(spotify_artist))
    ]

    if matches.empty:
        print("❌ Song not found in dataset for feature comparison.")
        continue

    song_index = matches.index[0]

    # 🎶 Recommend
    recommendations = recommend_similar_songs(song_index, 5)

    print("\n🎶 Recommended Songs:")
    print(recommendations)
