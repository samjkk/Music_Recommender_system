import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv("dataset.csv")

df = df[
    [
        "track_name",
        "artists",
        "track_genre",
        "popularity",
        "danceability",
        "energy",
        "tempo",
        "valence"
    ]
]

df.columns = [
    "song",
    "artist",
    "genre",
    "popularity",
    "danceability",
    "energy",
    "tempo",
    "valence"
]

df.dropna(inplace=True)
df = df[df["popularity"] >= 50]  
# df = df.sample(3000, random_state=42) 
df.reset_index(drop=True, inplace=True)
df["song_clean"] = df["song"].str.lower().str.strip()

features = df[["danceability", "energy", "tempo", "valence"]]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

def recommend_similar_songs(song_name, n=5):
    song_name = song_name.lower().strip()

    if song_name not in df["song_clean"].values:
        print(f" Song '{song_name}' not found. Try another one.")
        return

    song_index = df[df["song_clean"] == song_name].index[0]
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

    print(f"\n Top {n} Songs similar to '{df.loc[song_index, 'song']}':\n")
    for _, row in recommendations.iterrows():
        print(f" {row['song']} — {row['artist']} (Genre: {row['genre']})")


mood_mapping = {
    "happy": {"energy": 0.8, "valence": 0.9},
    "sad": {"energy": 0.3, "valence": 0.2},
    "energetic": {"energy": 0.9, "valence": 0.7},
    "chill": {"energy": 0.4, "valence": 0.5}
}

def recommend_by_mood(genre, mood, n=5):
    if mood not in mood_mapping:
        print(" Invalid mood.")
        return

    genre_df = df[df["genre"].str.lower() == genre.lower()]

    if genre_df.empty:
        print(" Genre not found.")
        return

    user_vector = [
        0.5,
        mood_mapping[mood]["energy"],
        genre_df["tempo"].mean(),
        mood_mapping[mood]["valence"]
    ]

    user_vector = scaler.transform([user_vector])

    genre_features = genre_df[["danceability", "energy", "tempo", "valence"]]
    genre_scaled = scaler.transform(genre_features)

    similarities = cosine_similarity(user_vector, genre_scaled)[0]
    genre_df["similarity"] = similarities

    recommendations = genre_df.sort_values(
        "similarity", ascending=False
    ).head(n)

    print(f"\n {mood.capitalize()} {genre.capitalize()} Songs:\n")
    for _, row in recommendations.iterrows():
        print(f"{row['song']} — {row['artist']}")

# ===============================
# 7. LIVE SPOTIFY RECOMMENDATION
# ===============================

load_dotenv("file.env")
client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

# Diagnostic check for credentials
def check_spotify_credentials():
    if not client_id or not client_secret:
        return False, "Missing credentials in file.env"
    if len(client_id) != 32 or len(client_secret) != 32:
        return False, f"Invalid credential lengths (Client ID: {len(client_id)}, Secret: {len(client_secret)}). Both should be 32."
    return True, "OK"

def recommend_from_live(n=10):
    is_ok, msg = check_spotify_credentials()
    if not is_ok:
        print(f"\n⚠️  Spotify Error: {msg}")
        print("Please fix 'file.env' to use this feature.")
        return

    try:
        sp = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret
            )
        )
        
        song_query = input("\nEnter a song name to search on Spotify: ")
        result = sp.search(q=song_query, type="track", limit=1)
        
        if not result["tracks"]["items"]:
            print(f"❌ Song '{song_query}' not found on Spotify.")
            return

        track = result["tracks"]["items"][0]
        audio_features = sp.audio_features([track["id"]])[0]
        
        if not audio_features:
            print(f"❌ Could not fetch details for '{track['name']}'.")
            return

        live_vector = [
            audio_features["danceability"],
            audio_features["energy"],
            audio_features["tempo"],
            audio_features["valence"]
        ]

        live_vector_scaled = scaler.transform([live_vector])
        similarities = cosine_similarity(live_vector_scaled, scaled_features)[0]

        temp_df = df.copy()
        temp_df["similarity"] = similarities
        recommendations = temp_df.sort_values(by="similarity", ascending=False).head(n)

        print(f"\n✨ Found on Spotify: '{track['name']}' by {track['artists'][0]['name']}")
        print(f"--- Top {n} Recommendations from local dataset ---\n")
        print(recommendations[["song", "artist", "genre", "popularity"]])

    except Exception as e:
        print(f"❌ Spotify API Error: {e}")

# ===============================
# 8. EDA – FEATURE VISUALIZATION
# ===============================

def run_eda():
    plt.figure(figsize=(6,4))
    sns.histplot(df["energy"], kde=True)
    plt.title("Energy Distribution")
    plt.show()

    plt.figure(figsize=(6,4))
    corr = df[["danceability", "energy", "tempo", "valence"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Audio Feature Correlation")
    plt.show()

    plt.figure(figsize=(6,4))
    sns.scatterplot(
        x=df["energy"],
        y=df["valence"],
        hue=df["genre"],
        legend=False
    )
    plt.title("Energy vs Valence")
    plt.show()


# ===============================
# 9. MENU SYSTEM
# ===============================

while True:
    print("\n MUSIC RECOMMENDATION SYSTEM ")
    print("1. Song-to-Song Recommendation (Local)")
    print("2. Mood-Based Recommendation (Local)")
    print("3. Live Spotify Search Recommendation")
    print("4. View Feature Analysis (EDA)")
    print("5. Exit")

    choice = input("Enter your choice (1-5): ")

    if choice == "1":
        print("\nSome songs you can try:")
        print(df["song"].sample(10).values)
        song = input("\nEnter song name: ")
        recommend_similar_songs(song)

    elif choice == "2":
        genre = input("Enter genre: ")
        mood = input("Enter mood (happy/sad/energetic/chill): ").lower()
        recommend_by_mood(genre, mood)

    elif choice == "3":
        recommend_from_live()

    elif choice == "4":
        run_eda()

    elif choice == "5":
        print(" Goodbye!")
        break

    else:
        print(" Invalid choice.")
