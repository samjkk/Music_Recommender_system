import pandas as pd
df = pd.read_csv("dataset.csv")
df.head()
df.info()
df.columns

df = df[[
    "track_name",
    "artists",
    "track_genre",
    "danceability",
    "energy",
    "tempo",
    "valence"
]]
df.columns = ["song","artist","genre","danceability","energy","tempo","mood"]
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
user_genre = input("Enter preferred genre: ")
genre_df = df[df["genre"].str.lower() == user_genre.lower()]
def recommend_songs(data, n=10):
    return data.sort_values(
        by=["energy", "danceability", "tempo", "mood"],
        ascending=False
    ).head(n)
recommendations = recommend_songs(genre_df)

print("\n Recommended Songs:")
for _, row in recommendations.iterrows():
    print(f"{row['song']} - {row['artist']}")
