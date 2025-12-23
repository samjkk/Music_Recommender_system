import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="🎵 Music Recommendation System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1DB954;
        font-size: 3em;
        margin-bottom: 10px;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.2em;
    }
    </style>
    """, unsafe_allow_html=True)

# Load environment variables
client_id = None
client_secret = None

# Try Streamlit secrets first
try:
    client_id = st.secrets.get("SPOTIFY_CLIENT_ID")
    client_secret = st.secrets.get("SPOTIFY_CLIENT_SECRET")
except:
    pass

# Fall back to .env file
if not client_id or not client_secret:
    load_dotenv("file.env")
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

# Load dataset
@st.cache_data
def load_data():
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
    df.columns = ["song", "artist", "genre", "popularity", "danceability", "energy", "tempo", "valence"]
    df.dropna(inplace=True)
    df = df[df["popularity"] >= 50]
    df.reset_index(drop=True, inplace=True)
    df["song_clean"] = df["song"].str.lower().str.strip()
    return df

@st.cache_resource
def get_scaler_and_features(df):
    features = df[["danceability", "energy", "tempo", "valence"]]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaler, scaled_features

# Load data
df = load_data()
scaler, scaled_features = get_scaler_and_features(df)

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-header">🎵 Music Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discover songs you\'ll love based on audio features</p>', unsafe_allow_html=True)

st.divider()

# Sidebar
with st.sidebar:
    st.markdown("## 🎛️ Navigation")
    page = st.radio("Select a feature:", [
        "🔎 Song-to-Song",
        "😊 Mood-Based",
        "🎧 Spotify Search",
        "📊 Feature Analysis",
        "ℹ️ About"
    ])

# ====== PAGE 1: SONG-TO-SONG RECOMMENDATION ======
if page == "🔎 Song-to-Song":
    st.header("🎵 Find Similar Songs")
    st.markdown("Enter a song name to discover similar tracks based on audio features")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        song_input = st.selectbox(
            "Select a song or type to search:",
            df["song"].unique(),
            key="song_search"
        )
    with col2:
        n_recommendations = st.number_input("Number of recommendations", min_value=1, max_value=20, value=5)
    
    if st.button("🔍 Find Similar Songs", key="btn_similar"):
        song_name = song_input.lower().strip()
        
        if song_name not in df["song_clean"].values:
            st.error(f"❌ Song '{song_input}' not found in dataset")
        else:
            song_index = df[df["song_clean"] == song_name].index[0]
            similarity_scores = cosine_similarity(
                scaled_features[song_index].reshape(1, -1),
                scaled_features
            )[0]
            
            temp_df = df.copy()
            temp_df["similarity"] = similarity_scores
            
            recommendations = temp_df.sort_values("similarity", ascending=False).iloc[1:n_recommendations+1]
            
            # Display original song
            orig_song = df.loc[song_index]
            st.info(f"🎵 **Found:** *{orig_song['song']}* by **{orig_song['artist']}** ({orig_song['genre']})")
            
            # Display recommendations
            st.subheader(f"✨ Top {n_recommendations} Similar Songs:")
            for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
                similarity = row['similarity'] * 100
                col1, col2, col3 = st.columns([0.5, 3, 1])
                with col1:
                    st.metric("", f"#{idx}")
                with col2:
                    st.write(f"**{row['song']}** • {row['artist']}")
                    st.caption(f"📂 {row['genre']} • ⭐ {row['popularity']}/100")
                with col3:
                    st.metric("Similarity", f"{similarity:.1f}%")

# ====== PAGE 2: MOOD-BASED RECOMMENDATION ======
elif page == "😊 Mood-Based":
    st.header("😊 Mood-Based Recommendations")
    st.markdown("Get song suggestions based on your mood and preferred genre")
    
    mood_mapping = {
        "happy": {"energy": 0.8, "valence": 0.9},
        "sad": {"energy": 0.3, "valence": 0.2},
        "energetic": {"energy": 0.9, "valence": 0.7},
        "chill": {"energy": 0.4, "valence": 0.5}
    }
    
    col1, col2, col3 = st.columns(3)
    with col1:
        genre = st.selectbox("Select a genre:", df["genre"].unique())
    with col2:
        mood = st.selectbox("Select a mood:", list(mood_mapping.keys()))
    with col3:
        n_recommendations = st.number_input("Number of recommendations", min_value=1, max_value=20, value=5, key="mood_n")
    
    if st.button("🎵 Get Recommendations", key="btn_mood"):
        genre_df = df[df["genre"].str.lower() == genre.lower()]
        
        if genre_df.empty:
            st.error(f"❌ Genre '{genre}' not found")
        else:
            user_vector = [
                0.5,
                mood_mapping[mood]["energy"],
                genre_df["tempo"].mean(),
                mood_mapping[mood]["valence"]
            ]
            
            user_vector_scaled = scaler.transform([user_vector])
            genre_features = genre_df[["danceability", "energy", "tempo", "valence"]]
            genre_scaled = scaler.transform(genre_features)
            
            similarities = cosine_similarity(user_vector_scaled, genre_scaled)[0]
            genre_df_copy = genre_df.copy()
            genre_df_copy["similarity"] = similarities
            
            recommendations = genre_df_copy.sort_values("similarity", ascending=False).head(n_recommendations)
            
            st.success(f"✨ {mood.capitalize()} {genre.capitalize()} Songs:")
            for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
                col1, col2 = st.columns([0.3, 3])
                with col1:
                    st.metric("", f"#{idx}")
                with col2:
                    st.write(f"**{row['song']}** • {row['artist']}")
                    st.caption(f"⭐ {row['popularity']}/100 • 🎵 {row['genre']}")

# ====== PAGE 3: SPOTIFY SEARCH ======
elif page == "🎧 Spotify Search":
    st.header("🎧 Spotify Search & Match")
    st.markdown("Search for a song on Spotify and find similar songs in our dataset")
    
    if not client_id or not client_secret:
        st.error("❌ Spotify credentials not configured")
        st.info("📝 Instructions: Add your Spotify credentials to the app settings")
    else:
        song_query = st.text_input("Enter a song name to search on Spotify:")
        n_recommendations = st.number_input("Number of recommendations", min_value=1, max_value=20, value=5, key="spotify_n")
        
        if st.button("🔍 Search on Spotify", key="btn_spotify"):
            if not song_query:
                st.warning("⚠️ Please enter a song name")
            else:
                try:
                    with st.spinner("Connecting to Spotify API..."):
                        sp = spotipy.Spotify(
                            auth_manager=SpotifyClientCredentials(
                                client_id=client_id,
                                client_secret=client_secret
                            )
                        )
                    
                    with st.spinner("Searching for your song..."):
                        result = sp.search(q=song_query, type="track", limit=1)
                    
                    if not result["tracks"]["items"]:
                        st.error(f"❌ Song '{song_query}' not found on Spotify")
                    else:
                        track = result["tracks"]["items"][0]
                        
                        with st.spinner("Fetching audio features..."):
                            audio_features = sp.audio_features([track["id"]])[0]
                        
                        if not audio_features:
                            st.error(f"❌ Could not fetch audio features for '{track['name']}'")
                        else:
                            live_vector = [
                                audio_features.get("danceability", 0.5),
                                audio_features.get("energy", 0.5),
                                audio_features.get("tempo", 120),
                                audio_features.get("valence", 0.5)
                            ]
                            
                            live_vector_scaled = scaler.transform([live_vector])
                            similarities = cosine_similarity(live_vector_scaled, scaled_features)[0]
                            
                            temp_df = df.copy()
                            temp_df["similarity"] = similarities
                            recommendations = temp_df.sort_values(by="similarity", ascending=False).head(n_recommendations)
                            
                            # Display found song
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                if track["album"]["images"]:
                                    st.image(track["album"]["images"][0]["url"], width=200)
                            with col2:
                                st.success(f"✨ Found on Spotify: **{track['name']}**")
                                st.write(f"Artist: {track['artists'][0]['name']}")
                                st.write(f"Album: {track['album']['name']}")
                                st.caption(f"🔗 [Open on Spotify](https://open.spotify.com/track/{track['id']})")
                            
                            st.divider()
                            st.subheader(f"🎵 Top {n_recommendations} Matches in Our Dataset:")
                            for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
                                col1, col2 = st.columns([0.3, 3])
                                with col1:
                                    st.metric("", f"#{idx}")
                                with col2:
                                    st.write(f"**{row['song']}** • {row['artist']}")
                                    st.caption(f"📂 {row['genre']} • ⭐ {row['popularity']}/100")
                        
                except spotipy.exceptions.SpotifyException as e:
                    st.error(f"❌ Spotify API Error: {str(e)}")
                    st.info("💡 Your credentials might be invalid. Check app settings → Secrets")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Please try again or refresh the page")

# ====== PAGE 4: FEATURE ANALYSIS ======
elif page == "📊 Feature Analysis":
    st.header("📊 Audio Feature Analysis")
    st.markdown("Explore the distribution and relationships of audio features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎵 Energy Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df["energy"], kde=True, ax=ax, color="#1DB954")
        ax.set_xlabel("Energy")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    
    with col2:
        st.subheader("😊 Valence Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df["valence"], kde=True, ax=ax, color="#1ED760")
        ax.set_xlabel("Valence (Happiness)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    
    st.subheader("🔗 Feature Correlation Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[["danceability", "energy", "tempo", "valence"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, cbar_kws={'label': 'Correlation'})
    st.pyplot(fig)
    
    st.subheader("⚡ Energy vs Valence by Genre")
    fig, ax = plt.subplots(figsize=(12, 6))
    genres = df["genre"].value_counts().head(10).index
    sns.scatterplot(data=df[df["genre"].isin(genres)], x="energy", y="valence", 
                    hue="genre", ax=ax, s=50, alpha=0.6)
    ax.set_xlabel("Energy")
    ax.set_ylabel("Valence (Happiness)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    st.pyplot(fig)
    
    # Statistics
    st.subheader("📈 Dataset Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Songs", len(df))
    with col2:
        st.metric("Genres", df["genre"].nunique())
    with col3:
        st.metric("Avg Popularity", f"{df['popularity'].mean():.1f}")
    with col4:
        st.metric("Avg Tempo (BPM)", f"{df['tempo'].mean():.0f}")

# ====== PAGE 5: ABOUT ======
elif page == "ℹ️ About":
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ### 🎵 Music Recommendation System
    
    A Python-based recommendation engine that suggests songs based on audio features and user preferences using Spotify's audio features.
    
    #### 🎯 Features
    - **Song-to-Song**: Find similar songs using cosine similarity
    - **Mood-Based**: Get recommendations based on mood and genre
    - **Spotify Integration**: Search Spotify and match with local dataset
    - **Feature Analysis**: Visualize audio feature distributions
    
    #### 🔧 How It Works
    The system uses **cosine similarity** to compare songs based on:
    - **Danceability** (0-1): How suitable for dancing
    - **Energy** (0-1): Intensity and activity
    - **Tempo**: Speed in beats per minute
    - **Valence** (0-1): Musical positivity
    
    #### 📊 Dataset
    - **Size**: ~3,700 songs
    - **Coverage**: Multiple genres
    - **Source**: Spotify audio features
    
    #### 🚀 Technologies
    - Streamlit (Web interface)
    - Scikit-learn (Machine learning)
    - Spotipy (Spotify API)
    - Pandas (Data processing)
    
    #### 📝 License
    Open source - MIT License
    
    #### 🔗 Links
    - [GitHub Repository](https://github.com/samjkk/Music_Recommender_system)
    - [Spotify Developer](https://developer.spotify.com)
    
    ---
    **Built with ❤️ by samjkk**
    """)

st.divider()
st.markdown("<p style='text-align: center; color: #888;'>Made with Streamlit • 🎵 Enjoy discovering music!</p>", unsafe_allow_html=True)
