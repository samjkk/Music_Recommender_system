# 🎵 Music Recommendation System

A Python-based music recommendation system that suggests songs based on similarity, mood, and user preferences using Spotify's audio features.

**🚀 [Try the Live App Here](https://musicrecommendersystem-adavrjma9jprqnsudffpt6.streamlit.app/)**

## ✨ Features

- **Song-to-Song Recommendations**: Find similar songs using cosine similarity on audio features
- **Mood-Based Recommendations**: Get song suggestions by genre and mood (happy/sad/energetic/chill)
- **Live Spotify Search**: Search for songs on Spotify and find similar tracks in the local dataset
- **User-Based Recommendations**: Recommendations based on genre preferences
- **Feature Analysis (EDA)**: Visualize audio feature distributions and correlations

## 🎯 How It Works

The system uses **cosine similarity** to compare songs based on their audio features:

- **Danceability** (0-1): How suitable a song is for dancing
- **Energy** (0-1): Intensity and activity level
- **Tempo**: Speed of the song (BPM)
- **Valence** (0-1): Musical positivity (happiness)

All features are normalized using StandardScaler before comparison.

## 📁 Project Structure

```
Music Recommendation System/
├── dataset.csv              # Song database (~3,700 songs)
├── file.env                 # Spotify API credentials (DO NOT commit)
├── .env.example             # Template for environment variables
├── full.py                  # Main application with menu system
├── new.py                   # Spotify API integration
├── song_based.py            # Song-to-song recommendation engine
├── mood_based.py            # Mood-based recommendation engine
├── User_based.py            # User-based recommendation engine
└── README.md                # This file
```

## 🚀 Installation

1. **Clone & Install**
   ```bash
   git clone https://github.com/samjkk/Music_Recommender_system.git
   pip install -r requirements.txt
   ```

2. **Add Spotify Credentials** (optional)
   - Copy `.env.example` to `.env`
   - Add your Spotify API credentials

3. **Run**
   ```bash
   python full.py        # Local version
   streamlit run app.py  # Web version (live)
   ```

## 📖 Usage

**Song-to-Song**: Find similar songs  
**Mood-Based**: Get recommendations by mood + genre  
**Spotify Search**: Search Spotify and get local matches  
**Feature Analysis**: Visualize audio features  

## 📊 Dataset Information

**File**: `dataset.csv`

**Columns**:
- `track_name`: Song title
- `artists`: Artist(s) name
- `track_genre`: Music genre
- `popularity`: Spotify popularity score (0-100)
- `danceability`: Audio feature (0-1)
- `energy`: Audio feature (0-1)
- `tempo`: Tempo in BPM
- `valence`: Audio feature (0-1)

**Size**: ~3,700 songs across multiple genres

## 🔐 Security

⚠️ **IMPORTANT**: Never commit `file.env` or credentials to GitHub!

- Add `file.env` to `.gitignore`
- Always use `.env.example` as a template
- Environment variables should be set on deployment platform (Heroku, AWS, etc.)

## 🛠️ Technologies Used

- **pandas**: Data manipulation
- **scikit-learn**: Machine learning (cosine similarity, StandardScaler)
- **spotipy**: Spotify API wrapper
- **matplotlib & seaborn**: Data visualization
- **numpy**: Numerical operations
- **python-dotenv**: Environment variable management

## 📋 Requirements

See `requirements.txt`

## 📝 License

MIT License

---

**Built by samjkk**
