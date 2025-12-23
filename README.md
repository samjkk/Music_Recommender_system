# 🎵 Music Recommendation System

A Python-based music recommendation system that suggests songs based on similarity, mood, and user preferences using Spotify's audio features.

## ✨ Features

- **Song-to-Song Recommendations**: Find similar songs based on audio features (danceability, energy, tempo, valence)
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

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/samjkk/Music-Recommendation-System.git
   cd Music-Recommendation-System
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Spotify API credentials**
   - Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
   - Create an app and get your Client ID and Secret
   - Copy `.env.example` to `.env`
   - Add your credentials to `.env`:
     ```
     SPOTIFY_CLIENT_ID=your_client_id_here
     SPOTIFY_CLIENT_SECRET=your_client_secret_here
     ```

## 📖 Usage

### Run the Main Application
```bash
python full.py
```

You'll see a menu with options:
```
1. Song-to-Song Recommendation (Local)
2. Mood-Based Recommendation (Local)
3. Live Spotify Search Recommendation
4. View Feature Analysis (EDA)
5. Exit
```

### Example Usage

**Song-to-Song:**
```
Enter song name: bohemian rhapsody
↓
Recommends 5 similar songs from the dataset
```

**Mood-Based:**
```
Enter genre: pop
Enter mood: happy
↓
Returns energetic, positive pop songs
```

**Spotify Search:**
```
Enter a song name to search on Spotify: blinding lights
↓
Finds the song on Spotify, then recommends similar songs from local dataset
```

## 🔧 Configuration

### Audio Feature Thresholds

Modify mood mappings in `full.py`:
```python
mood_mapping = {
    "happy": {"energy": 0.8, "valence": 0.9},
    "sad": {"energy": 0.3, "valence": 0.2},
    "energetic": {"energy": 0.9, "valence": 0.7},
    "chill": {"energy": 0.4, "valence": 0.5}
}
```

### Dataset Filtering

Change the popularity threshold in `full.py`:
```python
df = df[df["popularity"] >= 50]  # Songs with popularity >= 50
```

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

See `requirements.txt` for full dependency list:
```
pandas
numpy
scikit-learn
spotipy
python-dotenv
matplotlib
seaborn
```

Install with:
```bash
pip install -r requirements.txt
```

## 🚢 Deployment

### Option 1: Streamlit (Recommended for Quick Deployment)
```bash
pip install streamlit
streamlit run app.py
```
Deploy on: [Streamlit Cloud](https://streamlit.io/cloud) (free)

### Option 2: Flask API
Build a REST API and deploy on Heroku, AWS, or DigitalOcean

### Option 3: Desktop App
Use PyInstaller to create an `.exe` for Windows

## 📈 Future Improvements

- [ ] Add collaborative filtering recommendations
- [ ] Implement user ratings and feedback system
- [ ] Create web interface with Streamlit
- [ ] Add more audio features (acousticness, instrumentalness, etc.)
- [ ] Implement user preference learning
- [ ] Add database instead of CSV
- [ ] Create mobile app

## 🤝 Contributing

Feel free to fork, modify, and submit pull requests!

## 📝 License

This project is open source and available under the MIT License.

## 📧 Contact

For questions or suggestions, feel free to reach out!

---

**Built with ❤️ by samjkk**
