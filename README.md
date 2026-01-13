# AI Music Homogenization
**Computational audit of generative music systems**

This project examines whether AI-generated music exhibits *algorithmic homogenization*—reduced within-genre diversity and diminished cross-genre distinction—compared to human music.  
We compare tracks from Suno (AI) and Spotify (human) across genres such as Afrobeats and House.

## 🔍 Research Questions
- Do AI systems generate music that is less diverse within a genre?
- Do genres collapse into a narrower sonic space under AI generation?

## ⚙️ Methods
- Feature extraction with **OpenL3** embeddings (512D, mean-pooled)
- Metrics: within-genre dispersion, cross-genre centroid distance, classifier confusability
- Analysis implemented in **Python 3.10** (`librosa`, `openl3`, `pandas`, `sklearn`)

## 📂 Structure
- `data/` – local raw audio (not versioned)
- `interim/` – feature files (`.parquet`)
- `src/` – scripts for data collection and analysis
- `reports/` – plots and summary tables


