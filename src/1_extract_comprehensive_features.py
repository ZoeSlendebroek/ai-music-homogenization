"""
Comprehensive Audio Feature Extraction for AI Music Homogenization Analysis
Extracts acoustic, timbral, rhythmic, harmonic, and structural features
"""

import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveFeatureExtractor:
    """Extract multi-dimensional features from audio files"""
    
    def __init__(self, sr=22050, n_mfcc=13, n_chroma=12):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma
    
    def extract_all_features(self, audio_path):
        """Extract comprehensive feature set from audio file"""
        y, sr = librosa.load(audio_path, sr=self.sr)

        # --- NEW: use the middle 30 seconds for comparability with iTunes previews ---
        target_len = int(30 * sr)

        if len(y) > target_len:
            # crop middle 30 seconds
            mid = len(y) // 2
            half = target_len // 2
            start = max(0, mid - half)
            end = start + target_len
            if end > len(y):
                end = len(y)
                start = end - target_len
            y = y[start:end]

        elif len(y) < target_len:
            # pad to exactly 30 seconds
            pad_total = target_len - len(y)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            y = np.pad(y, (pad_left, pad_right), mode="constant")

        # else: exactly 30s → do nothing


        features = {}
        
        # === SPECTRAL FEATURES ===
        features.update(self._extract_spectral_features(y, sr))
        
        # === TIMBRAL FEATURES ===
        features.update(self._extract_timbral_features(y, sr))
        
        # === RHYTHMIC FEATURES ===
        features.update(self._extract_rhythmic_features(y, sr))
        
        # === HARMONIC FEATURES ===
        features.update(self._extract_harmonic_features(y, sr))
        
        # === STRUCTURAL FEATURES ===
        features.update(self._extract_structural_features(y, sr))
        
        # === DYNAMIC FEATURES ===
        features.update(self._extract_dynamic_features(y, sr))
        
        # === MFCC STATISTICS ===
        features.update(self._extract_mfcc_statistics(y, sr))
        
        return features
    
    def _extract_spectral_features(self, y, sr):
        """Extract spectral characteristics"""
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spec_flatness = librosa.feature.spectral_flatness(y=y)[0]
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        return {
            'spectral_centroid_mean': np.mean(spec_cent),
            'spectral_centroid_std': np.std(spec_cent),
            'spectral_centroid_var': np.var(spec_cent),
            'spectral_bandwidth_mean': np.mean(spec_bw),
            'spectral_bandwidth_std': np.std(spec_bw),
            'spectral_rolloff_mean': np.mean(spec_rolloff),
            'spectral_rolloff_std': np.std(spec_rolloff),
            'spectral_flatness_mean': np.mean(spec_flatness),
            'spectral_flatness_std': np.std(spec_flatness),
            'spectral_contrast_mean': np.mean(spec_contrast),
            'spectral_contrast_std': np.std(spec_contrast),
        }
    
    def _extract_timbral_features(self, y, sr):
        """Extract timbral characteristics"""
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # Harmonic-percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Energy ratio
        harmonic_energy = np.sum(y_harmonic**2)
        percussive_energy = np.sum(y_percussive**2)
        total_energy = harmonic_energy + percussive_energy
        
        return {
            'zero_crossing_rate_mean': np.mean(zcr),
            'zero_crossing_rate_std': np.std(zcr),
            'harmonic_ratio': harmonic_energy / total_energy if total_energy > 0 else 0,
            'percussive_ratio': percussive_energy / total_energy if total_energy > 0 else 0,
        }
    
    def _extract_rhythmic_features(self, y, sr):
        """Extract rhythmic characteristics"""
        # Tempo and beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo) # force to float
        
        # Onset detection
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        
        # Inter-onset intervals
        if len(onset_frames) > 1:
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            ioi = np.diff(onset_times)
            onset_density = len(onset_frames) / (len(y) / sr)
        else:
            ioi = np.array([0])
            onset_density = 0
        
        return {
            'tempo': tempo,
            'onset_density': onset_density,
            'onset_strength_mean': np.mean(onset_env),
            'onset_strength_std': np.std(onset_env),
            'ioi_mean': np.mean(ioi) if len(ioi) > 0 else 0,
            'ioi_std': np.std(ioi) if len(ioi) > 0 else 0,
            'ioi_cv': np.std(ioi) / np.mean(ioi) if len(ioi) > 0 and np.mean(ioi) > 0 else 0,
        }
    
    def _extract_harmonic_features(self, y, sr):
        """Extract harmonic/chroma characteristics"""
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        # Tonal centroid features (Harte's 6-dimensional tonal space)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        
        return {
            'chroma_stft_mean': np.mean(chroma),
            'chroma_stft_std': np.std(chroma),
            'chroma_stft_var': np.var(chroma),
            'chroma_cqt_mean': np.mean(chroma_cqt),
            'chroma_cqt_std': np.std(chroma_cqt),
            'tonnetz_mean': np.mean(tonnetz),
            'tonnetz_std': np.std(tonnetz),
        }
    
    def _extract_structural_features(self, y, sr):
        """Extract structural/repetition characteristics"""
        # Self-similarity matrix
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mfcc_sync = librosa.util.sync(mfcc, librosa.beat.beat_track(y=y, sr=sr)[1])
        
        # Compute recurrence matrix
        R = librosa.segment.recurrence_matrix(mfcc_sync, mode='affinity')
        
        # Structural features from recurrence
        repetition_score = np.mean(R) - np.mean(np.diag(R))
        
        return {
            'repetition_score': repetition_score,
            'self_similarity_mean': np.mean(R),
            'self_similarity_std': np.std(R),
        }
    
    def _extract_dynamic_features(self, y, sr):
        """Extract dynamic/loudness characteristics"""
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        
        # Dynamic range
        db = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        
        return {
            'rms_mean': np.mean(rms),
            'rms_std': np.std(rms),
            'rms_var': np.var(rms),
            'dynamic_range_db': np.max(db) - np.min(db),
            'crest_factor': np.max(np.abs(y)) / np.sqrt(np.mean(y**2)) if np.mean(y**2) > 0 else 0,
        }
    
    def _extract_mfcc_statistics(self, y, sr):
        """Extract detailed MFCC statistics"""
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        features = {}
        
        # MFCC statistics
        for i in range(self.n_mfcc):
            features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
            features[f'mfcc_{i}_std'] = np.std(mfcc[i])
        
        # Delta statistics (dynamics)
        features['mfcc_delta_mean'] = np.mean(mfcc_delta)
        features['mfcc_delta_std'] = np.std(mfcc_delta)
        features['mfcc_delta2_mean'] = np.mean(mfcc_delta2)
        features['mfcc_delta2_std'] = np.std(mfcc_delta2)
        
        return features


def process_audio_directory(audio_dir, output_csv, label):
    """Process all audio files in directory and save features"""
    extractor = ComprehensiveFeatureExtractor()
    
    audio_files = list(Path(audio_dir).glob('*.wav')) + list(Path(audio_dir).glob('*.mp3'))
    
    all_features = []
    
    for i, audio_file in enumerate(audio_files):
        print(f"Processing {label} {i+1}/{len(audio_files)}: {audio_file.name}")
        
        try:
            features = extractor.extract_all_features(str(audio_file))
            features['filename'] = audio_file.name
            features['label'] = label
            features['track_index'] = i
            all_features.append(features)
        except Exception as e:
            print(f"  ERROR processing {audio_file.name}: {e}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} tracks to {output_csv}")
    print(f"Extracted {len(df.columns)-3} features per track")
    
    return df


if __name__ == "__main__":
    # Configuration
    project_root = Path(__file__).resolve().parent.parent  # goes from src/ → project root

    AI_AUDIO_DIR = project_root / "data/ai/afrobeat/audio"
    HUMAN_AUDIO_DIR = project_root / "data/human/afrobeat/audio"
    
    # Extract features
    print("="*60)
    print("EXTRACTING AI TRACK FEATURES")
    print("="*60)
    ai_features = process_audio_directory(AI_AUDIO_DIR, 'ai_features_comprehensive.csv', 'AI')
    
    print("\n" + "="*60)
    print("EXTRACTING HUMAN TRACK FEATURES")
    print("="*60)
    human_features = process_audio_directory(HUMAN_AUDIO_DIR, 'human_features_comprehensive.csv', 'Human')
    
    # Combine and save
    combined = pd.concat([ai_features, human_features], ignore_index=True)
    combined.to_csv('all_features_comprehensive.csv', index=False)
    
    print("\n" + "="*60)
    print("FEATURE EXTRACTION COMPLETE")
    print("="*60)
    print(f"Total tracks processed: {len(combined)}")
    print(f"AI tracks: {len(ai_features)}")
    print(f"Human tracks: {len(human_features)}")
    print(f"Total features: {len(combined.columns)-3}")