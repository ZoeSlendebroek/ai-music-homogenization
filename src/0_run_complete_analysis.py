"""
MASTER PIPELINE: Computational Audit of AI Music Homogenization
Runs complete analysis from feature extraction to visualization
"""

import sys
import subprocess
from pathlib import Path

print("""
================================================================================
COMPUTATIONAL AUDIT: AI MUSIC HOMOGENIZATION
================================================================================
This pipeline performs a comprehensive computational analysis of whether
AI-generated Afrobeats tracks exhibit reduced diversity compared to human-
produced tracks.

Analysis Pipeline:
1. Feature extraction (using pre-extracted matched_pairs_comprehensive.csv)
2. Test homogenization hypotheses (variance, clustering, coverage, entropy)
3. Validate matched AI-human pairs
4. Train classifiers and extract feature importance
5. Generate publication-quality visualizations

Prerequisites:
- Feature data in data/afrobeat/matched_pairs_comprehensive.csv
- Python packages: librosa, scikit-learn, scipy, pandas, matplotlib, seaborn
================================================================================
""")

def check_data_file():
    """Check if the matched pairs CSV exists"""
    print("\nChecking data file...")
    
    project_root = Path(__file__).resolve().parent.parent
    data_file = project_root / "data/afrobeat/matched_pairs_comprehensive.csv"
    
    if not data_file.exists():
        print(f" ERROR: Data file not found: {data_file}")
        print("   Please run extract_matched_comprehensive_features.py first")
        return False
    
    print(f"✓ Found data file: {data_file}")
    
    # Check if it has data
    import pandas as pd
    try:
        df = pd.read_csv(data_file)
        print(f"✓ Loaded {len(df)} rows")
        print(f"  AI tracks: {len(df[df['side']=='AI'])}")
        print(f"  Human tracks: {len(df[df['side']=='Human'])}")
        return True
    except Exception as e:
        print(f" ERROR loading data file: {e}")
        return False

def check_dependencies():
    """Check if required Python packages are installed"""
    print("\nChecking dependencies...")
    
    required = [
        'librosa', 'numpy', 'pandas', 'scipy', 
        'sklearn', 'matplotlib', 'seaborn'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"   {package}")
    
    if missing:
        print(f"\n Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    return True

def run_analysis_step(script_name, description):
    """Run a single analysis script"""
    print("\n" + "="*80)
    print(f"STEP: {description}")
    print("="*80)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                               capture_output=False, text=True)
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            return True
        else:
            print(f"{description} failed")
            return False
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        return False

def main():
    """Run complete analysis pipeline"""
    
    # Pre-flight checks
    if not check_data_file():
        print("\nData file check failed. Please generate the matched pairs CSV first.")
        return
    
    if not check_dependencies():
        print("\nDependency check failed. Please install missing packages.")
        return
    
    print("\n" + "="*80)
    print("STARTING ANALYSIS PIPELINE")
    print("="*80)
    
    # Define analysis steps (skip feature extraction)
    steps = [
        ("2_homogenization_analysis.py", 
         "Homogenization Analysis"),
        ("3_validate_matched_pairs.py", 
         "Pair Validation"),
        ("4_classification_and_feature_importance.py", 
         "Classification Analysis"),
        ("5_visualization_suite.py", 
         "Visualization Generation")
    ]
    
    # Run each step
    results = []
    for script, description in steps:
        script_path = Path(script)
        if script_path.exists():
            success = run_analysis_step(script, description)
            results.append((description, success))
        else:
            print(f"Warning: {script} not found, skipping...")
            results.append((description, False))
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS PIPELINE COMPLETE")
    print("="*80)
    
    print("\nStep Results:")
    for step, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {status}: {step}")
    
    successful = sum(1 for _, s in results if s)
    total = len(results)
    
    print(f"\nOverall: {successful}/{total} steps completed successfully")
    
    if successful == total:
        print("\n🎉 All analyses completed! Check output files and figures.")
        print("\nGenerated files:")
        print("  - homogenization_*.csv (variance, distances, coverage, entropy)")
        print("  - pair_validation_results.csv")
        print("  - pair_statistics.csv")
        print("  - classification_*.csv (comparison, importance, tracks, pairs)")
        print("  - fig_*.png (all visualizations)")
    else:
        print("\n⚠️  Some steps failed. Check error messages above.")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()