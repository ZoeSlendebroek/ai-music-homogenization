#!/usr/bin/env python3
"""
Collect a large pool of human Afrobeats candidates from Spotify playlists.

- Filters by release year and presence of preview_url
- Fetches Spotify audio features (danceability, energy, valence, tempo, etc.)
- Writes one CSV with one row per track (no clustering, no downloading yet)

Usage:
  python collect_human_afrobeat_candidates.py \
    --playlists 1dUGRxuSyKCHsI3dnXYcYc,0SCVMfqFZrLuAYKXJVYwUF,1mHunEEIPUZwV6EfX1so5e \
    --out_meta ../data/human/afrobeat/candidates_spotify.csv \
    --year_min 2015 --year_max 2025
"""

import os, time, argparse
import numpy as np
import pandas as pd
from pathlib import Path
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
import urllib.parse

def itunes_preview_url(artist, title):
    """
    Search iTunes for a matching track and return the previewUrl if available.
    """
    query = f"{artist} {title}"
    url = "https://itunes.apple.com/search?term=" + urllib.parse.quote(query) + "&entity=song&limit=5"

    try:
        resp = requests.get(url, timeout=10).json()
        results = resp.get("results", [])
        if not results:
            return None

        # Find best match: prioritize artist match first
        for r in results:
            if "previewUrl" in r:
                return r["previewUrl"]

        return None
    except Exception:
        return None

# Argument configuration
ap = argparse.ArgumentParser()
ap.add_argument("--playlists", required=True, help="comma-separated playlist IDs")
ap.add_argument("--out_meta", default="data/human/afrobeat/candidates_spotify.csv")
ap.add_argument("--year_min", type=int, default=2015) # 2015 - 2025
ap.add_argument("--year_max", type=int, default=2025)
ap.add_argument("--max_per_playlist", type=int, default=500)
args = ap.parse_args()

# Spotify authentication
# Store your own credentials in secrets!
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.environ["SPOTIFY_CLIENT_ID"],
    client_secret=os.environ["SPOTIFY_CLIENT_SECRET"]
))

def year_ok(d):
    y = (d or "")[:4]
    return y.isdigit() and args.year_min <= int(y) <= args.year_max

# Candidate Collection
playlist_ids = [s.strip() for s in args.playlists.split(",")]
cands = []
seen = set()

for pid in playlist_ids:
    print(f"Scanning playlist {pid}...")
    results = sp.playlist_items(pid, additional_types=["track"], limit=100)
    grabbed = 0
    while results and grabbed < args.max_per_playlist:
        for it in results["items"]:
            t = it.get("track") or {}
            tid = t.get("id")
            if not tid or tid in seen:
                continue

            rdate = (t.get("album") or {}).get("release_date", "")
            if not year_ok(rdate):
                continue

            purl = itunes_preview_url(
                artist=", ".join(a["name"] for a in t.get("artists", [])),
                title=t.get("name", "")
            )
            if not purl:
                continue

            seen.add(tid)
            grabbed += 1
            cands.append({
                "track_id": tid,
                "artist": ", ".join(a["name"] for a in t.get("artists", [])),
                "title": t.get("name", ""),
                "release_date": rdate,
                "preview_url": purl
            })
        results = sp.next(results) if results and results.get("next") else None
        time.sleep(0.05)

C = pd.DataFrame(cands)
print(f"Raw candidates with previews: {len(C)}")

if C.empty:
    raise SystemExit("No candidates with previews found. Check playlist IDs or year range.")

# Save big pool of potential candidates
out_path = Path(args.out_meta)
out_path.parent.mkdir(parents=True, exist_ok=True)

# Just use the metadata + iTunes preview URLs
C.to_csv(out_path, index=False)
print(f"Saved human candidate pool (no Spotify audio_features) to: {out_path}")

