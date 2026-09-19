"""
genre_pair_analysis.py

Section 2 of the FMA analysis: fit one global PCA on standardized audio
features, bucket tracks into 5-year release periods, and track how far apart
(or close together) each pair of top genres sits in PCA-2D over time.

Usage (as called from report.ipynb):

    pair_result = GenrePairAnalyzer(tracks, features, min_samples=10).analyze()
    pairs = interesting_pairs(pair_result, threshold=0.3)
"""

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from feature_engineering import get_cross_features

logger = logging.getLogger("genre_pair_analysis")

# 5-year period start years covered by the FMA catalog's release-date range.
PERIOD_STARTS = [1980, 1985, 1990, 1995, 2000, 2005, 2010]
PERIOD_WIDTH = 5


@dataclass
class GenrePairResult:
    """Everything the Section 2 plots (2.1-2.4) need."""

    X_pca: pd.DataFrame                      # index=track_id, cols=['PC1','PC2','period','genre']
    genres: List[str]
    periods: List[int]
    pair_distances: pd.DataFrame             # index=period, columns='GenreA|GenreB'
    explained_variance: Tuple[float, float]  # (PC1 %, PC2 %)
    centroids: Dict[Tuple[int, str], np.ndarray] = field(default_factory=dict)


class GenrePairAnalyzer:
    """
    Fits one global PCA on standardized FMA audio features (+ cross-feature
    derivations from feature_engineering.get_cross_features), buckets tracks
    into 5y release periods, and computes the pairwise Euclidean distance
    between top-genre centroids in PCA-2D for each period where a cell has
    at least `min_samples` tracks.
    """

    def __init__(self, tracks: pd.DataFrame, features: pd.DataFrame, min_samples: int = 10):
        self.tracks = tracks
        self.features = features
        self.min_samples = min_samples

    # ------------------------------------------------------------------
    def _prepare_tracks(self) -> pd.DataFrame:
        """Pull genre_top + release year; drop tracks missing either; bucket into 5y periods."""
        df = pd.DataFrame({
            "genre": self.tracks[("track", "genre_top")],
            "date_released": self.tracks[("album", "date_released")],
        })
        df = df.dropna(subset=["genre", "date_released"])

        df["year"] = pd.to_datetime(df["date_released"], errors="coerce").dt.year
        df = df.dropna(subset=["year"])
        df["year"] = df["year"].astype(int)

        df["period"] = (df["year"] // PERIOD_WIDTH) * PERIOD_WIDTH
        df = df[df["period"].isin(PERIOD_STARTS)]

        return df

    # ------------------------------------------------------------------
    def _prepare_features(self, track_ids: pd.Index) -> pd.DataFrame:
        """Flatten + standardize raw and cross-derived audio features for the given tracks."""
        feat = self.features.loc[self.features.index.intersection(track_ids)].copy()

        # flatten the 3-level MultiIndex columns (statistic, feature, coefficient) -> single strings
        feat.columns = ["_".join(map(str, c)) for c in feat.columns]

        cross = get_cross_features(feat)
        feat_full = pd.concat([feat, cross], axis=1).replace([np.inf, -np.inf], np.nan).dropna()

        return feat_full

    # ------------------------------------------------------------------
    def analyze(self) -> GenrePairResult:
        meta = self._prepare_tracks()
        feat_full = self._prepare_features(meta.index)

        common_idx = meta.index.intersection(feat_full.index)
        meta = meta.loc[common_idx]
        feat_full = feat_full.loc[common_idx]

        logger.info(f"Fitting global PCA on {len(feat_full)} tracks (after period+feature filters)")

        X_std = StandardScaler().fit_transform(feat_full.values)

        pca = PCA(n_components=2, random_state=0)
        X_2d = pca.fit_transform(X_std)
        explained = tuple(round(v * 100, 1) for v in pca.explained_variance_ratio_)

        X_pca = pd.DataFrame(X_2d, index=feat_full.index, columns=["PC1", "PC2"])
        X_pca["period"] = meta["period"]
        X_pca["genre"] = meta["genre"]

        genres = sorted(X_pca["genre"].unique())
        periods = sorted(X_pca["period"].unique())

        logger.info(
            f"Computing centroids over {len(periods)} periods x {len(genres)} genres "
            f"(min_samples={self.min_samples} per cell)"
        )

        centroids: Dict[Tuple[int, str], np.ndarray] = {}
        for period in periods:
            for genre in genres:
                cell = X_pca[(X_pca["period"] == period) & (X_pca["genre"] == genre)]
                if len(cell) >= self.min_samples:
                    centroids[(period, genre)] = cell[["PC1", "PC2"]].mean().to_numpy()

        pair_cols = [f"{a}|{b}" for a, b in combinations(genres, 2)]
        pair_distances = pd.DataFrame(index=periods, columns=pair_cols, dtype=float)

        for period in periods:
            for a, b in combinations(genres, 2):
                ca, cb = centroids.get((period, a)), centroids.get((period, b))
                if ca is not None and cb is not None:
                    pair_distances.loc[period, f"{a}|{b}"] = float(np.linalg.norm(ca - cb))

        return GenrePairResult(
            X_pca=X_pca,
            genres=genres,
            periods=periods,
            pair_distances=pair_distances,
            explained_variance=explained,
            centroids=centroids,
        )


def interesting_pairs(pair_result: GenrePairResult, threshold: float = 0.3) -> List[Tuple[str, str]]:
    """
    Return genre pairs, as (genre_a, genre_b) tuples, whose centroid distance
    changed by at least `threshold` (relative, e.g. 0.3 = 30%) between the
    first and last period that has data for that pair. Pairs with fewer than
    two observed periods, or a zero-valued first observation, are skipped.
    """
    out: List[Tuple[str, str]] = []
    for col in pair_result.pair_distances.columns:
        vals = pair_result.pair_distances[col].dropna()
        if len(vals) < 2 or vals.iloc[0] == 0:
            continue
        rel_change = abs(vals.iloc[-1] - vals.iloc[0]) / vals.iloc[0]
        if rel_change >= threshold:
            genre_a, genre_b = col.split("|")
            out.append((genre_a, genre_b))
    return out
