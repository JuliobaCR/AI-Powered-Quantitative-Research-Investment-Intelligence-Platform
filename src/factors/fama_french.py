"""
Fama-French 5-Factor Model — factor data ingestion + OLS regression.

Factor data source: Kenneth French Data Library (Dartmouth Tuck), daily
5-factor dataset (Mkt-RF, SMB, HML, RMW, CMA, RF). Network/parsing failures
degrade gracefully to an empty DataFrame / None, matching the rest of the
platform's "best-effort, never crash the dashboard" data-fetching pattern
(see src.market_data.fetcher.fetch_ohlcv, src.fundamentals.analyzer).
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

FF5_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)

FACTOR_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


@lru_cache(maxsize=1)
def fetch_ff5_factors() -> pd.DataFrame:
    """
    Download and parse the Fama-French 5-factor daily dataset.

    Returns a DataFrame indexed by date (DatetimeIndex) with columns
    Mkt-RF, SMB, HML, RMW, CMA, RF as decimal daily returns (0.01 == 1%).
    Returns an empty DataFrame on any network or parsing failure.
    """
    try:
        resp = requests.get(FF5_DAILY_URL, timeout=20)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            with zf.open(csv_name) as f:
                lines = f.read().decode("utf-8", errors="ignore").splitlines()

        header_idx = next(i for i, line in enumerate(lines) if "Mkt-RF" in line)
        raw = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    except Exception:
        return pd.DataFrame()

    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.rename(columns={raw.columns[0]: "Date"})

    raw["Date"] = pd.to_numeric(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"])
    raw = raw[raw["Date"] > 19000000]  # drop annual-block rows (4-digit years) & footer

    raw["Date"] = pd.to_datetime(raw["Date"].astype(int).astype(str), format="%Y%m%d")
    raw = raw.set_index("Date")

    for col in FACTOR_COLUMNS + ["RF"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce") / 100.0

    return raw[FACTOR_COLUMNS + ["RF"]].dropna()


@dataclass
class FactorExposure:
    ticker: str
    alpha_annual_pct: float
    betas: dict
    r_squared: float
    n_obs: int


def _ols_betas(excess: np.ndarray, factor_matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit excess ~ 1 + factors via OLS. Returns (coeffs, r_squared)."""
    X_design = np.column_stack([np.ones(len(factor_matrix)), factor_matrix])
    coeffs, *_ = np.linalg.lstsq(X_design, excess, rcond=None)

    fitted = X_design @ coeffs
    residuals = excess - fitted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((excess - excess.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return coeffs, r_squared


def run_factor_regression(
    stock_returns: pd.Series,
    ticker: str = "",
    factors: pd.DataFrame | None = None,
) -> FactorExposure | None:
    """
    Full-sample Fama-French 5-factor regression for a single asset.

    Returns None if factor data is unavailable or there is insufficient
    overlapping history (<60 observations).
    """
    if factors is None:
        factors = fetch_ff5_factors()
    if factors.empty:
        return None

    df = pd.DataFrame({"asset": stock_returns}).join(factors, how="inner").dropna()
    if len(df) < 60:
        return None

    excess = (df["asset"] - df["RF"]).to_numpy()
    factor_matrix = df[FACTOR_COLUMNS].to_numpy()
    coeffs, r_squared = _ols_betas(excess, factor_matrix)

    alpha_daily, betas = coeffs[0], coeffs[1:]

    return FactorExposure(
        ticker=ticker,
        alpha_annual_pct=round(alpha_daily * 252 * 100, 3),
        betas={f: round(float(b), 3) for f, b in zip(FACTOR_COLUMNS, betas)},
        r_squared=round(float(r_squared), 4),
        n_obs=len(df),
    )


def rolling_factor_exposures(
    stock_returns: pd.Series,
    factors: pd.DataFrame | None = None,
    window: int = 126,
    step: int = 5,
) -> pd.DataFrame:
    """
    Rolling-window Fama-French 5-factor betas, for a 3D exposure surface
    (time x factor x beta).

    Returns a DataFrame indexed by date with columns Mkt-RF, SMB, HML, RMW, CMA.
    Returns an empty DataFrame if factor data is unavailable or there is
    insufficient history for at least one window.
    """
    if factors is None:
        factors = fetch_ff5_factors()
    if factors.empty:
        return pd.DataFrame()

    df = pd.DataFrame({"asset": stock_returns}).join(factors, how="inner").dropna()
    if len(df) < window + step:
        return pd.DataFrame()

    rows = []
    for end in range(window, len(df) + 1, step):
        win = df.iloc[end - window:end]
        excess = (win["asset"] - win["RF"]).to_numpy()
        factor_matrix = win[FACTOR_COLUMNS].to_numpy()
        coeffs, _ = _ols_betas(excess, factor_matrix)

        row = dict(zip(FACTOR_COLUMNS, coeffs[1:]))
        row["Date"] = win.index[-1]
        rows.append(row)

    return pd.DataFrame(rows).set_index("Date")
