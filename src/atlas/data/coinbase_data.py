from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd
from atlas.coinbase.client import CoinbaseClient
from atlas.config import CoinbaseSettings, get_coinbase_settings
from atlas.logging_utils import get_logger
from atlas.market import safe_filename_symbol

logger = get_logger(__name__)


def _cache_path(symbol: str, start: datetime, end: datetime, granularity: str) -> Path:
    cache_dir = Path("outputs") / "cache" / "coinbase"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a unique hash for the request key
    key = f"{symbol}_{start.isoformat()}_{end.isoformat()}_{granularity}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    
    safe_symbol = safe_filename_symbol(symbol)
    return cache_dir / f"{safe_symbol}_{h}.csv.gz"


def load_coinbase_bars_cached(
    *,
    symbol: str,
    start: datetime,
    end: datetime,
    granularity: str = "ONE_MINUTE",
    settings: CoinbaseSettings = None,
) -> pd.DataFrame:
    """
    Fetch Coinbase bars with local caching.
    """
    path = _cache_path(symbol, start, end, granularity)
    
    if path.exists():
        logger.info(f"Loading cached Coinbase bars for {symbol} from {path}")
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=[0])
            if df.index.tz is None:
                df.index = pd.to_datetime(df.index, utc=True)
            return df
        except Exception as e:
            logger.warning(f"Failed to read cache {path}: {e}, re-fetching")

    if settings is None:
        settings = get_coinbase_settings()
        
    client = CoinbaseClient(settings)
    
    # Resolve product ID if needed?
    # For now assume symbol IS the product_id (e.g. BTC-PERP)
    # The market module handles normalization to generic names, but 
    # Coinbase requires specific IDs. 
    # If symbol comes in as BTC-PERP, that matches Coinbase Advanced Trade.
    
    logger.info(f"Fetching Coinbase bars for {symbol} ({start} -> {end})")
    df = client.get_product_candles(
        product_id=symbol,
        start=start,
        end=end,
        granularity=granularity
    )
    
    if not df.empty:
        # Cache results
        try:
            df.to_csv(path, compression="gzip")
        except Exception as e:
            logger.warning(f"Failed to write cache {path}: {e}")
            
    return df
