import logging
import sys
from datetime import datetime, timedelta
from atlas.config import get_coinbase_settings
from atlas.coinbase.client import CoinbaseClient

def main():
    # Setup basic logging to stdout
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Checking Coinbase settings...")
    try:
        settings = get_coinbase_settings()
        if not settings.api_key:
            logger.error("COINBASE_API_KEY is missing or empty.")
            return
        logger.info(f"API Key found: {settings.api_key[:4]}...")
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        return

    client = CoinbaseClient(settings)

    logger.info("Listing 'PERP' products...")
    try:
        products = client.list_products()
        perp_products = [p for p in products if "PERP" in p.get("product_id", "")]
        
        if not perp_products:
            logger.warning("No products with 'PERP' in ID found. Listing first 5 products:")
            for p in products[:5]:
                logger.info(f" - {p.get('product_id')}")
        else:
            logger.info(f"Found {len(perp_products)} PERP products.")
            for p in perp_products[:5]:
                logger.info(f" - {p.get('product_id')} ({p.get('status', 'unknown')})")
    except Exception as e:
        logger.error(f"Failed to list products: {e}")
        # Continue to try fetching candle anyway?
    
    logger.info("Attempting to fetch 1 candle for BTC-PERP...")
    try:
        end = datetime.now()
        start = end - timedelta(minutes=100)
        candles = client.get_product_candles("BTC-PERP", start, end, "ONE_MINUTE")
        if candles.empty:
            logger.warning("Candles dataframe is empty!")
        else:
            logger.info(f"Success! Fetched {len(candles)} bars.")
            print(candles.head())
    except Exception as e:
        logger.error(f"Failed to fetch candles: {e}")

if __name__ == "__main__":
    main()
