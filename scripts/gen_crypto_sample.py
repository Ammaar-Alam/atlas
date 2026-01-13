import pandas as pd
from pathlib import Path

def create_crypto_samples():
    data_dir = Path("data/sample")
    spy_path = data_dir / "SPY_1min_sample.csv"
    
    if not spy_path.exists():
        print("SPY sample not found!")
        return

    df = pd.read_csv(spy_path)
    
    # Create BTC-PERP (Scale SPY x 200 approx, Vol x 0.1)
    btc = df.copy()
    btc[["open", "high", "low", "close"]] = btc[["open", "high", "low", "close"]] * 200
    btc["volume"] = (btc["volume"] * 0.1).astype(int)
    btc.to_csv(data_dir / "BTC-PERP_1min_sample.csv", index=False)
    print(f"Created BTC-PERP sample at {data_dir / 'BTC-PERP_1min_sample.csv'}")

    # Create ETH-PERP (Scale SPY x 10 approx)
    eth = df.copy()
    eth[["open", "high", "low", "close"]] = eth[["open", "high", "low", "close"]] * 10
    eth["volume"] = (eth["volume"] * 0.5).astype(int)
    eth.to_csv(data_dir / "ETH-PERP_1min_sample.csv", index=False)
    print(f"Created ETH-PERP sample at {data_dir / 'ETH-PERP_1min_sample.csv'}")

if __name__ == "__main__":
    create_crypto_samples()
