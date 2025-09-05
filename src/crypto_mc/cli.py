
import argparse, os, json, time
import yaml
import numpy as np
import pandas as pd
from typing import List
from .utils.io import find_data_files, load_coin_csv
from .simulation import simulate_single, simulate_multi
from .risk import value_at_risk, conditional_var, max_drawdown, ruin_probability

def load_config(cfg_path: str) -> dict:
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_prices(data_path: str, symbol: str, date_col: str="Date", price_col: str="Close") -> pd.Series:
    files = find_data_files(data_path)
    for f in files:
        df = load_coin_csv(f, symbol)
        if df is not None:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
            return df.set_index(date_col)[price_col].astype(float)
    raise FileNotFoundError(f"Symbol {symbol} not found in {data_path}")

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def cmd_simulate(args):
    config = load_config(args.config)
    outdir = config.get('output_path', './crypto_mc/outputs')
    ensure_outdir(outdir)

    if args.symbols and len(args.symbols)>0:
        prices_map = {}
        for s in args.symbols:
            prices_map[s] = get_prices(args.data_path, s, config.get('date_column','Date'), config.get('price_column','Close'))
        paths, port = simulate_multi(prices_map, args.model, args.horizon, args.n_paths, dt=1.0, seed=args.seed, config=config, weights=args.weights)
        if port is not None:
            pnl = np.log(port[-1,:]/port[0, :])
        else:
            port = paths.mean(axis=2)
            pnl = np.log(port[-1,:]/port[0, :])
        m = summarize_metrics(pnl, port, args)
    else:
        prices = get_prices(args.data_path, args.symbol, config.get('date_column','Date'), config.get('price_column','Close'))
        paths = simulate_single(prices, args.model, args.horizon, args.n_paths, dt=1.0, config=config, seed=args.seed)
        pnl = np.log(paths[-1,:]/paths[0,:])
        m = summarize_metrics(pnl, paths, args)

    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(outdir, f"metrics_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    sample = min(200, args.n_paths)
    if args.save_paths:
        if isinstance(args.symbols, list) and len(args.symbols)>0:
            pd.DataFrame(port[:, :sample]).to_csv(os.path.join(outdir, f"paths_sample_{ts}.csv"), index=False)
        else:
            pd.DataFrame(paths[:, :sample]).to_csv(os.path.join(outdir, f"paths_sample_{ts}.csv"), index=False)
    print(json.dumps(m, indent=2))

def summarize_metrics(pnl: np.ndarray, paths: np.ndarray, args) -> dict:
    levels = [0.95, 0.99]
    res = {
        "horizon_days": int(args.horizon),
        "n_paths": int(args.n_paths),
        "model": args.model,
        "VaR": {str(l): float(value_at_risk(pnl, l)) for l in levels},
        "CVaR": {str(l): float(conditional_var(pnl, l)) for l in levels},
        "expected_log_return": float(pnl.mean()),
        "prob_loss": float((pnl<0).mean()),
    }
    if isinstance(paths, np.ndarray):
        res["avg_max_drawdown"] = float(max_drawdown(paths))
        res["ruin_prob_50pct"] = float(ruin_probability(paths, threshold=paths[0,0]*0.5))
    return res

def main():
    parser = argparse.ArgumentParser(description="Crypto Monte Carlo CLI")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("simulate", help="Run simulations and output risk stats")
    p.add_argument("--symbol", type=str, help="Single symbol like BTC")
    p.add_argument("--symbols", nargs="*", help="Multiple symbols (e.g., BTC ETH)")
    p.add_argument("--weights", nargs="*", type=float, default=None, help="Weights for multi-asset portfolio")
    p.add_argument("--model", type=str, choices=["gbm","garch","jump","bootstrap"], default="gbm")
    p.add_argument("--horizon", type=int, default=365)
    p.add_argument("--n-paths", type=int, default=20000)
    p.add_argument("--data-path", type=str, default="./crypto_mc/data/raw")
    p.add_argument("--config", type=str, default="./crypto_mc/config.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-paths", action="store_true", help="Save sample of simulated paths")
    p.set_defaults(func=cmd_simulate)

    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
    else:
        args.func(args)

if __name__ == "__main__":
    main()
