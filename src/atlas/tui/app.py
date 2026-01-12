from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event, Thread
from typing import Optional

import pandas as pd
from rich.table import Table
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.suggester import SuggestFromList
from textual.widgets import Header, Input, Log, Static

from atlas.backtest.engine import BacktestConfig, run_backtest
from atlas.config import get_alpaca_settings, get_default_max_position_notional_usd
from atlas.data.alpaca_data import load_stock_bars_cached
from atlas.data.csv_loader import load_bars_csv
from atlas.paper.runner import PaperConfig, run_paper_loop
from atlas.strategies.registry import build_strategy
from atlas.utils.time import now_ny, parse_iso_datetime


@dataclass
class TuiState:
    symbol: str = "SPY"
    data_source: str = "sample"
    csv_path: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    timeframe: Optional[str] = None
    strategy: str = "ma_crossover"
    fast_window: int = 10
    slow_window: int = 30
    initial_cash: float = 100_000.0
    max_position_notional_usd: float = 10_000.0
    slippage_bps: float = 0.0
    allow_short: bool = False
    paper_lookback_bars: int = 200
    paper_poll_seconds: int = 60
    paper_max_position_notional_usd: float = 1_000.0
    paper_allow_trading_when_closed: bool = False
    paper_dry_run: bool = False

    @classmethod
    def from_dict(cls, raw: dict) -> "TuiState":
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        data = {k: v for k, v in raw.items() if k in fields}
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_relative_timeframe(spec: str) -> timedelta:
    spec = spec.strip().lower()
    if spec.endswith("min"):
        return timedelta(minutes=int(spec[:-3]))
    if spec.endswith("h"):
        return timedelta(hours=int(spec[:-1]))
    if spec.endswith("d"):
        return timedelta(days=int(spec[:-1]))
    if spec.endswith("w"):
        return timedelta(weeks=int(spec[:-1]))
    if spec.endswith("m"):
        return timedelta(days=30 * int(spec[:-1]))
    if spec.endswith("y"):
        return timedelta(days=365 * int(spec[:-1]))
    raise ValueError("unsupported timeframe spec")


def _infer_bar_minutes(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 0.0
    diffs = index.to_series().diff().dropna().dt.total_seconds() / 60.0
    median = float(diffs.median())
    return median if median > 0 else 0.0


def _resolve_time_window(
    *,
    bars: pd.DataFrame,
    timeframe: Optional[str],
    start: Optional[str],
    end: Optional[str],
) -> tuple[pd.DataFrame, Optional[str], Optional[str]]:
    start_dt = parse_iso_datetime(start) if start else None
    end_dt = parse_iso_datetime(end) if end else None

    if timeframe:
        delta = _parse_relative_timeframe(timeframe)
        end_dt = pd.Timestamp(bars.index[-1]).to_pydatetime()
        start_dt = end_dt - delta

    if start_dt is not None:
        bars = bars[bars.index >= start_dt]
    if end_dt is not None:
        bars = bars[bars.index <= end_dt]

    return (
        bars,
        start_dt.isoformat() if start_dt else None,
        end_dt.isoformat() if end_dt else None,
    )


class AtlasTui(App):
    COMMANDS = [
        "/help",
        "/backtest",
        "/paper start",
        "/paper stop",
        "/timeframe 7d",
        "/timeframe 6h",
        "/timeframe 1m",
        "/timeframe 1y",
        "/timeframe clear",
        "/algorithm ma_crossover",
        "/data sample",
        "/data csv",
        "/data alpaca",
        "/csv /path/to/bars.csv",
        "/symbol SPY",
        "/start 2024-01-02T09:30:00-05:00",
        "/end 2024-01-02T16:00:00-05:00",
        "/save",
        "/load",
    ]
    CSS = """
    Screen { layout: vertical; background: $surface; }
    #body { height: 1fr; }
    #settings { width: 38%; border: solid $accent; padding: 1; }
    #results { width: 62%; border: solid $accent; padding: 1; }
    #lower { height: 11; border: solid $accent; padding: 1; }
    #log { height: 1fr; background: $surface; }
    #suggestions {
        height: auto;
        max-height: 4;
        border: none;
        padding: 0 1;
        color: $text-muted;
        text-style: dim;
        background: $surface;
        overflow-y: auto;
    }
    #input { height: 3; border: solid $accent; padding: 0 1; }
    Input > .input--suggestion { color: $text-muted; text-style: dim; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.state = TuiState(
            max_position_notional_usd=get_default_max_position_notional_usd(
                mode="backtest"
            ),
            paper_max_position_notional_usd=get_default_max_position_notional_usd(
                mode="paper"
            ),
        )
        self._ctrl_c_armed_at: Optional[float] = None
        self._paper_thread: Optional[Thread] = None
        self._paper_stop: Optional[Event] = None
        self._last_run_dir: Optional[Path] = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            settings = Static(id="settings")
            settings.border_title = "Settings"
            yield settings
            results = Static(id="results")
            results.border_title = "Results"
            yield results
        lower = Vertical(id="lower")
        lower.border_title = "Console"
        with lower:
            yield Log(id="log")
            yield Static(id="suggestions")
            cmd = Input(
                id="input",
                placeholder="/help",
                suggester=SuggestFromList(self.COMMANDS, case_sensitive=False),
            )
            cmd.border_title = "Command"
            yield cmd

    def on_mount(self) -> None:
        log_widget = self.query_one("#log", Log)
        log_widget.auto_scroll = True
        log_widget.display = False
        suggestions = self.query_one("#suggestions", Static)
        suggestions.display = False
        self.query_one("#input", Input).focus()
        self._render_settings()
        self._render_results(None)

    def on_key(self, event) -> None:
        if event.key == "ctrl+c":
            event.stop()
            input_box = self.query_one("#input", Input)
            if input_box.value:
                input_box.value = ""
                return
            now = time.monotonic()
            if self._ctrl_c_armed_at and now - self._ctrl_c_armed_at < 2.0:
                self.exit()
                return
            self._ctrl_c_armed_at = now

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        self._update_suggestions("")
        if not text:
            return
        if not text.startswith("/"):
            self._write_log("commands must start with /")
            return
        self._handle_command(text)

    def on_input_changed(self, event: Input.Changed) -> None:
        self._update_suggestions(event.value)

    def _update_suggestions(self, value: str) -> None:
        suggestions = self.query_one("#suggestions", Static)
        value = value.strip()
        if not value.startswith("/"):
            suggestions.display = False
            return
        if value == "/":
            matches = list(self.COMMANDS)
        else:
            matches = [cmd for cmd in self.COMMANDS if cmd.startswith(value)]
        if not matches:
            suggestions.display = False
            return
        suggestions.display = True
        suggestions.update("\n".join(matches[:6]))

    def _handle_command(self, text: str) -> None:
        parts = text.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in {"/help", "/?"}:
            self._write_log(
                "commands: /backtest, /paper start|stop, /timeframe <7d|6h|1m|1y|clear>, "
                "/algorithm <name>, /data <sample|csv|alpaca>, /csv <path>, "
                "/symbol <SPY>, /start <iso>, /end <iso>, /save [path], /load [path]"
            )
            return

        if cmd == "/symbol" and args:
            self.state.symbol = args[0].upper()
            self._render_settings()
            return

        if cmd == "/data" and args:
            value = args[0].lower()
            if value not in {"sample", "csv", "alpaca"}:
                self._write_log("data source must be sample|csv|alpaca")
                return
            self.state.data_source = value
            self._render_settings()
            return

        if cmd == "/csv" and args:
            self.state.csv_path = " ".join(args)
            self._render_settings()
            return

        if cmd == "/timeframe" and args:
            if args[0].lower() == "clear":
                self.state.timeframe = None
            else:
                self.state.timeframe = args[0]
            self._render_settings()
            return

        if cmd == "/start" and args:
            self.state.start = " ".join(args)
            self._render_settings()
            return

        if cmd == "/end" and args:
            self.state.end = " ".join(args)
            self._render_settings()
            return

        if cmd in {"/algorithm", "/strategy", "/aglorithm"} and args:
            self.state.strategy = args[0]
            self._render_settings()
            return

        if cmd == "/backtest":
            self._run_backtest()
            return

        if cmd == "/paper":
            action = args[0].lower() if args else "start"
            if action == "start":
                self._start_paper()
            elif action == "stop":
                self._stop_paper()
            else:
                self._write_log("paper command must be /paper start|stop")
            return

        if cmd == "/save":
            path = Path(args[0]) if args else Path(".atlas_tui.json")
            path.write_text(json.dumps(self.state.to_dict(), indent=2))
            self._write_log(f"saved config: {path}")
            return

        if cmd == "/load":
            path = Path(args[0]) if args else Path(".atlas_tui.json")
            if not path.exists():
                self._write_log(f"config not found: {path}")
                return
            raw = json.loads(path.read_text())
            self.state = TuiState.from_dict(raw)
            self._render_settings()
            self._write_log(f"loaded config: {path}")
            return

        self._write_log(f"unknown command: {cmd}")

    def _write_log(self, message: str) -> None:
        log_widget = self.query_one("#log", Log)
        if not log_widget.display:
            log_widget.display = True
        log_widget.write_line(message)

    def _render_settings(self) -> None:
        table = Table(title="Settings", show_header=False)
        table.add_column("k", style="bold")
        table.add_column("v")
        table.add_row("symbol", self.state.symbol)
        table.add_row("data_source", self.state.data_source)
        table.add_row("csv_path", self.state.csv_path or "-")
        table.add_row("timeframe", self.state.timeframe or "-")
        table.add_row("start", self.state.start or "-")
        table.add_row("end", self.state.end or "-")
        table.add_row(
            "strategy",
            f"{self.state.strategy} (fast={self.state.fast_window} slow={self.state.slow_window})",
        )
        table.add_row("initial_cash", f"{self.state.initial_cash:.2f}")
        table.add_row("max_notional", f"{self.state.max_position_notional_usd:.2f}")
        table.add_row("slippage_bps", f"{self.state.slippage_bps:.2f}")
        table.add_row("allow_short", str(self.state.allow_short))
        table.add_row("paper_running", str(self._paper_thread is not None))
        table.add_row("paper_lookback", str(self.state.paper_lookback_bars))
        table.add_row("paper_poll_s", str(self.state.paper_poll_seconds))
        table.add_row(
            "paper_max_notional",
            f"{self.state.paper_max_position_notional_usd:.2f}",
        )
        table.add_row("paper_dry_run", str(self.state.paper_dry_run))

        self.query_one("#settings", Static).update(table)

    def _render_results(self, summary: Optional[Table]) -> None:
        widget = self.query_one("#results", Static)
        if summary is None:
            table = Table(show_header=False)
            table.add_column("k", style="bold")
            table.add_column("v")
            table.add_row("status", "no backtest yet")
            table.add_row("hint", "run /backtest to generate a summary")
            widget.update(table)
            return
        widget.update(summary)

    def _run_backtest(self) -> None:
        self._write_log("running backtest...")
        try:
            run_dir, summary = self._run_backtest_sync()
        except Exception as exc:
            self._write_log(f"backtest error: {exc}")
            return
        self._last_run_dir = run_dir
        self._render_results(summary)
        self._write_log(f"backtest complete: {run_dir}")

    def _run_backtest_sync(self) -> tuple[Path, Table]:
        run_dir = (
            Path("outputs")
            / "backtests"
            / f"tui_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        if self.state.data_source == "sample":
            path = Path("data") / "sample" / f"{self.state.symbol}_1min_sample.csv"
            bars = load_bars_csv(path)
            data_hint = str(path)
        elif self.state.data_source == "csv":
            if not self.state.csv_path:
                raise ValueError("csv path not set")
            bars = load_bars_csv(Path(self.state.csv_path))
            data_hint = self.state.csv_path
        elif self.state.data_source == "alpaca":
            if self.state.timeframe and not (self.state.start and self.state.end):
                delta = _parse_relative_timeframe(self.state.timeframe)
                end_dt = now_ny()
                start_dt = end_dt - delta
                self.state.start = start_dt.isoformat()
                self.state.end = end_dt.isoformat()
            if not self.state.start or not self.state.end:
                raise ValueError("start/end required for alpaca data")
            settings = get_alpaca_settings(require_keys=True)
            bars = load_stock_bars_cached(
                settings=settings,
                symbol=self.state.symbol,
                start=parse_iso_datetime(self.state.start),
                end=parse_iso_datetime(self.state.end),
                timeframe="1Min",
            )
            data_hint = f"{self.state.start} -> {self.state.end}"
        else:
            raise ValueError("invalid data source")

        bars, start_dt, end_dt = _resolve_time_window(
            bars=bars,
            timeframe=self.state.timeframe,
            start=self.state.start,
            end=self.state.end,
        )

        if len(bars) < 3:
            raise ValueError("backtest window has too few bars")

        strat = build_strategy(
            name=self.state.strategy,
            params_path=None,
            fast_window=self.state.fast_window,
            slow_window=self.state.slow_window,
        )

        cfg = BacktestConfig(
            symbol=self.state.symbol,
            initial_cash=self.state.initial_cash,
            max_position_notional_usd=self.state.max_position_notional_usd,
            slippage_bps=self.state.slippage_bps,
            allow_short=self.state.allow_short,
        )

        run_backtest(bars=bars, strategy=strat, cfg=cfg, run_dir=run_dir)

        metrics = json.loads((run_dir / "metrics.json").read_text())
        equity_curve = pd.read_csv(
            run_dir / "equity_curve.csv", parse_dates=["timestamp"]
        )
        final_equity = float(equity_curve["equity"].iloc[-1])
        bar_minutes = _infer_bar_minutes(bars.index)
        days = int(pd.Series(bars.index.date).nunique())
        duration = pd.Timestamp(bars.index[-1]) - pd.Timestamp(bars.index[0])
        trades = pd.read_csv(run_dir / "trades.csv")
        gross_notional = float(trades["notional"].sum()) if len(trades) else 0.0

        summary = Table(title="Backtest summary", show_header=False)
        summary.add_column("k", style="bold")
        summary.add_column("v")
        summary.add_row("run_dir", str(run_dir))
        summary.add_row("symbol", self.state.symbol)
        summary.add_row("data", f"{self.state.data_source} ({data_hint})")
        summary.add_row(
            "window",
            f"{start_dt or bars.index[0].isoformat()} -> {end_dt or bars.index[-1].isoformat()}  |  bars={len(bars)}  days={days}  bar={bar_minutes:.2f}m",
        )
        summary.add_row("duration", str(duration))
        summary.add_row(
            "strategy",
            f"{self.state.strategy} (fast={self.state.fast_window} slow={self.state.slow_window})",
        )
        summary.add_row("warmup_bars", str(strat.warmup_bars()))
        summary.add_row(
            "config",
            "  ".join(
                [
                    f"initial_cash={cfg.initial_cash:.2f}",
                    f"max_notional={cfg.max_position_notional_usd:.2f}",
                    f"slippage_bps={cfg.slippage_bps:.2f}",
                    f"allow_short={cfg.allow_short}",
                ]
            ),
        )
        summary.add_row(
            "results",
            "  ".join(
                [
                    f"final_equity={final_equity:.2f}",
                    f"total_return={metrics['total_return']:.4%}",
                    f"max_drawdown={metrics['max_drawdown']:.4%}",
                    f"sharpe={metrics['sharpe']:.2f}",
                    f"fills={metrics['trades']}",
                    f"gross_notional={gross_notional:.2f}",
                ]
            ),
        )

        return run_dir, summary

    def _start_paper(self) -> None:
        if self._paper_thread is not None:
            self._write_log("paper loop already running")
            return
        settings = get_alpaca_settings(require_keys=True)
        run_dir = (
            Path("outputs")
            / "paper"
            / f"tui_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        strat = build_strategy(
            name=self.state.strategy,
            params_path=None,
            fast_window=self.state.fast_window,
            slow_window=self.state.slow_window,
        )
        cfg = PaperConfig(
            symbols=[self.state.symbol],
            lookback_bars=self.state.paper_lookback_bars,
            poll_seconds=self.state.paper_poll_seconds,
            max_position_notional_usd=self.state.paper_max_position_notional_usd,
            allow_trading_when_closed=self.state.paper_allow_trading_when_closed,
            dry_run=self.state.paper_dry_run,
        )

        self._paper_stop = Event()

        def _runner() -> None:
            try:
                run_paper_loop(
                    settings=settings,
                    strategy=strat,
                    cfg=cfg,
                    run_dir=run_dir,
                    max_loops=None,
                    stop_event=self._paper_stop,
                )
            except Exception as exc:
                self.call_from_thread(self._write_log, f"paper loop error: {exc}")
            finally:
                self._paper_thread = None
                self._paper_stop = None
                self.call_from_thread(self._render_settings)

        self._paper_thread = Thread(target=_runner, daemon=True)
        self._paper_thread.start()
        self._write_log(f"paper loop started: {run_dir}")
        self._render_settings()

    def _stop_paper(self) -> None:
        if self._paper_stop is None:
            self._write_log("paper loop is not running")
            return
        self._paper_stop.set()
        self._write_log("paper loop stop requested")


def run_tui() -> None:
    AtlasTui().run()
