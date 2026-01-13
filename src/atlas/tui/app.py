from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event, Thread
from typing import Any, Optional

import pandas as pd
from rich.console import Group
from rich.table import Table
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Input, Log, Static

from atlas.backtest.engine import BacktestConfig, run_backtest
from atlas.config import get_alpaca_settings, get_default_max_position_notional_usd
from atlas.data.bars import parse_bar_timeframe
from atlas.data.universe import load_universe_bars
from atlas.paper.runner import PaperConfig, run_paper_loop
from atlas.strategies.registry import build_strategy, list_strategy_names
from atlas.utils.time import now_ny, parse_iso_datetime


@dataclass
class TuiState:
    symbols: str = "SPY"
    data_source: str = "sample"
    csv_path: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    timeframe: Optional[str] = None
    bar_timeframe: str = "1Min"
    alpaca_feed: str = "delayed_sip"
    paper_feed: str = "iex"
    strategy: str = "spy_open_close"
    fast_window: int = 10
    slow_window: int = 30
    strategy_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    initial_cash: float = 100_000.0
    max_position_notional_usd: float = 10_000.0
    slippage_bps: float = 0.0
    allow_short: bool = False
    paper_lookback_bars: int = 200
    paper_poll_seconds: int = 60
    paper_max_position_notional_usd: float = 1_000.0
    paper_regular_hours_only: bool = True
    paper_allow_trading_when_closed: bool = False
    paper_limit_offset_bps: float = 5.0
    paper_dry_run: bool = False

    @classmethod
    def from_dict(cls, raw: dict) -> "TuiState":
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        data = {k: v for k, v in raw.items() if k in fields}
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


STRATEGY_PARAM_SPECS: dict[str, dict[str, type]] = {
    "spy_open_close": {},
    "no_trade": {},
    "ema_crossover": {
        "fast_window": int,
        "slow_window": int,
    },
    "ma_crossover": {
        "fast_window": int,
        "slow_window": int,
    },
    "nec_x": {
        "M": int,
        "V": int,
        "Wcorr": int,
        "rho_min": float,
        "strength_entry": float,
        "strength_exit": float,
        "H_max": int,
        "k_cost": float,
        "spread_floor_bps": float,
        "slip_bps": float,
        "daily_loss_limit": float,
        "kill_switch": float,
    },
    "nec_pdt": {
        "M": int,
        "V": int,
        "eps": float,
        "H": int,
        "base_thr_bps": float,
        "budget_step_bps": float,
        "atr_lookback_bars": int,
        "stop_atr_mult": float,
        "trail_atr_mult": float,
        "min_hold_bars": int,
        "flip_confirm_bars": int,
        "max_day_trades_per_rolling_5_days": int,
        "half_spread_bps": float,
        "slippage_bps": float,
        "fee_bps": float,
    },
    "orb_trend": {
        "orb_minutes": int,
        "orb_breakout_bps": float,
        "confirm_bars": int,
        "atr_window": int,
        "er_window": int,
        "er_min": float,
        "expected_hold_bars": int,
        "k_cost": float,
        "slippage_bps": float,
        "min_hold_bars": int,
        "daily_loss_limit": float,
        "kill_switch": float,
    },
}

STRATEGY_DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "spy_open_close": {},
    "no_trade": {},
    "ema_crossover": {
        "fast_window": 10,
        "slow_window": 30,
    },
    "ma_crossover": {
        "fast_window": 10,
        "slow_window": 30,
    },
    "nec_x": {
        "M": 6,
        "V": 12,
        "Wcorr": 12,
        "rho_min": 0.60,
        "strength_entry": 0.80,
        "strength_exit": 0.20,
        "H_max": 6,
        "k_cost": 1.25,
        "spread_floor_bps": 0.50,
        "slip_bps": 0.75,
        "daily_loss_limit": 0.010,
        "kill_switch": 0.025,
    },
    "nec_pdt": {
        "M": 6,
        "V": 12,
        "eps": 1e-8,
        "H": 12,
        "base_thr_bps": 10.0,
        "budget_step_bps": 4.0,
        "atr_lookback_bars": 12,
        "stop_atr_mult": 2.0,
        "trail_atr_mult": 2.5,
        "min_hold_bars": 4,
        "flip_confirm_bars": 3,
        "max_day_trades_per_rolling_5_days": 3,
        "half_spread_bps": 1.5,
        "slippage_bps": 2.0,
        "fee_bps": 0.3,
    },
    "orb_trend": {
        "orb_minutes": 30,
        "orb_breakout_bps": 4.0,
        "confirm_bars": 2,
        "atr_window": 20,
        "er_window": 12,
        "er_min": 0.35,
        "expected_hold_bars": 12,
        "k_cost": 2.0,
        "slippage_bps": 1.25,
        "min_hold_bars": 3,
        "daily_loss_limit": 0.010,
        "kill_switch": 0.025,
    },
}


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


def _parse_bool_arg(value: str) -> Optional[bool]:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return None


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
    SUGGESTION_MAX_LINES = 8
    COMMAND_ALIASES = {
        "/?": "/help",
        "/bars": "/bar",
        "/strategy": "/algorithm",
        "/aglorithm": "/algorithm",
        "/algorithim": "/algorithm",
        "/initialcash": "/cash",
        "/initial_cash": "/cash",
        "/max": "/maxnotional",
        "/notional": "/maxnotional",
        "/max_notional": "/maxnotional",
        "/slip": "/slippage",
        "/allowshort": "/short",
        "/allow_short": "/short",
        "/paperlookbackbars": "/paperlookback",
        "/paperpolls": "/paperpoll",
        "/papermax": "/papermaxnotional",
    }
    BASE_COMMANDS = [
        "/help",
        "/?",
        "/backtest",
        "/paper",
        "/timeframe",
        "/bar",
        "/bars",
        "/algorithm",
        "/strategy",
        "/param",
        "/params",
        "/fast",
        "/slow",
        "/cash",
        "/maxnotional",
        "/slippage",
        "/short",
        "/data",
        "/feed",
        "/paperfeed",
        "/paperlookback",
        "/paperpoll",
        "/papermaxnotional",
        "/paperclosed",
        "/paperrth",
        "/paperlimitbps",
        "/paperdry",
        "/csv",
        "/symbol",
        "/symbols",
        "/start",
        "/end",
        "/save",
        "/load",
    ]
    CSS = """
    Screen { layout: vertical; background: $surface; }
    #body { height: 1fr; }
    #settings { width: 38%; border: solid $accent; padding: 1; }
    #results { width: 62%; border: solid $accent; padding: 1; }
    #lower { height: 18; border: solid $accent; padding: 1; }
    #log { height: 1fr; background: $surface; }
    #suggestions {
        height: 0;
        max-height: 8;
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
        self._paper_run_dir: Optional[Path] = None
        self._last_run_dir: Optional[Path] = None
        self._last_live_decision_ts: Optional[str] = None
        self._suggestion_matches: list[str] = []
        self._suggestion_index: int = 0
        self._suggestion_scroll: int = 0
        self._config_path = Path(os.getenv("ATLAS_TUI_CONFIG", ".atlas_tui.json"))
        self._autosave_enabled = True
        self._config_load_failed = False

    def _load_config(self) -> None:
        path = self._config_path
        if not path.exists():
            self._config_load_failed = False
            return
        try:
            raw = json.loads(path.read_text())
            self.state = TuiState.from_dict(raw)
            self._config_load_failed = False
        except Exception as exc:
            self._config_load_failed = True
            self._autosave_enabled = False
            self._write_log(f"failed to load config ({path}): {exc}")

    def _save_config(self) -> None:
        if not self._autosave_enabled or self._config_load_failed:
            return
        path = self._config_path
        try:
            path.write_text(json.dumps(self.state.to_dict(), indent=2))
        except Exception as exc:
            self._autosave_enabled = False
            self._write_log(f"failed to save config ({path}): {exc}")

    def _canonicalize_strategy_name(self, name: str) -> str:
        name = name.strip()
        alias_map = {
            "nec-x": "nec_x",
            "nec-pdt": "nec_pdt",
            "orb-trend": "orb_trend",
            "ema-crossover": "ema_crossover",
            "spy-open-close": "spy_open_close",
            "no-trade": "no_trade",
        }
        if name in alias_map:
            return alias_map[name]
        return name

    def _strategy_param_spec(self, strategy: str) -> dict[str, type]:
        return STRATEGY_PARAM_SPECS.get(strategy, {})

    def _ensure_strategy_params(self, strategy: str) -> None:
        alias_map = {
            "nec_x": "nec-x",
            "nec_pdt": "nec-pdt",
            "orb_trend": "orb-trend",
            "ema_crossover": "ema-crossover",
            "spy_open_close": "spy-open-close",
            "no_trade": "no-trade",
        }
        alias = alias_map.get(strategy)
        if alias and alias in self.state.strategy_params and strategy not in self.state.strategy_params:
            self.state.strategy_params[strategy] = self.state.strategy_params.pop(alias)

        if strategy == "ma_crossover":
            if not self.state.strategy_params:
                defaults = {
                    "fast_window": self.state.fast_window,
                    "slow_window": self.state.slow_window,
                }
            else:
                defaults = STRATEGY_DEFAULT_PARAMS.get(strategy, {})
        else:
            defaults = STRATEGY_DEFAULT_PARAMS.get(strategy, {})

        if strategy not in self.state.strategy_params:
            self.state.strategy_params[strategy] = dict(defaults)
        else:
            params = self.state.strategy_params[strategy]
            for key, value in defaults.items():
                params.setdefault(key, value)

        if strategy in {"ma_crossover", "ema_crossover"}:
            params = self.state.strategy_params.get(strategy, {})
            if "fast_window" in params:
                self.state.fast_window = int(params["fast_window"])
            if "slow_window" in params:
                self.state.slow_window = int(params["slow_window"])

    def _normalize_param_key(self, strategy: str, key: str) -> str:
        spec = self._strategy_param_spec(strategy)
        if spec:
            for candidate in spec:
                if candidate.lower() == key.lower():
                    return candidate
        return key

    def _format_param_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def _format_strategy_params(self, strategy: str, params: dict[str, Any]) -> str:
        if not params:
            return "-"
        spec = self._strategy_param_spec(strategy)
        ordered = list(spec.keys())
        for key in params:
            if key not in ordered:
                ordered.append(key)
        parts = [f"{k}={self._format_param_value(params[k])}" for k in ordered if k in params]
        per_line = 4
        lines = [
            "  ".join(parts[i : i + per_line])
            for i in range(0, len(parts), per_line)
        ]
        return "\n".join(lines)

    def _parse_int_value(self, raw: str) -> int:
        try:
            return int(raw)
        except ValueError:
            value = float(raw)
            if not value.is_integer():
                raise ValueError("expected integer") from None
            return int(value)

    def _parse_param_value(self, strategy: str, key: str, raw: str) -> Any:
        spec = self._strategy_param_spec(strategy)
        if spec and key in spec:
            expected = spec[key]
            if expected is int:
                return self._parse_int_value(raw)
            return float(raw)
        try:
            return self._parse_int_value(raw)
        except Exception:
            try:
                return float(raw)
            except Exception:
                return raw

    def _canonicalize_command(self, token: str) -> str:
        token = token.strip().lower()
        return self.COMMAND_ALIASES.get(token, token)

    def _arg_options(self, command: str) -> list[str]:
        cmd = self._canonicalize_command(command)
        if cmd == "/paper":
            return ["start", "stop"]
        if cmd == "/fast":
            return ["5", "10", "20"]
        if cmd == "/slow":
            return ["20", "30", "50", "100"]
        if cmd == "/cash":
            return ["10000", "50000", "100000", "250000"]
        if cmd == "/maxnotional":
            return ["1000", "5000", "10000", "25000", "100000"]
        if cmd == "/slippage":
            return ["0", "0.5", "1.25", "2.0", "5.0"]
        if cmd == "/short":
            return ["true", "false"]
        if cmd == "/data":
            return ["sample", "csv", "alpaca"]
        if cmd in {"/feed", "/paperfeed"}:
            return ["iex", "delayed_sip", "sip"]
        if cmd == "/paperlookback":
            return ["50", "100", "200", "500"]
        if cmd == "/paperpoll":
            return ["5", "15", "30", "60"]
        if cmd == "/papermaxnotional":
            return ["1000", "5000", "10000"]
        if cmd in {"/paperclosed", "/paperrth", "/paperdry"}:
            return ["true", "false"]
        if cmd == "/paperlimitbps":
            return ["0", "1", "5", "10", "25"]
        if cmd == "/bar":
            return ["1Min", "5Min"]
        if cmd == "/algorithm":
            return list_strategy_names()
        if cmd == "/param":
            strategy = self._canonicalize_strategy_name(self.state.strategy)
            spec = self._strategy_param_spec(strategy)
            return list(spec.keys()) if spec else []
        if cmd == "/timeframe":
            return ["6h", "7d", "30d", "180d", "1y", "clear"]
        if cmd == "/csv":
            return ["data/sample"]
        if cmd == "/symbol":
            return ["SPY", "QQQ"]
        if cmd == "/symbols":
            return ["SPY,QQQ"]
        return []

    def _compute_suggestions(self, raw: str) -> list[str]:
        raw = raw.rstrip("\n")
        raw_l = raw.lstrip()
        if not raw_l.startswith("/"):
            return []

        ends_with_ws = bool(raw_l) and raw_l[-1].isspace()
        tokens = raw_l.strip().split()
        if not tokens:
            return []

        cmd_token = tokens[0]
        cmd = self._canonicalize_command(cmd_token)
        base_prefix = cmd_token.strip().lower()
        known_base = {self._canonicalize_command(c) for c in self.BASE_COMMANDS}

        # If user has typed more than one arg and then whitespace, we intentionally stop.
        if len(tokens) >= 2 and ends_with_ws:
            return []

        # Completing the command itself (no args yet).
        if len(tokens) == 1 and cmd not in known_base:
            return [c for c in self.BASE_COMMANDS if c.startswith(base_prefix)]

        # Completing the first arg for a known command.
        options = self._arg_options(cmd)
        if not options:
            return []

        arg_prefix = ""
        if len(tokens) >= 2:
            arg_prefix = tokens[1].strip().lower()

        matches = [o for o in options if o.lower().startswith(arg_prefix)]
        return [f"{cmd_token.strip()} {m}" for m in matches]

    def on_key(self, event: events.Key) -> None:
        input_box = self.query_one("#input", Input)
        if not input_box.has_focus:
            return

        if not self._suggestion_matches:
            return

        if event.key in {"up", "down"}:
            delta = -1 if event.key == "up" else 1
            self._suggestion_index = (self._suggestion_index + delta) % len(
                self._suggestion_matches
            )
            self._render_suggestions()
            event.prevent_default()
            event.stop()
            return

        if event.key == "tab":
            selected = self._suggestion_matches[self._suggestion_index]
            input_box.value = selected
            try:
                input_box.cursor_position = len(selected)
            except Exception:
                pass
            self._update_suggestions(selected)
            event.prevent_default()
            event.stop()
            return

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
        self._load_config()
        self.state.strategy = self._canonicalize_strategy_name(self.state.strategy)
        self._ensure_strategy_params(self.state.strategy)
        self._render_settings()
        self._render_results(None)
        self.set_interval(0.5, self._refresh_live_view)

    def on_unmount(self) -> None:
        self._save_config()

    def action_help_quit(self) -> None:
        input_box = self.query_one("#input", Input)
        if input_box.value:
            input_box.value = ""
            self._update_suggestions("")
            self._ctrl_c_armed_at = None
            return

        now = time.monotonic()
        if self._ctrl_c_armed_at and now - self._ctrl_c_armed_at < 2.0:
            self.exit()
            return
        self._ctrl_c_armed_at = now
        self._write_log("press ctrl+c again to quit")

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

    def _clear_suggestions(self) -> None:
        suggestions = self.query_one("#suggestions", Static)
        self._suggestion_matches = []
        self._suggestion_index = 0
        self._suggestion_scroll = 0
        suggestions.styles.height = 0
        suggestions.update("")
        suggestions.display = False

    def _render_suggestions(self) -> None:
        suggestions = self.query_one("#suggestions", Static)
        if not self._suggestion_matches:
            self._clear_suggestions()
            return

        matches = self._suggestion_matches
        self._suggestion_index = max(0, min(int(self._suggestion_index), len(matches) - 1))

        max_lines = int(self.SUGGESTION_MAX_LINES)
        if self._suggestion_index < self._suggestion_scroll:
            self._suggestion_scroll = self._suggestion_index
        if self._suggestion_index >= self._suggestion_scroll + max_lines:
            self._suggestion_scroll = self._suggestion_index - max_lines + 1
        self._suggestion_scroll = max(0, min(int(self._suggestion_scroll), max(len(matches) - 1, 0)))

        window = matches[self._suggestion_scroll : self._suggestion_scroll + max_lines]

        lines: list[str] = []
        for idx, cmd in enumerate(window):
            global_idx = self._suggestion_scroll + idx
            prefix = "> " if global_idx == self._suggestion_index else "  "
            lines.append(prefix + cmd)

        suggestions.display = True
        suggestions.styles.height = len(lines)
        suggestions.update("\n".join(lines))

    def _update_suggestions(self, value: str) -> None:
        raw = value
        if not raw.lstrip().startswith("/"):
            self._clear_suggestions()
            return

        matches = self._compute_suggestions(raw)
        if not matches:
            self._clear_suggestions()
            return
        if matches != self._suggestion_matches:
            self._suggestion_index = 0
            self._suggestion_scroll = 0
        self._suggestion_matches = matches
        self._render_suggestions()

    def _handle_command(self, text: str) -> None:
        parts = text.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in {"/help", "/?"}:
            self._write_log(
                "commands: /backtest, /paper start|stop, /timeframe <7d|6h|1m|1y|clear>, "
                "/bar <1Min|5Min>, /algorithm <name>, /data <sample|csv|alpaca>, "
                "/param <key> <value>, /params, "
                "/fast <int>, /slow <int>, /cash <usd>, /maxnotional <usd>, /slippage <bps>, /short <true|false>, "
                "/feed <iex|delayed_sip|sip>, /paperfeed <iex|delayed_sip|sip>, /csv <path>, "
                "/paperlookback <bars>, /paperpoll <seconds>, /papermaxnotional <usd>, "
                "/paperclosed <true|false>, /paperrth <true|false>, /paperlimitbps <float>, /paperdry <true|false>, "
                "/symbol <SPY>, /symbols <SPY,QQQ>, /start <iso>, /end <iso>, /save [path], /load [path]"
            )
            return

        if cmd == "/symbol" and args:
            self.state.symbols = args[0].upper()
            self._render_settings()
            return

        if cmd == "/symbols" and args:
            self.state.symbols = args[0].upper()
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

        if cmd == "/feed" and args:
            value = args[0].lower()
            if value not in {"iex", "delayed_sip", "sip"}:
                self._write_log("feed must be one of: iex | delayed_sip | sip")
                return
            self.state.alpaca_feed = value
            self._render_settings()
            return

        if cmd == "/paperfeed" and args:
            value = args[0].lower()
            if value not in {"iex", "delayed_sip", "sip"}:
                self._write_log("paperfeed must be one of: iex | delayed_sip | sip")
                return
            self.state.paper_feed = value
            self._render_settings()
            return

        if cmd == "/fast" and args:
            try:
                value = int(args[0])
            except ValueError:
                self._write_log("fast must be an integer")
                return
            if value <= 0:
                self._write_log("fast must be > 0")
                return
            active = self._canonicalize_strategy_name(self.state.strategy)
            target = "ema_crossover" if active == "ema_crossover" else "ma_crossover"
            self._ensure_strategy_params(target)
            if active in {"ma_crossover", "ema_crossover"}:
                self.state.fast_window = value
            self.state.strategy_params[target]["fast_window"] = value
            self._render_settings()
            return

        if cmd == "/slow" and args:
            try:
                value = int(args[0])
            except ValueError:
                self._write_log("slow must be an integer")
                return
            if value <= 0:
                self._write_log("slow must be > 0")
                return
            active = self._canonicalize_strategy_name(self.state.strategy)
            target = "ema_crossover" if active == "ema_crossover" else "ma_crossover"
            self._ensure_strategy_params(target)
            if active in {"ma_crossover", "ema_crossover"}:
                self.state.slow_window = value
            self.state.strategy_params[target]["slow_window"] = value
            self._render_settings()
            return

        if cmd == "/cash" and args:
            try:
                value = float(args[0])
            except ValueError:
                self._write_log("cash must be a number (usd)")
                return
            if value <= 0:
                self._write_log("cash must be > 0")
                return
            self.state.initial_cash = value
            self._render_settings()
            return

        if cmd == "/maxnotional" and args:
            try:
                value = float(args[0])
            except ValueError:
                self._write_log("maxnotional must be a number (usd)")
                return
            if value <= 0:
                self._write_log("maxnotional must be > 0")
                return
            self.state.max_position_notional_usd = value
            self._render_settings()
            return

        if cmd == "/slippage" and args:
            try:
                value = float(args[0])
            except ValueError:
                self._write_log("slippage must be a number (bps)")
                return
            if value < 0:
                self._write_log("slippage must be >= 0")
                return
            self.state.slippage_bps = value
            strategy = self._canonicalize_strategy_name(self.state.strategy)
            if strategy == "orb_trend":
                self._ensure_strategy_params(strategy)
                self.state.strategy_params[strategy]["slippage_bps"] = float(value)
            elif strategy == "nec_x":
                self._write_log(
                    "note: for nec_x, slippage is derived from spread_floor_bps + slip_bps (edit via /param to keep gating consistent)"
                )
            elif strategy == "nec_pdt":
                self._write_log(
                    "note: for nec_pdt, slippage is derived from half_spread_bps + slippage_bps + fee_bps (edit via /param to keep gating consistent)"
                )
            self._render_settings()
            return

        if cmd == "/short" and args:
            value = _parse_bool_arg(args[0])
            if value is None:
                self._write_log("short must be true|false")
                return
            self.state.allow_short = value
            self._render_settings()
            return

        if cmd == "/paperclosed" and args:
            value = _parse_bool_arg(args[0])
            if value is None:
                self._write_log("paperclosed must be true|false")
                return
            self.state.paper_allow_trading_when_closed = value
            self._render_settings()
            return

        if cmd == "/paperrth" and args:
            value = _parse_bool_arg(args[0])
            if value is None:
                self._write_log("paperrth must be true|false")
                return
            self.state.paper_regular_hours_only = value
            self._render_settings()
            return

        if cmd == "/paperlookback" and args:
            try:
                value = int(args[0])
            except ValueError:
                self._write_log("paperlookback must be an integer (bars)")
                return
            if value <= 0:
                self._write_log("paperlookback must be > 0")
                return
            self.state.paper_lookback_bars = value
            self._render_settings()
            return

        if cmd == "/paperpoll" and args:
            try:
                value = int(args[0])
            except ValueError:
                self._write_log("paperpoll must be an integer (seconds)")
                return
            if value <= 0:
                self._write_log("paperpoll must be > 0")
                return
            self.state.paper_poll_seconds = value
            self._render_settings()
            return

        if cmd == "/papermaxnotional" and args:
            try:
                value = float(args[0])
            except ValueError:
                self._write_log("papermaxnotional must be a number (usd)")
                return
            if value <= 0:
                self._write_log("papermaxnotional must be > 0")
                return
            self.state.paper_max_position_notional_usd = value
            self._render_settings()
            return

        if cmd == "/paperlimitbps" and args:
            try:
                value = float(args[0])
            except ValueError:
                self._write_log("paperlimitbps must be a number (bps)")
                return
            if value < 0:
                self._write_log("paperlimitbps must be >= 0")
                return
            self.state.paper_limit_offset_bps = value
            self._render_settings()
            return

        if cmd == "/paperdry" and args:
            value = _parse_bool_arg(args[0])
            if value is None:
                self._write_log("paperdry must be true|false")
                return
            self.state.paper_dry_run = value
            self._render_settings()
            return

        if cmd == "/csv" and args:
            self.state.csv_path = " ".join(args)
            self._render_settings()
            return

        if cmd == "/timeframe" and args:
            if args[0].lower() == "clear":
                self.state.timeframe = None
                self._render_settings()
                return

            try:
                delta = _parse_relative_timeframe(args[0])
            except Exception:
                self._write_log("unsupported timeframe spec (examples: 7d, 6h, 1m, 1y)")
                return
            self.state.timeframe = args[0]
            if self.state.data_source == "alpaca":
                end_dt = now_ny()
                start_dt = end_dt - delta
                self.state.start = start_dt.isoformat()
                self.state.end = end_dt.isoformat()
            self._render_settings()
            return

        if cmd in {"/bar", "/bars"} and args:
            self.state.bar_timeframe = args[0]
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

        if cmd in {"/algorithm", "/strategy", "/aglorithm", "/algorithim"} and args:
            strategy = self._canonicalize_strategy_name(args[0])
            self.state.strategy = strategy
            self._ensure_strategy_params(strategy)
            if strategy in {"nec_x", "nec_pdt", "orb_trend"}:
                self.state.symbols = "SPY,QQQ"
                self.state.bar_timeframe = "5Min"
                params = self.state.strategy_params.get(strategy, {})
                if strategy == "nec_x":
                    self.state.slippage_bps = float(params.get("spread_floor_bps", 0.0)) + float(
                        params.get("slip_bps", 0.0)
                    )
                elif strategy == "nec_pdt":
                    self.state.slippage_bps = float(params.get("half_spread_bps", 0.0)) + float(
                        params.get("slippage_bps", 0.0)
                    ) + float(params.get("fee_bps", 0.0))
                else:
                    self.state.slippage_bps = float(params.get("slippage_bps", 1.25))
            elif strategy == "spy_open_close":
                self.state.symbols = "SPY"
            self._render_settings()
            return

        if cmd in {"/param", "/params"}:
            strategy = self._canonicalize_strategy_name(self.state.strategy)
            self._ensure_strategy_params(strategy)
            params = self.state.strategy_params.get(strategy, {})
            if cmd == "/params" or not args:
                rendered = self._format_strategy_params(strategy, params)
                self._write_log(f"{strategy} params: {rendered}")
                return
            if len(args) != 2:
                self._write_log("param usage: /param <key> <value>")
                return
            key = self._normalize_param_key(strategy, args[0])
            spec = self._strategy_param_spec(strategy)
            if strategy in STRATEGY_PARAM_SPECS and not spec:
                self._write_log(f"{strategy} has no editable params")
                return
            if spec and key not in spec:
                valid = ", ".join(spec.keys())
                self._write_log(f"unknown param for {strategy}: {args[0]} (valid: {valid})")
                return
            try:
                value = self._parse_param_value(strategy, key, args[1])
            except ValueError:
                self._write_log(f"invalid value for {key}")
                return
            params[key] = value
            if strategy in {"ma_crossover", "ema_crossover"}:
                if key == "fast_window":
                    self.state.fast_window = int(value)
                if key == "slow_window":
                    self.state.slow_window = int(value)
            if strategy == "orb_trend" and key == "slippage_bps":
                self.state.slippage_bps = float(value)
            if strategy == "nec_x" and key in {"spread_floor_bps", "slip_bps"}:
                self.state.slippage_bps = float(params.get("spread_floor_bps", 0.0)) + float(
                    params.get("slip_bps", 0.0)
                )
            if strategy == "nec_pdt" and key in {"half_spread_bps", "slippage_bps", "fee_bps"}:
                self.state.slippage_bps = float(params.get("half_spread_bps", 0.0)) + float(
                    params.get("slippage_bps", 0.0)
                ) + float(params.get("fee_bps", 0.0))
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
            path = Path(" ".join(args)) if args else self._config_path
            self._config_path = path
            self._autosave_enabled = True
            self._config_load_failed = False
            try:
                path.write_text(json.dumps(self.state.to_dict(), indent=2))
            except Exception as exc:
                self._write_log(f"failed to save config ({path}): {exc}")
                return
            self._write_log(f"saved config: {path}")
            return

        if cmd == "/load":
            path = Path(" ".join(args)) if args else self._config_path
            self._config_path = path
            if not path.exists():
                self._write_log(f"config not found: {path}")
                return
            try:
                raw = json.loads(path.read_text())
                self.state = TuiState.from_dict(raw)
            except Exception as exc:
                self._write_log(f"failed to load config ({path}): {exc}")
                return
            self.state.strategy = self._canonicalize_strategy_name(self.state.strategy)
            self._ensure_strategy_params(self.state.strategy)
            self._autosave_enabled = True
            self._config_load_failed = False
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
        self.state.strategy = self._canonicalize_strategy_name(self.state.strategy)
        self._ensure_strategy_params(self.state.strategy)
        table = Table(title="Settings", show_header=False)
        table.add_column("k", style="bold")
        table.add_column("v")
        table.add_row("symbols", self.state.symbols)
        table.add_row("data_source", self.state.data_source)
        table.add_row("alpaca_feed", self.state.alpaca_feed)
        table.add_row("csv_path", self.state.csv_path or "-")
        table.add_row("timeframe", self.state.timeframe or "-")
        table.add_row("bar_timeframe", self.state.bar_timeframe)
        table.add_row("start", self.state.start or "-")
        table.add_row("end", self.state.end or "-")
        table.add_row(
            "strategy",
            (
                f"{self.state.strategy} (fast={self.state.fast_window} slow={self.state.slow_window})"
                if self.state.strategy in {"ma_crossover", "ema_crossover"}
                else self.state.strategy
            ),
        )
        params = self.state.strategy_params.get(self.state.strategy, {})
        if params:
            table.add_row(
                "params",
                self._format_strategy_params(self.state.strategy, params),
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
        table.add_row("paper_feed", self.state.paper_feed)
        table.add_row("paper_rth_only", str(self.state.paper_regular_hours_only))
        table.add_row("paper_when_closed", str(self.state.paper_allow_trading_when_closed))
        table.add_row("paper_limit_bps", f"{self.state.paper_limit_offset_bps:.2f}")
        table.add_row("paper_dry_run", str(self.state.paper_dry_run))
        table.add_row("config", str(self._config_path))

        self.query_one("#settings", Static).update(table)
        self._save_config()

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

    def _tail_last_jsonl(self, path: Path) -> Optional[dict]:
        if not path.exists():
            return None
        try:
            with path.open("rb") as f:
                f.seek(0, 2)
                size = f.tell()
                if size <= 0:
                    return None
                read_size = min(size, 131072)
                f.seek(-read_size, 2)
                chunk = f.read().decode("utf-8", errors="ignore")
        except OSError:
            chunk = path.read_text(errors="ignore")
        lines = [ln for ln in chunk.splitlines() if ln.strip()]
        if not lines:
            return None
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError:
            return None

    def _render_paper_live(self, decision: dict) -> None:
        results = self.query_one("#results", Static)
        results.border_title = "Paper (live)"

        ts = str(decision.get("timestamp", ""))
        targets = decision.get("targets", {}) or {}
        positions = decision.get("positions", {}) or {}
        debug = decision.get("debug", {}) or {}

        summary = Table(show_header=False)
        summary.add_column("k", style="bold")
        summary.add_column("v")
        summary.add_row("run_dir", str(self._paper_run_dir) if self._paper_run_dir else "-")
        summary.add_row("timestamp", ts or "-")
        summary.add_row("strategy", self.state.strategy)
        summary.add_row("symbols", self.state.symbols)
        summary.add_row("bar_timeframe", self.state.bar_timeframe)
        summary.add_row("paper_feed", self.state.paper_feed)
        summary.add_row("paper_rth_only", str(self.state.paper_regular_hours_only))
        summary.add_row(
            "paper_when_closed",
            str(self.state.paper_allow_trading_when_closed),
        )
        summary.add_row("paper_limit_bps", f"{self.state.paper_limit_offset_bps:.2f}")
        summary.add_row("reason", str(decision.get("reason") or "-"))
        summary.add_row(
            "targets",
            "  ".join(f"{k}={float(v):+.2f}" for k, v in targets.items()) if targets else "-",
        )
        summary.add_row(
            "positions",
            "  ".join(f"{k}={float(v):+.4f}" for k, v in positions.items()) if positions else "-",
        )
        if "equity" in decision:
            summary.add_row("equity", f"{float(decision['equity']):.2f}")
        if "cash" in decision:
            summary.add_row("cash", f"{float(decision['cash']):.2f}")

        gates = Table(show_header=False)
        gates.add_column("k", style="bold")
        gates.add_column("v")
        if "rho" in debug:
            gates.add_row("rho", f"{float(debug['rho']):.3f}")
        if "agree" in debug:
            gates.add_row("agree", str(bool(debug["agree"])))
        if "strength" in debug:
            gates.add_row("strength", f"{float(debug['strength']):.3f}")
        if "vol_ratio_max" in debug:
            gates.add_row("vol_ratio_max", f"{float(debug['vol_ratio_max']):.3f}")
        if "chosen" in debug:
            gates.add_row("chosen", str(debug["chosen"]))
        if "chosen_netEdge_bps" in debug:
            gates.add_row("netEdge_bps", f"{float(debug['chosen_netEdge_bps']):.3f}")
        if "dir" in debug:
            gates.add_row("dir", str(int(debug["dir"])))
        if "expMove_bps" in debug and isinstance(debug["expMove_bps"], dict):
            gates.add_row(
                "expMove_bps",
                "  ".join(f"{k}={float(v):.2f}" for k, v in debug["expMove_bps"].items()),
            )
        if "costRT_bps" in debug and isinstance(debug["costRT_bps"], dict):
            gates.add_row(
                "costRT_bps",
                "  ".join(f"{k}={float(v):.2f}" for k, v in debug["costRT_bps"].items()),
            )
        if "netEdge_bps" in debug and isinstance(debug["netEdge_bps"], dict):
            gates.add_row(
                "netEdge_bps_all",
                "  ".join(f"{k}={float(v):.2f}" for k, v in debug["netEdge_bps"].items()),
            )

        if "spy" in debug and isinstance(debug["spy"], dict):
            spy = debug["spy"]
            gates.add_row(
                "SPY score/m/v",
                f"{float(spy.get('score', 0.0)):+.3f}  {float(spy.get('m', 0.0)):+.6f}  {float(spy.get('v', 0.0)):.6f}",
            )
            gates.add_row(
                "SPY vwap_dev/volr",
                f"{float(spy.get('vwap_dev', 0.0)):+.4f}  {float(spy.get('vol_ratio', 0.0)):.3f}",
            )
        if "qqq" in debug and isinstance(debug["qqq"], dict):
            qqq = debug["qqq"]
            gates.add_row(
                "QQQ score/m/v",
                f"{float(qqq.get('score', 0.0)):+.3f}  {float(qqq.get('m', 0.0)):+.6f}  {float(qqq.get('v', 0.0)):.6f}",
            )
            gates.add_row(
                "QQQ vwap_dev/volr",
                f"{float(qqq.get('vwap_dev', 0.0)):+.4f}  {float(qqq.get('vol_ratio', 0.0)):.3f}",
            )

        last_order = None
        last_fill = None
        if self._paper_run_dir:
            last_order = self._tail_last_jsonl(self._paper_run_dir / "orders.jsonl")
            last_fill = self._tail_last_jsonl(self._paper_run_dir / "fills.jsonl")

        tape = Table(show_header=False)
        tape.add_column("k", style="bold")
        tape.add_column("v")
        if last_order:
            tape.add_row(
                "last_order",
                f"{last_order.get('symbol')} {last_order.get('side')} qty={last_order.get('qty')} id={last_order.get('order_id')}",
            )
        if last_fill:
            tape.add_row(
                "last_fill",
                f"{last_fill.get('symbol')} {last_fill.get('side')} qty={last_fill.get('filled_qty')} px={last_fill.get('filled_avg_price')} status={last_fill.get('status')}",
            )

        results.update(Group(summary, "", gates, "", tape))

    def _refresh_live_view(self) -> None:
        if self._paper_run_dir is None:
            return
        decision = self._tail_last_jsonl(self._paper_run_dir / "decisions.jsonl")
        if not decision:
            return
        ts = str(decision.get("timestamp", ""))
        if ts and ts == self._last_live_decision_ts:
            return
        self._last_live_decision_ts = ts
        self._render_paper_live(decision)

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

        symbols = [s.strip().upper() for s in self.state.symbols.split(",") if s.strip()]
        if not symbols:
            raise ValueError("symbols not set")

        tf = parse_bar_timeframe(self.state.bar_timeframe)
        start_dt = parse_iso_datetime(self.state.start) if self.state.start else None
        end_dt = parse_iso_datetime(self.state.end) if self.state.end else None

        alpaca_settings = None
        if self.state.data_source == "alpaca":
            if self.state.timeframe:
                try:
                    delta = _parse_relative_timeframe(self.state.timeframe)
                except Exception as exc:
                    raise ValueError(f"invalid timeframe: {self.state.timeframe}") from exc
                end_dt = now_ny()
                start_dt = end_dt - delta
                self.state.start = start_dt.isoformat()
                self.state.end = end_dt.isoformat()
            if not (start_dt and end_dt):
                raise ValueError("start/end required for alpaca data")
            alpaca_settings = get_alpaca_settings(require_keys=True)

        csv_path = Path(self.state.csv_path) if self.state.csv_path else None
        csv_dir = csv_path if (csv_path and csv_path.is_dir()) else None
        csv_file = csv_path if (csv_path and csv_path.is_file()) else None

        universe = load_universe_bars(
            symbols=symbols,
            data_source=self.state.data_source,
            timeframe=tf,
            start=start_dt,
            end=end_dt,
            csv_path=csv_file,
            csv_dir=csv_dir,
            alpaca_settings=alpaca_settings,
            alpaca_feed=self.state.alpaca_feed,
        )
        data_hint = universe.hint
        bars_by_symbol = universe.bars_by_symbol

        if self.state.timeframe and self.state.data_source != "alpaca":
            delta = _parse_relative_timeframe(self.state.timeframe)
            end_dt = min(pd.Timestamp(df.index[-1]).to_pydatetime() for df in bars_by_symbol.values())
            start_dt = end_dt - delta
            for s in list(bars_by_symbol):
                df = bars_by_symbol[s]
                df = df[(df.index >= start_dt) & (df.index <= end_dt)]
                bars_by_symbol[s] = df

        common_index: Optional[pd.DatetimeIndex] = None
        for s in symbols:
            idx = bars_by_symbol[s].index
            common_index = idx if common_index is None else common_index.intersection(idx)
        if common_index is None or len(common_index) < 3:
            raise ValueError("backtest window has too few aligned bars")
        common_index = common_index.sort_values()

        strat = build_strategy(
            name=self.state.strategy,
            params_path=None,
            symbols=symbols,
            fast_window=self.state.fast_window,
            slow_window=self.state.slow_window,
            params=self.state.strategy_params.get(self.state.strategy),
        )

        cfg = BacktestConfig(
            symbols=symbols,
            initial_cash=self.state.initial_cash,
            max_position_notional_usd=self.state.max_position_notional_usd,
            slippage_bps=self.state.slippage_bps,
            allow_short=self.state.allow_short,
        )

        run_backtest(bars_by_symbol=bars_by_symbol, strategy=strat, cfg=cfg, run_dir=run_dir)

        metrics = json.loads((run_dir / "metrics.json").read_text())
        equity_curve = pd.read_csv(
            run_dir / "equity_curve.csv", parse_dates=["timestamp"]
        )
        final_equity = float(equity_curve["equity"].iloc[-1])
        bar_minutes = _infer_bar_minutes(common_index)
        sessions = int(pd.Series(common_index.date).nunique())
        duration = pd.Timestamp(common_index[-1]) - pd.Timestamp(common_index[0])
        trades = pd.read_csv(run_dir / "trades.csv")
        gross_notional = float(trades["notional"].sum()) if len(trades) else 0.0

        summary = Table(title="Backtest summary", show_header=False)
        summary.add_column("k", style="bold")
        summary.add_column("v")
        summary.add_row("run_dir", str(run_dir))
        summary.add_row("symbols", ",".join(symbols))
        summary.add_row("data", f"{self.state.data_source} ({data_hint})")
        summary.add_row(
            "window",
            f"{(start_dt.isoformat() if start_dt else common_index[0].isoformat())} -> {(end_dt.isoformat() if end_dt else common_index[-1].isoformat())}  |  bars={len(common_index)}  sessions={sessions}  bar={bar_minutes:.2f}m",
        )
        summary.add_row("duration", str(duration))
        summary.add_row(
            "strategy",
            (
                f"{self.state.strategy} (fast={self.state.fast_window} slow={self.state.slow_window})"
                if self.state.strategy in {"ma_crossover", "ema_crossover"}
                else self.state.strategy
            ),
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
        self._paper_run_dir = run_dir
        self._last_live_decision_ts = None

        placeholder = Table(show_header=False)
        placeholder.add_column("k", style="bold")
        placeholder.add_column("v")
        placeholder.add_row("status", "starting paper loop…")
        placeholder.add_row("run_dir", str(run_dir))
        placeholder.add_row("symbols", self.state.symbols)
        placeholder.add_row("bar_timeframe", self.state.bar_timeframe)
        placeholder.add_row("strategy", self.state.strategy)
        results = self.query_one("#results", Static)
        results.border_title = "Paper (live)"
        results.update(placeholder)

        strat = build_strategy(
            name=self.state.strategy,
            params_path=None,
            symbols=[s.strip().upper() for s in self.state.symbols.split(",") if s.strip()],
            fast_window=self.state.fast_window,
            slow_window=self.state.slow_window,
            params=self.state.strategy_params.get(self.state.strategy),
        )
        cfg = PaperConfig(
            symbols=[s.strip().upper() for s in self.state.symbols.split(",") if s.strip()],
            bar_timeframe=self.state.bar_timeframe,
            alpaca_feed=self.state.paper_feed,
            lookback_bars=self.state.paper_lookback_bars,
            poll_seconds=self.state.paper_poll_seconds,
            max_position_notional_usd=self.state.paper_max_position_notional_usd,
            allow_short=self.state.allow_short,
            regular_hours_only=self.state.paper_regular_hours_only,
            allow_trading_when_closed=self.state.paper_allow_trading_when_closed,
            limit_offset_bps=self.state.paper_limit_offset_bps,
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
