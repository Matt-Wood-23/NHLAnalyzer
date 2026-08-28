"""
Discord notifications for NHL predictions.

Two modes:
  1. Webhook mode (simple, no bot token needed):
       POST prediction embeds to a Discord webhook URL.
       Requires: DISCORD_WEBHOOK_URL env var.

  2. Bot mode (persistent, slash commands):
       Full py-cord bot with /predictions and /props commands.
       Requires: DISCORD_BOT_TOKEN env var.

Usage:
    # One-shot webhook post for today's games
    python -m bot.discord_bot --webhook

    # One-shot webhook post with SOG props too
    python -m bot.discord_bot --webhook --props

    # One-shot webhook post for a specific date
    python -m bot.discord_bot --webhook --date 2026-03-10

    # Persistent bot (slash commands)
    python -m bot.discord_bot --bot
"""

import argparse
import logging
import os
from datetime import date
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
from dotenv import load_dotenv

from config.season import current_season as _current_season
from pipeline.evaluate_history import confidence_label

logger = logging.getLogger(__name__)

# Discord embed colours
_COLOR_HOME  = 0x1F8EF1   # blue  — home team favoured
_COLOR_AWAY  = 0xE74C3C   # red   — away team favoured
_COLOR_EVEN  = 0x95A5A6   # grey  — roughly even
_COLOR_PROPS = 0xF1C40F   # gold  — SOG props
_COLOR_ELO   = 0x9B59B6   # purple — ELO rankings
_COLOR_HIST  = 0x3498DB   # blue   — prediction history
_COLOR_RECAP = 0x2ECC71   # green  — team recap

_TEAM_EMOJI: dict[str, str] = {
    "ANA": "🦆", "BOS": "🐻", "BUF": "🦬", "CAR": "🌀", "CBJ": "💥",
    "CGY": "🔥", "CHI": "🪶", "COL": "🏔️", "DAL": "⭐", "DET": "🐙",
    "EDM": "🛢️", "FLA": "🐆", "LAK": "👑", "MIN": "🌲", "MTL": "🔵",
    "NJD": "😈", "NSH": "🎸", "NYI": "🗽", "NYR": "🗽", "OTT": "⚔️",
    "PHI": "🟠", "PIT": "🐧", "SEA": "🦑", "SJS": "🦈", "STL": "🎵",
    "TBL": "⚡", "TOR": "🍁", "UTA": "🏔️", "VAN": "🐳", "VGK": "♟️",
    "WPG": "✈️", "WSH": "🦅",
}


def _team_str(team: str) -> str:
    return f"{_TEAM_EMOJI.get(team, '🏒')} {team}"


def _prob_bar(prob: float, width: int = 10) -> str:
    filled = round(prob * width)
    return "█" * filled + "░" * (width - filled)


def _confidence_label(prob: float) -> str:
    """Return pick strength label for the favoured side's probability.

    Delegates to pipeline.evaluate_history so the labels shown on a live pick
    and the tiers reported by /history are the same thresholds — otherwise the
    history breakdown would not actually be checking these labels.
    """
    return confidence_label(prob)


# ---------------------------------------------------------------------------
# Embed builders — game predictions
# ---------------------------------------------------------------------------

def format_daily_header(pred_date: date, n_games: int) -> dict:
    return {
        "title": f"🏒 NHL Predictions — {pred_date.strftime('%A, %B %d %Y')}",
        "description": f"{n_games} regular-season game{'s' if n_games != 1 else ''} today",
        "color": 0x2ECC71,
    }


def _build_odds_index(odds_list: list[dict]) -> dict[tuple[str, str], dict]:
    """Index Action Network odds by (home_team, away_team) for fast lookup."""
    return {(g["home_team"], g["away_team"]): g for g in odds_list}


def _edge_str(model_prob: float, market_prob: float, team: str) -> str:
    """Return a formatted edge string, e.g. '+5.7% ↑' or '-2.1% ↓'."""
    edge = model_prob - market_prob
    arrow = "↑" if edge >= 0 else "↓"
    sign  = "+" if edge >= 0 else ""
    return f"{_team_str(team)} {sign}{edge:.1%} {arrow}"


def format_game_embed(row: dict, odds: Optional[dict] = None) -> dict:
    """
    Build a Discord embed dict for a single game prediction.
    Expected row keys: game_id, home_team, away_team, prob_home_win,
                       home_back_to_back, away_back_to_back, rest_advantage.
    Optional odds: Action Network game dict (from fetch_odds).
    """
    home   = row["home_team"]
    away   = row["away_team"]
    p_home = float(row["prob_home_win"])
    p_away = 1.0 - p_home

    if p_home > 0.55:
        color = _COLOR_HOME
    elif p_away > 0.55:
        color = _COLOR_AWAY
    else:
        color = _COLOR_EVEN

    p_fav = max(p_home, p_away)
    pick  = home if p_home >= p_away else away
    conf  = _confidence_label(p_fav)
    pick_line = f"**Pick: {_team_str(pick)}** — {conf} ({p_fav:.1%})\n\n"

    # ELO ratings
    home_elo = row.get("home_elo")
    away_elo = row.get("away_elo")
    if home_elo is not None and away_elo is not None:
        elo_str = f"**ELO:** {_team_str(home)} {int(home_elo)} · {_team_str(away)} {int(away_elo)}"
    else:
        elo_str = ""

    flags = []
    if row.get("home_back_to_back"):
        flags.append(f"{_team_str(home)} on B2B")
    if row.get("away_back_to_back"):
        flags.append(f"{_team_str(away)} on B2B")
    rest_adv = row.get("rest_advantage")
    if rest_adv is not None and abs(float(rest_adv)) >= 1:
        adv_team = home if float(rest_adv) > 0 else away
        flags.append(f"{_team_str(adv_team)} rest advantage ({int(abs(float(rest_adv)))}d)")

    context_str = " · ".join(flags) if flags else "—"

    description = (
        f"**{_team_str(away)}** @ **{_team_str(home)}**\n\n"
        f"{pick_line}"
        f"**{_team_str(home)} win:**  {p_home:.1%}  `{_prob_bar(p_home)}`\n"
        f"**{_team_str(away)} win:**  {p_away:.1%}  `{_prob_bar(p_away)}`\n\n"
    )
    if elo_str:
        description += f"{elo_str}\n"
    description += f"**Context:** {context_str}"

    # Market odds block
    if odds and odds.get("consensus"):
        con = odds["consensus"]
        mkt_home = con["prob_home_win"]
        mkt_away = con["prob_away_win"]
        n_books  = con["n_books"]
        total    = con.get("total")

        # Best available moneyline (first book with data)
        home_ml = away_ml = None
        for b in odds.get("books", {}).values():
            if b.get("home_ml") is not None and b.get("away_ml") is not None:
                home_ml = b["home_ml"]
                away_ml = b["away_ml"]
                break

        ml_str    = f"ML `{home_ml:+d}` / `{away_ml:+d}`  · " if home_ml else ""
        total_str = f"O/U `{total}`  · " if total else ""
        edge_home = _edge_str(p_home, mkt_home, home)
        edge_away = _edge_str(p_away, mkt_away, away)

        description += (
            f"\n\n**Market** ({n_books} books, no-vig):\n"
            f"{_team_str(home)} `{mkt_home:.1%}` / {_team_str(away)} `{mkt_away:.1%}`\n"
            f"{ml_str}{total_str}\n"
            f"**Edge:** {edge_home}  ·  {edge_away}"
        )

    return {
        "title": f"{_team_str(away)} @ {_team_str(home)}",
        "description": description,
        "color": color,
        "footer": {"text": "NHL Analyzer • ML v3"},
    }


def format_odds_embed(game: dict) -> dict:
    """Build a Discord embed showing lines for a single game (no model output)."""
    home = game["home_team"]
    away = game["away_team"]
    con  = game.get("consensus", {})

    lines = [f"**{_team_str(away)} @ {_team_str(home)}**\n"]

    if con:
        mkt_home = con["prob_home_win"]
        mkt_away = con["prob_away_win"]
        total    = con.get("total")
        lines.append(
            f"**Consensus ({con['n_books']} books):** "
            f"{_team_str(home)} `{mkt_home:.1%}` / {_team_str(away)} `{mkt_away:.1%}`"
            + (f"  ·  O/U `{total}`" if total else "")
        )

    if game.get("books"):
        lines.append("")
        lines.append("`{:<12} {:>7} {:>7} {:>5} {:>5}`".format("Book", home, away, "O/U", "Sprd"))
        lines.append("`" + "-" * 38 + "`")
        for book, b in game["books"].items():
            home_ml   = f"{b['home_ml']:+d}"  if b.get("home_ml")    is not None else "—"
            away_ml   = f"{b['away_ml']:+d}"  if b.get("away_ml")    is not None else "—"
            total_str = str(b["total"])        if b.get("total")      is not None else "—"
            sprd_str  = f"{b['home_spread']:+g}" if b.get("home_spread") is not None else "—"
            lines.append("`{:<12} {:>7} {:>7} {:>5} {:>5}`".format(
                book[:12], home_ml, away_ml, total_str, sprd_str
            ))

    start_local = game["start_time"].strftime("%I:%M %p UTC")
    return {
        "title": f"{_team_str(away)} @ {_team_str(home)}",
        "description": "\n".join(lines),
        "color": _COLOR_EVEN,
        "footer": {"text": f"Action Network  ·  {start_local}  ·  NHL Analyzer"},
    }


# ---------------------------------------------------------------------------
# Embed builders — SOG props
# ---------------------------------------------------------------------------

def format_props_header(pred_date: date, n_players: int) -> dict:
    return {
        "title": f"🎯 SOG Projections — {pred_date.strftime('%A, %B %d %Y')}",
        "description": (
            f"Model projections for {n_players} skater"
            f"{'s' if n_players != 1 else ''} \u00b7 No book lines \u00b7 Research use only"
        ),
        "color": _COLOR_PROPS,
    }


def format_props_embeds(props_df, pred_date: date) -> list[dict]:
    """
    Build a list of Discord embed dicts for SOG props.

    Layout: compact monospace text table, 10 players per embed.
    Returns: [header_embed, data_embed_1, data_embed_2, ...]
    """
    if props_df is None or len(props_df) == 0:
        return []

    rows   = props_df.to_dict("records")
    embeds = [format_props_header(pred_date, len(rows))]

    col_header = "`{:<22} {:>3} {:>4} {:>5} {:>5} {:>5}`".format(
        "Player", "Pos", "Opp", "xSOG", "L10", "TOI"
    )
    separator = "`" + "-" * 48 + "`"

    BATCH = 10
    for i in range(0, len(rows), BATCH):
        chunk = rows[i : i + BATCH]
        lines = [col_header, separator]
        for r in chunk:
            name = str(r.get("player_name", ""))[:22]
            pos  = str(r.get("position",    ""))[:3]
            opp  = str(r.get("opponent",    ""))[:4]
            xsog = f"{r.get('expected_sog',  0.0):.2f}"
            l10  = f"{r.get('sog_l10',       0.0):.1f}"
            toi  = f"{r.get('toi_min_l10',   0.0):.1f}"
            lines.append(
                "`{:<22} {:>3} {:>4} {:>5} {:>5} {:>5}`".format(
                    name, pos, opp, xsog, l10, toi
                )
            )
        embeds.append({
            "description": "\n".join(lines),
            "color": _COLOR_PROPS,
            "footer": {"text": "NHL Analyzer \u2022 SOG Model \u00b7 Model projections only"},
        })

    return embeds


# ---------------------------------------------------------------------------
# Embed builders — team recap
# ---------------------------------------------------------------------------

_PARQUET_DIR = Path(__file__).parent.parent / "data" / "parquet"


def build_team_recap(team: str, n_games: int = 10) -> dict:
    """
    Pull last N games for a team from parquet and return a structured recap dict.

    Returns:
        {
          "team": str,
          "games": list[dict],   # per-game rows, chronological
          "record": (W, L),
          "avg_gf": float, "avg_ga": float,
          "avg_xgf_pct": float,
          "avg_shots_for": float, "avg_shots_against": float,
          "goalie_name": str | None,
          "goalie_sv_pct_l10": float | None,
        }
    """
    mp_path = _PARQUET_DIR / "moneypuck_team_game_stats.parquet"
    if not mp_path.exists():
        return {}

    mp = pd.read_parquet(mp_path)
    team_games = mp[mp["team"] == team.upper()].copy()
    if team_games.empty:
        return {}

    team_games["game_num"] = team_games["game_id"].str[-4:].astype(int)
    team_games = team_games.sort_values(["season", "game_num"])
    recent = team_games.tail(n_games)

    eps = 1e-9
    recent = recent.copy()
    recent["won"] = (recent["goals_for"] > recent["goals_against"]).astype(int)
    recent["xgf_pct"] = recent["xg_for"] / (recent["xg_for"] + recent["xg_against"] + eps)

    games = []
    for _, row in recent.iterrows():
        venue = "vs" if row["is_home"] else "@"
        result = "W" if row["won"] else "L"
        games.append({
            "game_id":   row["game_id"],
            "result":    result,
            "venue":     venue,
            "opponent":  row["opp_team"],
            "gf":        int(row["goals_for"]),
            "ga":        int(row["goals_against"]),
            "sf":        int(row.get("shots_for", 0) or 0),
            "sa":        int(row.get("shots_against", 0) or 0),
            "xgf_pct":   float(row["xgf_pct"]),
        })

    w = int(recent["won"].sum())
    l = n_games - w
    avg_gf = float(recent["goals_for"].mean())
    avg_ga = float(recent["goals_against"].mean())
    avg_xgf = float(recent["xgf_pct"].mean())
    avg_sf = float(recent["shots_for"].mean()) if "shots_for" in recent.columns else 0.0
    avg_sa = float(recent["shots_against"].mean()) if "shots_against" in recent.columns else 0.0

    # Goalie: most recent starter's last 10 save%
    goalie_name = None
    goalie_sv = None
    g_path = _PARQUET_DIR / "goalie_game_stats.parquet"
    if g_path.exists():
        g_df = pd.read_parquet(g_path)
        starters = g_df[(g_df["team"] == team.upper()) & g_df["is_starter"]].copy()
        if not starters.empty:
            starters = starters.sort_values(["season", "game_num"])
            last_starter = starters.iloc[-1]
            goalie_name = last_starter.get("goalie_name", None)
            recent_starts = starters[starters["goalie_id"] == last_starter["goalie_id"]].tail(10)
            if len(recent_starts) > 0:
                goalie_sv = float(recent_starts["save_pct"].mean())

    return {
        "team": team.upper(),
        "games": games,
        "record": (w, l),
        "avg_gf": avg_gf,
        "avg_ga": avg_ga,
        "avg_xgf_pct": avg_xgf,
        "avg_shots_for": avg_sf,
        "avg_shots_against": avg_sa,
        "goalie_name": goalie_name,
        "goalie_sv_pct_l10": goalie_sv,
    }


def format_recap_embed(recap: dict) -> list[dict]:
    """Build Discord embeds for a team recent-form recap."""
    if not recap:
        return [{"description": "Team not found or no data available.", "color": _COLOR_RECAP}]

    team = recap["team"]
    emoji = _TEAM_EMOJI.get(team, "🏒")
    w, l = recap["record"]
    n = w + l

    # Per-game result lines
    result_lines = []
    for g in recap["games"]:
        icon = "🟢" if g["result"] == "W" else "🔴"
        result_lines.append(
            f"{icon} {g['result']} {g['gf']}-{g['ga']} {g['venue']} {_TEAM_EMOJI.get(g['opponent'], '')} {g['opponent']}"
            f"  `SF {g['sf']} SA {g['sa']}`"
        )

    # Summary stats block
    xgf_pct = recap["avg_xgf_pct"]
    xgf_bar = _prob_bar(xgf_pct)
    goalie_str = ""
    if recap.get("goalie_name") and recap.get("goalie_sv_pct_l10") is not None:
        goalie_str = f"\n**Last starter:** {recap['goalie_name']} · SV% L10: `{recap['goalie_sv_pct_l10']:.3f}`"

    description = (
        f"**{emoji} {team} — Last {n} Games**\n"
        f"**Record:** {w}–{l}  ·  "
        f"**GF/G:** {recap['avg_gf']:.2f}  ·  "
        f"**GA/G:** {recap['avg_ga']:.2f}\n"
        f"**xGF%:** {xgf_pct:.1%}  `{xgf_bar}`  ·  "
        f"**SF/G:** {recap['avg_shots_for']:.1f}  **SA/G:** {recap['avg_shots_against']:.1f}"
        f"{goalie_str}\n\n"
        + "\n".join(result_lines)
    )

    return [{
        "title": f"{emoji} {team} Recent Form",
        "description": description,
        "color": _COLOR_RECAP,
        "footer": {"text": f"NHL Analyzer • Last {n} games · MoneyPuck data"},
    }]


# ---------------------------------------------------------------------------
# Embed builders — ELO rankings
# ---------------------------------------------------------------------------


def format_elo_embeds(top_n: int = 32) -> list[dict]:
    """Build embeds showing current ELO rankings from saved parquet."""
    path = _PARQUET_DIR / "elo_ratings.parquet"
    if not path.exists():
        return [{"description": "No ELO ratings found. Run `python -m pipeline.backfill` first.", "color": _COLOR_ELO}]

    elo_df = pd.read_parquet(path).sort_values("elo", ascending=False).head(top_n)

    lines = ["`{:<4} {:<4} {:>6}`".format("Rank", "Team", "ELO")]
    lines.append("`" + "-" * 16 + "`")
    for i, (_, row) in enumerate(elo_df.iterrows(), 1):
        team = str(row["team"])
        elo = int(row["elo"])
        emoji = _TEAM_EMOJI.get(team, "🏒")
        lines.append("`{:<4}`{} `{:<4} {:>5}`".format(f"#{i}", emoji, team, elo))

    mean_elo = int(elo_df["elo"].mean())
    top_elo = int(elo_df.iloc[0]["elo"])
    bot_elo = int(elo_df.iloc[-1]["elo"])

    embeds = [{
        "title": "📊 NHL ELO Rankings",
        "description": "\n".join(lines),
        "color": _COLOR_ELO,
        "footer": {"text": f"Mean: {mean_elo} · Range: {bot_elo}–{top_elo} · NHL Analyzer"},
    }]
    return embeds


# ---------------------------------------------------------------------------
# Embed builders — prediction history / accuracy
# ---------------------------------------------------------------------------

def _pct(value: float) -> str:
    return f"{value:.1%}"


def _pts_delta(value: float, baseline: float) -> str:
    """Signed gap in percentage points, e.g. "+2.4 pts" / "-1.4 pts"."""
    return f"{(value - baseline) * 100:+.1f} pts"


def _brier_delta(value: float, baseline: float) -> str:
    """Gap against a Brier baseline, worded because lower is better."""
    gap = baseline - value
    return f"{abs(gap):.4f} {'better' if gap >= 0 else 'worse'}"


def format_history_embed(
    season: str | None = None,
    last_n: int = 25,
) -> dict:
    """Build an embed summarizing prediction accuracy from local history.

    Scoped to one season by default.  Pooling seasons into a single lifetime
    number hides the thing you actually want to see — whether the model is
    working *this* year — and mixes results from different trained models.
    """
    from pipeline.evaluate_history import (
        calibration_table, clv_summary, confidence_breakdown, coverage_warning,
        filter_season, load_history, market_comparison, recent_form, summarize,
    )

    hist = load_history()
    if hist.empty:
        return {
            "title": "📈 Prediction History",
            "description": (
                "No prediction history yet. Predictions are logged "
                "automatically when the pipeline runs."
            ),
            "color": _COLOR_HIST,
        }

    if season is None:
        season = _current_season()
        if filter_season(hist, season).empty:
            logged = sorted(hist["season"].dropna().unique())
            if logged:
                season = logged[-1]

    scoped = filter_season(hist, season)
    if scoped.empty:
        available = sorted(hist["season"].dropna().unique())
        return {
            "title": "📈 Prediction History",
            "description": (
                f"No predictions logged for **{season}**.\n"
                f"Available: {', '.join(available) if available else 'none'}"
            ),
            "color": _COLOR_HIST,
        }

    scope_label = "all seasons" if str(season).lower() == "all" else season
    stats = summarize(scoped)

    if not stats["n"]:
        return {
            "title": f"📈 Prediction History — {scope_label}",
            "description": (
                f"**{stats['n_logged']}** predictions logged, none scored yet.\n"
                "Outcomes are filled in once the games are played."
            ),
            "color": _COLOR_HIST,
        }

    # ---- headline ----
    acc_gap = _pts_delta(stats["accuracy"], stats["baseline_accuracy"])
    brier_gap = _brier_delta(stats["brier"], stats["baseline_brier"])
    beating = stats["accuracy"] >= stats["baseline_accuracy"]

    description = (
        f"**Record:** {stats['wins']}-{stats['losses']} "
        f"({stats['n']} of {stats['n_logged']} logged predictions scored)\n\n"
        f"{'✅' if beating else '⚠️'} **Accuracy:** {_pct(stats['accuracy'])}  "
        f"· always-home {_pct(stats['baseline_accuracy'])} ({acc_gap})\n"
        f"**Brier:** {stats['brier']:.4f}  "
        f"· no-skill {stats['baseline_brier']:.4f} ({brier_gap})\n"
    )
    if not beating:
        description += (
            "\n*Picking the home team every night would have done better "
            "over this sample.*\n"
        )

    fields = []

    # ---- pick strength: does the bot's own label mean anything? ----
    tiers = confidence_breakdown(scoped)
    if not tiers.empty:
        lines = [f"{'Tier':<12}{'N':>4}{'Acc':>8}{'Brier':>9}"]
        for tier, row in tiers.iterrows():
            lines.append(
                f"{tier:<12}{int(row['n']):>4}"
                f"{row['accuracy']:>7.1%}{row['brier']:>9.4f}"
            )
        fields.append({
            "name": "By pick strength",
            "value": "```\n" + "\n".join(lines) + "\n```",
            "inline": False,
        })

    # ---- calibration ----
    cal = calibration_table(scoped)
    if not cal.empty:
        lines = [f"{'Bucket':<9}{'N':>4}{'Said':>8}{'Actual':>9}"]
        for bucket, row in cal.iterrows():
            lines.append(
                f"{str(bucket):<9}{int(row['n']):>4}"
                f"{row['predicted']:>8.0%}{row['actual']:>9.0%}"
            )
        fields.append({
            "name": "Calibration (said vs actual home win rate)",
            "value": "```\n" + "\n".join(lines) + "\n```",
            "inline": False,
        })

    # ---- vs the market: the benchmark that decides whether this is worth it ----
    mkt = market_comparison(scoped)
    if mkt["n"]:
        gap = mkt["model_brier"] - mkt["market_brier"]
        lines = [
            f"{'':<10}{'model':>9}{'market':>9}",
            f"{'Brier':<10}{mkt['model_brier']:>9.4f}{mkt['market_brier']:>9.4f}",
            f"{'Accuracy':<10}{mkt['model_accuracy']:>8.1%}{mkt['market_accuracy']:>9.1%}",
        ]
        note = (
            f"{'ahead by' if gap < 0 else 'behind by'} {abs(gap):.4f} Brier "
            f"on {mkt['n']} games"
        )
        clv = clv_summary(scoped)
        if clv["n"]:
            note += (
                f"\nLine moved our way {clv['beat_close_rate']:.0%} of the time "
                f"({clv['mean_clv']:+.1%} avg)"
            )
        fields.append({
            "name": "vs the closing line",
            "value": "```\n" + "\n".join(lines) + "\n```" + note,
            "inline": False,
        })

    # ---- recent form ----
    recent = recent_form(scoped, last_n=last_n)
    if recent["n"] >= 5:
        fields.append({
            "name": f"Last {recent['n']}",
            "value": (
                f"{recent['wins']}-{recent['losses']} · "
                f"{_pct(recent['accuracy'])} accuracy · "
                f"{recent['brier']:.4f} Brier"
            ),
            "inline": False,
        })

    footer_bits = []
    if "model_name" in scoped.columns:
        models = ", ".join(sorted(scoped["model_name"].dropna().unique()))
        if models:
            footer_bits.append(models)
    warning = coverage_warning(scoped)
    if warning:
        footer_bits.append(warning)
    footer_bits.append("small samples — read the trend, not the decimal")

    return {
        "title": f"📈 Prediction History — {scope_label}",
        "description": description,
        "color": _COLOR_HIST,
        "fields": fields,
        "footer": {"text": " • ".join(footer_bits)},
    }


# ---------------------------------------------------------------------------
# Webhook posting
# ---------------------------------------------------------------------------

def post_webhook(webhook_url: str, embeds: list[dict]) -> bool:
    """POST embeds to a Discord webhook (max 10 per request)."""
    try:
        for i in range(0, len(embeds), 10):
            chunk = embeds[i : i + 10]
            resp = httpx.post(webhook_url, json={"embeds": chunk}, timeout=15)
            resp.raise_for_status()
        logger.info("Webhook POST successful (%d embeds)", len(embeds))
        return True
    except httpx.HTTPStatusError as e:
        logger.error(
            "Webhook POST failed %s: %s", e.response.status_code, e.response.text
        )
        return False
    except Exception as e:
        logger.error("Webhook POST error: %s", e)
        return False


def send_predictions_webhook(
    predictions_df,
    webhook_url: str,
    pred_date: Optional[date] = None,
    props_df=None,
) -> bool:
    """Format and post daily predictions (and optionally SOG props) to a Discord webhook."""
    if pred_date is None:
        pred_date = date.today()

    # Fetch odds and build lookup index (best-effort — don't fail if unavailable)
    odds_index = {}
    try:
        from ingestion.action_network import fetch_odds
        odds_index = _build_odds_index(fetch_odds(pred_date))
        logger.info("Fetched odds for %d games", len(odds_index))
    except Exception as e:
        logger.warning("Could not fetch odds: %s", e)

    # Game predictions
    if predictions_df is not None and len(predictions_df) > 0:
        rows   = predictions_df.to_dict("records")
        embeds = [format_daily_header(pred_date, len(rows))]
        embeds += [
            format_game_embed(r, odds=odds_index.get((r["home_team"], r["away_team"])))
            for r in rows
        ]
        if not post_webhook(webhook_url, embeds):
            return False
    else:
        logger.info("No game predictions to post")

    # SOG props (separate POST to avoid hitting the 10-embed/request limit)
    if props_df is not None and len(props_df) > 0:
        props_embeds = format_props_embeds(props_df, pred_date)
        if props_embeds and not post_webhook(webhook_url, props_embeds):
            return False

    return True


# ---------------------------------------------------------------------------
# Bot mode (py-cord)
# ---------------------------------------------------------------------------

def run_bot(token: str) -> None:
    """Start the py-cord slash-command bot."""
    try:
        import discord
    except ImportError:
        raise ImportError("py-cord not installed. Run: pip install 'py-cord>=2.0'")

    intents = discord.Intents.default()
    bot = discord.Bot(intents=intents)

    @bot.slash_command(name="predictions", description="Show NHL win probabilities for a given date")
    async def predictions_cmd(
        ctx: discord.ApplicationContext,
        date_str: discord.Option(str, "Date (YYYY-MM-DD)", required=False, default=""),
    ):
        await ctx.defer()
        try:
            from pipeline.live import run as live_run
            from ingestion.action_network import fetch_odds
            target = date.fromisoformat(date_str) if date_str else date.today()
            preds  = live_run(target_date=target, dry_run=True)

            if preds.empty:
                await ctx.followup.send(
                    f"No regular-season games found for {target}."
                )
                return

            # Fetch odds best-effort
            odds_index = {}
            try:
                odds_index = _build_odds_index(fetch_odds(target))
            except Exception as oe:
                logger.warning("Odds fetch failed: %s", oe)

            rows   = preds.to_dict("records")
            embeds = [format_daily_header(target, len(rows))]
            embeds += [
                format_game_embed(r, odds=odds_index.get((r["home_team"], r["away_team"])))
                for r in rows
            ]
            for i in range(0, len(embeds), 10):
                chunk = [discord.Embed.from_dict(e) for e in embeds[i : i + 10]]
                await ctx.followup.send(embeds=chunk)
        except Exception as e:
            logger.error("predictions_cmd error: %s", e)
            await ctx.followup.send(f"Error generating predictions: {e}")

    @bot.slash_command(name="odds", description="Show current NHL lines from Action Network")
    async def odds_cmd(
        ctx: discord.ApplicationContext,
        date_str: discord.Option(str, "Date (YYYY-MM-DD)", required=False, default=""),
    ):
        await ctx.defer()
        try:
            from ingestion.action_network import fetch_odds
            target = date.fromisoformat(date_str) if date_str else date.today()
            games  = fetch_odds(target)

            if not games:
                await ctx.followup.send(f"No lines found for {target}.")
                return

            embeds = [{
                "title": f"🎰 NHL Lines — {target.strftime('%A, %B %d %Y')}",
                "description": f"{len(games)} game{'s' if len(games) != 1 else ''} · Action Network",
                "color": _COLOR_EVEN,
            }]
            embeds += [format_odds_embed(g) for g in games]
            for i in range(0, len(embeds), 10):
                chunk = [discord.Embed.from_dict(e) for e in embeds[i : i + 10]]
                await ctx.followup.send(embeds=chunk)
        except Exception as e:
            logger.error("odds_cmd error: %s", e)
            await ctx.followup.send(f"Error fetching odds: {e}")

    @bot.slash_command(name="props", description="Show SOG projections for today's NHL skaters")
    async def props_cmd(
        ctx: discord.ApplicationContext,
        date_str: discord.Option(str, "Date (YYYY-MM-DD)", required=False, default=""),
        min_sog: discord.Option(float, "Minimum SOG threshold", required=False, default=1.5),
    ):
        await ctx.defer()
        try:
            from pipeline.props_live import run as props_run
            target = date.fromisoformat(date_str) if date_str else date.today()
            props  = props_run(target_date=target, min_sog=min_sog)

            if props.empty:
                await ctx.followup.send(
                    f"No SOG projections found for {target}."
                )
                return

            props_embeds = format_props_embeds(props, target)
            for i in range(0, len(props_embeds), 10):
                chunk = [discord.Embed.from_dict(e) for e in props_embeds[i : i + 10]]
                await ctx.followup.send(embeds=chunk)
        except Exception as e:
            logger.error("props_cmd error: %s", e)
            await ctx.followup.send(f"Error generating SOG projections: {e}")

    @bot.slash_command(name="recap", description="Show recent form for an NHL team (last N games)")
    async def recap_cmd(
        ctx: discord.ApplicationContext,
        team: discord.Option(str, "Team abbreviation (e.g. EDM, TOR, BOS)", required=True),
        games: discord.Option(int, "Number of games (default 10)", required=False, default=10),
    ):
        await ctx.defer()
        try:
            recap = build_team_recap(team.upper(), n_games=min(max(games, 5), 20))
            embeds = format_recap_embed(recap)
            await ctx.followup.send(
                embeds=[discord.Embed.from_dict(e) for e in embeds]
            )
        except Exception as e:
            logger.error("recap_cmd error: %s", e)
            await ctx.followup.send(f"Error generating recap: {e}")

    @bot.slash_command(name="elo", description="Show current NHL team ELO rankings")
    async def elo_cmd(ctx: discord.ApplicationContext):
        await ctx.defer()
        try:
            embeds = format_elo_embeds()
            await ctx.followup.send(
                embeds=[discord.Embed.from_dict(e) for e in embeds]
            )
        except Exception as e:
            logger.error("elo_cmd error: %s", e)
            await ctx.followup.send(f"Error loading ELO rankings: {e}")

    @bot.slash_command(
        name="history",
        description="Prediction accuracy vs baseline, calibration and pick strength",
    )
    async def history_cmd(
        ctx: discord.ApplicationContext,
        season: discord.Option(
            str, 'Season, e.g. "2025-2026", or "all" (default: current)',
            required=False, default="",
        ),
        last: discord.Option(
            int, "How many recent predictions to summarize (default 25)",
            required=False, default=25,
        ),
    ):
        await ctx.defer()
        try:
            embed = format_history_embed(
                season=season or None, last_n=max(1, last),
            )
            await ctx.followup.send(embed=discord.Embed.from_dict(embed))
        except Exception as e:
            logger.error("history_cmd error: %s", e)
            await ctx.followup.send(f"Error loading prediction history: {e}")

    @bot.event
    async def on_ready():
        logger.info("Bot ready as %s", bot.user)
        print(f"Logged in as {bot.user} — slash commands synced")

    bot.run(token)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="NHL Discord notifications")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--webhook", action="store_true",
                      help="One-shot webhook post (DISCORD_WEBHOOK_URL env var)")
    mode.add_argument("--bot",     action="store_true",
                      help="Persistent bot mode (DISCORD_BOT_TOKEN env var)")
    parser.add_argument("--date", default=None,
                        help="Date for predictions (YYYY-MM-DD). Default: today.")
    parser.add_argument("--props", action="store_true",
                        help="Also post SOG props projections (webhook mode only)")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()

    if args.webhook:
        webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
        if not webhook_url:
            raise SystemExit("Set the DISCORD_WEBHOOK_URL environment variable")

        conn = None
        db_url = os.environ.get("DATABASE_URL")
        if db_url:
            try:
                import psycopg2
                conn = psycopg2.connect(db_url)
            except Exception as e:
                logger.warning("DB connect failed: %s — running without DB", e)

        from pipeline.live import run as live_run
        preds = live_run(target_date=target, dry_run=False, conn=conn)

        props = None
        if args.props:
            from pipeline.props_live import run as props_run
            logger.info("Running SOG props pipeline...")
            props = props_run(target_date=target)

        success = send_predictions_webhook(preds, webhook_url, pred_date=target, props_df=props)

        if conn:
            conn.close()
        if not success:
            raise SystemExit(1)

        n = len(preds) if preds is not None else 0
        p = len(props) if props is not None else 0
        print(f"Posted {n} prediction(s) and {p} SOG prop(s) for {target}")

    elif args.bot:
        token = os.environ.get("DISCORD_BOT_TOKEN")
        if not token:
            raise SystemExit("Set the DISCORD_BOT_TOKEN environment variable")
        run_bot(token)
