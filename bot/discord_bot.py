"""
Discord notifications for NHL predictions.

Two modes:
  1. Webhook mode (simple, no bot token needed):
       POST prediction embeds to a Discord webhook URL.
       Requires: DISCORD_WEBHOOK_URL env var.

  2. Bot mode (persistent, slash commands):
       Full discord.py bot with /predictions and /props commands.
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

logger = logging.getLogger(__name__)

# Discord embed colours
_COLOR_HOME  = 0x1F8EF1   # blue  — home team favoured
_COLOR_AWAY  = 0xE74C3C   # red   — away team favoured
_COLOR_EVEN  = 0x95A5A6   # grey  — roughly even
_COLOR_PROPS = 0xF1C40F   # gold  — SOG props
_COLOR_ELO   = 0x9B59B6   # purple — ELO rankings
_COLOR_HIST  = 0x3498DB   # blue   — prediction history

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
    """Return pick strength label for the favoured side's probability."""
    if prob >= 0.60:
        return "Strong Pick"
    elif prob >= 0.55:
        return "Lean"
    else:
        return "Toss-Up"


# ---------------------------------------------------------------------------
# Embed builders — game predictions
# ---------------------------------------------------------------------------

def format_daily_header(pred_date: date, n_games: int) -> dict:
    return {
        "title": f"🏒 NHL Predictions — {pred_date.strftime('%A, %B %d %Y')}",
        "description": f"{n_games} regular-season game{'s' if n_games != 1 else ''} today",
        "color": 0x2ECC71,
    }


def format_game_embed(row: dict) -> dict:
    """
    Build a Discord embed dict for a single game prediction.
    Expected row keys: game_id, home_team, away_team, prob_home_win,
                       home_back_to_back, away_back_to_back, rest_advantage.
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

    return {
        "title": f"{_team_str(away)} @ {_team_str(home)}",
        "description": description,
        "color": color,
        "footer": {"text": "NHL Analyzer • ML v3"},
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
# Embed builders — ELO rankings
# ---------------------------------------------------------------------------

_PARQUET_DIR = Path(__file__).parent.parent / "data" / "parquet"
_HISTORY_DIR = Path(__file__).parent.parent / "data" / "predictions"


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

def format_history_embed(last_n: int = 50) -> dict:
    """Build an embed summarizing prediction accuracy from local history."""
    path = _HISTORY_DIR / "prediction_history.parquet"
    if not path.exists():
        return {
            "description": "No prediction history yet. Predictions are logged automatically when you run the pipeline.",
            "color": _COLOR_HIST,
        }

    # Try to backfill outcomes before displaying
    try:
        from pipeline.evaluate_history import backfill_outcomes
        hist = backfill_outcomes()
    except Exception:
        hist = pd.read_parquet(path)

    if hist.empty:
        hist = pd.read_parquet(path)

    total = len(hist)
    unique_dates = hist["predicted_at"].str[:10].nunique() if "predicted_at" in hist.columns else "?"
    models_used = ", ".join(hist["model_name"].unique()) if "model_name" in hist.columns else "unknown"

    description = (
        f"**Total predictions logged:** {total}\n"
        f"**Prediction dates:** {unique_dates}\n"
        f"**Models used:** {models_used}\n"
    )

    # Show accuracy if outcomes are available
    has_outcome = hist.get("actual_home_win")
    if has_outcome is not None and has_outcome.notna().any():
        evaluated = hist[hist["actual_home_win"].notna()]
        n_eval = len(evaluated)
        accuracy = evaluated["correct"].mean() if "correct" in evaluated.columns else None
        brier = evaluated["brier"].mean() if "brier" in evaluated.columns else None

        description += f"\n**Evaluated:** {n_eval} / {total} games\n"
        if accuracy is not None:
            description += f"**Accuracy:** {accuracy:.1%}\n"
        if brier is not None:
            description += f"**Brier score:** {brier:.4f}\n"

        # Recent trend
        if n_eval >= 10 and "correct" in evaluated.columns:
            recent = evaluated.tail(last_n)
            r_acc = recent["correct"].mean()
            r_brier = recent["brier"].mean() if "brier" in recent.columns else None
            description += f"\n**Last {len(recent)}:** {r_acc:.1%} accuracy"
            if r_brier is not None:
                description += f", {r_brier:.4f} Brier"
            description += "\n"
    else:
        description += (
            "\n*No outcomes yet. Run `python -m pipeline.evaluate_history` "
            "after games complete to see accuracy.*"
        )

    return {
        "title": "📈 Prediction History",
        "description": description,
        "color": _COLOR_HIST,
        "footer": {"text": "NHL Analyzer • Prediction Tracker"},
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

    # Game predictions
    if predictions_df is not None and len(predictions_df) > 0:
        rows   = predictions_df.to_dict("records")
        embeds = [format_daily_header(pred_date, len(rows))]
        embeds += [format_game_embed(r) for r in rows]
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
# Bot mode (discord.py)
# ---------------------------------------------------------------------------

def run_bot(token: str) -> None:
    """Start the discord.py slash-command bot."""
    try:
        import discord
        from discord import app_commands
    except ImportError:
        raise ImportError("discord.py not installed. Run: pip install 'discord.py>=2.0'")

    intents = discord.Intents.default()
    client  = discord.Client(intents=intents)
    tree    = app_commands.CommandTree(client)

    @tree.command(name="predictions", description="Show NHL win probabilities for a given date")
    async def predictions_cmd(interaction: discord.Interaction, date_str: str = ""):
        await interaction.response.defer()
        try:
            from pipeline.live import run as live_run
            target = date.fromisoformat(date_str) if date_str else date.today()
            preds = live_run(target_date=target, dry_run=True)

            if preds.empty:
                await interaction.followup.send(
                    f"No regular-season games found for {target}."
                )
                return

            rows   = preds.to_dict("records")
            embeds = [format_daily_header(target, len(rows))]
            embeds += [format_game_embed(r) for r in rows]
            await interaction.followup.send(
                embeds=[discord.Embed.from_dict(e) for e in embeds[:10]]
            )
        except Exception as e:
            logger.error("predictions_cmd error: %s", e)
            await interaction.followup.send(f"Error generating predictions: {e}")

    @tree.command(name="props", description="Show SOG projections for today's NHL skaters")
    async def props_cmd(
        interaction: discord.Interaction,
        date_str: str = "",
        min_sog: float = 1.5,
    ):
        await interaction.response.defer()
        try:
            from pipeline.props_live import run as props_run
            target = date.fromisoformat(date_str) if date_str else date.today()
            props  = props_run(target_date=target, min_sog=min_sog)

            if props.empty:
                await interaction.followup.send(
                    f"No SOG projections found for {target}."
                )
                return

            props_embeds = format_props_embeds(props, target)
            # Send in chunks of 10 (each chunk becomes a separate Discord message)
            for i in range(0, len(props_embeds), 10):
                chunk = [discord.Embed.from_dict(e) for e in props_embeds[i : i + 10]]
                await interaction.followup.send(embeds=chunk)
        except Exception as e:
            logger.error("props_cmd error: %s", e)
            await interaction.followup.send(f"Error generating SOG projections: {e}")

    @tree.command(name="elo", description="Show current NHL team ELO rankings")
    async def elo_cmd(interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            embeds = format_elo_embeds()
            await interaction.followup.send(
                embeds=[discord.Embed.from_dict(e) for e in embeds]
            )
        except Exception as e:
            logger.error("elo_cmd error: %s", e)
            await interaction.followup.send(f"Error loading ELO rankings: {e}")

    @tree.command(name="history", description="Show prediction history summary")
    async def history_cmd(interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            embed = format_history_embed()
            await interaction.followup.send(embed=discord.Embed.from_dict(embed))
        except Exception as e:
            logger.error("history_cmd error: %s", e)
            await interaction.followup.send(f"Error loading prediction history: {e}")

    @client.event
    async def on_ready():
        await tree.sync()
        logger.info("Bot ready as %s", client.user)
        print(f"Logged in as {client.user} — slash commands synced")

    client.run(token)


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
