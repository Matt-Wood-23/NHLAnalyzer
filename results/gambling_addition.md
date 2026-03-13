# Gambling Extension — Player Props

**Date:** 2026-03-09
**Current model scope:** Game-level win probability (team stats only)
**Proposed extension:** Player prop modeling, starting with shots on goal (SOG)

---

## Current Stack vs. Props Requirements

The existing pipeline is entirely team-level. Every feature (xG%, Corsi, shot share) is
aggregated per team per game — no player identities anywhere in the pipeline.

Props require a **player-level data layer**, which is a new ingestion + feature + model track.

| Prop Type | Modelability | Data Needed | Status |
|-----------|-------------|-------------|--------|
| Shots on goal (SOG) | ★★★★☆ — high volume, stable rate | Player game logs + opponent shot suppression | ❌ not ingested |
| Goalie saves | ★★★★☆ — very predictable | Goalie logs + opponent xG | Schema exists, table empty |
| Points (G+A) | ★★★☆☆ — real signal via TOI/PP usage | Player logs + line/PP usage | ❌ not ingested |
| Goals | ★★☆☆☆ — high variance, rare events | Same as points | ❌ not ingested |

**SOG is the highest-priority prop to model** — highest volume, books set thin lines (~0.5u
over/under), and individual shot attempt rates are among the most stable per-player metrics
in hockey.

---

## Data Sources Available

### MoneyPuck Shot-Level CSVs (already downloaded)
The `shots_YYYY.csv` files we download contain per-shot rows including:
- `shooter_id`, `shooter_name` — player identity
- `xGoal` — shot quality (expected goal probability)
- `shotType`, `distance`, `angle` — shot characteristics
- `isHomeTeam`, `period`, `manpowerSituation` — game context
- `goalieIdForShot`, `goalieNameForShot` — goalie faced

These can be aggregated to **per-player per-game SOG + xG** with zero new data fetching.

### NHL API Player Game Logs (free, no key needed)
```
GET https://api-web.nhle.com/v1/player/{playerId}/game-log/{season}/{gameType}
```
Returns per-game: TOI, goals, assists, shots, PP time, hits, blocks.
This is the primary source for TOI and PP usage — critical features for SOG models.

### The Odds API — Player Props Market
Same API key as game lines, different market parameter:
```python
"markets": "player_shots_on_goal"   # SOG over/under lines
"markets": "player_points"          # points over/under
"markets": "player_goal_scorer"     # anytime/first/last goal
```
**Cost caveat:** Free tier (500 req/month) burns quickly with player props since you need
separate calls per market. Each daily props pull = ~10–20 requests. A paid plan (~$79/month)
is needed for sustained daily prop tracking and CLV measurement.

---

## SOG Model Design

### Feature Set (per player per game)
```
Player rolling features (last 10, 20 games):
  - avg_sog_l10, avg_sog_l20           — raw shot rate
  - avg_icf_l10                        — individual Corsi (shot attempt proxy)
  - avg_toi_l10                        — more ice = more shots
  - avg_pp_toi_pct_l10                 — power play time share
  - avg_ixg_l10                        — individual xG per game

Opponent features (from existing team pipeline):
  - opp_sf_pct_against_l10             — how many shots the opponent allows
  - opp_hd_chances_against_l10         — high-danger chance suppression
  - opp_xg_against_l10                 — overall shot quality allowed

Context (from existing context features):
  - home_away                          — minor SOG split
  - back_to_back                       — fatigue
  - rest_advantage                     — team-level rest
```

### Model
A **Poisson regression** or XGBoost fit to the expected SOG count.
Shots are count events with a stable per-game rate — the Poisson distribution is a
natural fit and produces calibrated probabilities for over/under thresholds.

```
edge = model_expected_SOG − book_line
Bet over if edge > +0.3, under if edge < −0.3 (threshold tuned on historical data)
```

### CLV Measurement (same framework as game lines)
```
clv = model_prob(over) − closing_line_prob(over)
```
Positive CLV = we moved in the right direction vs. the market.

---

## Build Plan (Phase 7)

| Step | File | Description |
|------|------|-------------|
| 1 | `ingestion/player_stats.py` | Ingest NHL API player game logs per season (shots, TOI, PP TOI) |
| 2 | `db/schema_props.sql` | New tables: `player_stats`, `player_predictions`, `prop_odds` |
| 3 | `features/player.py` | Rolling SOG, TOI, xG, iCF features per player |
| 4 | `models/sog_model.py` | Poisson/XGBoost regression for expected SOG |
| 5 | `ingestion/odds_api.py` | Extend to fetch `player_shots_on_goal` market |
| 6 | `pipeline/props_live.py` | Daily: model xSOG vs line → edge table → Discord |

Estimated effort: similar scope to Phases 3+4 combined.

---

## Practical Blockers

1. **The Odds API cost** — without closing lines, CLV measurement is impossible.
   You'd only get pre-game lines and have no way to grade edge quality post-hoc.
   Need a paid plan before building the CLV side.

2. **Lineup/injury data** — props are useless if the player is scratched.
   The NHL API daily roster endpoint provides this but adds another daily fetch step.
   ```
   GET https://api-web.nhle.com/v1/roster-season/{teamAbbrev}
   ```

3. **Sample size per player** — new players, call-ups, or injured returnees have
   few recent games. Need a minimum `games_played_l20 >= 5` filter before betting.

4. **Line availability** — SOG props aren't posted for every player on every slate.
   Books focus on stars and high-ice-time forwards. Need to handle missing lines gracefully.

---

## What the Current Model Contributes

The existing team-level pipeline feeds directly into props as **opponent quality inputs**:

- `opp_sf_pct_against_l20` → how suppressive the opposing defence is
- `opp_hd_chances_against_l20` → shot quality allowed
- `opp_xg_against_l20` → overall defensive quality

These are already computed and saved in the feature matrix — they'd be merged onto
player prop rows by `game_id` + `is_home` to get the opponent context.

---

## Summary

SOG props are the most tractable starting point:
- Stable individual rates → models well
- High volume books post lines on → enough data to evaluate edge
- MoneyPuck shot data already downloaded → minimal new ingestion for training
- Integrates cleanly with the existing team-level feature pipeline

The main gate is **Odds API cost** for closing lines. Until then, the game-line CLV
framework (Phase 5) is the better use of the free tier.
