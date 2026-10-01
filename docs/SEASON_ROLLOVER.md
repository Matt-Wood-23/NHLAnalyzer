# Season Rollover Checklist

What to do before a new NHL season starts. The short version: **one edit is
required, everything else is verification.**

## The one required edit

Add the new season's opening date to `config/season.py`:

```python
SEASON_STARTS: dict[int, str] = {
    ...
    2025: "2025-10-08",
    2026: "2026-09-29",   # <- add the real opener
}
```

Read the date from the API rather than from a slate you happen to be looking
at — the 2026 entry was first written as October 5th, labelled "confirmed",
and was six days late:

```bash
curl -s https://api-web.nhle.com/v1/schedule/2026-10-01 | python -c "import sys, json; print(json.load(sys.stdin)['regularSeasonStartDate'])"
```

That is it. The season list, the MoneyPuck download map, the NHL API season
strings, the model training folds, and the XGBoost tuning split all derive
from this table plus the current date.

If you forget, nothing crashes: `season_start()` estimates October 8th and
logs a warning, and dates end up a few days off. That is a deliberate
trade — the previous design left the date table a season behind and silently
produced `NaT`, which blanked out every rest, back-to-back, and season-day
feature for the entire newest season.

### How "current season" is decided

`current_season()` rolls over on **September 1st**, so pre-season work in
September already targets the upcoming season. To pin a season for a
backfill or a replay, set the environment variable:

```bash
NHL_CURRENT_SEASON=2025-2026 python -m pipeline.live --date 2026-03-15
```

Any format works: `2025-2026`, `20252026`, or `2025`.

## Pre-season sequence

Run these in order once MoneyPuck publishes the new season's shot file.

```bash
# 1. Confirm the config resolves the way you expect
python -c "from config.season import *; print(current_season(), season_start(current_season()))"

# 2. Tests must be green before touching data
pytest

# 3. Pull the new season's data and rebuild the feature matrix.
#    Safe to run before opening night — MoneyPuck 404s are skipped with a
#    warning rather than aborting the run.
python -m pipeline.backfill

# 4. Re-tune ELO on the now-longer history and persist the parameters.
#    Writes models/saved/elo_params.json, which backfill and live both read.
python -m features.elo

# 5. Rebuild the feature matrix so it uses the newly tuned ELO
python -m pipeline.backfill

# 6. Re-run model selection on the extra season of data
python -m models.baseline
python -m models.xgboost_model --trials 150 --ensemble --calibration none --feature-selection

# 7. Retrain and serialize the production model
python -m pipeline.train --model random_forest

# 7a. Pull the new season's boxscores -- the dressed-lineup spine the SOG
#     model trains on, plus goalie starter flags.  Final games cache
#     permanently, so this only fetches the season just played (~40s).
python -m ingestion.boxscore --backfill --build --merge

# 7b. Retrain the SOG props model on the extra season.
#     --save is required; without it the script only evaluates.
#     (7a rebuilds player_game_stats_full.parquet for it)
python -m models.sog_model --save

# 7c. Retrain the goalie saves model, likewise.
#     (pipeline.backfill rebuilds goalie_game_stats.parquet for it)
python -m models.saves_model --save

# 8. Dry-run a prediction
python -m pipeline.live --dry-run
```

Steps 6 and 7 matter: the walk-forward folds and the XGBoost tuning split
both shift forward automatically, so the previous holdout season becomes
training data. Skipping the retrain means serving a model that has never
seen the most recent season.

## What to verify on opening night

| Check | Expected |
|---|---|
| `python -m pipeline.live --dry-run` | Games appear, no "Missing snapshot" warnings |
| `season_opener` column | True for every game on night one, then only for teams yet to play |
| Second night, a team that played the first | `home_back_to_back` / `away_back_to_back` is 1 — if not, game dates are not syncing |
| "Game dates are behind" warning | Absent. If present, run `python -m pipeline.daily --refresh-only` |
| `home_elo` / `away_elo` | Near 1500 and clustered — ratings regressed at the season boundary |
| Rolling features | Mostly NaN on day one, and that is correct |
| `python -m pipeline.saves_live` | `in_season_factor` is 1.0 for ~3 weeks, then drifts below it |
| `python -m pipeline.props_live --dry-run` | Silent until skaters reach 5 games |
| Scratch log line in `props_live` | "No scratches published yet" until roughly an hour before puck drop, then "dropped N of M skaters" |
| `starter_basis` in `saves_live` | Reads "N of last 10 starts"; N will be small in the opening week because the window is still filling |

### Opening week: predictions are published, and they are thin

Rolling features make up the overwhelming majority of the model's inputs, and
for a team's first game none of them exist. Coverage is about 7% — in the
training matrix as well, every season.

Those games are published anyway, on ELO, goalie form and rest, and flagged
`season_opener`. They used to be dropped: the coverage guard could not tell
"nothing has been played" from "a snapshot failed", so in 2026 it refused the
whole opening week and suggested re-running the command that had just run.

Do not read much into them. Walk-forward Brier on first games is 0.2456
against 0.2496 for the base rate (n = 78, −0.004 ± 0.010) — no measurable
skill. Exclude them from evaluation by `feature_coverage`.

From a team's second game coverage is 77% or better and the ordinary 70%
guard applies. Below that with both teams having played means a snapshot
failed to build; the log says which.

### Run it before puck drop

A game already under way is predicted and displayed but never written to the
prediction log. So a run made in the evening records nothing for that night,
by design: the price it would attach is a live line, and once results are
ingested the features contain the game being "predicted".

That last row is intentional. Training rolls stats strictly within a season,
so on opening night the model has no rolling history and leans on ELO and
context — exactly what it was trained to do for game one. The live pipeline
matches this by scoping its snapshot to the current season. If you see
last season's form carried into October predictions, something has
regressed; `tests/test_team_features.py::TestTrainServeParity` guards it.

Teams that have not played yet still appear on the slate with NaN features
rather than being dropped, so no game silently disappears from the first
few days of the schedule.

## Carry the prediction log forward

`data/predictions/prediction_history.parquet` is committed to the repo and is
the one file that cannot be rebuilt — everything else regenerates from
MoneyPuck. Commit it after a run so the backup stays current, and keep a single
machine writing it: it is an append-only binary, so two diverged copies cannot
be merged.

Rolling into a new season does not need it cleared. Predictions are tagged
with the season derived from their game ID, and `/history` scopes to one
season by default, so last year's results stay available under
`/history season:2025-2026` without diluting this year's numbers.

## Structural changes to watch for

These are rare, but they are the things a new season can actually break.

- **Relocation, rename, or expansion.** Update `_DIVISIONS` in
  `features/context.py` and `TEAM_ABBREVS` in `ingestion/nhl_api.py`.
  `tests/test_context.py::TestLeagueStructure` fails if the two disagree or
  if a division no longer has eight teams. Add the emoji in
  `bot/discord_bot.py` too.
- **Schedule length change.** Add an entry to `SCHEDULE_SHAPES` in
  `config/season.py` — `(total games, days)`, keyed by the first season it
  applies to. It drives the game-id-to-date estimate used when Postgres is
  unavailable. 2026-27 went from 82 games to 84 and nothing flagged it.
- **MoneyPuck column changes.** `ingestion/moneypuck.py` names its columns
  explicitly; a rename surfaces as a `KeyError` during aggregation.
