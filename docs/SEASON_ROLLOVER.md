# Season Rollover Checklist

What to do before a new NHL season starts. The short version: **one edit is
required, everything else is verification.**

## The one required edit

Add the new season's opening date to `config/season.py`:

```python
SEASON_STARTS: dict[int, str] = {
    ...
    2025: "2025-10-08",
    2026: "2026-10-06",   # <- add the real opener
}
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

# 7b. Retrain the SOG props model on the extra season.
#     --save is required; without it the script only evaluates.
#     (pipeline.backfill rebuilds player_game_stats.parquet for it)
python -m models.sog_model --save

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
| `home_elo` / `away_elo` | Near 1500 and clustered — ratings regressed at the season boundary |
| Rolling features | Mostly NaN on day one, and that is correct |
| `python -m pipeline.props_live --dry-run` | Silent until skaters reach 5 games |

### Expect no predictions for the first few days

This is the one behaviour change worth deciding on before opening night.

Rolling features make up the overwhelming majority of the model's inputs, and
on day one none of them exist — every team has zero games in the new season.
Feature coverage is therefore very low, and the guard drops those games
rather than publishing a number that is almost entirely column means wearing
a probability's clothes.

Coverage climbs as games accumulate and predictions resume on their own,
typically within the first week. If you would rather publish ELO-and-context
predictions from night one, lower the threshold:

```bash
python -m pipeline.live --dry-run --min-coverage 0.05
```

Those picks are real — ELO, rest, back-to-back and division flags are all
populated on day one — but they are much weaker than mid-season ones, so
treat the early-season Brier score accordingly.

Later in the season the opposite reading applies: low coverage means a
snapshot failed to rebuild, not that it is early. Check the `feature_coverage`
column in the prediction history to tell the two apart.

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
- **Schedule length change.** `TOTAL_GAMES` and `SEASON_DAYS` in
  `config/season.py` drive the game-id-to-date estimate used when Postgres
  is unavailable.
- **MoneyPuck column changes.** `ingestion/moneypuck.py` names its columns
  explicitly; a rename surfaces as a `KeyError` during aggregation.
