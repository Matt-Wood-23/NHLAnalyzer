# Handoff — pre-season work, August 2026

Written for whoever picks this up next, human or agent. Everything described
here is on the branch **`claude/pre-season-improvements-pn03zy`**, four
commits ahead of `main`, 36 files, +3048/-581, 131 tests passing.

None of it has been run against real data. See
[Verified vs. not](#verified-vs-not) before trusting any of it.

---

## Do this first

The prediction log (`data/predictions/prediction_history.parquet`) is the one
irreplaceable file in this repo — every other artifact rebuilds from
MoneyPuck. One of these commits stops tracking it in git. Back it up before
pulling:

```bash
cp data/predictions/prediction_history.parquet ~/nhl_history_backup.parquet
git pull
ls data/predictions/          # must still be there
```

If the pull reports a conflict on that path, resolve it with
`git rm --cached data/predictions/prediction_history.parquet` — the working
file stays put. After this, git never touches it again.

Then rebuild and check the numbers:

```bash
python -m pipeline.daily                          # now also builds player stats
python -m pipeline.evaluate_history --season 2025-2026
```

---

## What changed and why

### 1. Season configuration — `config/season.py` (new)

The season list, current season, and opening dates were duplicated across ten
files in three formats (`2025-2026`, `20252026`, `2025`). One copy — the start
dates in `features/context.py` — had fallen a season behind, so every 2025-26
game got a `NaT` date and **NaN rest / back-to-back / season_day features for
the entire most recent season of training data**. Silent; no error.

Everything now derives from one table plus the current date. Rolling to a new
season is a one-line edit (`SEASON_STARTS`), and forgetting it degrades to an
October estimate with a warning rather than `NaT`. `docs/SEASON_ROLLOVER.md`
has the checklist.

Also fixed here: the game-id→date estimator used 1230 total games (a 31-team
league; it should be 1312), and the XGBoost tuning split validated on a season
that is now two years stale.

`tests/test_season_wiring.py` fails the build if any module hardcodes a season
literal again.

### 2. Train/serve parity — the big one

Training rolled team stats strictly within a season; `pipeline/live.py`
re-implemented the same logic and rolled **across** season boundaries, with a
different venue-split window. Live predictions were built on inputs the model
had never been trained on — worst in October, when every carried-over value
came from the previous season.

**The pattern to preserve:** serving features are produced by appending a
placeholder row to the history and running *the exact function that builds the
training matrix*, then reading the placeholder's row. Not a parallel
implementation that happens to agree today.

- Teams: `features/team.py::pregame_snapshot` → used by
  `pipeline/live.py::_build_team_snapshot`
- Players: `features/player.py::pregame_player_snapshot` → used by
  `pipeline/props_live.py`

Both are covered by parity tests that compare every feature column against the
training path. **If you add a feature, add it to the training function and let
the snapshot inherit it.** Do not compute it separately on the serving side.

Teams with no games yet get an all-NaN row instead of being dropped, matching
what training saw on opening night, so no game vanishes from the early
schedule.

### 3. `/props` was running on 5 of its 11 features

`build_player_features_from_log` hardcoded `xg`, `shot_attempts` and
`xg_per_attempt` to `NaN` across both windows, commented "not in NHL API".
True — but the SOG model doesn't train on the NHL API. It trains on
`player_game_stats.parquet`, aggregated from MoneyPuck shot CSVs already on
disk, which carry per-player `xGoal`. Six of eleven features were replaced by
training-set means on every projection.

Player features now come from that parquet. The NHL API is still used for
rosters and ice time; neither feeds the model, so a failed game-log fetch costs
a display column rather than distorting a projection. The roster stays
authoritative on which team a player is on (a traded player's history carries
his old one).

Opponent context had a quieter version of the same bug — training reads the
opponent's pre-game `sf_pct_l20` / `xg_against_l20` /
`hd_chances_against_l20` from the team features, while props re-derived them
as a flat cross-season mean of the last 20 games. Now reuses the live team
snapshot.

**`player_game_stats.parquet` was only ever built by running
`ingestion.player_stats` by hand** — not in `backfill`, not in `daily` — so the
props training data drifted out of date with the team data refreshed daily
beside it. Both now rebuild it.

### 4. Feature-coverage guards

Both pipelines mean-impute missing values, so an upstream break surfaced as a
plausible ~50% prediction rather than an error. Every prediction now carries a
`feature_coverage` score; anything under 70% is withheld
(`--min-coverage` to override, `MIN_FEATURE_COVERAGE` in `pipeline/live.py`
and `pipeline/props_live.py`).

**Expect no game predictions on opening night.** With zero games played,
rolling features don't exist, coverage is ~5%, and the guard withholds. That
is deliberate — those picks would be almost entirely imputed means. Coverage
climbs and predictions resume within the first week. The owner has said he
doesn't plan to use it opening night, so this is fine as-is; `--min-coverage
0.05` publishes ELO-and-context-only picks if that changes.

### 5. ELO — overtime discount and persisted tuning

Roughly a quarter of NHL games are decided after regulation and those finishes
are near coin flips, but they were credited as full wins — pushing that noise
into the model's single most important feature. OT/SO winners now receive
`ot_win_value` (default 0.6). `1.0` reproduces the old behaviour and is in the
tuning grid, so tuning can only pick the change if it helps.

The tuner used to print its results while the pipeline carried on using module
defaults. Parameters now persist to `models/saved/elo_params.json` and are read
back by backfill and live. **Run `python -m features.elo` on real data before
the retrain** — it will confirm or reject the OT discount empirically.

### 6. `/history` rework

Was a lifetime accuracy number with nothing to read it against. Now:

- **Season-scoped.** Predictions are tagged with a season derived from the
  game_id prefix, so 2026-27 won't pool into 2025-26. Works retroactively.
- **Baselines.** Accuracy against always-pick-home, Brier against the no-skill
  score, both from the same games. This is the point — home teams win ~54% of
  NHL games, so raw accuracy is unreadable alone.
- **Calibration table**, which `print_accuracy_report` already computed but
  `/history` never displayed.
- **Pick-strength breakdown**, answering the question the `/predictions`
  labels invite. `bot/discord_bot.py::_confidence_label` now delegates to
  `pipeline/evaluate_history.py::confidence_label`, so the breakdown checks the
  labels actually shown rather than a parallel copy.
- Options: `/history season:2025-2026`, `/history season:all`, `/history last:50`.

Fixed along the way: the command called `backfill_outcomes()`, **rewriting the
prediction parquet on every invocation** — a file write triggered from Discord,
racing the daily pipeline. Scoring is read-only in the bot now. Re-scoring also
used to erase stored outcomes when the feature matrix was absent; played
results are now preserved and only refreshed.

### 7. Housekeeping

- `mlflow`, `shap`, `optuna` and `psycopg2` are now lazy imports, so training
  and serving don't pull in experiment-tracking or Postgres dependencies —
  matching the README's claim that the pipeline runs on Parquet alone.
- MoneyPuck ingestion tolerates a season it hasn't published yet (404 → skip
  with a warning) instead of aborting. This happens every October.
- `pytest.ini`, `tests/` (131 tests, synthetic data only — no network, no
  Postgres), and a GitHub Actions workflow on 3.11 and 3.12.

---

## Verified vs. not

**Verified** — synthetic data end-to-end, both pipelines:

- Feature matrix builds; context NaN rates 0.00 across every season including
  the newest (the bug in §1 is fixed).
- Train/serve parity: all 70 team rolling columns and all 8 player feature
  columns identical between training and serving paths.
- Props coverage 45% → 100%, 11/11 features live.
- ELO: pre-game ratings carry no leakage, zero-sum, `ot_win_value=1.0`
  reproduces old behaviour exactly.
- `/history` embed renders and fits Discord's size limits.
- An old-schema history file (205 rows, no `season`, no `feature_coverage`,
  object-dtype `correct`) upgrades with no row or outcome loss.

**Not verified** — needs real data:

- Whether the ELO overtime discount actually improves Brier. Unvalidated;
  `python -m features.elo` will settle it.
- Whether any of this changes real accuracy. The parity fixes are correctness
  fixes, not necessarily accuracy wins.
- The numbers quoted in the `/history` commit message (55.7% vs a 57.1%
  baseline; Strong Pick 72%, Lean 39%) come from the **stale 85-row March
  snapshot** that was committed to the repo, not from the real season. Treat
  them as illustrative only. Re-derive on the real log.

---

## Gotchas

- **`models/saved/*.pkl` are tracked in git.** Running `pipeline.train` or
  `models.sog_model` overwrites them. I clobbered them twice during testing and
  had to `git checkout --`. Check `git status` after any training run.
- **`data/predictions/` is now gitignored.** The repo is no longer a backup for
  the prediction log. Back it up elsewhere; carry it across machines.
- `data/parquet/` and `data/raw/` are gitignored and rebuilt by
  `pipeline.daily`. Deleting them is safe; it just costs a re-download.

---

## Open items

Ordered roughly by value. The owner's stated goal is **beating the market**,
with the bot being fun and useful; he picked the props fix first, which is
done.

### Confirmed goalie starters

Probably the largest remaining addressable accuracy gain. `_build_goalie_snapshot`
uses the team's *most recent* starter; actual starters are announced pre-game
and backups start roughly a third of games. Goaltending is among the biggest
single-game swing factors. Needs a reliable pre-game source — the NHL API
doesn't publish confirmed starters cleanly.

### Odds / closing-line value

`ingestion/odds_api.py` is a complete, working client that **nothing imports**,
and it's Postgres-only so it can't run in the Parquet setup actually in use.
`models/evaluate.py` still carries a comment saying "CLV vs closing lines is
tracked in Phase 5 once odds are loaded" — that never happened.

This is the only baseline that answers "beat the market". Constraint: the owner
will not pay for anything. The Odds API free tier (500 req/month, no card) is
enough to log closing lines *going forward* — so CLV accumulates from October
rather than being backtestable across 2021-26. If odds are wanted, this needs a
Parquet path, not the existing DB-only one. If not, delete or clearly mark
`odds_api.py` as unused rather than leaving dead code implying a working
feature.

### `h2h_home_win_rate_l3` may be a dead feature

Without Postgres, `features/context.py` skips the H2H block entirely and the
feature is NaN in both training and serving. **Check whether `DATABASE_URL` is
actually set** — the owner wasn't sure. If it isn't, either reimplement H2H on
Parquet (the data is there; it just needs game results keyed by matchup) or
drop the feature rather than leave a permanently-empty input in the model.

### Automating the daily run

Owner is deciding between his PC and a Raspberry Pi. `pipeline.daily` already
does the whole loop in one command, so either is a single cron line. What's
missing for unattended operation: a lockfile so overlapping runs can't both
write the parquets, failure notification (the Discord webhook is right there),
and retry on network blips during the MoneyPuck download. The Pi's real
constraint is the pandas rebuild over ~5 seasons of shot data — fine on a Pi 4
with enough RAM, slow on older hardware.

### Smaller

- **Residual season-boundary skew** in `_build_special_teams_snapshot` and
  `_build_opponent_quality_snapshot` (`pipeline/live.py`) — both take
  `.iloc[-1]` across all seasons, so in the opening week they serve last
  season's final values. Same class as §2 but much smaller in effect; the team
  snapshot that dominates the feature count is fixed.
- **Props projects scratched and injured players.** Any skater with ≥5 games
  gets a line. Filtering on recent appearances would help.
- **290 features on ~5000 games.** Selecting the top 20 improved Brier from
  0.2416 to 0.2404, which suggests the rest is mostly noise. Worth a more
  principled reduction.
- `pipeline/daily.py` imports `sys` and never uses it.
