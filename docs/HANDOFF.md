# Handoff — pre-season work, August 2026

Written for whoever picks this up next, human or agent.

**Session 1** (branch `claude/pre-season-improvements-pn03zy`, since merged)
was written before any of it had touched real data — see
[Verified vs. not](#verified-vs-not).

**Sessions 2, 3 and 4** (below) were all run against real data.

**Session 5** ([below](#session-5--opening-week-30-september-2026)) is the
first with the season under way, and it found that three of the five live
snapshots had been serving stale features all along. Read it before trusting
anything Sessions 1-4 say about train/serve parity. 393 tests passing.

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

**Measured on the real log** (179 scored predictions, 2026-03-13 to 04-13),
after merging to `main` on the owner's machine:

```
Record:    97-82
Accuracy:  54.2%   (always-home baseline: 53.1%)   ->  +1.1 pts
Brier:     0.2403  (no-skill baseline:   0.2491)   ->  0.0088 better

Strong Pick   71.9%   n=64
Lean          48.1%   n=54
Toss-Up       41.0%   n=61
```

The model beats the naive baseline, and the signal is concentrated almost
entirely in the Strong Pick tier — 71.9% on 64 games is roughly 3.5 standard
deviations above a coin flip. Lean and Toss-Up carry nothing. Calibration is
good above 60% and poor in the 50-60% band, which is the same story from a
different angle.

**Not verified** — needs a rebuild or a season of new data:

- Whether the ELO overtime discount actually improves Brier. Unvalidated;
  `python -m features.elo` will settle it.
- Whether any of this changes real accuracy. The parity fixes are correctness
  fixes, not necessarily accuracy wins.
- Whether the residual open items below change anything. They are unstarted.

---

## Gotchas

- **`models/saved/*.pkl` are tracked in git.** Running `pipeline.train` or
  `models.sog_model` overwrites them. I clobbered them twice during testing and
  had to `git checkout --`. Check `git status` after any training run.
- **`data/predictions/` is tracked**, deliberately — the repo is the backup for
  the prediction log. It is an append-only binary, so keep one machine writing
  it; two diverged copies cannot be merged by git.
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

### Watch the market comparison as data accumulates

**Built.** Every prediction now records the market's no-vig consensus price,
`pipeline.daily` fills in the closing price once games are played, and
`/history` reports model-vs-market Brier, closing-line value, and whether a
claimed edge actually materialised.

Backfilling closing prices onto the existing log recovered 91 of 179 games.
That retroactive sample is **not** evidence of an edge: the model appears
0.0053 Brier ahead of the market, but the 95% bootstrap CI is
[-0.0195, +0.0088] and spans zero. The mechanism is plain — the model averages
P(home)=0.500 against the market's 0.538, and home teams won only 45.1% over
those 91 games, so the less home-leaning forecaster wins on luck. A flat 50%
forecast would have scored 0.2500 against the market's 0.2533.

The prospective capture is the trustworthy one. Revisit after a few hundred
games with a price attached.

`ingestion/odds_api.py` is a paid-tier client that nothing imports and which
is Postgres-only. Delete it or mark it unused rather than leaving dead code
implying a second odds path.

### Refresh the stale Postgres `games` table

**Answered:** Postgres is running, `DATABASE_URL` is set in `.env`, and the
connection works, so `h2h_home_win_rate_l3` is a live feature (83% populated) —
not the dead one previously suspected. `goalie_stats`, `odds` and `predictions`
are all empty; goalie features come from MoneyPuck instead, which is what the
code does.

But `games` stops at **2026-03-09**, holding 1,011 of roughly 1,312 games for
2025-26. Games past that date are absent, so they fall out of the context join:

```
season      total  with_date   context NaN in feature matrix
2024-2025    1398       1312                           6.2%
2025-2026    1011       1011                          21.0%
```

A fifth of the newest season trains with no rest / back-to-back / season-day.
Fix is a re-run of `python -m ingestion.nhl_api`. (The ~88 dateless games in
each older season are playoffs, and harmless.)

Note this also corrects §1 above: the missing 2025-26 opener date only affected
the **no-database fallback**, which this setup does not use. The config fix is
still correct and worth having as the safety net, but the 21% NaN has a
different cause.

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

*(This list is carried forward and corrected under [Open items after Session 2](#open-items-after-session-2).)*

---

# Session 2 — props overhaul, 29 August 2026

Three commits on `main`, unpushed. 224 tests passing (was 147). Everything
below was run against real data, unlike Session 1.

The through-line: two of the three findings started as "add a feature" and
turned into "the number you were reading was wrong."

## 8. The SOG model was scored against a baseline it never had to beat

`evaluate_sog` used `mean(y_true)` as the line — taken from the test set, and
not a line any book posts — and reported accuracy against 0.500. At a 2.5
line roughly 72% of player-games go under, so the majority class is the bar.
Measured properly the model was overstated about threefold, and in the first
walk-forward fold a constant "under" beat it outright (0.694 vs 0.672).

Scoring now runs at 1.5 / 2.5 / 3.5. Since it was already a Poisson model,
`over_probability` turns the rate into P(SOG > line) — both what a bet needs
and something Brier can score:

| Line | Brier | no-skill | skill | Accuracy | majority |
|-----:|------:|---------:|------:|---------:|---------:|
| 1.5  | 0.2277 | 0.2502 |  9.0% | 62.4% | 52.9% |
| 2.5  | 0.1787 | 0.2013 | 11.2% | 74.1% | 72.2% |
| 3.5  | 0.1038 | 0.1159 | 10.4% | 86.7% | 86.7% |

The skill is real and it lives in the probabilities, not the binary call — at
3.5 the model ties always-under on accuracy while beating it by 10% on Brier.

## 9. League drift — the finding that generalises

Combined shots per game fell from 63.2 in 2021-22 to 55.6 in 2025-26, a 12%
decline that is **not** a roster-size artifact (skaters per game is flat at
~30.7). Starter saves fell 28.4 to 24.8 over the same span.

Any expanding-window model inherits the high-volume seasons and over-projects,
with the error growing as the gap widens: +0.065 shots in 2022-23 rising to
+0.120 in 2024-25.

Three corrections, now shared in `models/count_model.py`:

1. **Recency weighting** (`RECENCY_HALF_LIFE_SEASONS = 0.5`) — a season-old
   row carries a quarter the weight of a current one.
2. **Newest-season calibration** — a multiplicative factor fit on the most
   recent training season, folded into the log-link intercept where it is
   exact (`exp(b0 + log f + bx) == f·exp(b0 + bx)`). The saved pickle is
   therefore already corrected and serving needs no extra step.
3. **In-season calibration** (`in_season_factor`) — a serve-time factor fit on
   the current season's completed games. The first two can only reach the
   newest *training* season; the season being served is a further step down.

SOG bias fell +0.092 → +0.051, and on the most recent fold +0.104 → +0.019.

**Read this before building the next counting-stat model.** Totals will have
the same problem.

## 10. Goalie saves — `models/saves_model.py`, `/saves`

Saves are shots faced times save rate. Save rate barely moves (league ~.905),
so nearly all the variance is workload, which is a team property before it is
a goalie one. Features are the goalie's own rolling shots-against plus how
much his team concedes and how much tonight's opponent generates.

| Line | Brier | no-skill | skill | Accuracy | majority |
|-----:|------:|---------:|------:|---------:|---------:|
| 22.5 | 0.2017 | 0.2168 | 7.0% | 69.8% | 69.4% |
| 24.5 | 0.2288 | 0.2489 | 8.1% | 61.9% | 58.8% |
| 27.5 | 0.2277 | 0.2488 | 8.5% | 63.2% | 59.3% |
| 30.5 | 0.1865 | 0.2008 | 7.1% | 72.8% | 72.9% |

Two things differ from SOG and both mattered:

**Saves are overdispersed** — variance runs 1.9x the mean, still 1.8x after
excluding short outings, so it is not just pulled goalies. Poisson tails come
out too tight and P(over) reads more confident than the data supports. The
transform is negative binomial with a dispersion estimated from training
residuals, worth 1-7% Brier at every line. It is saved with the model because
P(over) is wrong without it.

**The drift is steeper.** Weighting and newest-season calibration still left
+0.68 saves of bias — almost exactly one season of drift. `saves_live` closes
most of the rest in-season, taking it to +0.36.

## 11. `/yesterday` — the reflection half of the daily loop

The scoring machinery already existed in `pipeline/daily.py`; everything it
produced was a season aggregate. `day_report` scopes it to one slate.

The per-game breakdown leads, not the record, because the tally alone is not
readable — 0-4 where every miss was a Toss-Up is close to noise, and the same
record with Strong Picks down is worth investigating. `strong_misses`
isolates exactly that case.

Three surfaces, one implementation: `/yesterday`, `evaluate_history --day`,
and `--results` in webhook mode.

**This needed a real fix underneath.** Nothing stored the date of the *games*,
only `predicted_at` — the run timestamp. Grouping by night off that is wrong
whenever a run crosses midnight UTC, which is every run after 20:00 Eastern.
`save_prediction_history` now takes `target_date` and stores `game_date`;
`game_dates()` falls back to `predicted_at` for older rows.

## What was rejected, and why

Three things looked like wins and did not survive checking. They are recorded
so nobody re-derives them:

- **Retroactive CLV.** Backfilled closing prices on 91 games showed the model
  0.0053 Brier ahead of the market. The bootstrap CI is [-0.0195, +0.0088] —
  spans zero. The mechanism is visible: model mean P(home) 0.500 vs market
  0.538 in a sample where home teams won only 45.1%. A flat 50% forecast would
  have scored 0.2500 against the market's 0.2533. **Not evidence of edge.**
- **Shrinking low-experience rolling averages toward the league mean.** Made
  the bias worse (+0.174 → +0.196). Fringe players are *below* league average,
  so the prior pulls them up — backwards.
- **Adding `log(games_played)` to the SOG features.** Cut low-experience bias
  40% but degraded overall bias and Brier at every line. Trading a targeted
  bias for a broader one is not a fix.

## Parlays — measured, and the answer is no

Same-game SOG legs are essentially independent:

```
teammate pairs (874k)   r = +0.0095   P(both 3+) 0.0720 vs 0.0707   lift 1.02
opponent pairs (1.85M)  r = -0.0011   P(both 3+) 0.0698 vs 0.0701   lift 1.00
```

Books shade SGPs for correlation that is not there, so a SOG parlay is
strictly worse than the legs singly. One correlation *is* real and runs
opposite to the folk wisdom: `corr(saves, goalie's team won) = +0.111`, and
shots faced barely predicts the result at all (-0.02). So "over saves + team
loses" has lift 0.91 — the correlated pairing is over saves + his own team's
moneyline, at 1.089.

---

## Open items after Session 2

*(Re-ranked and mostly resolved under [Open items after Session 3](#open-items-after-session-3).)*

Ordered by what would change the numbers most.

### Zero-shot games are missing entirely

`player_game_stats.parquet` has `min(shot_attempts) == 1` and no rows at 0.
MoneyPuck's shot file only contains players who shot, so a skater held to zero
attempts vanishes rather than recording a 0 — about 30.7 rows per game against
36 dressed skaters, so roughly 5 skaters a night disappear.

Every rolling average is therefore computed over a survivor set, which biases
it up, in the same direction as the league drift and on the same prop. Fixing
it means pulling the dressed lineup from the NHL API boxscore and left-joining
zeros. **That same call also gives confirmed starting goalies and unlocks
points/assists props**, which makes it the highest-leverage single change left.

### Confirmed goalie starters

`/saves` infers the starter from each team's most recent start — right most
nights, wrong on the back half of a back-to-back. Every row carries
`starter_basis` and the embed footer says so, but this is the weakest link in
the newest model. Same boxscore work as above.

### Residual low-experience bias in SOG

+0.17 shots for skaters under 20 career games against +0.003 over 200. Two
fixes tried and rejected (above). Untried: a hierarchical model, or simply
refusing to publish projections under ~20 games. Documented in the module
docstring and README as a known limitation.

### `--dry-run` does not stop the prediction log being written

`pipeline/live.py:792` — `dry_run` only guards the Postgres save.
`save_prediction_history` at line 789 runs unconditionally, so a dry run still
appends to the one irreplaceable file in the repo. This has been known for two
sessions and is a two-line fix.

### Season-boundary skew in three snapshots

`_build_goalie_snapshot`, `_build_special_teams_snapshot` and
`_build_opponent_quality_snapshot` (`pipeline/live.py`, lines 140/191/~253)
all take `.iloc[-1]` across every season, so in the opening week they serve
last season's final values. `_build_team_snapshot` — which dominates the
feature count — was fixed in Session 1, so the pipeline is now *inconsistent*
about whether snapshots reset at the boundary. Worth making uniform before
opening night.

### Not yet built

- **`/edge`** — the disagreements between model and market, ranked. Now
  possible because `market_prob_home` is captured at prediction time. Build it
  before opening night so it logs from game one. Note in the embed that
  disagreement is a hypothesis, not edge, until CLV confirms it.
- **Game totals.** The natural third model: mean 6.03 goals, P(over 5.5) =
  0.541, and the team matrix already carries xGF/xGA. Attractive because it is
  a *main* market, where the vig is a fraction of what props carry — a small
  edge actually survives. Use `count_model`; expect the same drift.
- **Anytime goalscorer** is derivable (17.6% base rate among shooters) but it
  is a longshot market with heavy hold and ~2 shots of signal. Skip it.

### Smaller

- **287 features on 6,877 games.** `run_feature_selection` exists in
  `models/xgboost_model.py` but production RF uses all of them. Top-20
  selection previously improved Brier 0.2416 → 0.2404.
- **Props still projects scratched and injured players** — any skater with ≥5
  games gets a line. The boxscore join fixes this too.
- **`ingestion/odds_api.py` is dead code** — Postgres-only, imported nowhere,
  superseded by `action_network.py`. Delete it or mark it unused.
- `pipeline/daily.py` imports `sys` and never uses it.

### Still true from Session 1

Automating the daily run is undecided (PC vs Pi). `pipeline.daily` is already
one command, so either is a single cron line; what is missing is a lockfile,
failure notification through the existing webhook, and retry on the MoneyPuck
download.


---

# Session 3 — the boxscore spine, 29 August 2026

Four commits on `main`, unpushed (seven total now). 264 tests passing (was
224). All run against real data.

Prompted by two repos: `Zmalski/NHL-API-Reference`, which documents endpoints
we were not using, and `peruukki/nhl-score-api`. The first was worth acting on;
the second wraps the same host we already call and adds nothing but a
dependency — useful only as a reference parser for the payload shapes.

## 12. The dressed lineup, not the shooters

`player_game_stats.parquet` came from MoneyPuck's shot files, which contain
only players who generated a shot event. A skater held to zero attempts
recorded no `0` — he vanished from the game. 30.7 rows per game against 36
dressed skaters, `min(shot_attempts) == 1` across all 200k rows.

`/gamecenter/{id}/boxscore` returns all 36 with `sog` present and zero where
earned, so it is now the spine and MoneyPuck contributes xG only. A skater who
does not match took no attempts: `xg` and `shot_attempts` are 0, a real
measurement. `xg_per_attempt` stays NaN — a rate is undefined with no
attempts, and the rolling means skip NaN, so a no-attempt game contributes a 0
to the SOG rate and nothing to shot quality.

    247,513 skater-games, up from 200,656            (+23%)
    6,877 games, up from 6,526 — 351 missing outright
    47,891 rows (19.3%) previously invisible

**The measured gain is much smaller than that row count suggests, and the
comparison is easy to get wrong.** Scored on the population the pipeline
actually serves — every dressed skater, target from the official boxscore,
with the old shooter-only history emulated on the same rows:

| Line | old Brier | new Brier | |
|-----:|----------:|----------:|---:|
| 1.5 | 0.2171 | 0.2155 | +0.78% |
| 2.5 | 0.1579 | 0.1570 | +0.54% |
| 3.5 | 0.0907 | 0.0903 | +0.41% |

Mean bias +0.040 → +0.030 shots. The added games are mostly easy unders, so
they move the base rate more than the skill.

**Do not use the intersection comparison.** Scoring both models only on
player-games present in the MoneyPuck file selects rows by whether the player
shot, which is the outcome — unknowable at bet time. On that biased subset the
new model looks *worse* (bias −0.197), purely because it is trained on a
population with a lower mean and evaluated on one selected for having shot.
This is the same class of error as Session 2's SOG baseline.

Likewise the README's SOG table is not comparable across the change: the new
numbers cover every dressed skater, and the majority baseline moved with them
(72.2% → 76.2% at 2.5).

The boxscore also carries per-skater `points`, `assists`, `toi`, `shifts`,
`hits`, `blockedShots`, and a `starter` flag per goalie.

## 13. Playoff games were sorting into the middle of the regular season

An NHL game ID is `SSSSTTNNNN`. Thirteen call sites derived a chronological key
as `game_id.str[-4:]`, dropping the game type — so `2024020311` and
`2024030311` both gave 311. Playoff games sorted as though they happened in
January, and every rolling feature keyed on that order saw a player's playoff
results while computing features for his *earlier* regular-season games.

That is leakage, not mis-ordering. All five matrices include playoffs — 351 of
6,877 games, affecting 643 of 1,471 skaters.

`config.season.game_number` takes `TTNNNN` instead: type order survives
(20311 < 30311), still a plain int, drop-in at all thirteen sites. A test
asserts no module derives the key by hand again.

SOG mean bias +0.030 → +0.021; Brier unchanged at every line, which is what a
4.6%-of-games leak should look like.

## 14. The starter rule was worse than a coin flip

`goalie_starts.parquet` — 13,754 starts carrying the boxscore's own flag —
made it possible to score the probable-starter inference for the first time.

| Rule | Correct |
|:--|--:|
| whoever started last (what we shipped) | 45.6% |
| most common of last 3 | 55.7% |
| most common of last 5 | 58.2% |
| **most common of last 10** | **59.7%** |
| most common of last 20 | 58.9% |
| season-to-date leader | 58.5% |

Tandems alternate: the busiest goalie takes only 60.2% of his team's starts, so
the previous game is nearly as much evidence *against* him going again as for
it. Past ten starts the window drifts onto goalies since traded or sent down.

`starter_basis` now reports the count ("7 of last 10 starts") with a
`starter_confidence` share.

**A confirmed pre-game starter does not exist in the NHL API.** Checked against
a real scheduled game (`2026020053`, `gameState=FUT`), not assumed:

- the pre-game `boxscore` has no `playerByGameStats` at all
- `right-rail` → `gameInfo.scratches` is `[]` until the lineup is filed
- `landing` → `matchup.goalieComparison` lists exactly 2 goalies per team, but
  ranked on *last* season's stats — the tandem, not a starter call

The `starter` flag appears only once the game is live, i.e. after bet time.

## 15. Scratches

Scratches *do* publish, roughly an hour before puck drop, so `props_live` now
drops them. A team roster is everyone under contract, so without this the model
posted lines on healthy scratches, injured players and anyone in the minors.

An empty response means "not filed yet", never "nobody is out". The run logs
which of the two it saw rather than silently treating an unfiled lineup as a
clean one.

## Open items after Session 3

*(The pre-season items below are done — see [Session 4](#session-4--pre-season-cleanup-29-august-2026).)*

Carried forward and re-ranked. Everything under Session 2's list that is not
repeated here is done.

### Points and assists props

The boxscore now carries them per skater with true zeros, and `count_model`
already has the Poisson/NB machinery. This is the cheapest new market left.
Assists are the more interesting of the two — heavily overdispersed, so expect
the negative binomial to matter the way it did for saves.

### Game totals

Still the best idea not yet built: a *main* market, where the vig is a fraction
of what props carry, so a small edge survives. Mean 6.03 goals, P(over 5.5) =
0.541, and the team matrix already carries xGF/xGA. Use `count_model`; expect
the same league drift.

### The saves model still trains on MoneyPuck goalie data

`goalie_starts.parquet` has official saves and shots-against for every goalie
appearance, including relievers, and a real starter flag. `features/goalie_mp.py`
predates it. Switching would also remove a source disagreement — the boxscore
and MoneyPuck differ on SOG in 0.82% of skater-games.

### Confirming the tandem

`landing.matchup.goalieComparison` names the two goalies a team is actually
carrying tonight. Intersecting that with the last-10 rule would catch injuries
and callups, which no historical rule can. Cheap; untested because it needs a
live slate.

### `--dry-run` still writes the prediction log

`pipeline/live.py:792` — `dry_run` guards only the Postgres save;
`save_prediction_history` runs unconditionally. Known for three sessions now.
Two-line fix.

### Season-boundary skew in three snapshots

`_build_goalie_snapshot`, `_build_special_teams_snapshot` and
`_build_opponent_quality_snapshot` still take `.iloc[-1]` across every season.
`_build_team_snapshot` was fixed in Session 1, so the pipeline is inconsistent
about whether snapshots reset at the boundary. Worth making uniform before
opening night, which is when it bites.

### Smaller

- **`/edge`** — offered twice, still not built. Build it before opening night
  so it logs disagreements from game one.
- **287 features on 6,877 games.** `run_feature_selection` exists but
  production RF ignores it; top-20 previously improved Brier 0.2416 → 0.2404.
- **NHL Edge endpoints** (skating speed, shot speed, zone time) are the only
  untapped *feature* source found — MoneyPuck has none of it. Speculative.
- **`ingestion/odds_api.py` is dead code**, superseded by `action_network.py`.
- `pipeline/daily.py` imports `sys` and never uses it.
- Automating the daily run is still undecided (PC vs Pi). `pipeline.daily` is
  one command; what is missing is a lockfile, failure notification through the
  existing webhook, and retry on the MoneyPuck download.


---

# Session 4 — pre-season cleanup, 29 August 2026

Four commits, 302 tests (was 264). The pre-season list from Session 3, cleared.

## 16. `--dry-run` was writing the prediction log

`dry_run` guarded only the Postgres write, so every dry run appended to
`prediction_history.parquet` — the one file here that cannot be rebuilt from
source data, because it records what the model believed *before* the games
were played.

**The flag could not just be widened.** `pipeline.daily` passed `dry_run=True`
meaning "skip Postgres" while relying on the history still being written.
Widening it would have silently stopped the daily loop recording anything —
the exact failure that is invisible until you go looking for results weeks
later. They are two flags now, and `daily.py` says which it wants.

This also stops `/predictions` overwriting the logged prediction: history
deduplicates on `(game_id, model_name)` keeping the last write, so a Discord
user asking for predictions was replacing the probability the daily run
recorded — which for CLV is the whole point, since it should be the belief at
price time.

## 17. The goalie snapshot picked the wrong goalie

`_build_goalie_snapshot` sorted by `("goalie_id", "season", "game_num")` and
took each team group's last row. That is the *highest goalie_id's* final start,
not the team's most recent. For BOS in 2025-26 it served Swayman's rolling
stats on a night Korpisalo had started last.

It also used "whoever started last" (45.6%, Session 3). Both serving paths now
share `features.goalie_props.likely_starters` (59.7%), so the outcome model and
the saves prop cannot put different goalies in net for the same game.

## 18. Snapshots reset where training resets — and only there

The Session 3 list called for making the snapshots "uniform". That was the
wrong instruction, and checking each feature family against its training code
is what caught it:

| Feature | Training | Live snapshot |
|:--|:--|:--|
| Team rolling | within season | scoped (fixed Session 1) |
| Special teams | within season | **was spanning — now scoped** |
| Opponent quality | within season | **was spanning — now scoped** |
| Goalie form | across seasons | spans — correct as-is |
| Rest / back-to-back | across seasons | spans — correct as-is |

`features.goalie_mp` deliberately rolls per goalie with no season reset, and
`features.context` computes rest with `groupby("team")` alone, so training saw
~180 days on opening night. Scoping either would have been the bug. Both are
documented in place with a test, so the next person pattern-matching on the
other three does not "fix" them.

## 19. `/edge`

Ranks tonight's disagreements, stated from the side the model favours. Reuses
`edge_realization` and `clv_summary`, both written in Session 2 and never
surfaced. Runs with `dry_run=True` so asking for it cannot overwrite the
canonical logged prediction.

The embed leads with the caveat rather than burying it, and a test asserts the
word "hypothesis" stays in the description — that is the line that decides
whether this reads as analysis or as a tip sheet.

**What the history actually says**, on 91 scored games with a closing price:

| bucket | n | model | market | actual | p vs market |
|:--|--:|--:|--:|--:|--:|
| < −5% | 11 | 55.9% | 63.8% | 72.7% | 0.756 |
| −5..−2% | 13 | 57.6% | 60.6% | 23.1% | 0.008 |
| −2..+2% | 28 | 57.1% | 56.9% | 46.4% | 0.340 |
| +2..+5% | 8 | 58.4% | 54.8% | 62.5% | 0.737 |
| > +5% | 31 | 59.7% | 49.9% | 64.5% | 0.110 |

The `> +5%` row is the one that looks like an edge and it does not survive
(p = 0.110). The single significant bucket points the wrong way and is about
what five tested buckets produce by chance. Same conclusion as the retroactive
CLV check: no evidence of edge yet.

## 20. The season opener

> **Wrong — corrected in Session 5.** The season opened 2026-09-29. The date
> below was never confirmed against anything.

`SEASON_STARTS` stopped at 2025. The 2026-27 season opens **2026-10-05**,
confirmed against the schedule API. Without it `season_start` fell back to
Oct 8 with a warning — a three-day error feeding `approximate_game_date` and
so the back-to-back flag.

**The rollover trips on 1 September, not opening night.**
`SEASON_ROLLOVER_MONTH` is 9, so `current_season()` flips to 2026-2027 then and
`all_seasons()` grows to six. That is when the rest of
`docs/SEASON_ROLLOVER.md` wants running.

## Open items after Session 4

### Not yet built

- **Points and assists props.** The boxscore carries them with true zeros and
  `count_model` has the machinery. Assists are the more interesting half —
  heavily overdispersed, so expect the negative binomial to matter as it did
  for saves.
- **Game totals.** Still the best untouched idea: a main market, where the vig
  is a fraction of what props carry. Mean 6.03 goals, P(over 5.5) = 0.541, and
  the team matrix already has xGF/xGA. Expect the same league drift.

### Wants a live slate

Neither can be tested until games are played, and both should be checked on
the first real one:

- Does `gameInfo.scratches` populate when expected (~1h before puck drop)?
- Can `landing.matchup.goalieComparison` confirm the tandem well enough to
  catch injuries and callups, which no historical rule can?

### Still open

- **The saves model trains on MoneyPuck goalie data.** `goalie_starts.parquet`
  has official saves and shots-against for every appearance plus a real starter
  flag; `features/goalie_mp.py` predates it.
- **287 features on 6,877 games.** `run_feature_selection` exists but
  production RF ignores it; top-20 previously improved Brier 0.2416 → 0.2404.
- **NHL Edge endpoints** — skating speed, shot speed, zone time. The only
  untapped *feature* source found. Speculative.
- `ingestion/odds_api.py` is dead code, superseded by `action_network.py`.
- `pipeline/daily.py` imports `sys` and never uses it.
- Automating the daily run is still undecided (PC vs Pi). `pipeline.daily` is
  one command; what is missing is a lockfile, failure notification through the
  existing webhook, and retry on the MoneyPuck download. **This is now the
  largest single gap** — everything it would run is built and tested.

---

# Session 5 — opening week, 30 September 2026

The season opened on 29 September. `python -m pipeline.daily` produced nothing
on either of the first two nights, ending in "No games scheduled" directly
under three lines naming the games it had dropped. 393 tests passing (was
305). Everything below was measured on the live season.

## 21. The coverage guard refused the whole opening week

`predict` drops any game with under 70% of the model's features populated.
For a team's first game of a season that is every game: rolling features reset
at the boundary, so coverage is ~7%.

That is not a serving fault. In the training matrix:

    min(games played)   n     coverage min / median / max
    0                 106     0.024 / 0.070 / 0.352
    1                  82     0.774 / 0.819 / 0.875
    5                  79     0.857 / 0.944 / 0.972
    25+             4,823     0.774 / 1.000 / 1.000

The split is clean — nothing between 0.35 and 0.77 — so the guard now asks
which side a game is on. A game where either team has yet to play is an
*opener*: it must have an ELO rating for both sides, and is published and
flagged `season_opener`. Everything else keeps the 70% rule, which still
means what it was written to mean.

`docs/SEASON_ROLLOVER.md` documented the old behaviour as a choice ("expect no
predictions for the first few days") with `--min-coverage 0.05` as the way
out. `pipeline.daily` never exposed that flag, and lowering it would have
switched the guard off for the rest of the season.

**Openers are published, not endorsed.** Walk-forward, production model:

| Games | n | model | base rate | difference |
|:--|--:|--:|--:|--:|
| first game of season | 78 | 0.2456 | 0.2496 | −0.004 ± 0.010 |
| games 2–5 | 267 | 0.2453 | 0.2500 | −0.005 ± 0.006 |
| games 6–10 | 325 | 0.2457 | 0.2499 | −0.004 ± 0.006 |
| game 11 onward | 4,807 | 0.2418 | 0.2502 | −0.008 ± 0.002 |

Nothing before game 11 is distinguishable from the base rate. Predictions on
openers have a standard deviation of 0.045 — they barely leave 50%.

The same run has an ELO-only logistic at 0.2413 against the 287-feature
forest's 0.2423 over all 5,477 games. ELO parameters are tuned on the full
history, so that comparison flatters ELO and is not clean — but it is
consistent with the note under "Still open" that feature selection has
previously helped, and it wants a proper look.

## 22. Three live snapshots were one game stale

Found while reading the guard, and the larger problem.

Training computes `shift(1).rolling(w)`, so the row for game *N* holds the
mean of games before *N*. The live builders for special teams, goalie form and
opponent quality computed that same shifted series and took the **last row** —
the pre-game value for the game already played. For the next game that is one
game stale at both ends: the most recent game missing, one too-old game still
in the window.

Replayed against the training matrix mid-way through 2025-26 (every team's
next game after game 767), before the fix:

| Family | cells matching training | median error |
|:--|--:|--:|
| team rolling | 100% | — |
| special teams | 21.6% | 0.20 sd |
| opponent quality | 16.0% | 0.24 sd |
| goalie form | 0% | 0.51 sd |

After: 100% in every family, at that cut and at two in the opening week
(games 20 and 40). Goalie form is compared for the goalie who actually
started; with the *guessed* starter it matches only where the guess is right,
which is the separate, known limit from section 14.

Opponent quality was worse than stale. It read each team's last **home** row
off the feature matrix, so it lagged by however many road games followed, and
a team that had only played away had no row at all.

It mattered most this week: a team's second game got NaN for every PP/PK
feature where training had its first game's numbers, which pushed coverage
under 70% and would have kept dropping games even with section 21 fixed.

Each module now has a `pregame_snapshot` built like
`features.team.pregame_snapshot` — append a placeholder game, run the training
function, read the placeholder. `opponent_quality` also saves the classified
games it rolls (`opponent_quality_games.parquet`) so live uses the same
strong/weak split as training instead of replaying ELO.

The refactor leaves the training matrix byte-identical.

**Why four sessions of tests missed it:** `TestSnapshotSeasonScoping` asserts
on the *source text* of these functions — that a string like
`st_games["season"] == season` appears. Those tests passed throughout.
`tests/test_pregame_snapshots.py` states the property instead: the snapshot
after *k* games equals the training row for game *k + 1*.

**Residual, not fixed:** opponent strength is classified against the *season
median* opponent ELO, which in training is computed over the whole season —
so October rows are classified using information from April. The median sits
near 1500 all year, so the effect is small, but it is look-ahead in a training
feature. Fixing it changes the matrix and needs a retrain.

## 23. Head-to-head meant two different things

`features.context` averages raw `home_win` over the last three meetings — the
home side's win rate in those games, whoever was home. `_load_h2h` flipped
each result to tonight's home team. The live definition is the more natural
quantity; it is not what the model learned. On 300 real games the flipped
version matched training on 33%, the un-flipped on 100%. Live now matches.

Whether the training definition is the *right* feature is a separate question
and probably deserves a no. Changing it means a retrain.

## 24. Nothing kept game dates current

Rest days, back-to-backs, season-day and head-to-head all read dates from the
Postgres `games` table. The only thing that ever wrote dates to it was
`python -m ingestion.nhl_api`, run by hand. `pipeline.daily` passes
`conn=None` to the MoneyPuck ingest on purpose and has no other write.

So 2026-27 opened with zero games in the table. Live, Toronto on the second
night of a back-to-back read as 168 days rested. In the matrix, the new
season's rows had no context features at all.

`sync_completed_games` upserts finished regular-season and playoff games from
the schedule endpoint — one request per week, so one request on a normal day —
and `refresh_data` calls it before the matrix rebuild. `pipeline.live` warns
when Postgres holds fewer current-season games than the stats do, which is the
state that used to pass silently.

Both date queries now take `before=target_date`. Without it, a game finished
earlier the same evening was the "last" one and rest came out as zero.

**The no-database fallback was also broken, by Session 3.** `game_number`
changed from `NNNN` to `TTNNNN` and `approximate_game_date` was still dividing
by 1,312, so game 311 landed in 2033. Only reachable without Postgres — which
is every option in `docs/deployment-options.html` except running it on the PC.
It now takes the sort key, and playoff keys sort after the regular season.

## 25. The opener, the schedule, and ELO on night one

- `SEASON_STARTS[2026]` said `2026-10-05`, with a comment claiming it was
  confirmed against the API. It was not. The season opened **2026-09-29**
  (`regularSeasonStartDate` on `/v1/schedule/{date}`), which made `season_day`
  negative for the first week.
- 2026-27 is an **84-game** season. `TOTAL_GAMES = 1312` was hard-coded;
  `schedule_shape(season)` now reads `SCHEDULE_SHAPES`.
- The ELO replay regresses ratings when it reaches a new season's first game,
  so until one exists the saved snapshot is last April's. Opening night would
  have been served un-regressed ratings on the one slate that rests on ELO
  alone. `ratings_entering` applies the regression when the last game played
  belongs to an earlier season. Too late to matter this year; a test pins it
  against the replay for next.

## 26. A started game is shown, never logged

`pipeline.live` now writes a game to the prediction log only while it is `FUT`
or `PRE`.

This was demonstrated rather than argued. During this session MoneyPuck
published the night's finished games; re-running moved Philadelphia's
"prediction" against Pittsburgh from 51% to 33%, because the features now
included the 7–0 loss in that same game. With `keep="last"` on the log, any
evening re-run of `pipeline.daily` had been free to replace a real pre-game
record with that, plus a live line as its `market_prob_home`.

**Consequence: 29 and 30 September are not in the log and should not be
added.** Nothing run now can reconstruct what the model would have said.

## Open items after Session 5

- **Automation is now the gap that costs something.** The log only records
  games predicted before puck drop, so a missed morning is a missed night.
  See `docs/deployment-options.html`. Whichever host is chosen needs Postgres
  or the date estimate, and the estimate is not good enough for back-to-backs.
- **Game dates should not need a database.** Every boxscore in
  `data/cache/boxscores/` carries `gameDate`. A `game_dates.parquet` built
  alongside the spine would make the pipeline self-contained and fill the ~350
  playoff games from 2021-24 that have no date in Postgres.
- **Retrain is due.** The matrix has grown by a season since the saved model
  was fitted, and the two training-side issues above (opponent-strength
  look-ahead, the head-to-head definition) can only be fixed with one.
- **Feature selection.** See the ELO-only comparison in section 21.
- **Source-text tests.** `tests/test_live_predict.py` still has several that
  assert a string appears in a function. Each is a candidate for the kind of
  miss in section 22.
- Everything under "Open items after Session 4" still stands.
