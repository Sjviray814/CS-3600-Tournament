# Tournament Bot Progress

## Setup (Done)
- Added `.gitignore` (matches, pycache, IDE files, pkl binaries)
- Fixed imports in Yolanda and RatGuesser example agents (`engine.game` → `game`) so they run locally
- Applied 2 bug fixes to `engine/gameplay.py` from Ed post #750:
  - Local play time now 240s (was 60s)
  - Search on last turn now correctly affects winner determination

## Challenger Agent (Done - v1)
Located in `3600-agents/Challenger/`

### Components
1. **HMM Rat Tracker** (numpy)
   - Bayesian belief over 64 cells, updated each turn
   - Uses transition matrix T, noise type (squeak/scratch/squeal), and noisy manhattan distance
   - Computes stationary distribution after 1000 steps as prior
   - Resets belief when rat is caught (by us or opponent)
   - Conservative search policy: only guesses when P(best) > 0.5, EV > 1.0, 4+ observations, and not in cooldown

2. **Greedy Move Scoring**
   - Carpet rolls scored highest (immediate points)
   - Prime moves scored by line-extension value (how much they extend existing primed runs)
   - Plain moves scored by positional value (proximity to open/primeable squares)

3. **Expectiminimax with Alpha-Beta Pruning**
   - Iterative deepening (depth 1-4)
   - Move ordering: carpet > prime > plain for better cutoffs
   - Time-managed: adapts depth based on remaining time per turn

4. **Heuristic**
   - Score differential (my points - opponent points)
   - Carpet potential from current position
   - Opponent carpet potential (threat)
   - Proximity to open/primeable squares

### Test Results (v1) — 120/120 wins (local)
| Matchup | As Player A | As Player B |
|---|---|---|
| vs Yolanda (random) | 20-0 | 20-0 |
| vs RatGuesser (HMM only) | 20-0 | 20-0 |
| vs BasicMovement (unfinished minimax) | 20-0 | 20-0 |

- Typical win margins: 10-30+ pts
- ~3-5 seconds per game total (well within 240s budget)

---

## Bytefight Tournament Iterations

### v1 (baseline) — uploaded, weak vs production bots
- Depth 1-4 iterative deepening, 0.64s / 240s used
- Lost 0-9 vs Albert Lite (34-47 point deficit)
- Diagnosis: barely thinking, weak heuristic, search too shallow

### v2 (major overhaul) — **40% baseline, current best**
- Negamax + alpha-beta + PVS + iterative deepening to depth 25+
- Transposition Table (LRU OrderedDict, 2M entries)
- Killer moves, history heuristic, aspiration windows
- 7-component phase-aware evaluation (score diff, carpet potential, threat, line quality, primeable adjacency, territory, mobility)
- Smart time management: 0.6x early, 1.4x mid, 0.8x late; uses ~120-180s of 240s
- Rat search thresholds: prob=0.35 even, 0.45 ahead, 0.25 behind
- **Result vs Albert Lite: 4W-5L-1T (≈40% win rate)**
- Zip: `~/Downloads/tournament_v2.zip` (9383 bytes, Apr 16 00:55)

### v3 (max_depth 25 + length-2 carpet penalty) — REGRESSION
- Added penalty on length-2 carpets to push for longer lines
- Raised max search depth
- **Result: 1W-9L vs Albert Lite**
- Diagnosis: penalty caused refusal to carpet even when len-2 was the best available; Albert scored anyway

### v4 (aggressive rat search) — REGRESSION (worst)
- Lowered rat thresholds: prob 0.35→0.25 (even), 0.45→0.30 (ahead), 0.25→0.18 (behind)
- `min_obs`: 3/2 → 2/1
- `search_cooldown`: 2 → 1
- **Result: 0W-6L vs Albert Lite (scrimmages 21-26)**
- **Overall v4 record: 4W-23L-0T, margin -570**
- Zip: `~/Downloads/tournament_v4.zip` (9692 bytes, Apr 16 05:30)

### Mathematical diagnosis of v4 failure
- Search EV formula: `prob × 4 − (1−prob) × 2 = 6×prob − 2`
- Breakeven: prob = 0.333
- v2 threshold prob=0.35 → EV = **+0.10 pts/search** (positive)
- v4 threshold prob=0.25 → EV = **−0.50 pts/search** (negative!)
- Lowering thresholds guaranteed point loss every search below breakeven

---

## Root-Cause Analysis vs Albert Lite

### The real gap (not rat search)
Turn-by-turn trace of production matches revealed:
- **Prime count**: Albert Lite primes ~25x per game, Challenger primes ~11x per game (2.3× gap)
- **Carpet length**: Albert Lite builds length 3-4 carpets (4-6 pts each), Challenger builds length 2 carpets (2 pts each)
- **Pts/turn from carpet loops**:
  - len-2 loop (prime-prime-carpet): 1.33 pts/turn
  - len-4 loop: 2.00 pts/turn
  - len-5 loop: 2.50 pts/turn
  - len-7 loop: 3.50 pts/turn

### Scoring confirmed from engine source
- PRIME move: +1 pt (only valid on SPACE cells)
- CARPET by length: `[0, -1, 2, 4, 6, 10, 15, 21]`
- SEARCH: hit → +4 (RAT_BONUS), miss → −2 (RAT_PENALTY)
- MAX_TURNS_PER_PLAYER = 40, total budget = 240s

### Analysis caveat (found this session)
- Trace scripts in `/tmp/analyze_v4*.py` and `/tmp/trace_scoring.py` assumed A always moves first (even indices = A's turns)
- Turn-by-turn inspection of actual JSON shows this assumption doesn't always hold
- All per-player stats from those scripts (searches, carpet counts) are **unreliable**
- Only aggregate fields (`a_points[-1]`, `b_points[-1]`, `result`, `reason`) are trusted

---

## Latest Bytefight Ladder Results (Apr 16 ~14:18) — 1W-16L catastrophe

Full ladder run across 17 matches vs varied opponents (Nemesis-1, KrishBot v3, HMMMiniAB, 225LBS, "Good Game!", "ff", other Challenger instances).

| # | We are | Opponent | Result | Score | Margin | Our t | Opp t | Reason |
|---|---|---|---|---|---|---|---|---|
| 0 | A | Unknown (errlog="13.4") | LOSS | 34-47 | -13 | **0.6s** | 147.7s | POINTS |
| 1 | B | Nemesis-1 | LOSS | 6-48 | -42 | 60.2s | 204.0s | POINTS |
| 2 | A | Unknown ("ff") | LOSS | 16-55 | -39 | 56.3s | 160.7s | POINTS |
| 3 | B | KrishBot v3 | **LOSS** | 7-32 | -25 | 75.6s | **2.0s** | POINTS |
| 4 | A | "Good Game!" | LOSS | -10-42 | -52 | 88.7s | 155.0s | POINTS |
| 5 | B | Unknown (minimax d=0) | LOSS | 23-39 | -16 | 72.1s | 73.2s | POINTS |
| 6 | A | HMMMiniAB d=8 | LOSS | 7-56 | -49 | 51.1s | 83.3s | POINTS |
| 7 | A | "Top rat hypothesis…" | LOSS | 22-48 | -26 | 73.9s | 42.9s | POINTS |
| 8 | A | "Top rat candidates…" | **LOSS** | 14-49 | -35 | 69.7s | **11.4s** | POINTS |
| 9 | B | 225LBS d=11.90 | LOSS | -15-43 | -58 | 67.1s | 219.9s | POINTS |
| 10 | A | Challenger ("12.4") | **WIN** | 45-43 | **+2** | 64.0s | 146.9s | POINTS |
| 11 | A | Challenger ("11.93") | LOSS | 8-55 | -47 | 68.8s | 148.4s | POINTS |
| 12 | A | Challenger ("12.82") | LOSS | -6-44 | -50 | 44.3s | 147.3s | POINTS |
| 13 | A | Challenger ("12.85") | LOSS | -22-47 | -69 | 49.6s | 144.8s | POINTS |
| 14 | A | Challenger ("12.85") | LOSS | 25-43 | -18 | 55.5s | 148.3s | POINTS |
| 15 | A | Challenger ("13.15") | LOSS | 18-37 | -19 | 61.7s | 144.2s | POINTS |
| 16 | A | Challenger ("12.47") | LOSS | -2-37 | -39 | 52.7s | 147.2s | POINTS |

**Record: 1W-16L-0T, total margin −595 (avg margin −35/game)**

### Damning evidence: heuristic is broken, not search depth

Two games the opponent used <12 seconds and STILL beat us by 25-35 pts while we used 70+ seconds:
- **Match 3**: KrishBot v3 used **2.0s**, we used 75.6s → we lost by 25 pts
- **Match 8**: Opponent used **11.4s**, we used 69.7s → we lost by 35 pts

Deep search with a bad heuristic loses to shallow greedy with a good heuristic. **The heuristic is the bottleneck**, not search depth.

### Aggregate gap
- Our avg score: ~10 pts/game
- Opponent avg score: ~45 pts/game
- In 16 self-scrims vs "Challenger" (7 games), we went 1W-6L as Player A

### Other patterns
- `ByteFight Match.json` (v1 era): we used **0.6s** — a completely untouched old match, confirms the v1 time-management bug is real but fixed in v2+
- Nobody is hitting INVALID_MOVE on us this run, but the trapped-state bug remains latent

---

## New Improvement Opportunities (repo audit + TA guidance)

### Critical safety fix (confirmed in code)
**agent.py:751** — `iterative_deepening_search` returns `Move.plain(Direction.RIGHT)` when `order_moves` is empty.
**agent.py:898** — `greedy_move` returns `Move.plain(Direction.RIGHT)` when no non-search moves exist.

Both fire when we're trapped (no legal non-search move). `PLAIN(RIGHT)` may be invalid → `EndReason.INVALID_MOVE` → instant forfeit. User verified this state is reachable locally. Fix: fall back to `Move.search()` (always legal) instead.

### TA-recommended (Apr 16)
Must implement TT + iterative deepening + null move pruning.
- ✅ Transposition Table — present (`agent.py:427`, LRU, 2M entries)
- ✅ Iterative deepening — present (`agent.py:738`, with aspiration windows)
- ⚠️ **Null move pruning — implemented but gated off** (`agent.py` negamax_pvs, R=2/3, non-PV only, turns_left ≥ 3 safety, allow_null recursion guard). Local A/B test vs Yolanda: NMP ON = 1W-2L at −60 pts avg; NMP OFF = 3W-0L at +37 pts avg. **Root cause: NMP trusts `evaluate()` as a cut gate (`static_eval ≥ beta`); with the currently-broken heuristic this over-prunes.** Re-enable AFTER v5 heuristic overhaul (toggle `NMP_ENABLED = True`).

### High-impact algorithmic upgrades (user-identified)
1. **Root-level SEARCH candidate** — SEARCH is gated outside tree search. Treat it as a first-class root move using belief-derived expected utility, so tactical-weak + search-strong turns are not missed.
2. **Belief-guided movement** — reward stepping toward high-P(rat) regions even on non-search turns (increases future search EV, shortens catch cycles). Belief is computed but only consulted at the search gate.
3. **Info-gain-aware rat policy** — complement immediate EV with posterior entropy reduction; sometimes a slightly negative-EV search is right because it concentrates belief for later.
4. **Make/unmake instead of `forecast_move` copy** — current negamax clones the board at every node. Mutable state + undo stack is the biggest single engine-side Elo lever.
5. **Selective search**:
   - **Null move pruning** (R=2 or R=3)
   - **Late Move Reductions (LMR)** on low-priority quiet moves
   - **Futility pruning** near leaves
   - **Check-style extensions** for tactical carpet threats
6. **Phase objective switching** — late + ahead: maximize win probability (minimize variance), not raw points. Avoid high-variance search when preserving lead is enough.
7. **Opponent-style adaptation** — infer from first 3-5 turns if opponent is search-heavy / carpet-heavy / line-builder; adjust coefficients live. Better than a static Albert clone for unknown tournament bots.

### Original gap analysis still stands
- Prime count gap: us 11 vs Albert ~25
- Carpet length gap: us len-2 vs Albert len-3/4
- Fix via prime-bias heuristic + patience on short carpets

---

## Current Status (as of Apr 16 afternoon)
- **agent.py** on disk: still contains v4 aggressive rat thresholds
- **Deployed zip**: v4 — confirmed 1W-16L across varied opponents
- **v2.zip (40% baseline)** — still the safest fallback to ship
- **Next session**: design v5 with a clear priority order (see below)

---

## TODO - Improvements Ordered by ROI

### Tier 0 — Safety / don't lose free games
- [ ] **Fix trapped-state fallback** (agent.py:751 and 898): return `Move.search()` not `Move.plain(Direction.RIGHT)` when no legal non-search move
- [ ] **Revert to v2 source** and ship v2.zip as an insurance baseline before attempting v5

### Tier 1 — Fix the heuristic (biggest win, per match data)
- [ ] **Prime-bias heuristic**: when no len-3+ carpet available, prefer prime-building over plain moves (close the 11→25 prime gap)
- [ ] **Patience on short carpets**: prefer len-3+ carpets; only fire len-2 carpet in endgame (last 5-8 turns) or when opponent threatens the line
- [ ] **Line-extension ordering**: prioritize primes that extend existing runs toward length 3, 4, 5
- [ ] **Belief-guided movement**: reward steps toward high-P(rat) regions
- [ ] **Phase-specific objective**: late + ahead → win-probability mode (low variance)

### Tier 2 — Search engine upgrades (TA-guided)
- [ ] **Null move pruning** (R=2, disable in zugzwang-like late-game)
- [ ] **Root-level SEARCH candidate** with belief EV
- [ ] **Make/unmake** instead of `forecast_move` cloning
- [ ] **Late Move Reductions (LMR)** on quiet tail moves
- [ ] **Futility pruning** near leaves
- [ ] **Check-style extensions** for carpet threats

### Tier 3 — Meta / infrastructure
- [ ] **Opponent-style adaptation**: infer opponent type from first 3-5 turns, tune live
- [ ] **Info-gain rat policy**: entropy-reduction term in search EV
- [ ] **Local Albert-Lite clone** or re-use observed opponents as heuristic sparring partners
- [ ] **Re-verify analysis scripts**: fix A/B indexing assumption (`turn i` is A iff `player_a_just_moved`, NOT `i % 2 == 0`)
- [ ] **Rat search: keep v2 thresholds** (prob=0.35 even) — positive EV confirmed mathematically
- [ ] Endgame strategy refinement (final 5-10 turns)
- [ ] Hyperparameter tuning across many games (needs sparring partners first)
