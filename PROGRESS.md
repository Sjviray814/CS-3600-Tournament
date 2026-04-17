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

## Decision (Apr 16 evening)
- **v4 archived** to `v4-archive` branch (commit 57baaea) — includes NMP gated off
- **Baseline reset**: extracted v2 source (1100 lines) from `~/Downloads/tournament_v2.zip` over agent.py on `arnav-trials` branch — committed as **v5 baseline**
- **v5 built on top of v2** with 3 surgical change-sets (CS-0 + CS-1 + CS-2). CS-3 (NMP retest) deferred until CS-1 validated — NMP over-prunes with broken heuristics.

---

## v5 Design — Approved Apr 16 evening

### Architecture
- Single file: `3600-agents/Challenger/agent.py`
- 4 change-sets, each ~5-40 lines, stacked on v2. No refactor of search loop, TT, killers, history, quiescence, aspiration windows, or time management.

### CS-0 — Trapped-state safety fix
**agent.py:752 (iterative_deepening_search) and :900 (greedy_move)**: when no non-search move is legal, return `Move.search((wx, wy))` (always legal, worst case -2 pts) instead of `Move.plain(Direction.RIGHT)` (may be invalid → `INVALID_MOVE` → instant forfeit).

### CS-1 — Heuristic patch (the big one)
- **CS-1a. Prime-bias amplification** (`evaluate`): compute `prime_amp = 2.0 if my_cv < 4.0 else 1.0` and multiply `my_chain * w_mch` and `prim * w_prim` by it. When the best immediate carpet is worth <4 pts (len-1 or len-2), we double the weight on chain potential + primeable bonus so the bot invests in priming rather than burning len-2 or playing positional-only moves. Closes the 11→25 prime-count gap.
- **CS-1b. Carpet-patience at root** (`iterative_deepening_search`): once per turn compute `patience_active = can_prime and turns_left > 5` at root. In the root move loop, apply `score -= 3.0` to every `CARPET len-2` move when `patience_active`. Existing len-1 penalty (-2) kept. Closes the carpet-length gap (us len-2 vs Albert len-3/4).
- **CS-1c. Belief-guided movement** (module var + `evaluate`): once per turn call `set_belief_proximity(rat_tracker.belief)` which precomputes a 64-cell proximity table `_BELIEF_PROX[i] = sum(belief[c] * max(0, 8 - manhattan(i, c)))`. `evaluate()` adds `(my_bp - opp_bp*0.4) * W_BELIEF[phase]` with small weights `(0.15, 0.20, 0.10)`. Precomputation is O(64*64) per turn; eval lookup is O(1).

### CS-2 — Root SEARCH post-gate
In `PlayerAgent.play()` after `iterative_deepening_search` returns: if best_move is non-carpet (positional only, 0 pts immediate) AND `rat_tracker.best_guess().prob >= 0.45` AND `rat_ev >= 2.0` AND `search_cooldown == 0` AND `observations_since_reset >= 2`, override with `Move.search(best_guess)`. Treats SEARCH as a first-class root candidate when the tree couldn't find a carpet.

### CS-3 — NMP retest (deferred)
Cherry-pick NMP from v4-archive, toggle `NMP_ENABLED = True`, A/B test vs Yolanda. Only attempt after CS-1 is validated as an improvement. Prior A/B test showed NMP regressed by 60 pts/game because it trusted `evaluate()` as a cut gate; fixing the eval is a prerequisite.

### Testing gates
1. **Gate 1 — smoke**: Challenger v5 vs Yolanda, 3 games each side. Must win all, no crashes.
2. **Gate 2 — sanity**: vs RatGuesser, BasicMovement, FirstIteration (3 each, both sides). Expect 100% WR.
3. **Gate 3 — upload v5** to bytefight. Run 10+ ladder matches. Target >40% WR (match v2) minimum, >50% WR goal.
4. **Gate 4 — CS-3 decision**: if Gate 3 passes, cherry-pick NMP and re-test.

---

## v6 — Data-Driven Overhaul (Apr 16, based on 57-game deep analysis)

### Deep analysis performed
- **27 games of Michael (#1, bot "Argghhhh")** — 17W-7L-3T
- **20 games of Carrie (TA bot)** — 10W-8L-2T
- **10 games of us (v5) vs Albert Lite** — 2W-8L, -14.0 margin/game

### Key correlations across 57 games
- Carpet pts → score: r = **+0.692** (strongest predictor)
- % search turns → score: r = **-0.471** (searching hurts)
- % scoring turns → score: r = **+0.561**

### v6 changes (on v5)
1. Raised rat search thresholds (prob 0.40-0.55, min_obs 3/5, cooldown 3) — target ~6 searches/game
2. Removed CS-2 post-gate (was adding unvetted searches after tree search)
3. Boosted carpet eval weights 3x (W_MY_CARPET 2.0/3.0/4.0)
4. Removed len-2 carpet-patience penalty (need MORE carpets, not longer)

### v6 results vs Albert Lite (10 games, matches 58-67)
- **5W-5L, +2.5 margin/game** (up from 2W-8L, -14.0)
- Searches: 6.1/g at 57% hit rate (was 13.3 at 35%)
- Carpets: 4.5/g, 12.8 pts/g (was 3.1, 9.5)
- Scoring: Us 39.5 vs Albert 37.0

### Remaining gap
- Albert: 7.5 carpets/g, 18.7 carpet pts/g vs our 4.5/g, 12.8 pts/g
- Root cause: our eval only values LOCAL carpet potential; top bots value board-wide primed infrastructure

---

## v7 — Board-Wide Infrastructure Eval (Apr 16, current)

### Structural insight
Michael (#1) has **43% of carpets with 0 consecutive primes immediately before** — he primes areas, leaves, returns later. His eval values ALL primed runs on the board weighted by distance. This drives emergent "prime area A, leave, return to carpet" behavior through search. Our eval only sees immediate/1-step carpet potential.

### v7 changes (on v6)
1. **Board-wide primed infrastructure eval** — new `_primed_infrastructure()` function scans ALL horizontal/vertical primed runs, weights by CARPET_POINTS[len] × distance discount to nearest endpoint. Cached by primed_mask (50k entry LRU). New weights W_MY_INFRA (0.15/0.25/0.10), W_OPP_INFRA (0.08/0.15/0.08). Integrated as component #10 in evaluate().
2. **Flattened time management** — 4.5s early, 4.0s mid, 3.5s late (was 0.7x/1.6x/0.9x adaptive). v6 used only 113s of 240s; targets 150-170s like Carrie (4.01s fixed) and Michael (3.25s avg).
3. **Full-line prime move ordering** — checks both horizontal AND vertical runs through newly primed cell (was only checking run behind). Better alpha-beta pruning for infrastructure-building moves.
4. **HMM belief zeroing on miss** — when player or opponent searches and misses, sets P(rat at cell) = 0 and renormalizes. Provably correct; improves belief sharpness for next search.

### v7 smoke test
- Beat Yolanda **45-1**, used 120.5s of 240s, reached depth 8
- Awaiting bytefight validation vs Albert Lite

---

## Current Status (as of Apr 16 night)
- **agent.py** on disk: v7 = v2 + CS-0 + CS-1(a/b/c) + v6 changes + v7 infra eval
- **Zip**: pending creation (`~/Downloads/tournament_v7.zip`)
- **Next**: upload to bytefight, run 10+ scrimmages vs Albert Lite, compare metrics

---

## TODO - Remaining Improvements

### Done
- [x] Fix trapped-state fallback (CS-0) — v5
- [x] Prime-bias heuristic (CS-1a) — v5
- [x] Patience on short carpets (CS-1b) — v5 (removed in v6)
- [x] Belief-guided movement (CS-1c) — v5
- [x] Root-level SEARCH candidate (CS-2) — v5 (removed in v6)
- [x] Raised search thresholds — v6
- [x] Boosted carpet eval weights — v6
- [x] Board-wide primed infrastructure eval — v7
- [x] Flattened time management — v7
- [x] Full-line prime move ordering — v7
- [x] HMM belief zeroing on miss — v7

### Next up
- [ ] **Null move pruning** — re-enable after infrastructure eval validated
- [ ] **Make/unmake** instead of `forecast_move` cloning (biggest speed lever)
- [ ] **Late Move Reductions (LMR)** on quiet tail moves
- [ ] **Phase-specific objective**: late + ahead → minimize variance
- [ ] **Opponent-style adaptation**: infer type from first 3-5 turns
- [ ] Endgame strategy refinement (final 5-10 turns)
- [ ] Hyperparameter tuning (needs scrimmage data)
