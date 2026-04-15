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

### Test Results (v1)
- 15/15 wins vs Yolanda (random), margins 10-30+ pts
- 5/5 wins vs RatGuesser
- ~3-5 seconds per game total

## TODO - Improvements Needed to Beat Carrie
- [ ] Better heuristic: cell potential scoring + distance awareness
- [ ] Deeper/faster minimax: transposition tables, better pruning
- [ ] Multi-turn prime line planning (plan sequences, not just single moves)
- [ ] Opponent disruption: detect and block opponent prime lines
- [ ] Adaptive rat search: more aggressive when leading, conservative when behind
- [ ] Endgame strategy (final 5-10 turns)
- [ ] Hyperparameter tuning across many games
- [ ] Stress test timing (must stay well under 240s)
