# goldenbloom

A Python simulator for **Zha Jin Hua** (炸金花, "Golden Flower"), a three-card poker game, modeled as a **dynamic Bayesian game**. Every player acts to maximize their own expected net payoff, using beliefs about opponents' hands that are updated with Bayes' rule as the game unfolds.

- 2 to 17 players (default 5), standard 52-card deck, 3 cards each
- Per-hand-type strategy statistics, cumulative profit per seat, progress bar with ETA
- Multi-process execution for large simulations
- Single file, only dependency: `numpy`

---

## Quick start

```bash
pip install numpy

python goldenbloom.py help                      # list every option and example
python goldenbloom.py --selftest                # verify the betting rules
python goldenbloom.py --games 3 --verbose       # watch 3 full games
python goldenbloom.py --players 6 --games 10000 --train 60 --rollouts 8 --hand-stats --workers 0
```

---

## Game rules (as implemented)

### Hands

Ranking, highest first:

| Rank | Type | Notes |
|---|---|---|
| 1 | Three of a kind | Three cards of equal rank |
| 2 | Straight flush | Straight and flush |
| 3 | Flush | Three cards of one suit |
| 4 | Straight | Three consecutive ranks; **Q-K-A counts, A-2-3 does not** |
| 5 | Pair | Exactly two cards of equal rank |
| 6 | High card | Everything else |

- Ace counts as 14. Suits have no rank.
- Hands of the same type are compared by their three ranks, highest first. This applies to pairs too: `A-2-2` beats `K-Q-Q`.
- **Special case:** a high-card hand `2-3-5` (not suited) beats any three of a kind. Against everything else it is an ordinary high-card hand.
- If two hands have identical ranks, the player who asked for the showdown loses.

### Betting

All amounts in logs and parameters are **actual chips paid**.

- Everyone pays the same ante, the starting player is random, and play goes around the table in seat order.
- A player who has **not looked** at their cards is betting *blind*. A player who has looked is a *seen* player. Looking is free, can be done on your own turn, and cannot be undone.
- A blind player pays the current **blind price** `x`. A seen player must pay `mult × x` (default `2x`). Seen payments are therefore always multiples of `mult`.
- The first player chooses any blind price (at least `--min-bet`). After that, the blind price may never be lower than the previous player's.
- Example with 5 players: ante pot = 5; the dealer bets blind 1 → 6; the next player, having looked, must pay 2 → 8; the next player, still blind, pays only 1 → 9.
- "Raise to N" means that player pays N. After a seen player raises to 10, seen players pay 10 and blind players pay 5.
- Every payment is independent; no one tops up earlier differences. There is no all-in and no rake.

### Actions

On their turn a player may:

| Action | Effect |
|---|---|
| **Fold** | Leave the hand; chips already paid are lost. **The folded hand is shown to everyone.** |
| **Call** | Pay the current amount without raising. |
| **Raise** | Raise the blind price and pay the new amount. |
| **Show (open)** | Pay the current amount and name any active player for a showdown. The target cannot refuse and pays nothing. **Only the two players see the cards;** everyone else just learns who was eliminated. |

- A blind player who opens sees both hands, so they count as having looked afterwards.
- After `--max-actions` actions (default 20, folds and shows included), the remaining players take turns, in original seat order, naming an opponent for a free showdown until one player remains. Each showdown compares only two hands, which avoids cycles caused by the 2-3-5 rule.
- Net payoff = chips won − total chips paid in. The sum over all seats is always zero.

---

## How the players decide

Each decision follows the structure of a Bayesian game.

1. **Beliefs.** Every player keeps a probability distribution over the 22,100 possible hands of each opponent, starting from the uniform deal distribution. After each observed action (fold, call, raise, show, blind or seen) the distribution is reweighted by the likelihood of that action under a behavior model `σ`. Folded hands (public) and showdown results (the winner's hand must beat the loser's) also update beliefs, and known cards are removed.
2. **Expected payoff.** For every legal action (fold, call, a few raise sizes, show against a chosen opponent, and for a blind player also "look first") the agent samples opponent hands from its beliefs and simulates the rest of the game to the end. The estimate is `V(a | I) = E[winnings − total paid in]`.
3. **Choice.** The agent plays `a* ∈ argmax V(a | I)`.
4. **Behavior model `σ`.** `σ(action | seen-status, stake level, hand-strength decile)` is not hard-coded. It starts from a mild heuristic prior and is **learned from self-play frequencies**: simulate, count what best-responding agents do, refresh `σ`, repeat. The `--train` games run this loop before the measured games, approximating the fixed point `(σ*, μ*)` of the Bayesian equilibrium.

Approximations to be aware of:

- Opponents' hands are treated as independent in the belief, with card conflicts resolved at sampling time.
- Win probability by hand strength is approximated as `q^(n−1)`, where `q` is the probability of beating a random hand.
- Inside simulations, a "show" action picks its target at random; real decisions choose targets deliberately.
- Results come from sampling, so more `--rollouts` means more accurate but slower decisions.

---

## Command-line options

Run `python goldenbloom.py help` for the same list in the terminal.

### Scale and speed

| Option | Default | Meaning |
|---|---|---|
| `--players N` | 5 | Number of players, 2–17 |
| `--games N` | 10 | Games to simulate and measure |
| `--train N` | 20 | Warm-up self-play games used to learn `σ` (per process when parallel) |
| `--rollouts N` | 24 | Monte Carlo simulations per decision. **Not a thread count.** |
| `--workers N` | 1 | Parallel processes; `0` uses all CPU cores |
| `--seed N` | 0 | Random seed |

### Rules

| Option | Default | Meaning |
|---|---|---|
| `--ante N` | 1 | Ante |
| `--min-bet N` | 1 | Lowest blind price for the first player |
| `--max-bet N` | 20 | Maximum single payment (a seen player's payment) |
| `--mult N` | 2 | Seen payment as a multiple of the blind price |
| `--max-actions N` | 20 | Actions before the free showdown phase |

### Output

| Option | Meaning |
|---|---|
| `--verbose` | Print every game step by step (disables the progress bar, forces one process) |
| `--hand-stats` | Strategy and results by starting hand type |
| `--report-every X` | Print cumulative net profit per seat every X games (default: about 10 rows) |
| `--csv FILE` | Also write the cumulative-profit table to a CSV file |

A one-line progress display (games done, elapsed time, estimated time remaining) is always shown during a run.

### Other modes

| Option | Meaning |
|---|---|
| `--baseline` | Seat 0 maximizes expected payoff; all others use a naive threshold strategy |
| `--stats` | Print hand-type counts and probabilities, then exit |
| `--selftest` | Check the blind/seen betting rules, then exit |
| `help`, `-h`, `--help` | Show the built-in help |

---

## Output

### Summary

```
共 600 局,5 人,平均底池 67.8,用时 00:58
座位  胜局  平均净收益
  0   118   +0.40
  ...
净收益总和 = +0 (应为 0)
行动统计(全部 4746 次行动,不含看牌与免费开牌): 弃牌1537(32%) 跟注1322(28%) 加注1030(22%) 开牌857(18%) | 闷牌时行动 1436 次,看牌后行动 3310 次
```

(The program prints in Chinese.) The action line counts every action by every player across all games: fold, call, raise and show. Looking at cards and free showdowns are not counted. `blind` plus `seen` actions add up to the same total.

### Strategy by starting hand type (`--hand-stats`)

For each of the six hand types: number dealt, how often the player looked before their first action, the distribution of the **first** action, the share of hands that won the pot, average chips paid in, and average net profit.

Illustrative run (5 players, 600 games, `--train 60 --rollouts 8`):

| Hand | Dealt | Looked first | Fold | Call | Raise | Show | Win | Avg paid | Avg net |
|---|---|---|---|---|---|---|---|---|---|
| Three of a kind | 8 | 88% | 0% | 38% | 62% | 0% | 100% | 52.0 | +79.00 |
| Straight flush | 6 | 67% | 0% | 33% | 67% | 0% | 100% | 59.3 | +69.17 |
| Flush | 166 | 61% | 1% | 30% | 51% | 17% | 89% | 45.7 | +36.76 |
| Straight | 78 | 56% | 3% | 24% | 47% | 27% | 76% | 35.2 | +17.88 |
| Pair | 472 | 63% | 12% | 23% | 41% | 24% | 42% | 24.4 | +1.85 |
| High card | 2270 | 59% | 47% | 15% | 22% | 16% | 8% | 8.0 | −4.15 |

Rare hands (three of a kind is about 0.24% of deals) need thousands of games before their rows are reliable. The strategy shown is endogenous: it depends on the rules, the number of players, the stake limits and how well `σ` has converged.

### Cumulative profit per seat

```
python goldenbloom.py --players 5 --games 100 --report-every 20 --csv profit.csv
```

```
    games  seat0  seat1  seat2  seat3  seat4  total
       20    -11    -84     -7    -10   +112     +0
       40    -31   -143    +43    +76    +55     +0
      ...
      100   -147   +208    -27   -314   +280     +0
```

Each row is the cumulative total after that many games. Seats use the same strategy, so large swings over short runs are luck and should shrink relative to volume over long runs. With `--workers > 1` each process records at 1/W of the interval and the records are summed, which is statistically equivalent but not one strictly sequential series; use `--workers 1` for a strictly ordered curve.

---

## Performance tips

- Use `--workers 0` to use every core; speed scales roughly with the number of cores.
- Lower `--rollouts` (8 to 12 is usually enough for aggregate statistics) for much faster runs at the cost of noisier decisions.
- `--train 60` is a reasonable warm-up; each worker repeats it.
- Fewer players run much faster. 17-player games are slow.
- Rough guide: about 0.13 s per 6-player game at `--rollouts 8` on a single core.

---

## Notes on the hand-type table

An early draft of the rules listed straights as 12 starting points (counting A-2-3). Since A-2-3 does not count, the program uses 11 starting points: 44 straight flushes and 660 straights out of 22,100 hands. Run `python goldenbloom.py --stats` to see the exact counts and probabilities.

| Type | Combinations | Probability |
|---|---|---|
| Three of a kind | 52 | 0.235% |
| Straight flush | 44 | 0.199% |
| Flush | 1,100 | 4.977% |
| Straight | 660 | 2.986% |
| Pair | 3,744 | 16.941% |
| High card | 16,500 | 74.661% |

---

## License

Provide your own license text here (for example MIT) before publishing.
