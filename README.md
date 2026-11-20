# gta6-trophy-rarity-forecast
[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wahyuda110/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wahyuda110/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
A forecaster for every GTA VI PlayStation trophy's earn rate on Exophase at 2026-11-26T00:00:00Z. It treats each rate as earners over players tracked and models the two forces separately: earners and tracked players are each fitted as straight lines in log time over the last 24 hours of reads and extrapolated to the target, averaged on the logit scale with a direct fit of the rate itself, then held to a soft cap per difficulty class read from the trophy's tier and description (story chapters rise fast, grinds and collectibles crawl, the platinum stays near the 0.01% floor since nobody finishes a 200-hour game in week one). The spread on the logit scale combines the fit's residual scatter, a random walk widening with the square root of the horizon, and a term for an overdue Exophase refresh; when the whole history is one unchanged page, as on launch night, it instead assumes a refresh is due, dilutes every rate by the expected jump in tracked players, and keeps the band wide enough to cover both no refresh and a full one.
## Run it
```sh
mkdir -p out && MAKEFUN_INPUT_DIR=~/workspace/makefun/gta6-trophy-rarity-forecast-01-kit/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
---
<sub>Entered on [make.fun](https://make.fun/?ref=g0v1qb0d&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
