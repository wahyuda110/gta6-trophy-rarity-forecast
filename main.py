"""GTA VI trophy rarity forecaster for make.fun gta6-trophy-rarity-forecast-01.

Forecasts, for every trophy of Grand Theft Auto VI on PS5, five quantiles of
its earn rate (%) on Exophase at a case's target time, from the case's history
of reads of that page. Standard library only, fully deterministic, no network.

Model
-----
An earn rate is a ratio: earners / players_tracked. Two forces move it:

  * earning -- players who already track the game earn more trophies, so the
    numerator grows; fast for early story trophies, slow for grinds and
    collectibles, ~zero for the platinum (about 200 hours of play);
  * dilution -- new players join Exophase tracking with zero trophies, so the
    denominator grows and every rate is pushed down. On launch night the
    page sat unchanged for hours and then one refresh multiplied the tracked
    players ~6x and cut nearly every rate, so dilution dominates in jumps.

The forecast splits the two forces instead of extrapolating the ratio:

  1. Time is measured in hours since the game's release, squashed as
     x = ln(1 + hours). Both the tracked-player count and each trophy's
     earner count are fitted, by least squares, as straight lines in x over
     the last 24 hours of reads (the launch-night page was a stale seed and
     is left behind), then read at the target. The ratio of the two fitted
     counts is one median estimate; a straight-line fit of the logit's rate
     itself is a second one. The median is their average on the logit scale,
     which hedges the two extrapolations against each other.
  2. The median is then held to what a trophy can plausibly reach by the
     target: a soft cap per difficulty class, read off the trophy's tier and
     description (story chapter, final mission, one-off activity, grind /
     collectible, 100%-completion, platinum). The platinum is capped near the
     0.01% floor -- nobody finishes a 200-hour game in week one.
  3. The spread on the logit scale combines the fit's residual scatter, a
     random-walk term growing with sqrt(horizon in days), and a term for an
     overdue Exophase refresh (the page unchanged for longer than its usual
     refresh gap means a jump is due). A trophy whose last read is 100%
     keeps a tight band near 100%: every tracked player had already earned
     it through a 6x refresh, so new tracked players have too.

When the whole history is flat -- every read the same page, as on launch
night -- no trend exists to fit. The forecast then assumes a refresh is
due: the expected jump in log players is the size of a launch-phase refresh
(about 1.6 log units) times how overdue one is, and each trophy's earners
are credited with a little growth in the meantime, fastest for story
trophies. The band is kept wide (about 1 logit unit) so it honestly covers
both "no refresh yet" and "a full refresh landed".

Quantiles are the median shifted by the standard normal's z at each level
times the spread, mapped back through the logistic, rounded to 4 decimals.
Every value is clamped to [0.01, 99.99] before any logit, matching the
scorer's clamp, and each row is non-decreasing by construction.
"""

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# How big a launch-phase Exophase refresh is, in log players (638 -> 3868 was
# ~1.8; a little shaved for caution), and the usual gap between refreshes.
REFRESH_LOG_JUMP = 1.6
REFRESH_GAP_HOURS = 5.0
# Spread of the random-walk part, in logit units per square-root day.
WALK_SIGMA = 0.45


def parse_time(text):
    """ISO 8601 timestamp (with a trailing Z) as an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(value):
    if value > 40:
        return 99.99
    if value < -40:
        return 0.01
    return 100.0 / (1.0 + math.exp(-value))


def round4(x):
    """Round to 4 decimals, half up, deterministically (no banker's rounding)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def ols(xs, ys):
    """Least-squares line y ~ a + b*x; returns (a, b, residual std, n).

    Returns None when there are too few points or no spread in x.
    """
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return a, b, resid, n


def classify(trophy):
    """Difficulty class from tier + description. Drives caps and earning speed.

    platinum -> endgame nobody reaches in week one; gold 100% completion ->
    same, slightly less extreme; story chapters / final mission -> players
    march through them; counted or "every/all" tasks -> grinds; the rest are
    one-off activities.
    """
    tier = trophy.get("tier", "bronze")
    desc = (trophy.get("description") or "").lower()
    if tier == "platinum":
        return "platinum"
    if "100% completion" in desc:
        return "completion"
    if "complete the final mission" in desc:
        return "story-final"
    if "complete chapter" in desc:
        return "story"
    count = re.search(r"(\d[\d,]*)", desc)
    big_count = count is not None and int(count.group(1).replace(",", "")) >= 8
    if big_count or " every " in desc or " all " in desc or "collectible" in desc:
        return "grind"
    return "activity"


# Soft cap on the median rate (%) a trophy can reach by the target, and how
# fast its earners grow while history is flat (logit units per hour).
CLASS_INFO = {
    "platinum": (0.5, 0.0),
    "completion": (3.0, 0.0),
    "story-final": (45.0, 0.30),
    "story": (75.0, 0.30),
    "activity": (65.0, 0.15),
    "grind": (20.0, 0.05),
}


def soft_cap(median, cap):
    """Pull a median above its cap most of the way back down to the cap."""
    if median <= cap:
        return median
    return cap + (median - cap) * 0.2


def quantiles(median, sigma):
    centre = logit(median)
    row = []
    for z in Z:
        value = round4(clamp_rate(inv_logit(centre + z * sigma)))
        row.append(value)
    # Rounding can only tie neighbours, never reverse them; enforce it anyway.
    for i in range(1, len(row)):
        if row[i] < row[i - 1]:
            row[i] = row[i - 1]
    return row


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    trophies = case["trophies"]
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    horizon_hours = case["horizon_minutes"] / 60.0
    horizon_days = case["horizon_minutes"] / 1440.0

    release = parse_time(case["game"]["release_date"] + "T00:00:00Z") \
        if case.get("game", {}).get("release_date") else parse_time(history[0]["at"])

    def hours_since_release(when):
        return max(0.0, (parse_time(when) - release).total_seconds() / 3600.0)

    # Per-read series: x = ln(1 + hours since release), players, rates.
    reads = []
    for read in history:
        players = read.get("players_tracked")
        rates = read.get("rates") or []
        if players is None or not rates:
            continue
        reads.append({
            "x": math.log1p(hours_since_release(read["at"])),
            "at": parse_time(read["at"]),
            "players": float(players),
            "rates": rates,
        })
    if not reads:
        raise ValueError(f"{name} has no usable reads")
    last = reads[-1]

    # How long the page has sat unchanged, and the usual gap between changes.
    changed_at = reads[0]["at"]
    gaps = []
    prev = None
    for read in reads:
        if prev is not None and (read["players"] != prev["players"]
                                 or read["rates"] != prev["rates"]):
            gaps.append((read["at"] - prev["at"]).total_seconds() / 3600.0)
            changed_at = read["at"]
        prev = read
    flat_hours = (reads[-1]["at"] - changed_at).total_seconds() / 3600.0
    refresh_gap = sorted(gaps)[len(gaps) // 2] if gaps else REFRESH_GAP_HOURS
    refresh_gap = min(max(refresh_gap, 1.0), 24.0)

    # Is the history moving at all? A flat page means no trend can be fitted.
    moving = any(
        read["players"] != reads[0]["players"] or read["rates"] != reads[0]["rates"]
        for read in reads
    )

    # How overdue a refresh is: 0 when one just landed, 1 when a full gap has
    # passed since the page last changed (plus the horizon itself).
    overdue = min(1.0, (flat_hours + horizon_hours) / refresh_gap)
    overdue_sigma = 0.5 * overdue if flat_hours > refresh_gap / 2 else 0.0

    # Recent window for the trend fits: the last 24 hours of reads.
    window_start = last["at"].timestamp() - 24 * 3600
    window = [r for r in reads if r["at"].timestamp() >= window_start]
    if len(window) < 3:
        window = reads

    target_x = math.log1p(hours_since_release(case["target"]))
    walk = WALK_SIGMA * math.sqrt(horizon_days)

    rows = []
    for i, trophy in enumerate(trophies):
        cls = classify(trophy)
        cap, earn_rate = CLASS_INFO[cls]

        # Last non-null rate for this trophy (the case's last read first).
        last_rate = None
        for read in reversed(reads):
            if i < len(read["rates"]) and read["rates"][i] is not None:
                last_rate = float(read["rates"][i])
                break
        if last_rate is None:
            last_rate = {"platinum": 0.05, "completion": 0.1}.get(cls, 1.0)

        # A trophy everyone already has stays near 100%.
        if last_rate >= 99.99:
            rows.append({"id": trophy["id"], "q": quantiles(99.9, 0.4)})
            continue

        if not moving:
            # Flat history: expect a refresh. The median rate is diluted by
            # the expected jump in tracked players, partly offset by earners
            # growing in the meantime (fastest for story trophies); the band
            # stays wide enough to cover no refresh or a full one.
            dilution = REFRESH_LOG_JUMP * overdue
            earning = min(0.5, earn_rate * horizon_hours)
            median = inv_logit(logit(last_rate) - dilution + earning)
            median = soft_cap(median, cap)
            sigma = 1.0
        else:
            # Trend regime: two median estimates, averaged on the logit scale.
            xs, lrs, les, lns = [], [], [], []
            for read in window:
                if i >= len(read["rates"]) or read["rates"][i] is None:
                    continue
                rate = float(read["rates"][i])
                if rate <= 0:
                    continue
                xs.append(read["x"])
                lrs.append(logit(rate))
                lns.append(math.log(read["players"]))
                les.append(math.log(max(1e-6, rate * read["players"] / 100.0)))
            fit_rate = ols(xs, lrs)
            fit_earn = ols(xs, les)
            fit_players = ols(xs, lns)

            estimates = []
            resid = 0.25
            if fit_rate is not None:
                a, b, resid, _ = fit_rate
                estimates.append(a + b * target_x)
            if fit_earn is not None and fit_players is not None:
                ea, eb, _, _ = fit_earn
                pa, pb, _, _ = fit_players
                earners = math.exp(ea + eb * target_x)
                players = math.exp(pa + pb * target_x)
                estimates.append(logit(100.0 * earners / max(players, 1.0)))
            if estimates:
                centre = sum(estimates) / len(estimates)
            else:
                centre = logit(last_rate)
            median = soft_cap(inv_logit(centre), cap)
            if cls == "platinum":
                median = min(median, 0.5)
            sigma = math.sqrt(min(resid, 1.0) ** 2 + walk ** 2 + overdue_sigma ** 2)
            sigma = min(max(sigma, 0.55), 1.6)
            if cls == "platinum":
                sigma = max(sigma, 0.7)

        rows.append({"id": trophy["id"], "q": quantiles(median, sigma)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
