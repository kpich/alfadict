"""Year statistics for viewer: KDE smoothing and variable-width bucketing."""

from collections import defaultdict
import math


def year_to_bucket(year: int) -> int:
    """Return the start year of the variable-width time bucket for a given year."""
    if year >= 2001:
        return year
    elif year >= 1901:
        return (year // 5) * 5
    elif year >= 1801:
        return (year // 10) * 10
    elif year >= 1601:
        return (year // 20) * 20
    else:
        return (year // 50) * 50


def _bucket_width(bucket_start: int) -> int:
    if bucket_start >= 2001:
        return 1
    elif bucket_start >= 1901:
        return 5
    elif bucket_start >= 1801:
        return 10
    elif bucket_start >= 1601:
        return 20
    else:
        return 50


def compute_year_buckets(
    sense_year_counts: dict[str, dict[int, int]],
    year_totals: dict[int, int],
) -> dict[str, list[tuple[float, float]]]:
    """Aggregate per-year counts into variable-width time buckets.

    Returns sense_key -> [(bucket_midpoint, proportion), ...] sorted by midpoint.
    Only buckets with data in year_totals are included (no zero imputation).
    """
    if not sense_year_counts or not year_totals:
        return {}

    bucket_totals: dict[int, int] = defaultdict(int)
    for year, total in year_totals.items():
        bucket_totals[year_to_bucket(year)] += total

    result: dict[str, list[tuple[float, float]]] = {}
    for sk, year_counts in sense_year_counts.items():
        bucket_counts: dict[int, int] = defaultdict(int)
        for year, count in year_counts.items():
            bucket_counts[year_to_bucket(year)] += count

        pts: list[tuple[float, float]] = []
        for bucket in sorted(bucket_counts):
            total = bucket_totals.get(bucket, 0)
            if total > 0:
                width = _bucket_width(bucket)
                pts.append((bucket + width / 2, bucket_counts[bucket] / total))
        result[sk] = pts

    return result


def compute_year_kde(
    sense_year_counts: dict[str, dict[int, int]],
    year_totals: dict[int, int],
    bandwidth: float = 2.5,
) -> dict[str, list[tuple[float, float]]]:
    """Nadaraya-Watson kernel smooth of per-year sense proportions.

    Returns sense_key -> [(year_float, proportion), ...] over a dense grid.
    """
    if not sense_year_counts or not year_totals:
        return {}

    all_years = list(year_totals.keys())
    year_min = min(all_years)
    year_max = max(all_years)

    n_grid = 300
    step = (year_max + bandwidth - (year_min - bandwidth)) / (n_grid - 1)
    grid = [year_min - bandwidth + i * step for i in range(n_grid)]

    def kernel(u: float) -> float:
        return math.exp(-0.5 * u * u)

    result: dict[str, list[tuple[float, float]]] = {}
    for sk, year_counts in sense_year_counts.items():
        pts: list[tuple[float, float]] = []
        for t in grid:
            num = sum(
                count * kernel((t - y) / bandwidth) for y, count in year_counts.items()
            )
            den = sum(
                total * kernel((t - y) / bandwidth) for y, total in year_totals.items()
            )
            if den > 1e-10:
                pts.append((t, num / den))
        result[sk] = pts

    return result
