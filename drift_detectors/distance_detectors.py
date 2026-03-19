from __future__ import annotations

import collections
import itertools
import math
from typing import Callable, Iterable

from river.base.drift_detector import BinaryDriftAndWarningDetector


def _validate_binary(x):
    if x not in (0, 1):
        raise ValueError(
            f"Distance-based detectors expect binary input 0/1, got {x!r}"
        )


def _mean(window: Iterable[float]) -> float:
    window = list(window)
    return sum(window) / len(window) if len(window) > 0 else 0.0


def _bernoulli_probs(window: Iterable[int]) -> tuple[float, float]:
    """
    Converts a binary window into a Bernoulli PMF:
    P(1) = mean(window), P(0) = 1 - mean(window)
    """
    p1 = _mean(window)
    p0 = 1.0 - p1
    return p0, p1


def _safe_log2(x: float) -> float:
    return math.log2(x) if x > 0 else 0.0


def js_distance_binary(ref_window: Iterable[int], cur_window: Iterable[int]) -> float:
    """
    Jensen-Shannon distance between two Bernoulli distributions estimated
    from binary windows.
    """
    p = _bernoulli_probs(ref_window)
    q = _bernoulli_probs(cur_window)
    m = ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)

    def kl(a, b):
        return sum(ai * (_safe_log2(ai) - _safe_log2(bi)) for ai, bi in zip(a, b) if ai > 0)

    js_div = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    js_div = max(js_div, 0.0)
    return math.sqrt(js_div)


def wasserstein_distance_binary(ref_window: Iterable[int], cur_window: Iterable[int]) -> float:
    """
    For binary support {0,1}, the 1D Wasserstein distance equals the absolute
    difference in means.
    """
    return abs(_mean(ref_window) - _mean(cur_window))


def ks_statistic(ref_window: Iterable[int], cur_window: Iterable[int]) -> float:
    """
    Two-sample Kolmogorov-Smirnov statistic on empirical samples.
    Works for binary input and also for generic scalar input if needed later.
    """
    x = sorted(ref_window)
    y = sorted(cur_window)
    n = len(x)
    m = len(y)

    if n == 0 or m == 0:
        return 0.0

    i = j = 0
    d = 0.0

    values = sorted(set(x) | set(y))
    for v in values:
        while i < n and x[i] <= v:
            i += 1
        while j < m and y[j] <= v:
            j += 1

        fx = i / n
        fy = j / m
        d = max(d, abs(fx - fy))

    return d


class WindowDistanceDriftDetector(BinaryDriftAndWarningDetector):
    """
    Generic two-window distance-based drift detector.

    It maintains a rolling window of size 2 * window_size.
    The first half is the reference window, and the second half is the current window.
    A drift is raised when the chosen distance/statistic exceeds drift_threshold.
    A warning is raised when it exceeds warning_threshold.
    """

    def __init__(
        self,
        window_size: int = 100,
        drift_threshold: float = 0.2,
        warning_threshold: float | None = None,
        distance_fn: Callable[[Iterable[int], Iterable[int]], float] | None = None,
    ):
        super().__init__()

        if window_size <= 1:
            raise ValueError("window_size must be > 1")
        if drift_threshold < 0:
            raise ValueError("drift_threshold must be >= 0")

        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.warning_threshold = (
            warning_threshold if warning_threshold is not None else drift_threshold * 0.6
        )
        self.distance_fn = distance_fn

        if self.warning_threshold < 0:
            raise ValueError("warning_threshold must be >= 0")
        if self.warning_threshold > self.drift_threshold:
            raise ValueError("warning_threshold must be <= drift_threshold")

        self._reset()

    def _reset(self):
        self._window = collections.deque(maxlen=2 * self.window_size)
        self._drift_detected = False
        self._warning_detected = False
        self.last_score = 0.0

    def _split_windows(self):
        ref = list(itertools.islice(self._window, 0, self.window_size))
        cur = list(itertools.islice(self._window, self.window_size, 2 * self.window_size))
        return ref, cur

    def update(self, x):
        # Preserve current-step signal; reset at the next update.
        if self.drift_detected:
            self._reset()

        _validate_binary(x)
        self._window.append(int(x))

        if len(self._window) < 2 * self.window_size:
            self._drift_detected = False
            self._warning_detected = False
            self.last_score = 0.0
            return self

        ref, cur = self._split_windows()
        self.last_score = self.distance_fn(ref, cur)

        self._warning_detected = self.last_score >= self.warning_threshold
        self._drift_detected = self.last_score >= self.drift_threshold

        return self


class JSDDW(WindowDistanceDriftDetector):
    """
    Jensen-Shannon Distance Drift + Warning detector.
    Good starting defaults for binary correctness stream.
    """
    def __init__(
        self,
        window_size: int = 100,
        drift_threshold: float = 0.18,
        warning_threshold: float = 0.10,
    ):
        super().__init__(
            window_size=window_size,
            drift_threshold=drift_threshold,
            warning_threshold=warning_threshold,
            distance_fn=js_distance_binary,
        )


class WassersteinDW(WindowDistanceDriftDetector):
    """
    Wasserstein Distance Drift + Warning detector.
    On binary correctness stream this is |mean(ref) - mean(cur)|.
    """
    def __init__(
        self,
        window_size: int = 100,
        drift_threshold: float = 0.15,
        warning_threshold: float = 0.08,
    ):
        super().__init__(
            window_size=window_size,
            drift_threshold=drift_threshold,
            warning_threshold=warning_threshold,
            distance_fn=wasserstein_distance_binary,
        )


class KSTestDW(WindowDistanceDriftDetector):
    """
    Kolmogorov-Smirnov statistic Drift + Warning detector.
    """
    def __init__(
        self,
        window_size: int = 100,
        drift_threshold: float = 0.20,
        warning_threshold: float = 0.10,
    ):
        super().__init__(
            window_size=window_size,
            drift_threshold=drift_threshold,
            warning_threshold=warning_threshold,
            distance_fn=ks_statistic,
        )
