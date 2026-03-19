from drift_detectors.distance_detectors import JSDDW, WassersteinDW, KSTestDW

stream = [1] * 120 + [0] * 120

detectors = {
    "JS": JSDDW(window_size=50, drift_threshold=0.18, warning_threshold=0.10),
    "WASS": WassersteinDW(window_size=50, drift_threshold=0.15, warning_threshold=0.08),
    "KS": KSTestDW(window_size=50, drift_threshold=0.20, warning_threshold=0.10),
}

for name, detector in detectors.items():
    print(f"\nTesting {name}")
    for i, x in enumerate(stream):
        detector.update(x)
        if detector.warning_detected:
            print(f"{name} warning at {i}, score={detector.last_score:.4f}")
        if detector.drift_detected:
            print(f"{name} drift at {i}, score={detector.last_score:.4f}")
            break
