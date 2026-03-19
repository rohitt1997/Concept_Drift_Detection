from river import tree, drift
from experiment import Experiment
from joblib import Parallel, delayed
import itertools
from glob import glob
import os

from drift_detectors import (
    RDDM_M,
    EDDM_M,
    STEPD_M,
    ECDDWT_M,
    ADWINDW,
    KSWINDW,
    PHDW,
    FHDDMDW,
    FHDDMSDW,
    JSDDW,
    WassersteinDW,
    KSTestDW,
)
from drift_detectors import ECDDWTConfig, EDDMConfig, RDDMConfig, STEPDConfig
from utils.csv import ARFFStream

models = [
    ("HT", tree.HoeffdingTreeClassifier()),
    ("AHT", tree.HoeffdingAdaptiveTreeClassifier()),
    ("HT_DW", drift.DriftRetrainingClassifier(model=tree.HoeffdingTreeClassifier())),
]

dds = [
    ("ADWIN", ADWINDW()),
    ("PageHinkley", PHDW()),
    ("HDDM", drift.binary.HDDM_W()),
    ("KSWIN", KSWINDW()),
    ("DDM", drift.binary.DDM()),
    ("RDDM", RDDM_M(RDDMConfig())),
    ("STEPD", STEPD_M(STEPDConfig())),
    ("ECDD", ECDDWT_M(ECDDWTConfig())),
    ("EDDM", EDDM_M(EDDMConfig())),
    ("JS", JSDDW(window_size=100, drift_threshold=0.18, warning_threshold=0.10)),
    ("WASS", WassersteinDW(window_size=100, drift_threshold=0.15, warning_threshold=0.08)),
    ("KS", KSTestDW(window_size=100, drift_threshold=0.20, warning_threshold=0.10)),
]


def task(stream_path, model, dd):
    stream = ARFFStream(stream_path)
    stream_name = os.path.splitext(os.path.basename(stream_path))[0]
    stream_output = "./real-world-output"
    os.makedirs(stream_output, exist_ok=True)

    model_name, model_obj = model
    model_obj = model_obj.clone()

    dd_name, dd_obj = dd
    dd_obj = dd_obj.clone()

    if isinstance(model_obj, drift.DriftRetrainingClassifier):
        model_obj.drift_detector = dd_obj.clone()

    exp_name = f"{model_name}_{dd_name}_{stream_name}"
    print(f"Running {exp_name}...")

    exp = Experiment(
        exp_name,
        stream_output,
        model_obj,
        dd_obj,
        stream,
        stream_size=stream.n_samples,
    )

    exp.run()
    exp.save()


PATH = "./real-world-datasets/"
EXT = "*.arff"
streams = [
    file
    for path, subdir, files in os.walk(PATH)
    for file in glob(os.path.join(path, EXT))
]

for model in models:
    Parallel(n_jobs=4)(
        delayed(task)(stream, model, dd)
        for stream, dd in itertools.product(streams, dds)
    )
