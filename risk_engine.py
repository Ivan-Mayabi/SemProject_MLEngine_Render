import csv, math
from collections import Counter
from pathlib import Path
import joblib
import numpy as np

FLOODING = 0.0   # not modelled
LIGHTING = 0.0   # not modelled
MAX_POINTS = 5000

class OutsideCoverage(Exception):
    pass

class RiskEngine:
    def __init__(self, bundle, rows):
        self.model = bundle["model"]
        self.cols = list(bundle["feature_cols"])
        self.factor = 10 ** int(bundle["grid_round_off"])
        self.ref_max = float(bundle["ref_max_rate"])
        self.hour_mult = np.asarray(bundle.get("hour_multiplier") or [1.0] * 24, dtype=float)
        self.cells = {k: np.log1p(np.array([r[c] for c in self.cols], dtype=float)) for k, r in rows.items()}
        lat = [k[0] for k in self.cells]; lng = [k[1] for k in self.cells]
        self.bbox = (min(lat), max(lat), min(lng), max(lng))
        self.zero = np.zeros(len(self.cols))

    @classmethod
    def load(cls, model_path, cells_path):
        bundle = joblib.load(model_path)
        if not all(c.startswith("osm_") for c in bundle["feature_cols"]):
            raise ValueError("Model uses accident-derived features; re-export from the OSM version")
        rows = {}
        with open(cells_path, newline="") as f:
            for r in csv.DictReader(f):
                rows[(int(r["ilat"]), int(r["ilng"]))] = {c: float(r[c]) for c in bundle["feature_cols"]}
        return cls(bundle, rows)

    def _covered(self, c):
        a, b, d, e = self.bbox
        return a <= c[0] <= b and d <= c[1] <= e

    def predict(self, coords, mode, hour, debug=False):
        if len(coords) > MAX_POINTS:
            coords = coords[::math.ceil(len(coords) / MAX_POINTS)]
        counts = Counter()
        for p in coords:
            if len(p) != 2 or not all(math.isfinite(v) for v in p):
                raise ValueError(f"bad coordinate: {p}")
            cell = (round(p[0] * self.factor), round(p[1] * self.factor))
            if self._covered(cell):
                counts[cell] += 1
        if not counts:
            raise OutsideCoverage("no point is inside the area this model covers")

        cells = list(counts)
        w = np.array([counts[c] for c in cells], dtype=float)
        X = np.vstack([self.cells.get(c, self.zero) for c in cells])
        rate = np.clip(np.expm1(self.model.predict(X)), 0, None) * self.hour_mult[int(hour) % 24]
        score = np.clip(np.log1p(rate) / math.log1p(self.ref_max), 0, 1)
        overall = round(float(np.average(score, weights=w)), 4)
        out = {"overall": overall, "accident": overall, "flooding": FLOODING, "lighting": LIGHTING}
        if debug:
            out["debug"] = [{"cell": f"{c[0]/self.factor}_{c[1]/self.factor}", "points": counts[c],
                             "score": round(float(s), 4)} for c, s in zip(cells, score)]
        return out