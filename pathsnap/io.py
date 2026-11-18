"""Prediction input loading. Reference labels are never loaded here."""
from dataclasses import dataclass
from pathlib import Path
import hashlib, json, re
import numpy as np
from pyproj import CRS, Transformer

@dataclass
class Trip:
    trip_id: str
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray


def matrix(path, columns, integer=False):
    a = np.loadtxt(path, ndmin=2)
    if not len(a) or a.shape[1] != columns or not np.isfinite(a).all():
        raise ValueError(f"Invalid data shape or values: {Path(path).name}")
    if integer:
        if not np.equal(a, np.floor(a)).all():
            raise ValueError("IDs must be integers")
        a = a.astype(np.int64)
    return a


def load_trip(folder):
    folder = Path(folder)
    trip_id = folder.name
    if not re.fullmatch(r"[0-9]{8}", trip_id):
        raise ValueError("Trip ID must contain eight decimal digits")
    track = matrix(folder / f"{trip_id}.track", 3)
    nodes = matrix(folder / f"{trip_id}.nodes", 2)
    arcs = matrix(folder / f"{trip_id}.arcs", 2, True)
    for a in (track, nodes):
        if np.any(np.abs(a[:, 0]) > 180) or np.any(np.abs(a[:, 1]) > 90):
            raise ValueError("Coordinates outside longitude/latitude bounds")
    if len(track) < 2 or np.any(np.diff(track[:, 2]) <= 0):
        raise ValueError("At least two strictly time-ordered GPS points required")
    if arcs.min() < 0 or arcs.max() >= len(nodes):
        raise ValueError("Arc references an invalid node")
    return Trip(trip_id, track, nodes, arcs)


class Projection:
    def __init__(self, lonlat):
        lon, lat = np.median(lonlat[:, :2], axis=0)
        self.lon0,self.lat0=float(lon),float(lat)
        crs = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs")
        self.forward = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        self.backward = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    def xy(self, a):
        return np.column_stack(self.forward.transform(a[:, 0], a[:, 1]))

    project = xy

    def lonlat(self, a):
        return np.column_stack(self.backward.transform(a[:, 0], a[:, 1]))


def fingerprint(folder):
    folder = Path(folder)
    h = hashlib.sha256()
    for suffix in ("track", "nodes", "arcs"):
        h.update((folder / f"{folder.name}.{suffix}").read_bytes())
    return h.hexdigest()


def json_save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))
