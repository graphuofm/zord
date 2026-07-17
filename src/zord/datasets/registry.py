"""Dataset registry: maps a short name to where/how to load it and its scale
tier. Paths point at the HetCluster staging dir; `url` lets `zord` re-download
anywhere. Scale tiers (per ZORD_VISION): small / medium / large / ultra.

Edge counts below are VERIFIED from the 2026-05-30 download (see
$ZORD_DATA/data/MANIFEST.txt on HetCluster).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

CLUSTER_DATA = "$ZORD_DATA/data"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    fmt: str                 # "snap_edgelist" | "bitcoin_csv" | "tgb_linkprop" | "jodie"
    tier: str                # small | medium | large | ultra
    edges: int               # verified edge/row count (0 = unknown)
    hetcluster_path: Optional[str] = None   # path under CLUSTER_DATA
    url: Optional[str] = None
    note: str = ""
    tags: tuple = field(default_factory=tuple)


_SNAP = "https://snap.stanford.edu/data"
_JODIE = "http://snap.stanford.edu/jodie"

DATASETS: dict[str, DatasetSpec] = {
    # ---- attributed temporal (JODIE): real per-edge features (efeat [E, Fe]) ----
    # CSV: user_id,item_id,timestamp,state_label, then comma-separated edge features.
    # wikipedia/reddit edge feats are 172-dim LIWC vectors; mooc 4-dim; lastfm featureless.
    "jodie-wikipedia": DatasetSpec("jodie-wikipedia", "jodie", "small", 157_474,
        f"{CLUSTER_DATA}/jodie/wikipedia.csv", f"{_JODIE}/wikipedia.csv",
        "JODIE; user-item edits; 172-dim edge feats", ("attributed", "temporal", "bipartite")),
    "jodie-reddit": DatasetSpec("jodie-reddit", "jodie", "medium", 672_447,
        f"{CLUSTER_DATA}/jodie/reddit.csv", f"{_JODIE}/reddit.csv",
        "JODIE; user-subreddit posts; 172-dim edge feats", ("attributed", "temporal", "bipartite")),
    "jodie-mooc": DatasetSpec("jodie-mooc", "jodie", "medium", 411_749,
        f"{CLUSTER_DATA}/jodie/mooc.csv", f"{_JODIE}/mooc.csv",
        "JODIE; student-course actions; 4-dim edge feats", ("attributed", "temporal", "bipartite")),
    "jodie-lastfm": DatasetSpec("jodie-lastfm", "jodie", "medium", 1_293_103,
        f"{CLUSTER_DATA}/jodie/lastfm.csv", f"{_JODIE}/lastfm.csv",
        "JODIE; user-song listens; featureless (Fe=0)", ("attributed", "temporal", "bipartite")),

    # ---- small (sanity / fast iteration) --------------------------------
    "collegemsg": DatasetSpec("collegemsg", "snap_edgelist", "small", 59_835,
        f"{CLUSTER_DATA}/snap/CollegeMsg.txt.gz", f"{_SNAP}/CollegeMsg.txt.gz",
        "UCI online msgs; src dst unixts", ("social", "temporal")),
    "email-eu": DatasetSpec("email-eu", "snap_edgelist", "small", 332_334,
        f"{CLUSTER_DATA}/snap/email-Eu-core-temporal.txt.gz",
        f"{_SNAP}/email-Eu-core-temporal.txt.gz", "logical time", ("email", "temporal")),
    "bitcoin-otc": DatasetSpec("bitcoin-otc", "bitcoin_csv", "small", 35_592,
        f"{CLUSTER_DATA}/snap/soc-sign-bitcoinotc.csv.gz",
        f"{_SNAP}/soc-sign-bitcoinotc.csv.gz", "source,target,rating,time", ("trust", "weighted")),
    "bitcoin-alpha": DatasetSpec("bitcoin-alpha", "bitcoin_csv", "small", 24_186,
        f"{CLUSTER_DATA}/snap/soc-sign-bitcoinalpha.csv.gz",
        f"{_SNAP}/soc-sign-bitcoinalpha.csv.gz", "source,target,rating,time", ("trust", "weighted")),

    # ---- medium ----------------------------------------------------------
    "mathoverflow": DatasetSpec("mathoverflow", "snap_edgelist", "medium", 506_550,
        f"{CLUSTER_DATA}/snap/sx-mathoverflow.txt.gz", f"{_SNAP}/sx-mathoverflow.txt.gz",
        "", ("qa", "temporal")),
    "askubuntu": DatasetSpec("askubuntu", "snap_edgelist", "medium", 964_437,
        f"{CLUSTER_DATA}/snap/sx-askubuntu.txt.gz", f"{_SNAP}/sx-askubuntu.txt.gz",
        "", ("qa", "temporal")),
    "superuser": DatasetSpec("superuser", "snap_edgelist", "medium", 1_443_339,
        f"{CLUSTER_DATA}/snap/sx-superuser.txt.gz", f"{_SNAP}/sx-superuser.txt.gz",
        "", ("qa", "temporal")),
    "tgbl-wiki": DatasetSpec("tgbl-wiki", "tgb_linkprop", "medium", 0,
        f"{CLUSTER_DATA}/tgb", None, "TGB discrete-time link-prop", ("tgb",)),
    "tgbl-review": DatasetSpec("tgbl-review", "tgb_linkprop", "medium", 4_873_540,
        f"{CLUSTER_DATA}/tgb", None, "TGB; Amazon reviews over time", ("tgb",)),

    # ---- large -----------------------------------------------------------
    "wiki-talk": DatasetSpec("wiki-talk", "snap_edgelist", "large", 7_833_140,
        f"{CLUSTER_DATA}/snap/wiki-talk-temporal.txt.gz",
        f"{_SNAP}/wiki-talk-temporal.txt.gz", "", ("social", "temporal")),

    # ---- ultra (overflows the 32GB card -> out-of-core matters) ----------
    "stackoverflow": DatasetSpec("stackoverflow", "snap_edgelist", "ultra", 63_497_050,
        f"{CLUSTER_DATA}/snap/sx-stackoverflow.txt.gz",
        f"{_SNAP}/sx-stackoverflow.txt.gz", "63.5M temporal edges", ("qa", "temporal")),
    # planned, not yet staged:
    "gdelt": DatasetSpec("gdelt", "custom", "ultra", 191_000_000, None, None,
        "PLANNED: ~191M edges; time-bucket into snapshots", ("planned",)),
    "tkgl-icews": DatasetSpec("tkgl-icews", "tgb_linkprop", "ultra", 15_513_446, None, None,
        "PLANNED: TGB knowledge graph, 10k+ snapshots", ("tgb", "planned")),
}


def get_spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")


def by_tier(tier: str) -> list[DatasetSpec]:
    return [s for s in DATASETS.values() if s.tier == tier]
