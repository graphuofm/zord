# NOTE (name-shadow footgun, #8): this package re-exports the FUNCTIONS `arrange` and `allocate`,
# which share a name with their submodules arrange.py / allocate.py. So `import zord.partition.arrange
# as X` binds X to the FUNCTION (the package attribute), not the module. Real usage is UNAFFECTED --
# reach module members via `from zord.partition.arrange import arrange_cpp, choose_axis, ...` (the
# from-import resolves the module namespace correctly). We do NOT rename (the function is the
# established public API; renaming ripples across many call sites): documented low-risk, not a bug.
from .base import Partition, Partitioner
from .baselines import (HashPartitioner, RandomPartitioner,
                        CapacityProportionalHash, FennelPartitioner, MetisPartitioner)
from .hetero import ZordPartitioner
from .cost_model import CostParams, CostReport, evaluate
from .arrange import arrange, ArrangeResult, predict_ms
from .feature_parallel import (
    feature_parallel_plan, FeatureParallelPlan, hybrid_plans, HybridPlan,
    fp_aggregate_consistency,
)
from .attr_cost import (
    decide_axis, AttrDecision, crossover_dim, feature_relief_inequality,
    node_parallel_cost_ms, feature_parallel_cost_ms, integration_cost_ms,
)
from .allocate import (
    allocate, AllocationPlan, supra_solve_run, build_supra_cells, count_cuts,
)

PARTITIONERS = {
    "hash": HashPartitioner,
    "random": RandomPartitioner,
    "caphash": CapacityProportionalHash,
    "fennel": FennelPartitioner,
    "metis": MetisPartitioner,
    "zord": ZordPartitioner,
}

__all__ = [
    "Partition", "Partitioner",
    "HashPartitioner", "RandomPartitioner", "CapacityProportionalHash",
    "FennelPartitioner", "MetisPartitioner", "ZordPartitioner",
    "CostParams", "CostReport", "evaluate", "PARTITIONERS",
    "arrange", "ArrangeResult", "predict_ms",
    "feature_parallel_plan", "FeatureParallelPlan", "hybrid_plans", "HybridPlan",
    "fp_aggregate_consistency",
    # attr_cost (M2: the derived node/feature/hybrid axis pre-filter)
    "decide_axis", "AttrDecision", "crossover_dim", "feature_relief_inequality",
    "node_parallel_cost_ms", "feature_parallel_cost_ms", "integration_cost_ms",
    # allocate (K2: the composed MIDDLE-END allocation; supra-cut wrapper)
    "allocate", "AllocationPlan", "supra_solve_run", "build_supra_cells", "count_cuts",
]
