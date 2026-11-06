"""
byoa_adk/dag.py
~~~~~~~~~~~~~~~
Typed DSL for declaring an agent's internal Kanban DAG.
Cycle detection runs at class definition time (fast-fail before platform contact).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class Step:
    """
    A single node in the agent's Kanban DAG.

    Parameters
    ----------
    name        : Unique step identifier within this DAG.
    depends_on  : List of step names that must complete before this one runs.
    description : Human-readable summary (appears in Agent.md).
    capability  : Optional platform capability tag for this step.
    """
    name        : str
    depends_on  : List[str]      = field(default_factory=list)
    description : str            = ""
    capability  : Optional[str]  = None


class DAGValidationError(Exception):
    """Raised when the DAG is malformed (cycle, unknown dependency, etc.)."""


class DAG:
    """
    Directed Acyclic Graph of Steps.

    Usage
    -----
    dag = DAG([
        Step("fetch",   depends_on=[]),
        Step("clean",   depends_on=["fetch"]),
        Step("analyse", depends_on=["clean"]),
        Step("report",  depends_on=["analyse"]),
    ])
    """

    def __init__(self, steps: List[Step]) -> None:
        self.steps: Dict[str, Step] = {}
        for s in steps:
            if s.name in self.steps:
                raise DAGValidationError(f"Duplicate step name: {s.name!r}")
            self.steps[s.name] = s
        self._validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Check for unknown dependencies and cycles (Kahn's algorithm)."""
        # 1. All depends_on references must resolve.
        for step in self.steps.values():
            for dep in step.depends_on:
                if dep not in self.steps:
                    raise DAGValidationError(
                        f"Step {step.name!r} depends on unknown step {dep!r}."
                    )

        # 2. No cycles — Kahn's topological sort.
        in_degree: Dict[str, int] = {n: 0 for n in self.steps}
        for step in self.steps.values():
            for dep in step.depends_on:
                in_degree[step.name] += 1

        queue   = [n for n, d in in_degree.items() if d == 0]
        visited = 0
        while queue:
            node = queue.pop()
            visited += 1
            for step in self.steps.values():
                if node in step.depends_on:
                    in_degree[step.name] -= 1
                    if in_degree[step.name] == 0:
                        queue.append(step.name)

        if visited != len(self.steps):
            raise DAGValidationError(
                "DAG contains a cycle. Check depends_on relationships."
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> List[dict]:
        """Serialise to the list format embedded in Agent.md manifest JSON."""
        return [
            {
                "name"       : s.name,
                "depends_on" : s.depends_on,
                "description": s.description,
                "capability" : s.capability,
            }
            for s in self.steps.values()
        ]

    def topological_order(self) -> List[str]:
        """Return steps in a valid execution order (one valid linearisation)."""
        in_degree: Dict[str, int] = {n: 0 for n in self.steps}
        for step in self.steps.values():
            for dep in step.depends_on:
                in_degree[step.name] += 1
        queue  = sorted([n for n, d in in_degree.items() if d == 0])
        result = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            for step in self.steps.values():
                if node in step.depends_on:
                    in_degree[step.name] -= 1
                    if in_degree[step.name] == 0:
                        queue.append(step.name)
                        queue.sort()
        return result

    def __repr__(self) -> str:
        order = self.topological_order()
        return f"DAG(steps={order})"

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, context: dict, step_funcs: Dict[str, "Callable[[dict], Optional[dict]]"]) -> dict:
        """
        Actually execute the DAG, in the order topological_order() decides --
        not whatever order the caller happens to write code in.

        Each entry in step_funcs is called with the shared `context` dict and
        may return a dict of new/updated keys, which get merged into context
        before the next step runs. If a step raises, execution stops there
        and the exception propagates to the caller.
        """
        for step_name in self.topological_order():
            print(f"[DAG] execute step: {step_name}", flush=True)  # TEMP DEBUG -- remove after confirming
            func = step_funcs.get(step_name)
            if func is None:
                raise DAGValidationError(
                    f"No function registered for step {step_name!r}. "
                    f"step_funcs must provide one callable per declared step."
                )
            result = func(context)
            if result:
                context.update(result)
        return context
