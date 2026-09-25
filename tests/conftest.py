from __future__ import annotations

import dataclasses
from typing import Any

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext


def make_ctx(inputs: Any = None, output: Any = "out", **kw: Any) -> EvaluatorContext[Any, Any, Any]:
    """Build a bare EvaluatorContext without running a real Dataset."""
    fields = {f.name for f in dataclasses.fields(EvaluatorContext)}
    payload: dict[str, Any] = {
        "name": "case",
        "inputs": inputs,
        "metadata": None,
        "expected_output": None,
        "output": output,
        "duration": 0.0,
        "attributes": {},
        "metrics": {},
    }
    payload.update(kw)
    if "_span_tree" in fields:
        payload.setdefault("_span_tree", None)
    return EvaluatorContext(**payload)  # type: ignore[arg-type]


@dataclasses.dataclass
class Recorder(Evaluator[Any, Any, Any]):
    """Inner evaluator that records what it was shown and returns a canned verdict."""

    verdicts: list[bool] = dataclasses.field(default_factory=list)
    seen: list[Any] = dataclasses.field(default_factory=list)
    as_reason: bool = False

    def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> Any:
        self.seen.append(ctx.inputs)
        verdict = self.verdicts[len(self.seen) - 1] if self.verdicts else True
        return EvaluationReason(value=verdict, reason="canned") if self.as_reason else verdict
