"""Project a case's inputs down to an explicit allowlist before a judge sees them."""

from __future__ import annotations

import copy
import dataclasses
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic_evals.evaluators import Evaluator, EvaluatorContext

__all__ = ["BlindedJudge", "BlindingError"]


class BlindingError(ValueError):
    """Raised when the inputs cannot be projected onto the requested allowlist."""


def _to_mapping(inputs: Any) -> Mapping[str, Any]:
    if isinstance(inputs, Mapping):
        return inputs
    model_dump = getattr(inputs, "model_dump", None)
    if callable(model_dump):
        dumped: Mapping[str, Any] = model_dump()
        return dumped
    if dataclasses.is_dataclass(inputs) and not isinstance(inputs, type):
        return dataclasses.asdict(inputs)
    raise BlindingError(
        f"cannot apply a field allowlist to inputs of type "
        f"{type(inputs).__name__}. Pass a mapping, a pydantic model or a "
        f"dataclass, or supply redact= to project the inputs yourself."
    )


@dataclass
class BlindedJudge(Evaluator[object, object, object]):
    """Wrap any evaluator so it only sees the input fields you name.

    ``LLMJudge(include_input=True)`` hands the judge the whole input object.
    For a single-turn question that is exactly right. For an agent it is not:
    the input carries the system prompt, the tool arguments and the retrieved
    context, so the system being graded gets to state its own case, in its own
    words, inside the grader's prompt. Verdicts drift toward what the agent was
    instructed to do rather than what the user actually asked for.

    ``BlindedJudge`` narrows the input first and delegates the judging
    unchanged::

        BlindedJudge(
            inner=LLMJudge(rubric="...", include_input=True),
            allow=("user_message", "tool_names_called"),
        )

    Supply exactly one of ``allow`` (a field allowlist) or ``redact`` (a
    callable that returns the projected inputs).
    """

    inner: Evaluator[Any, Any, Any]
    allow: tuple[str, ...] = ()
    redact: Callable[[Any], Any] | None = None

    def __post_init__(self) -> None:
        if bool(self.allow) == (self.redact is not None):
            raise BlindingError(
                "provide exactly one of allow= (a field allowlist) or redact= "
                "(a projection callable)"
            )

    def project(self, inputs: Any) -> Any:
        if self.redact is not None:
            return self.redact(inputs)
        source = _to_mapping(inputs)
        missing = [key for key in self.allow if key not in source]
        if missing:
            # Skipping absent keys would let a blinding policy silently decay
            # into no blinding at all -- the exact failure it exists to prevent.
            raise BlindingError(
                f"allowlisted field(s) {missing} are absent from the inputs "
                f"(available: {sorted(source)}). Fix the allowlist rather than "
                f"letting the judge fall back to seeing everything."
            )
        return {key: source[key] for key in self.allow}

    async def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> Any:
        blinded = copy.copy(ctx)
        blinded.inputs = self.project(ctx.inputs)
        result = self.inner.evaluate(blinded)
        if inspect.isawaitable(result):
            result = await result
        return result
