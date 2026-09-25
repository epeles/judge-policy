"""Repeated-sampling majority vote, with the policy validated before any tokens are spent."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

__all__ = ["MajorityVote", "UnsoundVotingPolicy", "assert_sound_policy"]


class UnsoundVotingPolicy(ValueError):
    """Raised when a (runs, threshold) pair does not describe a majority."""


def assert_sound_policy(runs: int, threshold: int) -> None:
    """Reject vote policies that are not majorities, before the votes are cast.

    Two mistakes are easy to make and expensive to discover in CI:

    * ``threshold == runs`` is unanimity wearing a majority's clothes. It reads
      like a quorum and behaves like a veto: one dissenting sample fails the
      case, so raising ``runs`` makes the gate *stricter*, not more stable.
    * ``threshold <= runs / 2`` is not a majority at all. With ``runs=4,
      threshold=2`` a case can pass and fail on the same evidence.
    """
    if runs < 1:
        raise UnsoundVotingPolicy(f"runs must be >= 1, got {runs}")
    if threshold < 1:
        raise UnsoundVotingPolicy(f"threshold must be >= 1, got {threshold}")
    if threshold > runs:
        raise UnsoundVotingPolicy(
            f"threshold {threshold} exceeds runs {runs}: the gate can never pass"
        )
    if runs > 1 and threshold == runs:
        raise UnsoundVotingPolicy(
            f"threshold {threshold} equals runs {runs}: that is unanimity, not a "
            f"majority. A single dissenting sample fails the case, so increasing "
            f"runs tightens the gate instead of stabilising it. Use "
            f"threshold={runs // 2 + 1} for a true majority, or state the intent "
            f"explicitly with runs=1."
        )
    if threshold * 2 <= runs:
        raise UnsoundVotingPolicy(
            f"threshold {threshold} is not a majority of {runs} runs: the same "
            f"evidence could satisfy both the pass and the fail condition. Use "
            f"threshold={runs // 2 + 1} or higher."
        )


def _as_bool(output: Any, key: str | None) -> bool:
    """Coerce an inner evaluator's output to a single boolean vote."""
    if isinstance(output, Mapping):
        if key is None:
            raise ValueError(
                "inner evaluator returned a mapping of results; set vote_on= to "
                f"name which one to count. Available keys: {sorted(output)}"
            )
        if key not in output:
            raise ValueError(
                f"vote_on={key!r} is absent from the inner evaluator's results "
                f"(got {sorted(output)}). If the inner evaluator omits this key "
                f"to signal 'no verdict', it cannot be majority-voted."
            )
        output = output[key]
    if isinstance(output, EvaluationReason):
        output = output.value
    if not isinstance(output, bool):
        raise TypeError(
            f"majority vote needs a boolean verdict, got {type(output).__name__}"
        )
    return output


@dataclass
class MajorityVote(Evaluator[object, object, object]):
    """Run an inner evaluator repeatedly and take a validated majority.

    Sampling is sequential and stops as soon as the outcome is decided, so a
    unanimous run costs ``threshold`` calls rather than ``runs``.
    """

    inner: Evaluator[Any, Any, Any]
    runs: int = 3
    threshold: int = 2
    vote_on: str | None = None
    evaluation_name: str = "majority"

    def __post_init__(self) -> None:
        assert_sound_policy(self.runs, self.threshold)

    async def _vote(self, ctx: EvaluatorContext[Any, Any, Any]) -> bool:
        result = self.inner.evaluate(ctx)
        if inspect.isawaitable(result):
            result = await result
        return _as_bool(result, self.vote_on)

    async def evaluate(
        self, ctx: EvaluatorContext[object, object, object]
    ) -> Mapping[str, EvaluationReason | int]:
        passes = 0
        failures = 0
        budget_for_failure = self.runs - self.threshold

        for cast in range(1, self.runs + 1):
            if await self._vote(ctx):
                passes += 1
            else:
                failures += 1
            if passes >= self.threshold or failures > budget_for_failure:
                break

        decided = passes >= self.threshold
        return {
            f"{self.evaluation_name}_pass": EvaluationReason(
                value=decided,
                reason=(
                    f"{passes}/{cast} samples passed "
                    f"(threshold {self.threshold} of {self.runs})"
                ),
            ),
            f"{self.evaluation_name}_samples_used": cast,
        }
