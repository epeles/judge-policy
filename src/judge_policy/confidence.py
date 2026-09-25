"""A judge that is allowed to say the rubric does not decide this case."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

__all__ = ["Confidence", "ConfidenceGatedJudge", "Decision", "JudgeVerdict", "decide"]


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class JudgeVerdict(BaseModel):
    """What the judge returns. Note the third field."""

    reason: str = Field(
        description="Why this verdict follows from the rubric. Cite the part of "
        "the rubric you applied and the part of the output it applies to."
    )
    passed: bool = Field(description="Whether the output satisfies the rubric.")
    confidence: Confidence = Field(
        description=(
            "How determinately the rubric settles this case. Report 'low' when a "
            "careful reviewer applying this same rubric to this same output could "
            "reasonably land on either verdict -- that is, when the rubric is "
            "underspecified here. Low confidence is a claim about the rubric, not "
            "a hedge about your own certainty, and it is the correct answer when "
            "the rubric genuinely does not reach this case."
        )
    )


_SYSTEM_PROMPT = """\
You are grading one output against one rubric.

Apply only the rubric. Do not import your own standards of quality, style or \
helpfulness, and do not reward or penalise anything the rubric does not mention.

You may report low confidence. Doing so is not a failure on your part and \
carries no penalty: it routes the case to a human instead of to a verdict. \
Choosing a side on a case the rubric does not settle is the more costly error, \
because a coin flip presented as a judgement is indistinguishable from a real \
one downstream.
"""


class Decision(str, Enum):
    """What a verdict plus a binding policy amounts to."""

    PASS = "pass"
    FAIL = "fail"
    QUARANTINE = "quarantine"


def decide(verdict: JudgeVerdict, binding: tuple[Confidence, ...]) -> Decision:
    """Map a verdict onto an outcome. Pure; this is where the policy lives."""
    if verdict.confidence not in binding:
        return Decision.QUARANTINE
    return Decision.PASS if verdict.passed else Decision.FAIL


@dataclass
class ConfidenceGatedJudge(Evaluator[object, object, object]):
    """An LLM judge whose low-confidence verdicts quarantine instead of failing.

    ``GradingOutput`` is ``reason``, ``pass_`` and ``score``. There is nowhere
    for the judge to record that the rubric does not reach a given output, so it
    has to pick a side; on a genuinely ambiguous case that pick approaches a coin
    flip. Across repeated runs the same input then yields different verdicts, and
    the suite starts reporting failures the team cannot act on. A suite nobody
    trusts is worse than no suite, because it costs CI minutes to be ignored.

    On quarantine this evaluator emits **no pass/fail key at all**. That is the
    point: the absence of a verdict has to be representable, or "the judge could
    not tell" gets reported as "the system under test is broken".
    """

    rubric: str
    model: Any = None
    include_input: bool = False
    binding: tuple[Confidence, ...] = (Confidence.HIGH, Confidence.MEDIUM)
    retries: int = 1
    evaluation_name: str = "judge"
    model_settings: ModelSettings | None = field(
        default_factory=lambda: ModelSettings(temperature=0.0)
    )

    def _agent(self) -> Agent[None, JudgeVerdict]:
        return Agent(
            self.model,
            output_type=JudgeVerdict,
            system_prompt=_SYSTEM_PROMPT,
            model_settings=self.model_settings,
        )

    def _prompt(self, ctx: EvaluatorContext[object, object, object]) -> str:
        parts = [f"<rubric>\n{self.rubric}\n</rubric>"]
        if self.include_input:
            parts.append(f"<input>\n{ctx.inputs}\n</input>")
        parts.append(f"<output>\n{ctx.output}\n</output>")
        return "\n\n".join(parts)

    async def _judge(
        self, ctx: EvaluatorContext[object, object, object]
    ) -> JudgeVerdict:
        result = await self._agent().run(self._prompt(ctx))
        return result.output

    async def evaluate(
        self, ctx: EvaluatorContext[object, object, object]
    ) -> Mapping[str, EvaluationReason | str]:
        name = self.evaluation_name
        attempts = self.retries + 1
        verdict: JudgeVerdict | None = None
        last_error: Exception | None = None

        for _ in range(attempts):
            try:
                verdict = await self._judge(ctx)
            except Exception as exc:  # noqa: BLE001 - an evaluator fault is not a test failure
                last_error, verdict = exc, None
                continue
            if decide(verdict, self.binding) is not Decision.QUARANTINE:
                break

        if verdict is None:
            # The evaluator broke. That is our problem, not the system's.
            return {
                f"{name}_quarantined": EvaluationReason(
                    value=True, reason=f"evaluator error: {last_error!r}"
                )
            }

        decision = decide(verdict, self.binding)
        if decision is Decision.QUARANTINE:
            return {
                f"{name}_quarantined": EvaluationReason(
                    value=True,
                    reason=(
                        f"confidence {verdict.confidence.value} is not binding "
                        f"after {attempts} attempt(s): {verdict.reason}"
                    ),
                ),
                f"{name}_confidence": verdict.confidence.value,
            }

        return {
            f"{name}_pass": EvaluationReason(
                value=decision is Decision.PASS, reason=verdict.reason
            ),
            f"{name}_quarantined": EvaluationReason(value=False),
            f"{name}_confidence": verdict.confidence.value,
        }
