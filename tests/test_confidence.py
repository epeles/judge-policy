from __future__ import annotations

import dataclasses
from typing import Any

from conftest import make_ctx

from judge_policy import (
    Confidence,
    ConfidenceGatedJudge,
    Decision,
    JudgeVerdict,
    decide,
)

BINDING = (Confidence.HIGH, Confidence.MEDIUM)


def verdict(passed: bool, confidence: Confidence, reason: str = "because") -> JudgeVerdict:
    return JudgeVerdict(passed=passed, confidence=confidence, reason=reason)


class TestDecisionPolicy:
    """`decide` is pure, so the policy is testable without a model."""

    def test_high_confidence_binds(self) -> None:
        assert decide(verdict(True, Confidence.HIGH), BINDING) is Decision.PASS
        assert decide(verdict(False, Confidence.HIGH), BINDING) is Decision.FAIL

    def test_medium_confidence_binds(self) -> None:
        assert decide(verdict(False, Confidence.MEDIUM), BINDING) is Decision.FAIL

    def test_low_confidence_quarantines_regardless_of_verdict(self) -> None:
        assert decide(verdict(True, Confidence.LOW), BINDING) is Decision.QUARANTINE
        assert decide(verdict(False, Confidence.LOW), BINDING) is Decision.QUARANTINE

    def test_binding_set_is_configurable(self) -> None:
        strict = (Confidence.HIGH,)
        assert decide(verdict(False, Confidence.MEDIUM), strict) is Decision.QUARANTINE


@dataclasses.dataclass
class StubJudge(ConfidenceGatedJudge):
    """ConfidenceGatedJudge with the model call replaced by a script."""

    script: list[JudgeVerdict | Exception] = dataclasses.field(default_factory=list)
    calls: int = 0

    async def _judge(self, ctx: Any) -> JudgeVerdict:
        step = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(step, Exception):
            raise step
        return step


class TestQuarantineBehaviour:
    async def test_bound_failure_reports_a_verdict(self) -> None:
        judge = StubJudge(rubric="r", script=[verdict(False, Confidence.HIGH, "missed the point")])
        result = await judge.evaluate(make_ctx())
        assert result["judge_pass"].value is False
        assert result["judge_pass"].reason == "missed the point"
        assert result["judge_quarantined"].value is False

    async def test_quarantine_emits_no_pass_key_at_all(self) -> None:
        judge = StubJudge(rubric="r", script=[verdict(False, Confidence.LOW)])
        result = await judge.evaluate(make_ctx())
        # The whole point: "could not tell" must not be reportable as "failed".
        assert "judge_pass" not in result
        assert result["judge_quarantined"].value is True
        assert result["judge_confidence"] == "low"

    async def test_low_confidence_retries_once_before_quarantining(self) -> None:
        judge = StubJudge(
            rubric="r",
            retries=1,
            script=[verdict(True, Confidence.LOW), verdict(True, Confidence.HIGH)],
        )
        result = await judge.evaluate(make_ctx())
        assert judge.calls == 2
        assert result["judge_pass"].value is True

    async def test_persistent_low_confidence_quarantines(self) -> None:
        judge = StubJudge(
            rubric="r",
            retries=1,
            script=[verdict(True, Confidence.LOW), verdict(True, Confidence.LOW)],
        )
        result = await judge.evaluate(make_ctx())
        assert judge.calls == 2
        assert "judge_pass" not in result

    async def test_evaluator_error_quarantines_rather_than_fails(self) -> None:
        judge = StubJudge(rubric="r", retries=1, script=[RuntimeError("502 from provider")])
        result = await judge.evaluate(make_ctx())
        # An outage in the grader is our problem, not a defect in the system under test.
        assert "judge_pass" not in result
        assert result["judge_quarantined"].value is True
        assert "502 from provider" in (result["judge_quarantined"].reason or "")

    async def test_error_then_success_recovers(self) -> None:
        judge = StubJudge(
            rubric="r",
            retries=1,
            script=[RuntimeError("transient"), verdict(True, Confidence.HIGH)],
        )
        result = await judge.evaluate(make_ctx())
        assert result["judge_pass"].value is True


class TestPromptShape:
    def test_input_is_withheld_by_default(self) -> None:
        judge = StubJudge(rubric="r")
        prompt = judge._prompt(make_ctx(inputs="SECRET SYSTEM PROMPT", output="hello"))
        assert "SECRET SYSTEM PROMPT" not in prompt
        assert "hello" in prompt

    def test_input_included_when_asked(self) -> None:
        judge = StubJudge(rubric="r", include_input=True)
        prompt = judge._prompt(make_ctx(inputs="the question", output="hello"))
        assert "the question" in prompt
