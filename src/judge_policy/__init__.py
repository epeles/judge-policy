"""Guardrails for using an LLM judge as a CI gate, built on pydantic-evals."""

from judge_policy.blinding import BlindedJudge, BlindingError
from judge_policy.confidence import (
    Confidence,
    ConfidenceGatedJudge,
    Decision,
    JudgeVerdict,
    decide,
)
from judge_policy.voting import MajorityVote, UnsoundVotingPolicy, assert_sound_policy

__version__ = "0.1.0"

__all__ = [
    "BlindedJudge",
    "BlindingError",
    "Confidence",
    "ConfidenceGatedJudge",
    "Decision",
    "JudgeVerdict",
    "MajorityVote",
    "UnsoundVotingPolicy",
    "assert_sound_policy",
    "decide",
]
