from __future__ import annotations

import pytest
from conftest import Recorder, make_ctx

from judge_policy import MajorityVote, UnsoundVotingPolicy, assert_sound_policy


class TestPolicySoundness:
    def test_unanimity_disguised_as_majority_is_rejected(self) -> None:
        with pytest.raises(UnsoundVotingPolicy, match="unanimity, not a"):
            assert_sound_policy(runs=3, threshold=3)

    def test_sub_majority_threshold_is_rejected(self) -> None:
        # 2 of 4 would let the same evidence satisfy both pass and fail.
        with pytest.raises(UnsoundVotingPolicy, match="not a majority"):
            assert_sound_policy(runs=4, threshold=2)

    def test_unreachable_threshold_is_rejected(self) -> None:
        with pytest.raises(UnsoundVotingPolicy, match="never pass"):
            assert_sound_policy(runs=3, threshold=4)

    @pytest.mark.parametrize(("runs", "threshold"), [(1, 1), (3, 2), (5, 3), (5, 4), (4, 3)])
    def test_sound_policies_pass(self, runs: int, threshold: int) -> None:
        assert_sound_policy(runs=runs, threshold=threshold)

    def test_policy_is_checked_at_construction_not_at_run_time(self) -> None:
        with pytest.raises(UnsoundVotingPolicy):
            MajorityVote(inner=Recorder(), runs=3, threshold=3)


class TestVoting:
    async def test_majority_passes(self) -> None:
        inner = Recorder(verdicts=[True, False, True])
        result = await MajorityVote(inner=inner, runs=3, threshold=2).evaluate(make_ctx())
        assert result["majority_pass"].value is True

    async def test_minority_fails(self) -> None:
        inner = Recorder(verdicts=[False, True, False])
        result = await MajorityVote(inner=inner, runs=3, threshold=2).evaluate(make_ctx())
        assert result["majority_pass"].value is False

    async def test_early_termination_on_decided_pass(self) -> None:
        inner = Recorder(verdicts=[True, True, True])
        result = await MajorityVote(inner=inner, runs=3, threshold=2).evaluate(make_ctx())
        assert result["majority_samples_used"] == 2
        assert len(inner.seen) == 2, "third sample cannot change the outcome"

    async def test_early_termination_on_decided_fail(self) -> None:
        inner = Recorder(verdicts=[False, False, True])
        result = await MajorityVote(inner=inner, runs=3, threshold=2).evaluate(make_ctx())
        assert result["majority_pass"].value is False
        assert len(inner.seen) == 2

    async def test_accepts_evaluation_reason_from_inner(self) -> None:
        inner = Recorder(verdicts=[True, True], as_reason=True)
        result = await MajorityVote(inner=inner, runs=3, threshold=2).evaluate(make_ctx())
        assert result["majority_pass"].value is True
