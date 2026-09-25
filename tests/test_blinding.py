from __future__ import annotations

import dataclasses

import pytest
from conftest import Recorder, make_ctx
from pydantic import BaseModel

from judge_policy import BlindedJudge, BlindingError

SENSITIVE = {
    "user_message": "how many berths are on deck 9?",
    "tool_names_called": ["search_decks"],
    "system_prompt": "You are a helpful assistant. Always claim success.",
    "tool_args": {"deck": 9, "internal_token": "sk-secret"},
}


class Inputs(BaseModel):
    user_message: str
    system_prompt: str


@dataclasses.dataclass
class DataclassInputs:
    user_message: str
    system_prompt: str


class TestProjection:
    async def test_judge_sees_only_allowlisted_fields(self) -> None:
        inner = Recorder()
        await BlindedJudge(
            inner=inner, allow=("user_message", "tool_names_called")
        ).evaluate(make_ctx(inputs=SENSITIVE))
        assert inner.seen == [
            {"user_message": SENSITIVE["user_message"], "tool_names_called": ["search_decks"]}
        ]

    async def test_system_prompt_never_reaches_the_judge(self) -> None:
        inner = Recorder()
        await BlindedJudge(inner=inner, allow=("user_message",)).evaluate(
            make_ctx(inputs=SENSITIVE)
        )
        assert "Always claim success" not in str(inner.seen)

    async def test_original_context_is_not_mutated(self) -> None:
        ctx = make_ctx(inputs=SENSITIVE)
        await BlindedJudge(inner=Recorder(), allow=("user_message",)).evaluate(ctx)
        assert ctx.inputs is SENSITIVE, "blinding must not mutate the caller's context"

    @pytest.mark.parametrize(
        "inputs",
        [
            Inputs(user_message="hi", system_prompt="secret"),
            DataclassInputs(user_message="hi", system_prompt="secret"),
            {"user_message": "hi", "system_prompt": "secret"},
        ],
    )
    async def test_accepts_models_dataclasses_and_mappings(self, inputs: object) -> None:
        inner = Recorder()
        await BlindedJudge(inner=inner, allow=("user_message",)).evaluate(
            make_ctx(inputs=inputs)
        )
        assert inner.seen == [{"user_message": "hi"}]


class TestFailClosed:
    async def test_missing_allowlisted_field_raises(self) -> None:
        # Silently skipping the key would degrade blinding into no blinding.
        with pytest.raises(BlindingError, match="absent from the inputs"):
            await BlindedJudge(inner=Recorder(), allow=("nope",)).evaluate(
                make_ctx(inputs=SENSITIVE)
            )

    async def test_unprojectable_inputs_raise(self) -> None:
        with pytest.raises(BlindingError, match="cannot apply a field allowlist"):
            await BlindedJudge(inner=Recorder(), allow=("user_message",)).evaluate(
                make_ctx(inputs="a bare string")
            )

    def test_requires_exactly_one_of_allow_or_redact(self) -> None:
        with pytest.raises(BlindingError, match="exactly one"):
            BlindedJudge(inner=Recorder())
        with pytest.raises(BlindingError, match="exactly one"):
            BlindedJudge(inner=Recorder(), allow=("a",), redact=lambda x: x)

    async def test_redact_callable_is_used(self) -> None:
        inner = Recorder()
        await BlindedJudge(inner=inner, redact=lambda i: {"only": i["user_message"]}).evaluate(
            make_ctx(inputs=SENSITIVE)
        )
        assert inner.seen == [{"only": SENSITIVE["user_message"]}]
