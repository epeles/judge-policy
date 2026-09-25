# judge-policy

Three small evaluators that make an LLM judge safe to gate CI on, built on
[pydantic-evals](https://pydantic.dev/docs/ai/evals/evals/).

## Why this exists

Pydantic's own [guide to LLM-as-a-judge](https://pydantic.dev/articles/llm-as-a-judge)
gets the architecture right, and I follow it: deterministic checks first, a judge
only for the semantic layer on top, always ask for reasoning. I've been running
that architecture as a merge gate for an agent platform — a Playwright/TypeScript
suite driving a FastAPI/LangGraph backend over the AG-UI protocol, deterministic
SSE wire-format assertions deciding pass/fail, a judge confirming the
natural-language reply.

The architecture held. What broke was everything around it.

On one tier the judge **inverted its own verdict three times across runs on
identical input**. Not a flaky agent — a flaky *evaluator*. I demoted it to
advisory there and let the deterministic assertions decide, which was the right
call for that afternoon but not an answer. The failure mode a flaky evaluator
produces isn't technical, it's social: the suite starts emitting verdicts nobody
can act on, and a suite nobody trusts is worse than no suite, because it costs CI
minutes in order to be ignored.

These are the three changes that made the judge gateable. None of them is a
better prompt.

This library sits on top of `pydantic-evals` and composes with it; nothing here
patches or replaces anything in it. The gaps below are the ones I hit using it
in CI, which is a narrower and more paranoid setting than most eval work.

## 1. `BlindedJudge` — the judge sees a field allowlist, not the input

`LLMJudge(include_input=True)` hands the judge the entire input object. For a
single-turn question that's exactly right. For an agent it isn't: the input
carries the system prompt, the tool arguments and the retrieved context — so the
system being graded gets to state its own case, in its own words, inside the
grader's prompt. Verdicts drift toward what the agent was *instructed* to do
rather than what the user actually asked for. A system prompt containing
"always confirm the task succeeded" is not neutral context for a grader.

`BlindedJudge` narrows the input first and delegates the judging unchanged:

```python
from pydantic_evals.evaluators import LLMJudge
from judge_policy import BlindedJudge

BlindedJudge(
    inner=LLMJudge(rubric="The reply answers the question asked.", include_input=True),
    allow=("user_message", "tool_names_called"),
)
```

It wraps **any** evaluator, not just `LLMJudge`, and accepts mappings, pydantic
models or dataclasses as inputs. Pass `redact=` instead of `allow=` for a custom
projection.

A missing allowlisted field raises rather than being skipped. Silently dropping
an absent key is how a blinding policy decays into no blinding at all — the
exact failure it was written to prevent.

## 2. `ConfidenceGatedJudge` — "couldn't tell" is not "failed"

`GradingOutput` is `reason`, `pass_` and `score`. There is nowhere for the judge
to record that the rubric doesn't reach a given output, so it has to pick a side;
on a genuinely ambiguous case that pick approaches a coin flip. Run it three
times and you get three different answers, which is precisely what I was looking
at.

```python
from judge_policy import ConfidenceGatedJudge, Confidence

ConfidenceGatedJudge(
    rubric="The reply answers the question asked.",
    model="google-vertex:gemini-2.5-flash",
    binding=(Confidence.HIGH, Confidence.MEDIUM),
    retries=1,
)
```

* High and medium confidence bind. Low confidence retries once, then quarantines.
* An exception from the provider quarantines too. An outage in the grader is our
  problem, not a defect in the system under test.
* **On quarantine the evaluator emits no pass/fail key at all.** That's the
  point. The absence of a verdict has to be representable in the report, or
  "the judge couldn't tell" gets read downstream as "the system is broken".

One calibration note that cost me a week: models report high confidence for
everything unless low confidence is given a definition that isn't hedging. The
prompt here frames it as a claim about the *rubric* rather than about the
model's feelings — report low when a careful reviewer applying this same rubric
to this same output could reasonably land either way — and tells the judge
explicitly that doing so carries no penalty, because choosing a side on a case
the rubric doesn't settle is the more expensive error.

## 3. `MajorityVote` — a policy check that runs before the tokens do

```python
from judge_policy import MajorityVote

MajorityVote(inner=my_judge, runs=3, threshold=2)   # fine
MajorityVote(inner=my_judge, runs=3, threshold=3)   # UnsoundVotingPolicy
```

A threshold equal to the run count is unanimity wearing a majority's clothes: it
reads like a quorum and behaves like a veto, so raising `runs` makes the gate
*stricter* instead of more stable. A threshold at or below half is not a majority
at all — with `runs=4, threshold=2` the same evidence satisfies both the pass and
the fail condition. `assert_sound_policy` rejects both at construction time,
before any tokens are spent, and the error message names the threshold you
probably meant.

Sampling is sequential with early termination, so a unanimous case costs
`threshold` calls rather than `runs`.

## What this is not

None of this makes a judge trustworthy enough to gate alone, and nothing here
argues it should. In the suite these came from, deterministic protocol assertions
decide pass/fail and the judge only confirms on top; a judge verdict by itself
never carries a test. These narrow the band in which the judge's opinion is
allowed to matter.

## Install

```bash
pip install judge-policy
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

36 tests, no network and no model calls: `decide()` is pure, and the model call
in `ConfidenceGatedJudge` is isolated behind a single overridable method so the
policy is testable on its own. Model-backed integration runs are deliberately
kept out of CI — an eval library whose own test suite is flaky would be a poor
advertisement.

## License

MIT
