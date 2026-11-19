"""Tests for cascade verification.

The failure mode that matters is a verifier that fires on everything: it would double
the cost of every request while looking like it was improving quality. So these tests
pin down both directions -- what must escalate, and what must not.
"""

from __future__ import annotations

import pytest

from gateway.cascade import verify

GOOD_ANSWER = (
    "Jane Austen wrote Pride and Prejudice, which was first published in 1813 "
    "and remains one of her best known novels."
)


# ------------------------------------------------------------------ escalates


@pytest.mark.parametrize(
    ("answer", "trigger"),
    [
        ("I can't help with that request.", "refusal"),
        ("As an AI, I do not have access to that information.", "refusal"),
        ("I'm not sure, it's hard to say without more context here.", "hedge"),
        ("I don't know the answer to this particular question.", "hedge"),
        ("42", "too_short"),
        ("", "too_short"),
    ],
)
def test_bad_answers_escalate(answer, trigger):
    verdict = verify("Who wrote Pride and Prejudice?", answer)
    assert verdict.escalate
    assert verdict.trigger == trigger


def test_a_numeric_question_answered_without_a_number_escalates():
    """The check that catches a plausible-sounding non-answer: the model writes
    fluent prose about the problem and never actually computes anything."""
    verdict = verify(
        "How many clips did Natalia sell altogether?",
        "To solve this, you would add the April sales to the May sales and that "
        "gives you the combined total across both months.",
    )
    assert verdict.escalate
    assert verdict.trigger == "missing_number"


# --------------------------------------------------------------- accepts


def test_a_good_answer_is_accepted():
    assert not verify("Who wrote Pride and Prejudice?", GOOD_ANSWER).escalate


def test_a_numeric_question_answered_with_a_number_is_accepted():
    verdict = verify(
        "How many clips did Natalia sell altogether?",
        "Natalia sold 48 clips in April and 24 in May, so she sold 72 clips in total.",
    )
    assert not verdict.escalate


def test_a_non_numeric_question_is_not_required_to_contain_digits():
    """Guards against the numeric check leaking onto ordinary prose questions and
    escalating every single one of them."""
    assert not verify("Explain why the sky appears blue.", GOOD_ANSWER).escalate


def test_the_word_cannot_inside_a_legitimate_answer_does_not_trip_the_refusal_check():
    answer = (
        "Light cannot travel faster than c in a vacuum, which is why the speed of "
        "light is treated as a universal constant in relativity."
    )
    assert not verify("What is the speed of light?", answer).escalate


# --------------------------------------------------------------- integration


def test_escalated_requests_are_reported_and_billed_for_both_calls(client):
    """An escalation pays for the cheap call *and* the expensive one, so it must cost
    more than the always-expensive baseline. If the accounting hid that, a trigger-happy
    verifier would silently destroy the savings the router exists to produce."""
    body = client.post(
        "/v1/chat", json={"query": "How many clips did Natalia sell altogether?"}
    ).json()

    if body["routing"]["escalated"]:
        assert "escalated" in body["routing"]["reason"]
        assert body["usage"]["cost_usd"] > body["usage"]["baseline_cost_usd"]
        assert body["usage"]["saved_usd"] < 0


def test_cascade_can_be_disabled_per_request(client):
    body = client.post(
        "/v1/chat",
        json={"query": "How many clips did Natalia sell altogether?", "no_cascade": True},
    ).json()
    assert body["routing"]["escalated"] is False
