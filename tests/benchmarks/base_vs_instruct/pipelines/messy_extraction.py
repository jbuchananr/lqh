"""Messy structured extraction datagen.

This task is deliberately harder than the clean extraction benchmark: the
user message is a noisy support-thread excerpt with distractors, corrections,
and stale values. The assistant target is one canonical JSON object containing
the latest applicable values only.

Each sample is a 3-message ChatML conversation:
  system    -> SYSTEM_PROMPT
  user      -> noisy thread text
  assistant -> canonical JSON object
"""

from __future__ import annotations

import json
import random

from lqh.pipeline import (
    ChatMLMessage,
    Conversation,
    GenerationError,
    Pipeline,
    safe_content,
    step,
)

_KEYS = [
    "customer_name",
    "account_id",
    "requested_action",
    "effective_date",
    "priority",
    "owner",
]

SYSTEM_PROMPT = (
    "You extract the current support request from a messy message thread. "
    "Output ONLY one JSON object with exactly these keys: "
    + ", ".join(f'"{k}"' for k in _KEYS)
    + ". Use the latest corrected values, ignore stale/distractor values, "
    "and use null for a missing value. No commentary, no code fences."
)

_NAMES = [
    "Mara Klein", "Owen Brooks", "Priya Shah", "Elena Novak",
    "Daniel Ford", "Nadia Rahman", "Theo Martin", "Lena Ortiz",
]
_OWNERS = [
    "Avery Chen", "Mina Patel", "Jon Bell", "Sofia Ruiz",
    "Eli Wagner", "Noah Kim", "Iris Stein", "Camila Torres",
]
_ACTIONS = [
    "pause subscription",
    "update billing contact",
    "expedite replacement",
    "reset administrator access",
    "change shipping address",
    "schedule onboarding call",
    "cancel duplicate order",
    "extend trial period",
]
_PRIORITIES = ["low", "normal", "high", "urgent"]
_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _date() -> str:
    return f"{random.choice(_MONTHS)} {random.randint(1, 28)}, 2026"


def _account_id() -> str:
    return f"AC-{random.randint(1000, 9999)}"


def _record() -> dict[str, str]:
    return {
        "customer_name": random.choice(_NAMES),
        "account_id": _account_id(),
        "requested_action": random.choice(_ACTIONS),
        "effective_date": _date(),
        "priority": random.choice(_PRIORITIES),
        "owner": random.choice(_OWNERS),
    }


class MessyExtractionPipeline(Pipeline):
    async def generate(self, client, input=None) -> Conversation:
        self.record = _record()
        self.stale_account = _account_id()
        while self.stale_account == self.record["account_id"]:
            self.stale_account = _account_id()
        self.stale_date = _date()
        self.seed = f"mx-{self.record['account_id']}-{random.randint(0, 1_000_000)}"
        await self._make_thread(client)
        gold = json.dumps(self.record, ensure_ascii=False)
        return [
            ChatMLMessage("system", SYSTEM_PROMPT),
            ChatMLMessage("user", self.thread),
            ChatMLMessage("assistant", gold),
        ]

    @step(retries=4)
    async def _make_thread(self, client) -> None:
        r = self.record
        prompt = (
            "Write a realistic but compact support-thread excerpt with 5-8 lines. "
            "It must contain stale values, then later corrections. The extraction "
            "target is the latest corrected request.\n\n"
            "Latest values that MUST appear verbatim:\n"
            f"- customer_name: {r['customer_name']}\n"
            f"- account_id: {r['account_id']}\n"
            f"- requested_action: {r['requested_action']}\n"
            f"- effective_date: {r['effective_date']}\n"
            f"- priority: {r['priority']}\n"
            f"- owner: {r['owner']}\n\n"
            "Distractor/stale values that MUST also appear verbatim but be "
            "clearly superseded later:\n"
            f"- stale account_id: {self.stale_account}\n"
            f"- stale effective_date: {self.stale_date}\n\n"
            "Mention that the latest message overrides the earlier one. "
            "Output only the thread text, no JSON, no labels."
        )
        resp = await client.chat.completions.create(
            model=f"random:medium:{self.seed}",
            messages=[{"role": "user", "content": prompt}],
        )
        text = safe_content(resp).strip()
        if len(text) < 120:
            raise GenerationError(f"Thread too short ({len(text)} chars)")
        for key, val in r.items():
            if val not in text:
                raise GenerationError(f"Thread dropped latest {key}={val!r}")
        for val in (self.stale_account, self.stale_date):
            if val not in text:
                raise GenerationError(f"Thread dropped stale value {val!r}")
        self.thread = text
