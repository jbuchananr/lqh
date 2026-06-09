"""Structured-extraction datagen — unstructured invite text -> JSON.

We generate the *gold JSON first* (random but well-formed field values), then
ask a model to write a natural-language event invitation that embeds those
exact values. The assistant target is the canonical JSON. Generating the text
from the values (not the other way round) guarantees the gold is correct, and
a containment check rejects any blurb that dropped a value.

Each sample is a 3-message ChatML conversation:
  system    -> SYSTEM_PROMPT
  user      -> a free-text event invitation
  assistant -> the canonical JSON object
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

_KEYS = ["title", "date", "time", "location", "organizer"]

SYSTEM_PROMPT = (
    "You extract event details from a message into JSON. Output ONLY a single "
    "JSON object with exactly these keys: "
    + ", ".join(f'"{k}"' for k in _KEYS)
    + ". Use the value as written in the message. No commentary, no code "
    "fences — just the JSON object."
)

_TITLES = [
    "Quarterly Planning Sync", "Design Review", "Team Offsite", "Budget Kickoff",
    "Product Demo", "All-Hands Meeting", "Security Training", "Roadmap Workshop",
    "Customer Onboarding Call", "Retrospective", "Hiring Debrief", "Launch Rehearsal",
]
_LOCATIONS = [
    "Conference Room B", "the main auditorium", "Zoom", "the rooftop terrace",
    "Office 4F", "the downtown WeWork", "Meeting Room Aurora", "the cafeteria",
    "Building 2, Room 210", "Google Meet",
]
_ORGANIZERS = [
    "Priya Nair", "Tom Becker", "Sofia Russo", "Daniel Okafor", "Mei Lin",
    "Jonas Weber", "Aisha Rahman", "Carlos Mendez", "Hannah Schmidt", "Liam Murphy",
]
_MONTHS = ["January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"]


def _random_record() -> dict[str, str]:
    return {
        "title": random.choice(_TITLES),
        "date": f"{random.choice(_MONTHS)} {random.randint(1, 28)}",
        "time": f"{random.randint(1, 12)}:{random.choice(['00', '15', '30', '45'])} "
                f"{random.choice(['AM', 'PM'])}",
        "location": random.choice(_LOCATIONS),
        "organizer": random.choice(_ORGANIZERS),
    }


class ExtractionPipeline(Pipeline):
    async def generate(self, client, input=None) -> Conversation:
        self.record = _random_record()
        self.seed = f"ex-{self.record['title']}-{random.randint(0, 1_000_000)}"
        await self._make_invite(client)
        gold = json.dumps(self.record, ensure_ascii=False)
        return [
            ChatMLMessage("system", SYSTEM_PROMPT),
            ChatMLMessage("user", self.invite),
            ChatMLMessage("assistant", gold),
        ]

    @step(retries=4)
    async def _make_invite(self, client) -> None:
        r = self.record
        prompt = (
            "Write a short, natural event-invitation message (2-4 sentences, "
            "email or chat style). It MUST mention each of these verbatim:\n"
            f"- title: {r['title']}\n- date: {r['date']}\n- time: {r['time']}\n"
            f"- location: {r['location']}\n- organizer: {r['organizer']}\n\n"
            "Do not output JSON or a label — just the invitation prose."
        )
        resp = await client.chat.completions.create(
            model=f"random:medium:{self.seed}",
            messages=[{"role": "user", "content": prompt}],
        )
        text = safe_content(resp).strip()
        if len(text) < 30:
            raise GenerationError(f"Invite too short ({len(text)} chars)")
        # Every gold value must appear verbatim or the extraction gold is a lie.
        for key, val in r.items():
            if val not in text:
                raise GenerationError(f"Invite dropped {key}={val!r}")
        self.invite = text
