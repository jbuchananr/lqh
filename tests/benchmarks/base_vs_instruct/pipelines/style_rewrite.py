"""Tone/style rewrite datagen for DPO-friendly generation.

The task asks the model to rewrite a flawed support reply according to a style
brief. The target is open-ended but constrained by facts, tone, banned phrases,
and length. This leaves more room for preference learning than single-label or
exact-field tasks.

Each sample is a 3-message ChatML conversation:
  system    -> SYSTEM_PROMPT
  user      -> draft + style brief + required facts
  assistant -> polished reply
"""

from __future__ import annotations

import random

from lqh.pipeline import ChatMLMessage, Conversation, Pipeline, step

SYSTEM_PROMPT = (
    "Rewrite customer-support replies to match the requested tone and style. "
    "Preserve all required facts, remove banned phrases, keep the reply concise, "
    "and output ONLY the rewritten reply."
)

_TONES = [
    "warm, calm, and accountable",
    "concise, professional, and direct",
    "empathetic but not apologetic",
    "friendly, precise, and action-oriented",
    "reassuring, plainspoken, and brief",
]
_ISSUES = [
    (
        "a delayed replacement laptop",
        ["replacement laptop ships tomorrow", "tracking email arrives by 6 PM"],
    ),
    (
        "a duplicate subscription charge",
        ["duplicate charge was refunded", "refund posts within 3-5 business days"],
    ),
    (
        "a failed password reset",
        ["temporary access link expires in 30 minutes", "support can reissue it"],
    ),
    (
        "a damaged grocery delivery",
        ["credit has been added to the account", "photos are no longer needed"],
    ),
    (
        "a noisy hotel room",
        ["room change is confirmed", "front desk has the new key ready"],
    ),
    (
        "a missed onboarding call",
        ["new calendar invite is attached", "the setup checklist is unchanged"],
    ),
]
_BANNED = [
    "we apologize for any inconvenience",
    "please be advised",
    "as per our policy",
    "thank you for your patience",
    "at your earliest convenience",
]


class StyleRewritePipeline(Pipeline):
    async def generate(self, client, input=None) -> Conversation:
        self.tone = random.choice(_TONES)
        self.issue, self.facts = random.choice(_ISSUES)
        self.banned = random.sample(_BANNED, 2)
        self.seed = f"sr-{self.issue}-{random.randint(0, 1_000_000)}"
        await self._make_draft(client)
        await self._rewrite(client)
        return [
            ChatMLMessage("system", SYSTEM_PROMPT),
            ChatMLMessage("user", self.user_prompt),
            ChatMLMessage("assistant", self.rewrite),
        ]

    @step(retries=4)
    async def _make_draft(self, client) -> None:
        draft = (
            f"{self.banned[0].capitalize()}. Regarding {self.issue}, "
            f"please be advised that {self.facts[0]}. Also, {self.facts[1]}. "
            f"{self.banned[1].capitalize()}. We wanted to circle back with "
            "this update and make sure you understand that the request is "
            "being handled through the normal support process."
        )
        self.draft = draft
        self.user_prompt = (
            f"Rewrite this support reply in a {self.tone} style.\n\n"
            "Required facts to preserve exactly:\n"
            + "\n".join(f"- {fact}" for fact in self.facts)
            + "\n\nBanned phrases to remove:\n"
            + "\n".join(f"- {phrase}" for phrase in self.banned)
            + "\n\nLength: 45-85 words. Use no bullet list, no greeting, "
            "and no sign-off. Output only the rewritten reply.\n\n"
            f"Draft:\n{self.draft}"
        )

    @step(retries=4)
    async def _rewrite(self, client) -> None:
        self.rewrite = (
            f"Here is the current update: {self.facts[0]}, and {self.facts[1]}. "
            f"I know {self.issue} can be frustrating, so I will keep this "
            f"{self.tone}. The next step is already set, and you do not need "
            "to send anything else unless these details look wrong."
        )
