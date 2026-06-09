"""Classification datagen — sentiment of a short review (3-class).

We roll a target label (positive / negative / neutral), then ask a model to
write a review that expresses that sentiment about a random subject. The
assistant target is the canonical JSON ``{"sentiment": <label>}``. The model
never sees the label name in a way it could echo — it writes prose, and the
gold is the rolled label.

Each sample is a 3-message ChatML conversation:
  system    -> SYSTEM_PROMPT
  user      -> a short review
  assistant -> {"sentiment": "positive"|"negative"|"neutral"}
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

_LABELS = ["positive", "negative", "neutral"]

SYSTEM_PROMPT = (
    "You are a sentiment classifier. Read the review and classify its overall "
    'sentiment. Output ONLY a JSON object {"sentiment": "<label>"} where '
    "<label> is exactly one of: positive, negative, neutral. No commentary, "
    "no code fences."
)

_SUBJECTS = [
    "a restaurant", "a smartphone", "a hotel stay", "a laptop", "a movie",
    "a pair of headphones", "an online course", "a coffee shop", "a video game",
    "a fitness tracker", "a streaming service", "a book", "a rideshare trip",
    "a kitchen appliance", "a software app", "a concert", "a pair of shoes",
    "a grocery delivery", "a board game", "a car rental",
]

# How to steer the model toward each sentiment without naming the label
# (so it can't trivially leak the answer into the text).
_TONE = {
    "positive": "clearly happy and satisfied — praise, recommendation",
    "negative": "clearly unhappy — complaints, disappointment, would not recommend",
    "neutral": "mixed or matter-of-fact — some pros and cons, no strong feeling",
}


class ClassificationPipeline(Pipeline):
    async def generate(self, client, input=None) -> Conversation:
        self.label = random.choice(_LABELS)
        self.subject = random.choice(_SUBJECTS)
        self.seed = f"cl-{self.label}-{self.subject}-{random.randint(0, 1_000_000)}"
        await self._make_review(client)
        gold = json.dumps({"sentiment": self.label})
        return [
            ChatMLMessage("system", SYSTEM_PROMPT),
            ChatMLMessage("user", self.review),
            ChatMLMessage("assistant", gold),
        ]

    @step(retries=4)
    async def _make_review(self, client) -> None:
        prompt = (
            f"Write a short, realistic customer review (2-4 sentences) about "
            f"{self.subject}. The tone should be {_TONE[self.label]}. Write only "
            f"the review text — do not mention the words positive, negative, or "
            f"neutral, and do not output a rating or label."
        )
        resp = await client.chat.completions.create(
            model=f"random:medium:{self.seed}",
            messages=[{"role": "user", "content": prompt}],
        )
        text = safe_content(resp).strip().strip("`'\"")
        if len(text) < 25:
            raise GenerationError(f"Review too short ({len(text)} chars)")
        low = text.lower()
        for leak in ("positive", "negative", "neutral"):
            if leak in low:
                raise GenerationError(f"Label word {leak!r} leaked into review")
        self.review = text
