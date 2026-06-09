"""Translation datagen — English -> German, strict format discipline.

A "non-standard" translation task in the sense the brief means: the value
being trained isn't the translation knowledge itself (most instruct models
can already translate) but *format discipline* — emit ONLY the German, no
preamble, no quotes, no commentary. Out of the box, chat-tuned models tend
to wrap the answer ("Sure, in German that's '...'."), which the scorer
punishes; base models tend to ramble or continue the English. After SFT the
model should snap to the bare-translation format. That gap is exactly what
discriminates a good fine-tuning base from a bad one.

Each sample is a 3-message ChatML conversation:
  system    -> SYSTEM_PROMPT (baked so train/eval/score all share it)
  user      -> an English sentence
  assistant -> its German translation, bare
"""

from __future__ import annotations

import random

from lqh.pipeline import (
    ChatMLMessage,
    Conversation,
    GenerationError,
    Pipeline,
    safe_content,
    step,
)

SYSTEM_PROMPT = (
    "You are a translation engine that translates English to German. "
    "Output ONLY the German translation — no preamble, no surrounding "
    "quotes, no explanation, no English. Just the German sentence."
)

# Topic seeds for sentence variety. Kept broad so the eval set generalises
# beyond any single register.
_TOPICS = [
    "everyday small talk", "ordering food at a restaurant", "asking for directions",
    "a weather report", "a short news headline", "a customer-support reply",
    "a travel itinerary note", "a recipe step", "a workplace email sentence",
    "a museum exhibit caption", "a product review line", "a sports commentary line",
    "a doctor's appointment reminder", "a hobby description", "a children's story line",
    "a technical instruction", "a real-estate listing line", "a job posting sentence",
    "a social-media post", "a historical fact",
]


class TranslationPipeline(Pipeline):
    async def generate(self, client, input=None) -> Conversation:
        self.topic = random.choice(_TOPICS)
        self.seed = f"tr-{self.topic}-{random.randint(0, 1_000_000)}"
        await self._make_english(client)
        await self._translate(client)
        return [
            ChatMLMessage("system", SYSTEM_PROMPT),
            ChatMLMessage("user", self.english),
            ChatMLMessage("assistant", self.german),
        ]

    @step(retries=4)
    async def _make_english(self, client) -> None:
        prompt = (
            f"Write ONE natural English sentence about: {self.topic}. "
            f"8 to 20 words. Output only the sentence, no quotes, no label."
        )
        resp = await client.chat.completions.create(
            model=f"random:medium:{self.seed}",
            messages=[{"role": "user", "content": prompt}],
        )
        text = safe_content(resp).strip().strip("`'\"")
        words = text.split()
        if not (5 <= len(words) <= 30):
            raise GenerationError(f"English sentence out of range ({len(words)} words)")
        self.english = text

    @step(retries=4)
    async def _translate(self, client) -> None:
        prompt = (
            "Translate this English sentence into natural German. Output ONLY "
            "the German translation, nothing else:\n\n"
            f"{self.english}"
        )
        resp = await client.chat.completions.create(
            model=f"random:large:{self.seed}",
            messages=[{"role": "user", "content": prompt}],
        )
        german = safe_content(resp).strip().strip("`'\"")
        if not german:
            raise GenerationError("Empty German translation")
        # Reject obvious English passthrough / preamble leakage.
        for bad in ("Sure", "Here", "The German", "In German", "Translation"):
            if german.startswith(bad):
                raise GenerationError(f"Preamble leaked into gold: {german[:40]!r}")
        self.german = german
