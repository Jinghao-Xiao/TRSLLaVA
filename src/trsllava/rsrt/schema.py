from __future__ import annotations

from pydantic import BaseModel, Field


class RSRTVariant(BaseModel):
    one_sentence: str = Field(min_length=1)
    feature: str = Field(min_length=1)
    caption: str = Field(min_length=1)


class RSRTRecord(BaseModel):
    image: str
    variants: list[RSRTVariant] = Field(min_length=5, max_length=5)
    model: str | None = None

