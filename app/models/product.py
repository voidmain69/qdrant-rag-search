from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AttrValue = str | int | float | bool


class ProductIn(BaseModel):
    """Product as accepted by the ingestion API."""

    # leading/trailing whitespace must never make two external_ids map to different points
    model_config = ConfigDict(str_strip_whitespace=True)

    external_id: str = Field(min_length=1, max_length=128, description="Stable id in the source system")
    name: str = Field(min_length=1, max_length=512)
    description: str | None = Field(default=None, max_length=10_000)
    brand: str | None = Field(default=None, max_length=128)
    category: str | None = Field(default=None, max_length=256)
    article: str | None = Field(default=None, max_length=64, description="Артикул / SKU")
    product_code: str | None = Field(default=None, max_length=64, description="Внутрішній код товару")
    ean13: str | None = Field(default=None, max_length=32, description="Штрихкод EAN-13 (цифри)")
    attributes: dict[str, AttrValue] = Field(default_factory=dict)
    price: float | None = Field(default=None, ge=0)
    currency: str = Field(default="UAH", max_length=8)
    in_stock: bool = True

    @field_validator("name", "description", "brand", "category", mode="after")
    @classmethod
    def _collapse_whitespace(cls, v: str | None) -> str | None:
        if v is None:
            return None
        collapsed = " ".join(v.split())
        return collapsed or None

    @field_validator("ean13", mode="after")
    @classmethod
    def _digits_only(cls, v: str | None) -> str | None:
        if v is None:
            return None
        digits = "".join(c for c in v if c.isdigit())
        return digits or None


class PriceUpdate(BaseModel):
    """A price / availability change for one product — a payload-only update that never
    touches the vectors (price and stock are not embedded). At least one mutable field
    must be present."""

    model_config = ConfigDict(str_strip_whitespace=True)

    external_id: str = Field(min_length=1, max_length=128)
    price: float | None = Field(default=None, ge=0)
    currency: str | None = Field(default=None, max_length=8)
    in_stock: bool | None = None

    @model_validator(mode="after")
    def _at_least_one_field(self) -> PriceUpdate:
        if self.price is None and self.currency is None and self.in_stock is None:
            raise ValueError("at least one of price, currency, in_stock must be provided")
        return self

    def changed_fields(self) -> dict[str, Any]:
        """The subset of payload keys this update actually changes."""
        changes: dict[str, Any] = {}
        if self.price is not None:
            changes["price"] = self.price
        if self.currency is not None:
            changes["currency"] = self.currency
        if self.in_stock is not None:
            changes["in_stock"] = self.in_stock
        return changes


class PriceUpdateBatch(BaseModel):
    items: list[PriceUpdate] = Field(min_length=1, max_length=1000)


class BatchUpsertRequest(BaseModel):
    items: list[ProductIn] = Field(min_length=1, max_length=1000)


class BatchItemResult(BaseModel):
    external_id: str
    ok: bool
    error: str | None = None


class BatchUpsertResult(BaseModel):
    total: int
    succeeded: int
    failed: int
    items: list[BatchItemResult]
