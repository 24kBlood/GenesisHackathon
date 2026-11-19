"""Canonical values shared by the v1 wire schema and Python code."""

from enum import StrEnum


class FindingType(StrEnum):
    DUPLICATE_INVOICE = "duplicate_invoice"
    PAYMENT_ALLOCATION = "payment_allocation"
    FX_CONVERSION = "fx_conversion"
    PERIOD_CUTOFF = "period_cutoff"
