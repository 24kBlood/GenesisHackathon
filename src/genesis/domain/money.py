"""FX conversion on integer amounts and decimal-string quotes."""

from decimal import Context, Decimal, localcontext


def convert_minor(
    amount: int, source: str, target: str, rate: str, exponents: dict[str, int], rounding: str,
) -> int:
    """Scale through both currency exponents, then round to target minor units."""
    with localcontext(Context(prec=64)):
        source_major = Decimal(amount).scaleb(-exponents[source])
        target_minor = (source_major * Decimal(rate)).scaleb(exponents[target])
        return int(target_minor.quantize(Decimal(1), rounding=rounding))
