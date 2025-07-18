import re
from typing import Optional, Tuple, List


def extract_mentioned_products(text: str) -> Optional[List[Tuple[str, str]]]:
    pattern = r'\[\s*"([^"]+)"\s*,\s*"([^"]*)"\s*\]'
    matches = re.findall(pattern, text)

    if not matches:
        return None

    from collections import defaultdict

    sums = defaultdict(int)
    empty_qty = {}

    for code, qty in matches:
        code = code.strip()
        if not code:
            continue
        qty = qty.strip()
        if qty.isdigit():
            sums[code] += int(qty)
        elif qty == "":
            empty_qty[code] = True

    result = [(code, str(sums[code])) for code in sums]

    for code in empty_qty:
        if code not in sums:
            result.append((code, ""))

    return result if result else None


def extract_response_text(text: str) -> Optional[str]:
    pattern = r'"responder"\s*:\s*true\s*,\s*"respuesta"\s*:\s*"?([^"]+)"?'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None


def is_order(output_text: str) -> bool:
    return '"order": true' in output_text.lower()


def is_order_confirmation(message: str) -> bool:
    pattern = r"(es\s*correct[oa]*)"
    return bool(re.search(pattern, message.lower()))
