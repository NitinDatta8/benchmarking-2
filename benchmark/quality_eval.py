"""
Rule-based quality checks for benchmark outputs.

Checks model output against the quality_checks spec from each prompt:
  - field_presence: required keywords appear in output
  - json_schema: valid JSON with required keys, array lengths, numeric fields, etc.
  - hybrid: field_presence + json_schema combined
"""

import json
import re


def _has_keywords(output, keywords):
    lower = output.lower()
    return [kw.lower() in lower for kw in keywords]


def _count_list_items(output):
    numbered = re.findall(r"^\s*\d+[.)]\s", output, re.MULTILINE)
    bulleted = re.findall(r"^\s*[-*•]\s", output, re.MULTILINE)
    return max(len(numbered), len(bulleted))


def _parse_json(output):
    clean = re.sub(r"^```(?:json)?\s*", "", output.strip(), flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean)
    return json.loads(clean)


def _resolve_path(obj, path):
    match = re.match(r"^(\w+)\[(\d+)]$", path)
    if match:
        key, idx = match.group(1), int(match.group(2))
        arr = obj.get(key) if isinstance(obj, dict) else None
        if isinstance(arr, list) and idx < len(arr):
            return arr[idx]
        return None
    return obj.get(path) if isinstance(obj, dict) else None


def _check_json(output, spec):
    results = []
    try:
        parsed = _parse_json(output)
        results.append(True)
    except (json.JSONDecodeError, ValueError):
        return [False]

    for key in spec.get("required_keys", []):
        results.append(key in parsed)

    for key, min_len in spec.get("array_fields", {}).items():
        val = parsed.get(key)
        results.append(isinstance(val, list) and len(val) >= min_len)

    for key in spec.get("numeric_fields", []):
        if key in parsed:
            results.append(isinstance(parsed[key], (int, float)))

    for path, keys in spec.get("nested_keys", {}).items():
        obj = _resolve_path(parsed, path)
        for k in keys:
            results.append(isinstance(obj, dict) and k in obj)

    for key, constraint in spec.get("value_constraints", {}).items():
        val = parsed.get(key)
        if not isinstance(val, (int, float)):
            results.append(False)
            continue
        lo = constraint.get("min", float("-inf"))
        hi = constraint.get("max", float("inf"))
        results.append(lo <= val <= hi)

    return results


def evaluate(model_output, quality_checks):
    """Returns an object with .passed (bool) and .pass_rate (float 0-1)."""
    if not model_output or not model_output.strip():
        return _result(0, 1)

    check_type = quality_checks.get("type", "field_presence")
    checks = []

    if check_type == "json_schema":
        checks.extend(_check_json(model_output, quality_checks))

    elif check_type in ("field_presence", "hybrid"):
        required = quality_checks.get("required_fields", [])
        if required:
            checks.extend(_has_keywords(model_output, required))

        min_items = quality_checks.get("min_list_items")
        if min_items is not None:
            checks.append(_count_list_items(model_output) >= min_items)

        constraints = quality_checks.get("constraints_to_verify", [])
        if constraints:
            checks.extend(_has_keywords(model_output, constraints))

        if check_type == "hybrid":
            checks.extend(_check_json(model_output, quality_checks))
    else:
        checks.append(True)

    passed = sum(checks)
    return _result(passed, len(checks))


class _result:
    __slots__ = ("passed", "pass_rate")

    def __init__(self, passed_count, total):
        self.pass_rate = passed_count / total if total > 0 else 0.0
        self.passed = self.pass_rate >= 0.8
