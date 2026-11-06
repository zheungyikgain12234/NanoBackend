"""
byoa_adk/response_validator.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Validates real data against a declared schema -- used for BOTH directions:

  - validate_task_result(): the TaskResult an agent's run() returns,
    checked against its declared output_schema.
  - validate_task_input(): the real inputs a request arrives with,
    checked against the agent's declared input_schema.

Both share ONE underlying check (_validate_properties) so the two
directions can never silently behave differently from each other --
there is exactly one implementation of "does this data match this
declared schema", used both ways.

Used automatically by server.py — agent developers never call this directly.

Checks:
  1. Required keys are present
  2. Present values match their declared type
  3. None is only allowed where explicitly marked nullable
  4. Extra, undeclared keys are warned about, not treated as errors
"""
from __future__ import annotations
from dataclasses import dataclass, field
from .schema import TaskResult, TaskStatus

# Map JSON Schema type strings to Python types
_TYPE_MAP: dict[str, type] = {
    'string':  str,
    'integer': int,
    'number':  (int, float),
    'boolean': bool,
    'object':  dict,
    'array':   list,
}


@dataclass
class ResponseValidationResult:
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def fail(self, msg: str):
        self.valid = False
        self.errors.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)


def _validate_properties(data: dict, properties: dict, label: str) -> ResponseValidationResult:
    """
    The one shared check, used identically for inputs and outputs.

    data       : the real dict to check (payload.inputs, or result.outputs)
    properties : flat dict of field_name -> {"type": ..., "required": ..., "nullable": ...}
    label      : "inputs" or "outputs", used only in error messages
    """
    r = ResponseValidationResult()

    if not properties:
        return r  # nothing declared -- nothing to validate

    for key, defn in properties.items():
        is_required = defn.get('required', True)

        if key not in data:
            if is_required:
                r.fail(f'{label} missing declared key: "{key}"')
            continue

        actual_value = data[key]

        if actual_value is None:
            if defn.get('nullable', False):
                continue
            r.fail(f'{label}["{key}"] is None but property is not marked nullable')
            continue

        expected_type_str = defn.get('type')
        if not expected_type_str:
            r.warn(f'{label} schema for "{key}" has no "type" — skipping type check')
            continue

        expected_type = _TYPE_MAP.get(expected_type_str)
        if expected_type is None:
            r.warn(f'Unknown type "{expected_type_str}" for key "{key}" — skipping')
            continue

        if not isinstance(actual_value, expected_type):
            r.fail(
                f'{label}["{key}"] expected {expected_type_str}, '
                f'got {type(actual_value).__name__}: {repr(actual_value)[:60]}'
            )

    for key in data:
        if key not in properties:
            r.warn(f'{label}["{key}"] not declared in schema (extra field)')

    return r


def validate_task_result(
    result: TaskResult,
    output_schema: dict,
) -> ResponseValidationResult:
    """
    Validate a TaskResult against the agent's declared output_schema.

    output_schema uses a {"type": "object", "properties": {...}} wrapper
    (matching JSON Schema convention for an object's shape).

    Per-property flags (both optional, both default False):
      nullable: True  -> a present value of None passes, regardless of type
      required: False -> the key may be omitted from outputs entirely

    Only validates on SUCCESS — FAILURE/DEGRADED may have empty outputs.
    """
    if result.status != TaskStatus.SUCCESS:
        return ResponseValidationResult()

    outputs = result.outputs or {}
    properties = output_schema.get('properties', {})
    return _validate_properties(outputs, properties, label="outputs")


def validate_task_input(
    inputs: dict,
    input_schema: dict,
) -> ResponseValidationResult:
    """
    Validate real incoming request inputs against the agent's declared
    input_schema -- the same kind of check as validate_task_result(),
    just applied to the OTHER side of the contract.

    input_schema is a FLAT dict of field_name -> {"type": ..., "required": ...}
    (no "properties" wrapper -- this matches how input_schema has always
    been written in Agent.md and AgentManifest, unlike output_schema).
    """
    properties = input_schema or {}
    return _validate_properties(inputs, properties, label="inputs")

