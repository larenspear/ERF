#!/usr/bin/env python3
"""Check UPWINDALL's Z interpolation reads only its selected stencil.

The checker reads the live InterpolateInZ branches and compares their conditional
m_phi loads with the corresponding standalone interpolation structs and declared
GetUpwindCellNumber() values. C++'s conditional operator evaluates only its selected
arm, so the sp2/sm3 ternaries determine whether the outer ghost cells are read.

usage: check_upwindall_z_reach.py [ERF_ROOT]
exit:  0 all checks pass, 1 a check failed
"""
import re
import sys
from pathlib import Path

UPW_H = "Source/Utils/ERF_Interpolation_UPW.H"
BRANCHES = {
    "Centered_2nd": "CENTERED2",
    "Upwind_3rd": "UPWIND3",
    "Centered_4th": "CENTERED4",
    "Upwind_5th": "UPWIND5",
    "Centered_6th": "CENTERED6",
}

failures = []


def check(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        failures.append(what)


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def brace_block(text, start):
    depth = 0
    for i in range(start, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return text[start + 1:i]
    raise ValueError("unbalanced braces in interpolation source")


def split_structs(text):
    return {m.group(1): m.group(2) for m in re.finditer(
        r"^struct (\w+)\s*\n\{(.*?)^\};", text, flags=re.S | re.M)}


def function_body(struct, name):
    match = re.search(rf"\b{name}\s*\([^)]*\)\s*const\s*\{{", struct, flags=re.S)
    if not match:
        raise ValueError(f"{name}() not found")
    return brace_block(struct, match.end() - 1)


def cell_offset(expr, base):
    match = re.fullmatch(rf"{base}\s*(?:([+-])\s*(\d+))?", expr.strip())
    if not match:
        raise ValueError(f"unexpected cell index {expr!r}; expected {base} +/- integer")
    if not match.group(1):
        return 0
    return int(match.group(2)) * (1 if match.group(1) == "+" else -1)


def read_offsets(expr):
    """Return k offsets of m_phi reads, validating that they stay in the Z line."""
    offsets = []
    for match in re.finditer(r"m_phi\s*\(([^)]*)\)", expr):
        args = [arg.strip() for arg in match.group(1).split(",")]
        if len(args) != 4 or args[0] != "i" or args[1] != "j" or args[3] != "qty_index":
            raise ValueError(f"unexpected off-axis or non-cell read: {match.group(0)}")
        offsets.append(cell_offset(args[2], "k"))
    return offsets


def enum_set(condition):
    condition = condition.strip()
    while condition.startswith("(") and condition.endswith(")"):
        condition = condition[1:-1].strip()
    names = []
    for clause in condition.split("||"):
        match = re.fullmatch(r"\s*adv_type\s*==\s*AdvType::(\w+)\s*", clause)
        if not match:
            raise ValueError(f"unsupported adv_type condition: {condition}")
        names.append(match.group(1))
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate enum in adv_type condition: {condition}")
    return set(names)


def half_width(offsets):
    if not offsets:
        raise ValueError("cannot compute reach of an empty stencil")
    return max(-min(offsets), max(offsets) + 1)


def standalone_stencil(body):
    interp = function_body(body, "InterpolateInZ")
    offsets = set(read_offsets(interp))
    declared = re.search(r"GetUpwindCellNumber\s*\(\)\s*const\s*\{\s*return\s*(\d+)\s*;", body)
    if not declared:
        raise ValueError("GetUpwindCellNumber() not found")
    declared = int(declared.group(1))
    actual = half_width(offsets)
    check(actual == declared,
          f"standalone GetUpwindCellNumber()={declared} matches Z read reach {actual} ({sorted(offsets)})")
    return offsets, declared


def analyze_upwindall(body, standalones):
    interp = function_body(body, "InterpolateInZ")
    first_if = re.search(r"\bif\s*\((.*?)\)\s*\{", interp, flags=re.S)
    if not first_if:
        raise ValueError("Centered_2nd early-return branch not found")
    early_cond = enum_set(first_if.group(1))
    early_block = brace_block(interp, first_if.end() - 1)
    check(early_cond == {"Centered_2nd"},
          f"early-return load path is selected only by Centered_2nd (found {sorted(early_cond)})")
    check(bool(re.search(r"\breturn\s*;", early_block)),
          "Centered_2nd path returns before the higher-order loads")
    early_offsets = set(read_offsets(early_block))

    else_match = re.search(r"\belse\s*\{", interp[first_if.end():])
    if not else_match:
        raise ValueError("non-Centered_2nd else block not found")
    else_open = first_if.end() + else_match.end() - 1
    else_block = brace_block(interp, else_open)

    declarations = {}
    for match in re.finditer(r"\bamrex::Real\s+(\w+)\s*=\s*(.*?);", else_block, flags=re.S):
        declarations[match.group(1)] = match.group(2).strip()
    check(set(declarations) == {"sp2", "sp1", "s", "sm1", "sm2", "sm3"},
          f"non-early path declares the six expected stencil values (found {sorted(declarations)})")

    common = {}
    gates = {}
    gated_offsets = {}
    for name, expression in declarations.items():
        if "?" in expression:
            condition, arms = expression.split("?", 1)
            true_expr, separator, false_expr = arms.partition(":")
            if not separator:
                raise ValueError(f"{name}: conditional load is missing its false arm")
            gates[name] = enum_set(condition)
            true_offsets = read_offsets(true_expr)
            false_offsets = read_offsets(false_expr)
            check(len(true_offsets) == 1,
                  f"{name}: true arm has exactly one m_phi read (found {true_offsets})")
            check(not false_offsets and bool(re.fullmatch(
                r"\s*amrex::Real\(\s*0(?:\.0*)?\s*\)\s*", false_expr)),
                f"{name}: false arm is zero and performs no m_phi read")
            gated_offsets[name] = set(true_offsets)
        else:
            offsets = read_offsets(expression)
            check(len(offsets) == 1,
                  f"{name}: unconditional value has exactly one m_phi read (found {offsets})")
            common[name] = set(offsets)

    expected_gated_names = {"sp2", "sm3"}
    check(set(gates) == expected_gated_names,
          f"only sp2 and sm3 use conditional outer-cell loads (found {sorted(gates)})")
    common_offsets = set().union(*common.values()) if common else set()
    expected_gate_sets = {}
    for gated_name in expected_gated_names:
        expected_gate_sets[gated_name] = {
            adv_type for adv_type, struct_name in BRANCHES.items()
            if standalones[struct_name][0] - common_offsets
            and gated_offsets.get(gated_name, set()) & standalones[struct_name][0]
        }
        check(gates.get(gated_name) == expected_gate_sets[gated_name],
              f"{gated_name}: conditional read is enabled exactly for schemes whose standalone "
              f"stencil needs it (expected {sorted(expected_gate_sets[gated_name])}, "
              f"found {sorted(gates.get(gated_name, set()))})")

    check(set(common_offsets) == {0, 1, -1, -2},
          f"unconditional non-early reads are the shared core stencil (found {sorted(common_offsets)})")

    print("\nUPWINDALL branch reaches")
    for adv_type, struct_name in BRANCHES.items():
        if adv_type in early_cond:
            actual = early_offsets
        else:
            actual = set(common_offsets)
            for name, gate in gates.items():
                if adv_type in gate:
                    actual.update(gated_offsets[name])
        expected, declared = standalones[struct_name]
        reach = half_width(actual)
        check(actual == expected,
              f"{adv_type}: live Z reads {sorted(actual)} match {struct_name} stencil {sorted(expected)}")
        check(reach == declared,
              f"{adv_type}: live Z read reach {reach} matches {struct_name} declaration {declared}")


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    text = strip_comments((root / UPW_H).read_text())
    structs = split_structs(text)
    expected_structs = set(BRANCHES.values()) | {"UPWINDALL"}
    check(expected_structs.issubset(structs),
          f"header contains UPWINDALL and standalone schemes {sorted(expected_structs)}")
    if not expected_structs.issubset(structs):
        return 1

    standalones = {name: standalone_stencil(structs[name]) for name in BRANCHES.values()}
    analyze_upwindall(structs["UPWINDALL"], standalones)
    print(f"\n{len(failures)} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print(f"FAIL checker could not parse source: {exc}")
        sys.exit(1)
