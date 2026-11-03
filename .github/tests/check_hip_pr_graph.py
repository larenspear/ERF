import json
import re
import sys
from pathlib import Path

def targets(path):
    deps = {}
    for line in Path(path).read_text().splitlines():
        match = re.match(r"^([^\s:#]+):(?: (.*))?$", line)
        if match:
            deps.setdefault(match[1], set()).update((match[2] or "").split())
    seen, todo = set(), ["all"]
    while todo:
        node = todo.pop()
        if node not in seen:
            seen.add(node)
            todo.extend(deps.get(node, ()))
    return {node for node in seen if node.endswith(".dir/all")}

if __name__ == "__main__":
    p = Path(sys.argv[1])
    before = targets(p / "baseline-Makefile2")
    after = targets(p / "candidate-Makefile2")
    removed = before - after
    assert len(removed) == 11, sorted(removed)
    assert all(t.startswith("Submodules/AMReX/Tools/Plotfile/") for t in removed), sorted(removed)
    assert not after - before, sorted(after - before)
    assert json.loads((p / "baseline-compile-commands.json").read_text()) == json.loads((p / "candidate-compile-commands.json").read_text()), "Compiler commands changed"
    result = {"baseline_targets": len(before), "candidate_targets": len(after), "removed_targets": sorted(removed), "preserved_targets": sorted(after), "compile_commands_identical": True}
    (p / "target-coverage.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
