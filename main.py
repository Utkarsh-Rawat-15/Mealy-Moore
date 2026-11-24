"""Mealy-Moore machine simulator: complete backend in one file.

    python main.py            built-in self-test
    streamlit run app.py      interface

M = (Q, Sigma, Delta, delta, lambda, q0); Moore reads lambda on the state.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from typing import Any

MEALY, MOORE, TRAP = "mealy", "moore", "__trap__"
EPS = {"", "e", "eps", "epsilon", "lambda", "\u03bb"}


class MachineError(Exception):
    """Anything the user can trigger."""


class DefinitionError(MachineError):
    """The machine is not well formed."""


class InputError(MachineError):
    """The input does not fit Sigma."""


class SimulationError(MachineError):
    """A needed transition is undefined."""


@dataclass(frozen=True)
class Machine:
    name: str
    kind: str
    states: tuple[str, ...]
    sigma: tuple[str, ...]
    gamma: tuple[str, ...]
    start: str
    delta: dict[tuple[str, str], str] = field(default_factory=dict)
    lam: dict[Any, str] = field(default_factory=dict)

    def validate(self, complete: bool = True) -> list[str]:
        """All problems at once, so one edit round can fix them."""
        e: list[str] = []
        if not self.name.strip():
            e.append("The machine name must not be empty.")
        if self.kind not in (MEALY, MOORE):
            e.append(f"Machine type must be '{MEALY}' or '{MOORE}'.")
        if not self.states:
            e.append("Q is empty: define at least one state.")
        elif len(set(self.states)) != len(self.states):
            e.append(f"Duplicate state(s) in Q: {_dupes(self.states)}.")
        if self.start not in self.states:
            e.append(f"Start state '{self.start}' is not in Q.")
        if not self.sigma:
            e.append("Sigma is empty: define at least one input symbol.")
        if not self.gamma:
            e.append("Delta is empty: define at least one output symbol.")
        for symbol in self.sigma:
            if symbol.lower() in EPS:
                e.append(f"Input symbol '{symbol}' is reserved for the empty string.")
            if any(c.isspace() for c in symbol):
                e.append(f"Input symbol '{symbol}' contains whitespace.")
        for (state, symbol), target in self.delta.items():
            if state not in self.states or symbol not in self.sigma:
                e.append(f"Transition ({state},{symbol}) uses an unknown state or symbol.")
            if target not in self.states:
                e.append(f"Transition {state} --{symbol}--> '{target}' leaves Q.")
        missing = [f"delta({s},{a})" for s in self.states for a in self.sigma
                   if (s, a) not in self.delta]
        if missing and complete:
            e.append("Missing transition(s): " + _short(missing) + ".")
        if self.kind == MEALY:
            absent = [f"({s},{a})" for (s, a) in self.delta if (s, a) not in self.lam]
            if absent and complete:
                e.append("Missing output(s): " + _short(absent) + ".")
        else:
            absent = [s for s in self.states if s not in self.lam]
            if absent:
                e.append("Missing Moore output for state(s): " + _short(absent) + ".")
        bad = sorted({v for v in self.lam.values() if v not in self.gamma})
        if bad:
            e.append(f"Output symbol(s) outside Delta: {', '.join(bad)}.")
        return e

    def check(self, complete: bool = True) -> None:
        if errors := self.validate(complete):
            raise DefinitionError(" ".join(errors))

    def with_trap(self) -> "Machine":
        """Fill every hole with one explicit non-accepting trap state."""
        missing = [(s, a) for s in self.states for a in self.sigma
                   if (s, a) not in self.delta]
        if not missing:
            return self
        fill = self.gamma[0]
        delta = dict(self.delta)
        lam = dict(self.lam)
        for state, symbol in missing:
            delta[(state, symbol)] = TRAP
            if self.kind == MEALY:
                lam[(state, symbol)] = fill
        for symbol in self.sigma:
            delta[(TRAP, symbol)] = TRAP
            if self.kind == MEALY:
                lam[(TRAP, symbol)] = fill
        if self.kind == MOORE:
            lam[TRAP] = fill
        return Machine(f"{self.name} + trap", self.kind, self.states + (TRAP,),
                       self.sigma, self.gamma, self.start, delta, lam)


def _dupes(items) -> str:
    seen, out = set(), []
    for item in items:
        if item in seen and item not in out:
            out.append(item)
        seen.add(item)
    return ", ".join(out)


def _short(items, limit: int = 6) -> str:
    head = ", ".join(items[:limit])
    return head if len(items) <= limit else f"{head} and {len(items) - limit} more"


def tokenize(text: str, sigma) -> list[str]:
    """Comma form, whitespace form, or greedy longest match for multi-char symbols."""
    raw = text.strip()
    if raw.lower() in EPS:
        return []
    if "," in raw:
        tokens, source = [part.strip() for part in raw.split(",")], "comma"
        if any(not token for token in tokens):
            raise InputError("Empty symbol in the comma-separated input.")
    elif len(raw.split()) > 1:
        tokens, source = raw.split(), "space"
    else:
        tokens, index = [], 0
        while index < len(raw):
            hit = next((s for s in sorted(sigma, key=len, reverse=True)
                        if raw.startswith(s, index)), None)
            if hit is None:
                raise InputError(f"Unknown symbol '{raw[index]}' at position {index + 1}.")
            tokens.append(hit)
            index += len(hit)
        return tokens
    for position, token in enumerate(tokens, 1):
        if token not in sigma:
            raise InputError(f"Unknown symbol '{token}' at position {position}; "
                             f"Sigma is {{{', '.join(sigma)}}}.")
    return tokens


def format_symbols(symbols) -> str:
    return " ".join(symbols) if symbols else "\u03b5 (empty)"


@dataclass(frozen=True)
class Step:
    index: int
    state: str
    symbol: str
    next_state: str
    output: str
    origin: str


@dataclass(frozen=True)
class RunResult:
    input_symbols: tuple[str, ...]
    steps: tuple[Step, ...]
    initial_output: str
    final_state: str
    output_symbols: tuple[str, ...]
    kind: str

    @property
    def output_text(self) -> str:
        return "".join(self.output_symbols)

    @property
    def expected_length(self) -> int:
        return len(self.input_symbols) + (self.kind == MOORE)

    @property
    def length_ok(self) -> bool:
        return len(self.output_symbols) == self.expected_length


def simulate(machine: Machine, text: str) -> RunResult:
    """One pass over the input: O(n) time, O(n) trace."""
    machine.check(complete=False)
    symbols = tokenize(text, machine.sigma)
    state = machine.start
    outputs = [machine.lam[state]] if machine.kind == MOORE else []
    steps = []
    for index, symbol in enumerate(symbols, 1):
        key = (state, symbol)
        if key not in machine.delta:
            raise SimulationError(f"Missing transition delta({state},{symbol}). "
                                  "Add it or enable trap completion.")
        target = machine.delta[key]
        output = machine.lam[key] if machine.kind == MEALY else machine.lam[target]
        steps.append(Step(index, state, symbol, target, output,
                          "transition" if machine.kind == MEALY else "state"))
        outputs.append(output)
        state = target
    return RunResult(tuple(symbols), tuple(steps),
                     outputs[0] if machine.kind == MOORE else "",
                     state, tuple(outputs), machine.kind)


def moore_to_mealy(m: Machine) -> Machine:
    return Machine(f"{m.name} -> Mealy", MEALY, m.states, m.sigma, m.gamma, m.start,
                   dict(m.delta),
                   {(s, a): m.lam[m.delta[(s, a)]] for s in m.states for a in m.sigma})


def mealy_to_moore(m: Machine, start_output: str | None = None) -> Machine:
    """Pairs (state, output) keep the output that was produced on entry."""
    start_output = start_output or m.gamma[0]
    if start_output not in m.gamma:
        raise DefinitionError(f"Start output '{start_output}' is not in Delta.")
    first = f"{m.start}/{start_output}"
    states, delta, lam = [first], {}, {first: start_output}
    queue = [(m.start, start_output)]
    while queue:
        state, out = queue.pop(0)
        here = f"{state}/{out}"
        for symbol in m.sigma:
            target, emitted = m.delta[(state, symbol)], m.lam[(state, symbol)]
            there = f"{target}/{emitted}"
            delta[(here, symbol)] = there
            if there not in lam:
                lam[there] = emitted
                states.append(there)
                queue.append((target, emitted))
    return Machine(f"{m.name} -> Moore", MOORE, tuple(states), m.sigma, m.gamma,
                   first, delta, lam)


def convert(machine: Machine, start_output: str | None = None) -> Machine:
    return moore_to_mealy(machine) if machine.kind == MOORE else mealy_to_moore(machine, start_output)


def compare_moore_mealy(moore: Machine, mealy: Machine, text: str) -> dict[str, Any]:
    """Compare output symbols, so multi-character outputs cannot collide."""
    a, b = simulate(moore, text), simulate(mealy, text)
    return {"matches": a.output_symbols[1:] == b.output_symbols, "moore": a, "mealy": b}


@dataclass(frozen=True)
class DiagramView:
    state: str
    previous: str | None = None
    symbol: str | None = None
    path: tuple[tuple[str, str], ...] = ()
    caption: str = ""


def active_view(run: RunResult, index: int) -> DiagramView:
    index = max(0, min(index, len(run.steps)))
    if index == 0:
        first = run.steps[0].state if run.steps else run.final_state
        return DiagramView(first, caption="Start state, before reading any symbol.")
    step = run.steps[index - 1]
    return DiagramView(step.next_state, step.state, step.symbol,
                       tuple((s.state, s.next_state) for s in run.steps[: index - 1]),
                       f"Step {index} of {len(run.steps)}: {step.state} -> {step.next_state} "
                       f"on {step.symbol} / output {step.output}")


def build_dot(m: Machine, view: DiagramView | None = None) -> str:
    """DOT with active state (amber), active edge (red) and path (violet)."""
    def esc(text: str) -> str:
        return text.replace("\\", "\\\\").replace('"', '\\"')

    out = ["digraph M {", "  rankdir=LR; bgcolor=\"white\";",
           "  node [shape=circle, style=filled, fillcolor=\"#EEF2FF\", color=\"#2F5597\","
           " fontname=\"Helvetica\", fontsize=13, penwidth=1.6];",
           "  edge [fontname=\"Helvetica\", fontsize=11, color=\"#56607A\"];",
           "  start [shape=point, width=0.12, color=\"#1F3864\"];"]
    for state in m.states:
        label = f"{state} / {m.lam[state]}" if m.kind == MOORE else state
        attrs = f'label="{esc(label)}"'
        if view and view.state == state:
            attrs += ', fillcolor="#FDBA74", color="#C2410C", penwidth=3'
        if state == m.start:
            attrs += ", peripheries=2"
        out.append(f'  "{esc(state)}" [{attrs}];')
    out.append(f'  start -> "{esc(m.start)}";')
    grouped: dict[tuple[str, str], list[str]] = {}
    for (state, symbol), target in m.delta.items():
        text = f"{symbol} / {m.lam[(state, symbol)]}" if m.kind == MEALY else symbol
        grouped.setdefault((state, target), []).append(text)
    for (state, target), labels in grouped.items():
        style = ""
        if view and view.previous == state and target == view.state:
            style = ', color="#DC2626", penwidth=3, fontcolor="#DC2626"'
        elif view and (state, target) in view.path:
            style = ', color="#7C3AED", penwidth=2'
        out.append(f'  "{esc(state)}" -> "{esc(target)}" '
                   f'[label="{esc(chr(10).join(labels))}"{style}];')
    return "\n".join(out + ["}"])


# ----------------------------------------------------------------------
# Examples
# ----------------------------------------------------------------------
def _machine(name, kind, states, sigma, gamma, start, rows, moore_outputs=None):
    delta = {(s, a): t for s, a, t, _ in rows}
    lam = ({s: o for s, o in moore_outputs.items()} if moore_outputs is not None
           else {(s, a): o for s, a, _, o in rows})
    return Machine(name, kind, tuple(states), tuple(sigma), tuple(gamma), start, delta, lam)


PRESETS = {
    "detect_11_mealy": _machine(
        "Detect 11 - Mealy", MEALY, ("q0", "q1"), ("0", "1"), ("0", "1"), "q0",
        [("q0", "0", "q0", "0"), ("q0", "1", "q1", "0"),
         ("q1", "0", "q0", "0"), ("q1", "1", "q1", "1")]),
    "detect_11_moore": _machine(
        "Detect 11 - Moore", MOORE, ("q0", "q1", "q2"), ("0", "1"), ("0", "1"), "q0",
        [("q0", "0", "q0", ""), ("q0", "1", "q1", ""),
         ("q1", "0", "q0", ""), ("q1", "1", "q2", ""),
         ("q2", "0", "q0", ""), ("q2", "1", "q2", "")],
        {"q0": "0", "q1": "0", "q2": "1"}),
    "parity_mealy": _machine(
        "Parity - Mealy", MEALY, ("even", "odd"), ("0", "1"), ("0", "1"), "even",
        [("even", "0", "even", "0"), ("even", "1", "odd", "1"),
         ("odd", "0", "odd", "1"), ("odd", "1", "even", "0")]),
    "ends_ab_mealy": _machine(
        "Ends with ab - Mealy", MEALY, ("start", "a seen", "ab seen"), ("a", "ab", "b"),
        ("0", "1"), "start",
        [("start", "a", "a seen", "0"), ("start", "ab", "ab seen", "1"),
         ("start", "b", "start", "0"),
         ("a seen", "a", "a seen", "0"), ("a seen", "ab", "ab seen", "1"),
         ("a seen", "b", "start", "0"),
         ("ab seen", "a", "a seen", "0"), ("ab seen", "ab", "ab seen", "1"),
         ("ab seen", "b", "start", "0")]),
}
PRESET_LABELS = {
    "detect_11_mealy": "Detect 11 - Mealy (M1 section 5.2)",
    "detect_11_moore": "Detect 11 - Moore (M1 section 5.3)",
    "parity_mealy": "Parity - Mealy",
    "ends_ab_mealy": "Ends with ab - multi-character",
}
DEFAULT_PRESET = "detect_11_mealy"


# ----------------------------------------------------------------------
# JSON and CSV
# ----------------------------------------------------------------------
def machine_json(m: Machine) -> str:
    return json.dumps({
        "format": "mealymoore/1", "name": m.name, "kind": m.kind,
        "states": list(m.states), "input_alphabet": list(m.sigma),
        "output_alphabet": list(m.gamma), "start_state": m.start,
        "transitions": [
            {"state": s, "symbol": a, "next": t,
             **({"output": m.lam[(s, a)]} if m.kind == MEALY else {})}
            for (s, a), t in m.delta.items()],
        **({"outputs": dict(m.lam)} if m.kind == MOORE else {}),
    }, indent=2)


def machine_from_json(text: str) -> Machine:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DefinitionError(f"Not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise DefinitionError("The JSON root must be an object.")

    def items(key):
        value = data.get(key, [])
        if isinstance(value, str):
            value = [part.strip() for part in value.split(",") if part.strip()]
        return tuple(value)

    rows = data.get("transitions", [])
    if isinstance(rows, dict):
        delta = {}
        for key, target in rows.items():
            state, _, symbol = key.partition(",")
            delta[(state.strip(), symbol.strip())] = target
        lam = dict(data.get("outputs", {}))
    else:
        delta = {(r["state"], r["symbol"]): r["next"] for r in rows}
        lam = (dict(data["outputs"]) if isinstance(data.get("outputs"), dict)
               else {(r["state"], r["symbol"]): r["output"] for r in rows if "output" in r})
    return Machine(str(data.get("name", "Imported machine")),
                   str(data.get("kind", MEALY)).lower(), items("states"),
                   items("input_alphabet"), items("output_alphabet"),
                   str(data.get("start_state", "")), delta, lam)


def machine_table(m: Machine) -> list[list[str]]:
    return [[s, a, m.delta.get((s, a), ""),
             m.lam.get((s, a), "") if m.kind == MEALY else ""]
            for s in m.states for a in m.sigma]


def trace_to_csv(run: RunResult) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["step", "from", "input", "to", "output", "origin"])
    for s in run.steps:
        writer.writerow([s.index, s.state, s.symbol, s.next_state, s.output, s.origin])
    return buffer.getvalue()


# ----------------------------------------------------------------------
# Self-test: python main.py
# ----------------------------------------------------------------------
def _eq(actual, expected):
    assert actual == expected, f"expected {expected!r}, got {actual!r}"


def _raises(error, fn):
    try:
        fn()
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def _error(machine, fragment):
    assert fragment in " ".join(machine.validate()), f"missing {fragment!r}"


def selftest() -> tuple[int, list[str]]:
    mealy, moore = PRESETS["detect_11_mealy"], PRESETS["detect_11_moore"]
    broken = Machine(mealy.name, MEALY, mealy.states, mealy.sigma, mealy.gamma, "q0",
                     {k: v for k, v in mealy.delta.items() if k != ("q1", "0")}, mealy.lam)
    checks = [
        ("mealy 110111", lambda: _eq(simulate(mealy, "110111").output_text, "010011")),
        ("moore 110111", lambda: _eq(simulate(moore, "110111").output_text, "0010011")),
        ("mealy empty", lambda: _eq(simulate(mealy, "").output_text, "")),
        ("moore empty", lambda: _eq(simulate(moore, "").output_text, "0")),
        ("mealy 1111", lambda: _eq(simulate(mealy, "1111").output_text, "0111")),
        ("mealy 0000", lambda: _eq(simulate(mealy, "0000").output_text, "0000")),
        ("mealy final state", lambda: _eq(simulate(mealy, "110111").final_state, "q1")),
        ("moore final state", lambda: _eq(simulate(moore, "110111").final_state, "q2")),
        ("mealy length", lambda: _eq(len(simulate(mealy, "110111").output_symbols), 6)),
        ("moore length", lambda: _eq(len(simulate(moore, "110111").output_symbols), 7)),
        ("length check flag", lambda: _eq(simulate(moore, "110").length_ok, True)),
        ("unknown symbol", lambda: _raises(InputError, lambda: simulate(mealy, "1102"))),
        ("comma input", lambda: _eq(simulate(mealy, "1,1,0").input_symbols, ("1", "1", "0"))),
        ("space input", lambda: _eq(simulate(mealy, "1 1 0").input_symbols, ("1", "1", "0"))),
        ("multi-character input", lambda: _eq(tokenize("aab", ("a", "ab", "b")), ["a", "ab"])),
        ("empty token", lambda: _raises(InputError, lambda: simulate(mealy, "1,,0"))),
        ("epsilon input", lambda: _eq(tokenize("epsilon", ("0", "1")), [])),
        ("missing transition", lambda: _raises(SimulationError, lambda: simulate(broken, "110111"))),
        ("trap completion", lambda: _eq(simulate(broken.with_trap(), "110111").final_state, TRAP)),
        ("duplicate states", lambda: _error(
            Machine("x", MEALY, ("q0", "q0"), ("0",), ("0",), "q0"), "Duplicate")),
        ("bad start state", lambda: _error(
            Machine("x", MEALY, ("q0",), ("0",), ("0",), "q9"), "not in Q")),
        ("epsilon reserved", lambda: _error(
            Machine("x", MEALY, ("q0",), ("epsilon",), ("0",), "q0"), "reserved")),
        ("output outside Delta", lambda: _error(
            Machine("x", MEALY, ("q0",), ("0",), ("0",), "q0",
                    {("q0", "0"): "q0"}, {("q0", "0"): "1"}), "outside Delta")),
        ("moore to mealy", lambda: all(
            compare_moore_mealy(moore, moore_to_mealy(moore), w)["matches"]
            for w in ("", "1", "11", "101", "110111", "0", "10"))),
        ("mealy to moore", lambda: all(
            compare_moore_mealy(mealy_to_moore(mealy, "0"), mealy, w)["matches"]
            for w in ("", "1", "11", "1111", "110111"))),
        ("mealy to moore states", lambda: _eq(len(mealy_to_moore(mealy, "0").states), 3)),
        ("json round trip", lambda: _eq(
            simulate(machine_from_json(machine_json(mealy)), "110111").output_text, "010011")),
        ("csv rows", lambda: _eq(len(trace_to_csv(simulate(mealy, "111")).strip().splitlines()), 4)),
        ("diagram highlight", lambda: _eq(
            "#C2410C" in build_dot(mealy, active_view(simulate(mealy, "11"), 2)), True)),
        ("multi-character run", lambda: _eq(
            simulate(PRESETS["ends_ab_mealy"], "aab").output_text, "01")),
        ("parity 1011", lambda: _eq(simulate(PRESETS["parity_mealy"], "1011").output_text, "1101")),
    ]
    failures = []
    for name, fn in checks:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    return len(checks) - len(failures), failures


if __name__ == "__main__":
    passed, failed = selftest()
    for line in failed:
        print("FAIL", line)
    print(f"{passed} checks passed, {len(failed)} failed")
    raise SystemExit(1 if failed else 0)

