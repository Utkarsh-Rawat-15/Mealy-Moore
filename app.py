"""Streamlit front end for the Mealy-Moore simulator (logic lives in main.py).

    streamlit run app.py
"""

from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

import main
from main import (MEALY, MOORE, PRESETS, PRESET_LABELS, Machine, MachineError,
                  active_view, build_dot, compare_moore_mealy, convert,
                  machine_from_json, machine_json, machine_table, simulate,
                  trace_to_csv)

st.set_page_config(page_title="Mealy-Moore Simulator", page_icon="\u2699", layout="wide")

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
h1 {font-size: 2rem !important; margin-bottom: 0.2rem;}
[data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

if "machine" not in st.session_state:
    st.session_state.machine = PRESETS[main.DEFAULT_PRESET]
    st.session_state.version = 0
    st.session_state.run = None


def adopt(machine: Machine) -> None:
    st.session_state.machine = machine
    st.session_state.version += 1
    st.session_state.run = None


def show_errors(errors: list[str]) -> None:
    st.error(f"The definition has {len(errors)} problem(s):")
    for error in errors:
        st.markdown(f"- {error}")


machine: Machine = st.session_state.machine
version = st.session_state.version

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("\u2699 Mealy-Moore")
    st.caption("Finite-state machines with output \u00b7 CSET302 Milestone II")
    choice = st.selectbox("Bundled examples", list(PRESETS),
                          format_func=lambda key: PRESET_LABELS[key])
    if st.button("Load example", use_container_width=True):
        adopt(PRESETS[choice])
        st.rerun()
    st.divider()
    upload = st.file_uploader("Load machine (.json)", type="json")
    if upload is not None and st.button("Import file", use_container_width=True):
        try:
            candidate = machine_from_json(upload.getvalue().decode("utf-8"))
            errors = candidate.validate(complete=False)
            if errors:
                show_errors(errors)
            else:
                adopt(candidate)
                st.rerun()
        except (MachineError, UnicodeDecodeError) as exc:
            st.error(str(exc))
    st.download_button("Download machine JSON", machine_json(machine),
                       file_name="machine.json", mime="application/json",
                       use_container_width=True)
    st.divider()
    st.caption("Runs locally. No account and no network access after install. "
               "Backend self-test: `python main.py`.")

# ----------------------------------------------------------------- header
st.title("Mealy-Moore Machine Simulator")
st.caption("Define a machine, simulate an input string symbol by symbol, replay the run on the "
           "state diagram, and convert between Mealy and Moore machines.")
st.info(f"**Working machine:** {machine.name} \u00b7 **{machine.kind}** \u00b7 "
        f"|Q| = {len(machine.states)} \u00b7 \u03a3 = {{{', '.join(machine.sigma)}}} \u00b7 "
        f"\u0394 = {{{', '.join(machine.gamma)}}} \u00b7 q0 = {machine.start} \u00b7 "
        f"{'complete' if not machine.validate() else 'incomplete'}")

define_tab, simulate_tab, convert_tab, guide_tab = st.tabs(
    ["1 \u00b7 Define machine", "2 \u00b7 Simulate", "3 \u00b7 Convert", "4 \u00b7 Guide"])

# ----------------------------------------------------------------- define
with define_tab:
    st.subheader("Machine definition")
    st.caption("Edit the fields, then validate. The transition table has one row per "
               "(state, symbol) pair.")

    left, right = st.columns(2)
    with left:
        name = st.text_input("Machine name", machine.name, key=f"name{version}")
        kind = st.selectbox("Machine type", [MEALY, MOORE],
                            index=0 if machine.kind == MEALY else 1, key=f"kind{version}")
        states_text = st.text_input("States Q (comma separated)", ", ".join(machine.states),
                                    key=f"states{version}")
    with right:
        sigma_text = st.text_input("Input alphabet \u03a3 (comma separated)",
                                   ", ".join(machine.sigma), key=f"sigma{version}")
        gamma_text = st.text_input("Output alphabet \u0394 (comma separated)",
                                   ", ".join(machine.gamma), key=f"gamma{version}")
        start = st.selectbox("Start state q0", machine.states, key=f"start{version}")

    def parse(text: str) -> tuple[str, ...]:
        return tuple(part.strip() for part in text.split(",") if part.strip())

    states, sigma, gamma = parse(states_text), parse(sigma_text), parse(gamma_text)
    rows = []
    for state in states or machine.states:
        for symbol in sigma or machine.sigma:
            key = (state, symbol)
            rows.append({
                "From state": state,
                "Input": symbol,
                "Next state": machine.delta.get(key, state if state in states else ""),
                "Output": (machine.lam.get(key, "") if kind == MEALY
                           else machine.lam.get(machine.delta.get(key, ""), "")),
            })
    table = pd.DataFrame(rows).rename(columns={"Output": "Output (\u03bb)"})
    edited = st.data_editor(
        table, hide_index=True, use_container_width=True, num_rows="fixed",
        key=f"table{version}_{kind}_{len(states)}_{len(sigma)}",
        column_config={
            "From state": st.column_config.TextColumn(disabled=True),
            "Input": st.column_config.TextColumn(disabled=True),
            "Next state": st.column_config.SelectboxColumn(options=list(states) or None),
            "Output (\u03bb)": st.column_config.TextColumn(help="Mealy: output per transition"),
        },
    )

    def build() -> Machine:
        delta, lam = {}, {}
        for _, row in edited.iterrows():
            key = (row["From state"], row["Input"])
            target = str(row["Next state"] or "")
            delta[key] = target
            if kind == MEALY:
                lam[key] = str(row["Output (\u03bb)"] or "")
        if kind == MOORE:
            for state in states:
                values = [str(row["Output (\u03bb)"]) for _, row in edited.iterrows()
                          if row["From state"] == state and str(row["Output (\u03bb)"])]
                lam[state] = values[0] if values else ""
        return Machine(name, kind, states, sigma, gamma, start, delta, lam)

    actions = st.columns([1, 1, 3])
    if actions[0].button("Validate definition", use_container_width=True):
        errors = build().validate()
        if errors:
            show_errors(errors)
        else:
            st.success("Valid definition: complete, consistent and ready to run.")
    if actions[1].button("Apply definition", type="primary", use_container_width=True):
        candidate = build()
        errors = candidate.validate()
        if errors:
            show_errors(errors)
        else:
            adopt(candidate)
            st.rerun()

    with st.expander("Current saved definition (JSON)"):
        st.code(machine_json(machine), language="json")

# ----------------------------------------------------------------- simulate
with simulate_tab:
    st.subheader("Run the machine")
    with st.form("run_form"):
        cols = st.columns([3, 1])
        word = cols[0].text_input("Input string w", "110111",
                                  help="Plain, comma separated or space separated. "
                                       "Empty or epsilon for the empty string.")
        trap = cols[1].checkbox("Complete with a trap state", value=False)
        submitted = st.form_submit_button("Run simulation", type="primary",
                                          use_container_width=True)

    if submitted:
        try:
            target = machine.with_trap() if trap else machine
            st.session_state.run = (target, simulate(target, word))
        except (MachineError,) as exc:
            st.session_state.run = None
            st.error(str(exc))

    stored = st.session_state.get("run")
    if not stored:
        st.info("Enter an input string and run the simulation.")
    else:
        used, run = stored
        if used is not machine:
            st.warning("This run used a different definition (trap completion or an earlier "
                       "machine). Re-run after editing the definition.")

        metrics = st.columns(4)
        metrics[0].metric("Input length |w|", len(run.input_symbols))
        metrics[1].metric("Output length", len(run.output_symbols))
        metrics[2].metric("Expected length", run.expected_length)
        metrics[3].metric("Final state", run.final_state)
        if run.length_ok:
            st.success(f"Length check passed: a {run.kind} machine produces "
                       f"{run.expected_length} output symbol(s) for |w| = {len(run.input_symbols)}.")
        else:
            st.error("Length check failed: the output does not match the formal model.")

        st.markdown("**Output string**")
        st.code(main.format_symbols(run.output_symbols), language=None)
        joined = run.output_text or "\u03b5"
        st.caption(f"Concatenated: {joined} \u00b7 start {used.start} \u00b7 final {run.final_state}")

        st.subheader("State diagram and playback")
        step = st.slider("Playback position (0 = start state)", 0, len(run.steps), len(run.steps))
        view = active_view(run, step)
        st.caption(view.caption)
        st.graphviz_chart(build_dot(used, view), use_container_width=True)

        st.subheader("Trace table")
        frame = pd.DataFrame([{
            "Step": s.index, "From": s.state, "Input": s.symbol, "To": s.next_state,
            "Output": s.output, "Output from": s.origin} for s in run.steps])
        st.dataframe(frame, hide_index=True, use_container_width=True)
        if run.kind == MOORE:
            st.caption(f"Step 0 is the start state: \u03bb({used.start}) = {run.initial_output}.")
        st.download_button("Download trace as CSV", trace_to_csv(run),
                           file_name="trace.csv", mime="text/csv")

# ----------------------------------------------------------------- convert
with convert_tab:
    st.subheader("Convert between Moore and Mealy")
    if machine.kind == MOORE:
        st.caption("Moore \u2192 Mealy: each transition takes the output of the state it enters.")
        start_output = None
    else:
        st.caption("Mealy \u2192 Moore: states are pairs (state, last output); only reachable "
                   "pairs are created, so a start output is needed.")
        start_output = st.selectbox("Output of the start state", machine.gamma)

    if st.button("Convert machine", type="primary"):
        try:
            st.session_state.converted = convert(machine, start_output)
        except MachineError as exc:
            st.error(str(exc))

    converted = st.session_state.get("converted")
    if converted is None:
        st.info("Convert the working machine to see the new table, diagram and equivalence check.")
    else:
        st.success(f"Converted to a {converted.kind} machine with {len(converted.states)} "
                   f"state(s): {', '.join(converted.states)}.")
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Converted transition table**")
            st.dataframe(pd.DataFrame(machine_table(converted),
                                      columns=["State", "Input", "Next state", "Output"]),
                         hide_index=True, use_container_width=True)
        with right:
            st.markdown("**Converted state diagram**")
            st.graphviz_chart(build_dot(converted), use_container_width=True)

        st.subheader("Equivalence check")
        test = st.text_input("Test input for the equivalence check", "110111")
        if st.button("Check equivalence"):
            try:
                if converted.kind == MEALY:
                    result = compare_moore_mealy(machine, converted, test)
                else:
                    result = compare_moore_mealy(converted, machine, test)
                if result["matches"]:
                    st.success("PASS: dropping the initial Moore output gives exactly the "
                               "Mealy output.")
                else:
                    st.error("FAIL: the machines disagree on this input.")
                st.dataframe(pd.DataFrame({
                    "Machine": [result["moore"].kind, result["mealy"].kind,
                                f"{result['moore'].kind} without first symbol"],
                    "Input": [test, test, test],
                    "Output": [result["moore"].output_text,
                               result["mealy"].output_text,
                               "".join(result["moore"].output_symbols[1:])],
                }), hide_index=True, use_container_width=True)
            except MachineError as exc:
                st.error(str(exc))

        actions = st.columns([1, 1, 3])
        if actions[0].button("Use converted machine", use_container_width=True):
            adopt(converted)
            st.session_state.pop("converted", None)
            st.rerun()
        actions[1].download_button("Download converted JSON", machine_json(converted),
                                   file_name="converted.json", mime="application/json",
                                   use_container_width=True)

# ----------------------------------------------------------------- guide
with guide_tab:
    st.subheader("Input syntax")
    st.markdown(
        "- `110111` \u2014 one symbol per character.\n"
        "- `1 1 0 1 1 1` \u2014 whitespace separated; use this for multi-character alphabets.\n"
        "- `1, 1, 0` \u2014 comma separated; empty tokens are rejected.\n"
        "- `aab` on \u03a3 = {a, ab, b} \u2014 greedy longest match gives `a`, `ab`.\n"
        "- empty box or `epsilon` \u2014 the empty string: no steps, and a Moore machine still "
        "emits \u03bb(q0)."
    )
    st.subheader("Semantics")
    st.markdown(
        "- A Mealy machine emits one symbol per transition, so |output| = |w|.\n"
        "- A Moore machine emits the start-state output first, so |output| = |w| + 1.\n"
        "- An incomplete machine is never completed silently: the run stops at the missing "
        "pair. The trap option adds one explicit non-accepting state instead.\n"
        "- Conversion uses only reachable (state, output) pairs; equivalence compares output "
        "symbols, not concatenated strings."
    )
    st.subheader("Workflow")
    st.markdown(
        "1. Edit the definition and validate it.\n"
        "2. Run an input and read the output, metrics and trace.\n"
        "3. Move the playback slider to follow the path on the diagram.\n"
        "4. Convert the machine, check equivalence, and adopt the result if it is useful.\n"
        "5. Download the machine as JSON or the trace as CSV."
    )
    st.subheader("Known limits")
    st.markdown(
        "- Converted machines are not minimised.\n"
        "- Diagrams become crowded beyond about twelve states.\n"
        "- The working machine is not saved between sessions; download it first."
    )
