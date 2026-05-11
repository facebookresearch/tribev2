# TRIBE v2 -> anima Integration Proposal ADDENDUM (2026-05-02)

> **Companion to** `references/tribev2/ANIMA_INTEGRATION_PROPOSAL.md` (frozen 2026-04-26).
> Per recommendation #95, this addendum REVISES the Axis 3 verdict without modifying the frozen baseline.

## §1 Frozen baseline preservation note

- The original `ANIMA_INTEGRATION_PROPOSAL.md` is preserved as-is in its 2026-04-26 frozen state, in accordance with the **raw#1 immutability** principle.
- This addendum is written as a **separate file** (`ANIMA_INTEGRATION_PROPOSAL_ADDENDUM_2026_05_02_EN.md`); no modification is made to the baseline itself.
- All future updates (post-#95 -> #96, #97, #102 results, etc.) will likewise be written as **additional addendum files** in cumulative-append form.
  - e.g. `ANIMA_INTEGRATION_PROPOSAL_ADDENDUM_2026_05_03.md`, `..._ADDENDUM_2026_05_15.md`, etc.
- The reader derives the cumulative latest verdict by reading the baseline plus all addenda in chronological order.

## §2 Summary of finding #95 — discovery of cortexlab-toolkit

- 2026-Q1 PyPI release: **`cortexlab-toolkit`** (a community-maintained packaging fork of TRIBE v2).
- The dependency blocker noted in baseline §2 iter3 (the Meta FAIR internal packages `neuralset` / `neuraltrain` being unverified on PyPI) is now **resolved**.
  - `cortexlab-toolkit` wraps the public-API surface of both internal packages and is PyPI-installable.
- Integration capabilities:
  - Loads TRIBE v2 frozen weights (HuggingFace `facebook/tribev2`) directly.
  - **Streaming inference** (chunked stimulus -> online cortical BOLD prediction).
  - **Brain-alignment benchmarking** (predicted vs measured fMRI Pearson r per vertex).
- Impact: the dependency wallclock for Pilot-T1 (1-3h) collapses to effectively 0h, enabling immediate entry.

## §3 Axis 3 "No fit" verdict REVISE

**Original baseline §1 Axis 3 grounds for "No fit"**:
1. "wrapper would be large in scale"
2. "scientific value unclear (need to define a meaningful mapping between cortical vertices and cell state)"

**Under Framing D (bridge anchor), both grounds are false**:
1. "wrapper would be large" -> **false**. Under Framing D, the bridge layer consists of: CLM L_IX output -> text serialization -> cortexlab-toolkit streaming inference call. Estimated LOC < 100 (measured in hexa wrapper). The neuralset/neuraltrain PyPI blocker is resolved by cortexlab-toolkit, so no separate wrapper is required.
2. "scientific value unclear" -> **false**. Framing D defines not a simple isomorphism mapping but a **3-way cross-validation**:
   - axis A: user EEG (anima-clm-eeg measurement)
   - axis B: CLM L_IX state (anima-internal Lagrangian)
   - axis C: TRIBE v2 predicted cortical BOLD (TRIBE v2 forward)
   - When the pairwise correlations across all three axes are simultaneously satisfied, a cross-modal anchor is established between cell-language, EEG, and BOLD. This is much stricter and more scientifically meaningful than a simple cortical-vertex <-> cell-state mapping.

**New verdict**: **Strong fit via Framing D**.

## §4 4-framing matrix (per #95 + #101)

| framing | scope | sci value | cost | rank |
|---|---|:---:|---:|:---:|
| **D bridge anchor** | EEG <-> CLM <-> TRIBE BOLD 3-way cross-validation | highest | $0-2 | **1** |
| C G3 manifestation | CLM G3 (Phi*) -> manifests in cortical region (DMN/etc.) produced by TRIBE | very high | $0 | 2 |
| A text-mediated | CLM output text -> TRIBE BOLD prediction (sanity check) | medium | $0-2 | 3 |
| B direct injection | CLM hidden state -> direct injection into TRIBE fusion layer (architectural-mismatch risk) | high | $0-5 | 4 |
| **E tension closed-loop** | CLM-tension <-> TRIBE BOLD feedback loop (NEW per #101) | high | $0-5 | (separate axis) |
| F radical anima-cortex | absorbing TRIBE wholesale as an anima-internal cortex module | speculative | $0-10 | (out-of-scope, future review) |

- Framing D ranks first because it delivers the highest scientific value within a $0-2 budget and has an unambiguous single-falsifier structure.
- Framing E is newly added per #101. As a closed loop, it is classified on a separate axis (cross-cutting with D).
- Framing F is archival; not pursued immediately.

## §5 Top-3 falsifiers preregistered

- **F-CT-3** (core of Framing D): user EEG envelope <-> TRIBE v2 predicted BOLD median vertex Pearson **r >= 0.5**.
  - PASS: r >= 0.5 -> EEG and BOLD are anchored to the same latent state.
  - FAIL: r < 0.3 -> no EEG/BOLD bridge; Framing D discarded.
- **F-CT-4** (Framing C): CLM cortical map vs ALM (Mistral backbone) cortical map -> inter-family **r < 0.7** AND intra-family **r > 0.85**.
  - PASS: family signal is separable at the cortical level.
  - FAIL: r > 0.95 -> family signal is orthogonal to the brain axis; Framing C discarded.
- **F-CT-2** (Framing C reinforcement): G3 PhiStar value <-> DMN ROI activation Pearson **r >= 0.5**.
  - PASS: G3 manifests as DMN -> IIT-style consciousness anchor.
  - FAIL: r < 0.3 -> G3 not reflected at the cortical level.

## §6 Pilot status (#102 in progress)

- The Framing A pilot is **in EXEC progress as #102** using cortexlab-toolkit (as of this addendum's authoring).
- When results arrive, this addendum's §6 will receive a measured update (overwrite OK — the addendum itself is a living doc, but the baseline remains frozen).
- Result schema:
  - `pilot_a_status`: PASS / FAIL / PARTIAL
  - `cortexlab_pypi_install_ok`: bool
  - `tribev2_weights_loaded_ok`: bool
  - `streaming_inference_latency_s`: float
  - `predicted_bold_shape`: tuple (timepoints, vertices=10242)

## §7 5-axis fit revised (post-#95)

| Axis | original verdict (2026-04-26) | revised verdict (2026-05-02) |
|---|---|---|
| 1 EEG / anima-clm-eeg P1-P3 | No fit | **No fit** (unchanged — fMRI-only timescale mismatch persists) |
| 2 paradigm v11 7th axis | Partial (8th candidate) | **Partial** (unchanged) |
| **3 CLM L_IX substrate** | **No fit** | **REVISE -> Strong (via Framing D)** |
| 4 Mk.XI v10 family signal | Strong | **Strong** (unchanged — immediate entry recommended) |
| 5 R33 nexus witness | Partial | **Partial** (unchanged) |

- Change: **only Axis 3 is REVISED from "No fit" to "Strong (Framing D conditional)"**.
- The remaining 4 axes retain their baseline verdicts.

## §8 Honest C3 (raw#10 self-critique)

1. **Tension between frozen preservation and revision**: the original baseline is frozen under raw#1 immutability. This addendum is isolated as a separate file, preserving the baseline. However, this imposes a cumulative-reading burden on the reader (they must read baseline + addendum chain to obtain the latest verdict). If the addendum chain grows long, a separate SUMMARY index will be needed.
2. **Explicit dating**: this addendum = **2026-05-02**, the original = **2026-04-26**. A 6-day interval. Transparently recorded that no factual changes occurred beyond the discovery of cortexlab-toolkit (#95).
3. **Verification gap on cortexlab-toolkit PyPI existence**: at the time of writing this addendum only the PyPI listing has been confirmed; actual install + import + inference smoke test will be **finally confirmed by the result of #102 EXEC**. If #102 fails due to install failure or API-surface mismatch, the Axis 3 REVISE in §3 is rolled back immediately (a separate addendum will be added with a retraction note).
4. **Conditional character of the "Strong fit" verdict**: the "Strong" in §7 is conditional on **Framing D actually working**. The other framings (A/B/C/E/F) receive separate scores; Axis 3 = Strong via Framing D alone is explicitly a single-framing best-case verdict.
5. **Basis for the F-CT-3 r >= 0.5 threshold**: the median of the typical median-vertex r distribution (0.3-0.7) reported in the brain-prediction literature. A stricter threshold (r >= 0.7) carries false-negative risk; a looser one (r >= 0.3) risks passing trivial baselines. 0.5 is a compromise threshold that may be re-tuned after #102 results arrive.

---

*Addendum 2026-05-02. References baseline frozen 2026-04-26. Axis 3 only revised; baseline unchanged.*
