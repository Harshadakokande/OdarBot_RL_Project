# OdorBot research notes

This is a living record for the OdorBot preliminary experiment. It separates implemented prototype mechanics from scientific claims that still require validation. Do not treat an empty result table, a single run, or a visually appealing trajectory as evidence for the hypothesis.

## Research question

**Does using recent permitted odor and wind observations help an RL agent recover odor tracking more quickly after a controlled change in wind direction?**

## Working hypothesis

A policy with a short history of permitted sensor readings may use temporal context to distinguish a short odor gap from an environmental change. It may therefore show better proxy-recovery outcomes than a current-observation policy under matched wind-change evaluations.

This is a hypothesis, not a prediction guaranteed by the code. It is plausible that history worsens a tabular method because the state space becomes sparse.

## Methods being compared

| Method | Policy input | Status and interpretation |
| --- | --- | --- |
| A. Current-observation Q-learning | Latest enabled odor, wind, and/or previous-action reading | Primary memory-free baseline. |
| B. Sensory-history Q-learning | Fixed-length ordered history of enabled readings | Primary temporal-context baseline. History length must be reported. |
| C. Generic context-representation candidate | A modular transformation of recent enabled readings before policy selection | Optional experimental placeholder. It is **not ACIL** and must not be described as an ACIL implementation. |

All methods should use the same reward, action set, source-reaching criterion, episode limit, training budget, and evaluation conditions unless a difference is explicitly recorded below.

## Prototype assumptions under test

- The world is a two-dimensional discrete grid with a hidden source.
- The policy never receives the true source location or absolute agent position.
- Odor readings come from a simplified, stochastic, wind-dependent detection field rather than fluid dynamics.
- Wind has a discrete direction and one controlled direction change can occur within an episode.
- Odor, wind, and previous-action sensor availability are configurable; a result is meaningful only relative to the selected sensor assumptions.
- The state encoder is finite and tabular, so longer histories can reduce sample efficiency.
- Source finding and proxy recovery are distinct outcomes.

## Operational recovery record

The simulator cannot by itself prove that an agent has resumed genuine odor tracking. It records a configurable **proxy** instead.

| Timestamp/outcome | Record for each evaluation episode |
| --- | --- |
| Wind change | Exact step at which the controlled change was applied. |
| First reacquisition | First qualifying odor reading after the change. |
| Proxy recovery | First post-change step satisfying the configured tracking/detection rule. |
| No observed recovery | Proxy rule not met before source success or maximum-step termination. |
| Source success | Agent reaches the hidden source. |

When reporting recovery time, state the rule, threshold/window, denominator, and whether summaries exclude no-recovery trials. Always also report the count and fraction without observed recovery.

## Experiment settings record

Copy a row for every executed run. The saved runs/<run_id>/config.json is the authoritative configuration; this table is a readable index, not a replacement for it.

| Run ID | Code version/date | Condition | Methods | Train seeds | Eval seeds/episodes | Grid/max steps | Wind-change step/magnitude | Sensor availability/noise/intermittency | History window | Recovery rule | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| _not yet recorded_ |  | wind change / no change |  |  |  |  |  |  |  |  |  |

## Results record

Do not fill this table from console impressions. Copy values from summary.csv and retain a link to the complete run directory.

| Run ID | Method | Condition | Success rate | Mean / median steps to source | Mean path length | Recovery observed (n/fraction) | No observed recovery (n/fraction) | Recovery-time summary and denominator | Across-seed variability / interval | Interpretation limits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| _not yet recorded_ |  |  |  |  |  |  |  |  |  |  |

## Analysis plan

1. Train each method for the pre-specified budget across independent training seeds.
2. Freeze policies for evaluation: no exploration and no Q-value updates.
3. Evaluate methods on matched wind-change scenarios/seeds where possible.
4. Include a no-wind-change condition when configured to distinguish general navigation from disturbance recovery.
5. Inspect episode-level outcomes before aggregating; retain failed recovery trials rather than removing them.
6. Aggregate across training seeds, report variation, and use confidence intervals only when the evaluation design supports their interpretation.
7. Describe unexpected behavior as a model/implementation observation, not a biological or real-robot finding.

## Implemented in the prototype

- A configurable grid-world odor-source-search environment with a hidden source, discrete actions, a maximum episode length, and a source-reaching terminal condition.
- Configurable wind change, simplified stochastic odor sensing, and explicit sensor-availability assumptions.
- Current-observation and sensory-history tabular Q-learning baselines.
- A separate generic context-representation interface/candidate that is optional and explicitly not ACIL.
- Separate training and non-learning evaluation, episode-level export, saved configurations/logs, and generated plots under a unique run directory.
- Unit tests covering core environment mechanics, history/evaluation safety, recovery calculation, and reproducibility expectations.

## Not yet implemented, validated, or claimed

- A validated turbulent-plume, silkmoth, or real-robot simulator.
- Evidence that a context representation corresponds to ACIL or any specific external method.
- A scientifically validated definition of odor-tracking recovery.
- Evidence that the result generalizes beyond the configured grid, sensors, reward, and seed distribution.
- A claim that Professor Fujii has reviewed, approved, or endorsed the design.
- A claim of novelty, superiority, or a confirmed research conclusion before appropriately designed multi-seed experiments have been run.

## Unresolved questions for research discussion

1. What is the full name, paper/reference, intended input/output, and expected role of ACIL in this research direction?
2. Is there an existing simulator, dataset, benchmark task, or evaluation protocol from Professor Fujii's group that should replace or calibrate this toy environment?
3. Which odor, wind, proprioceptive, and action-history signals would be realistic for the intended robot or biological comparison?
4. What operational definition would make recovery scientifically defensible beyond the current proxy?
5. What wind perturbations, source layouts, and noise regimes are realistic and worth controlling first?
6. Which baseline methods and statistical comparison design are expected for a credible preliminary study?
7. Does the reward function unintentionally give an agent a shortcut that would not exist in the intended task?

## Decision and change log

| Date | Decision or observation | Evidence / run ID | Consequence for next experiment |
| --- | --- | --- | --- |
| _not yet recorded_ |  |  |  |

## References to keep with this notebook

- Full configuration: runs/<run_id>/config.json
- Episode-level evidence: runs/<run_id>/evaluation_episodes.csv
- Aggregate results: runs/<run_id>/summary.csv
- Detailed study design: docs/RESEARCH_DESIGN.md
- Discussion briefing: docs/PROFESSOR_DISCUSSION.md

