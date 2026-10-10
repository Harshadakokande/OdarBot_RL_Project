# Briefing for a discussion with Professor Fujii

## The problem identified

Odor-source search can become difficult when odor observations are intermittent and the wind direction changes. A policy that responds only to its newest reading may lose useful temporal context after such a disturbance. I built a small, transparent prototype to formulate this issue as a controlled RL experiment rather than to claim that it models a real plume or animal.

## Why begin with wind-change recovery?

A single scheduled wind-direction change creates a simple, inspectable disturbance. It lets the experiment distinguish ordinary source navigation from the period after the odor field has changed. The setup is intentionally narrow: one can inspect the event timing, sensor readings, trajectories, and policy outcomes without first needing a large neural model or a highly complex fluid simulator.

## Working hypothesis

Recent permitted odor and wind observations may help an RL policy recover after a wind change because they provide temporal context unavailable to a current-observation policy. This is only a working hypothesis. A history can also harm a tabular learner by making its state space too large.

## Proposed comparisons

| Comparison | Input to policy | Role |
| --- | --- | --- |
| A. Current-observation Q-learning | Latest enabled odor, wind, and/or previous-action reading | Baseline without explicit temporal history. |
| B. Sensory-history Q-learning | Fixed-length history of enabled readings | Simple test of whether temporal context changes outcomes. |
| C. Generic context-representation candidate | A modular transformation of recent readings | Optional future representation comparison. |

The third item is deliberately generic. It is **not ACIL**, and I do not infer or claim an ACIL architecture, definition, or result without its confirmed reference and intended role.

## Measures I plan to use

- Source-finding success rate.
- Steps to source: mean and median, with failed episodes handled explicitly.
- Path length.
- First post-change odor reacquisition.
- A configurable post-change recovery proxy, with both observed-recovery time and the fraction without observed recovery.
- Before/after wind-change behavior where the episode timing makes this meaningful.
- Variation across independent training seeds and matched evaluation scenarios where feasible.

The recovery measure is intentionally labelled a proxy. A sensor-based rule does not automatically demonstrate sustained odor tracking. The complete per-episode event record is retained so the definition can be revised and reanalysed.

## What is implemented in this prototype

- A configurable two-dimensional grid environment with a hidden source, discrete movement, maximum episode length, and a source-reaching terminal condition.
- A discrete wind direction with one controlled wind-change event.
- A simplified stochastic odor-detection field with configurable noise, intermittency, and sensor availability.
- Protection against exposing the true source location or agent absolute position to the policy.
- Current-observation and sensory-history tabular Q-learning baselines, plus a separate optional generic context-representation interface.
- Separate training and frozen-policy evaluation, run-specific configurations, episode-level CSVs, summaries, and figures.
- A no-wind-change control option, multi-seed experiment settings, and tests for core mechanics and evaluation safety.

## What is not implemented or validated

- A validated turbulent-fluid, silkmoth, or real-robot odor simulator.
- Evidence that the simplified field matches a laboratory simulator or dataset.
- A validated scientific definition of recovery or tracking.
- An implementation of ACIL or a claim about what ACIL means.
- A demonstrated performance advantage before carefully designed multi-seed experiments are executed and inspected.
- Evidence of novelty, biological relevance, or real-world generalization.

I do not claim that Professor Fujii has approved this design or that preliminary results prove the hypothesis.

## Guidance I would value

1. Is there an existing simulator, dataset, benchmark, or lab protocol that should replace or calibrate this simplified environment?
2. What is the exact ACIL reference: full name, paper, intended input/output, and its appropriate role in the comparison?
3. Which sensory signals are realistic and legitimate inputs for the intended task: odor, wind, previous motion/action, position estimates, or others?
4. What would count as a scientifically defensible recovery or sustained tracking measure after a wind perturbation?
5. Which wind changes, source locations, noise conditions, and episode scales best resemble the intended problem while remaining experimentally controlled?
6. Which baseline methods, seed counts, and statistical comparisons would make a preliminary result useful for the next research step?
7. Are there reward-design or evaluation pitfalls that could create an uninteresting shortcut in this toy task?

## Honest next step

The immediate goal is to run the documented baseline protocol, retain every configuration and episode-level outcome, and use the results to refine the research question with domain guidance. A favorable result would justify a more realistic and better-specified follow-up; it would not by itself validate the hypothesis outside this prototype.

