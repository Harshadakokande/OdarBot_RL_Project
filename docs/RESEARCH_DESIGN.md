# Research design: sensory history for wind-change recovery

## Purpose

OdorBot is a controlled preliminary experiment for the following question:

> Does access to recent permitted odor and wind observations help an RL policy recover more effectively after a controlled change in wind direction during odor-source search?

The aim is to make a research discussion concrete: define a transparent task, compare modest baselines fairly, save all evidence needed to inspect the comparison, and identify the assumptions that require expert guidance. It is not intended to reproduce an external simulator, establish a biological model, or demonstrate a deployable robot system.

## Motivation and hypothesis

Odor observations can be intermittent, and a wind shift can change where odor is likely to be detected. A policy that sees only the most recent sensor value may not distinguish a transient gap from a changed environment. A policy with recent observations may have useful temporal context.

**Working hypothesis:** under matched simplified wind-change episodes, a fixed-history policy may have better source-finding or proxy-recovery outcomes than a current-observation policy.

This is deliberately falsifiable. A negative result is informative about the chosen prototype and may reveal that the history encoding is too sparse, the task does not need memory, or the recovery proxy is inadequate. It is not a general conclusion about all odor search.

## Simplified task and scope boundary

The environment is a finite two-dimensional grid containing a hidden odor source and a mobile agent. The agent uses discrete moves: up, right, down, left, and stay/sample. Coordinates use (x, y), with x increasing rightward and y increasing downward in visualizations. The source is visible to the environment only; it is never supplied as a policy feature.

Wind is a discrete direction. One scheduled wind-direction change is introduced in the primary condition. Odor readings are sampled from a configurable, heuristic detection field that is higher in a downwind-biased area near the source and falls with crosswind offset and distance. Noise and intermittency make the readings stochastic. This is a convenience model for a controlled baseline, not turbulent-flow simulation or a validated sensory model.

The episode stops on source arrival or at its configured maximum step count. The shared reward gives terminal credit for source arrival and a small cost for remaining steps. Its exact numeric values are a run-level configuration and must be retained because reward design can change behavior materially.

## Information boundary

The experiment varies only information intended to be plausible sensor input.

| Available only if enabled | Never supplied to the policy |
| --- | --- |
| Current odor reading | True source coordinate |
| Current wind direction or wind-relative reading | Agent absolute location |
| Previous selected action | Ground-truth odor-field parameters, unless explicitly modelled as sensors |
| Fixed-length history of enabled readings | Evaluation labels such as success/recovery timestamps |

The reset representation for previous action and the history-padding rule are part of the configuration/implementation contract. They must be used the same way during training and evaluation.

## Methods

| Method | Representation | Role in the comparison | Important limitation |
| --- | --- | --- | --- |
| A. Current observation | Latest encoded permitted observation | Memory-free tabular Q-learning baseline | Cannot directly represent temporal pattern. |
| B. Sensory history | Ordered fixed-length sequence of recent encoded permitted observations | Tests simple temporal context | State count grows quickly with window length. |
| C. Generic context candidate | Separate transform from history to policy input | Extensible research interface, optionally evaluated | Placeholder only; it is **not ACIL**. |

The full name, publication, definition, and intended role of ACIL have not been confirmed. The prototype does not infer them. Any future ACIL comparison should be specified from an authoritative reference before being named or interpreted as such.

## Variables and experimental conditions

### Independent variables

| Variable | Levels/examples | Why it matters |
| --- | --- | --- |
| Policy representation | Current observation; fixed history; optional generic context candidate | Primary comparison. |
| Wind condition | One controlled change; no-change control when enabled | Separates disturbance recovery from ordinary navigation. |
| History window | Pre-specified fixed-length windows | Tests temporal context while exposing state-space cost. |
| Sensor assumptions | Enabled odor/wind/previous-action inputs; noise and intermittency settings | Defines which information is available to each policy. |

The first study should not vary all of these at once. Select a small, pre-specified comparison and label any later sweep as exploratory.

### Controlled variables

| Variable | Control rule |
| --- | --- |
| Environment geometry/source/start policy | Same configured distribution for every method in a condition. |
| Actions and reward | Identical across methods. |
| Source-reaching rule and episode cap | Identical across methods. |
| Training budget and Q-learning schedule | Comparable unless a documented experimental question changes one. |
| Evaluation scenarios | Matched seeds/scenarios where feasible. |
| Evaluation behavior | Exploration disabled; learned values frozen; history reset on every episode. |

### Dependent variables

| Outcome | Operational measurement | Caution |
| --- | --- | --- |
| Source-finding success | Fraction of episodes ending at the source | Does not isolate recovery by itself. |
| Steps to source | Mean and median, with failures treated explicitly | Averages can conceal timeouts. |
| Path length | Grid distance travelled over an episode | Stay actions may affect steps differently from distance. |
| First odor reacquisition | Post-change elapsed step to first qualifying reading | Sensor event, not necessarily stable tracking. |
| Proxy recovery time | Post-change elapsed step to configured recovery criterion | Requires validation; report only observed cases with denominator. |
| No observed recovery | Count/fraction failing the proxy before termination | Must never be silently removed. |
| Pre/post-change behavior | Configured episode windows or event-based summaries | Interpret only when both intervals exist. |

## Recovery event definition

The prototype labels recovery as a configurable operational proxy, not as a verified scientific construct. The event sequence is:

1. The environment applies the wind change at the configured step.
2. The evaluator records the first qualifying odor reading after the event.
3. The evaluator records the first time the configured sustained-detection or tracking proxy is satisfied.
4. The episode ends through source arrival or maximum-step termination.

The saved episode row must preserve all relevant timestamps and flags. If step 3 never occurs, recovery is censored as “not observed within the episode”; it is not assigned a fabricated finite time and is not dropped from the denominator. When a mean or median recovery time is reported only among observed-recovery episodes, that restriction must appear beside the value. Report the complementary no-recovery fraction as well.

An earlier implementation based on a small number of odor detections is only a proxy. Whether detections must be consecutive, how long tracking must persist, and whether spatial behavior should be included are research decisions needing validation.

## Training and evaluation protocol

1. Pre-specify the settings to compare: methods, sensor availability, history windows, wind-change schedule, reward, training budget, seeds, and recovery rule.
2. For each independent training seed, train a fresh policy in the designated environment distribution. Save training metrics separately from final evaluation.
3. Freeze the policy for evaluation. Disable exploratory actions and do not update Q-values or other learned parameters.
4. Reset the environment and agent/context history at the start of every evaluation episode. Update history consistently after every evaluated action.
5. Evaluate all methods on matched scenario/evaluation seeds where possible. Record any randomness that cannot be paired.
6. Run the wind-change condition and, where configured, a no-change control.
7. Save each episode outcome, aggregate by method and condition, then aggregate across independent training seeds rather than selecting a best run.

Training seeds and evaluation seeds should be recorded separately. Matching does not make data independent; it makes between-method differences easier to inspect because some environmental variation is shared.

## Analysis and uncertainty

Use episode-level data to calculate source success, movement, and event timing. Summarize first within a training seed and then across independent training seeds. Keep the number of seeds and evaluation episodes visible in every table and figure.

Uncertainty estimates should match the data-generating design. For example, variation across independently trained policies addresses a different question from variation across evaluation episodes of the same policy. Confidence intervals, bootstrap intervals, or paired comparisons may be useful only when the sample size and pairing are stated. Avoid best-seed reporting and do not turn a noisy difference into a claim of superiority.

## Data and visual evidence

Every run creates runs/<run_id>/ and retains:

- config.json and run_log.txt for provenance;
- training_metrics.csv separate from final evaluation data;
- evaluation_episodes.csv for every episode and its recovery flags;
- summary.csv for method/condition aggregates; and
- figures for the field, trajectories, wind timeline, success, recovery, learning progress, and pre/post comparison.

Plots represent computed run data except the explicitly labelled field illustration. Plot titles, legends, axes, units or step labels, and captions must identify the condition, methods, and treatment of missing recovery.

## Threats to validity

| Threat | Why it matters | Mitigation in this prototype |
| --- | --- | --- |
| Simplified plume dynamics | A heuristic field may not resemble real odor transport. | Label it honestly; seek a validated simulator/dataset before generalizing. |
| Privileged-information leakage | Source/position inputs would invalidate the intended comparison. | Keep them outside observations and test the observation schema. |
| Reward shortcuts | Policies may exploit artefacts rather than odor tracking. | Inspect trajectories, vary/check reward assumptions, document them. |
| Sparse history state space | A longer history can look worse because it is under-trained. | Report training budget and state encoding; avoid overinterpreting negatives. |
| Recovery-proxy validity | Sensor detections may not imply sustained tracking. | Save event details and call the metric a proxy. |
| Seed dependence | One run may be unusually easy or lucky. | Use independent training seeds and matched evaluation scenarios where possible. |
| Evaluation leakage | Exploration or updates during evaluation bias results. | Disable both and cover the rule with tests. |
| Multiple tuning attempts | Selecting the best setting after inspection inflates apparent effects. | Record settings/decisions and distinguish confirmatory from exploratory runs. |

## Conclusions this experiment can and cannot support

With sufficient carefully matched runs, the prototype can support a limited statement such as: “Under these explicitly documented grid, sensor, reward, and wind-change assumptions, this representation produced these observed source-finding and proxy-recovery outcomes.”

It cannot establish that the same representation improves real robotic odor search, validates a biological mechanism, reproduces Professor Fujii's work, or implements ACIL. Those questions require the correct reference, a more appropriate simulator or physical data, and a separately justified study.

