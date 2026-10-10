# OdorBot: sensory-history reinforcement learning for odor-source search

OdorBot is a small, reproducible research prototype for one focused question in robotic odor-source search:

> Does access to recent odor and wind observations help a reinforcement-learning (RL) agent recover after a controlled change in wind direction?

It is designed as a transparent preliminary experiment for research discussion, not as a claim of a new odor-search method. The environment uses a deliberately simplified two-dimensional odor-detection field. It does **not** reproduce Professor Keisuke Fujii's simulator, validate silkmoth biology, model turbulent fluid dynamics, or establish real-robot performance.

## What the prototype compares

All methods use the same environment, action set, reward definition, episode limit, training budget, and evaluation scenarios.

| Method | Information used for decision making | Purpose |
| --- | --- | --- |
| A. Current-observation DQN baseline | The latest permitted sensor reading | Tests the memory-free discrete-action DQN baseline. |
| B. Sensory-history DQN baseline | A fixed-length window of permitted recent readings | Tests whether a short sensor history helps after the wind change. |
| C. Generic context-representation candidate | A separately defined transformation of recent permitted readings | Provides an optional interface for a future representation experiment. It is **not ACIL**. |

Method C is intentionally a generic placeholder. No definition, architecture, or performance claim is attributed to ACIL until the full reference, name, and intended research role have been confirmed.

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer
- Packages listed in requirements.txt

## Install and run on Windows with VS Code

1. Open the project folder in VS Code.
2. Select **Terminal → New Terminal**.
3. Optional but recommended: create and activate a virtual environment.

   ~~~powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ~~~

4. Install project dependencies.

   ~~~powershell
   python -m pip install -r requirements.txt
   ~~~

5. Run a small smoke-test experiment.

   ~~~powershell
   python main.py --quick
   ~~~

6. Run the default configured experiment.

   ~~~powershell
   python main.py
   ~~~

7. Run the automated tests.

   ~~~powershell
   python -m pytest
   ~~~

If python is not recognized, replace it with py. The --quick run is useful for checking that training, evaluation, CSV export, and plots work end to end; it is not suitable for drawing scientific conclusions.

## Research question and scope

The experiment introduces one controlled wind-direction change during an episode. It then compares source-finding and a clearly labelled recovery proxy between a policy with only the current reading and a policy that can use a short sensory history.

The working hypothesis is directional rather than a result claim: a history-based policy *may* distinguish a brief odor loss from a changed downwind odor region better than a current-observation policy. A result from this simulator would support or challenge that hypothesis **within this simplified setting only**.

## Environment design

### World, coordinates, and termination

The world is a finite square grid. A coordinate is written as (x, y): x increases to the right and y increases downward in the plotted grid. The odor source is hidden from the policy. It is used internally only to generate odor readings, determine source-reaching success, and annotate evaluation plots.

At reset, the environment samples or uses the configured source location, agent start location, and initial wind direction. The source and starting location are kept separate. An episode ends when either:

- the agent reaches the source, or
- the configured maximum number of steps is reached.

The exact values used in a run are saved in that run's config.json.

### Actions

The agent chooses one discrete action per step.

| ID | Action | Grid displacement |
| --- | --- | --- |
| 0 | Move up | (0, -1) |
| 1 | Move right | (1, 0) |
| 2 | Move down | (0, 1) |
| 3 | Move left | (-1, 0) |
| 4 | Stay / sample | (0, 0) |

Moves are clipped at the grid boundary, so an attempted move outside the world leaves the agent at the boundary cell.

### Wind and simplified odor sensing

Wind is represented by a discrete direction. A configurable wind change occurs at a configured step; the run configuration records both the timing and direction change. The prototype uses a heuristic downwind-biased, crosswind-decaying detection field rather than a computational-fluid-dynamics or biological plume model. In broad terms, detection is more likely in a downwind region near the source and less likely with crosswind offset and distance.

Odor measurements are stochastic and may be intermittent. Detection strength, intermittency, sensor noise, and false-positive/false-negative behavior are configuration-controlled experiment assumptions, not measurements from a real sensor. The odor-field figure is therefore illustrative of the configured mathematical field, not a validated plume simulation.

### What the policy is allowed to observe

Sensor availability is explicit in the configuration. A run can make the following signals available or unavailable to the policy:

| Signal | Meaning | Notes |
| --- | --- | --- |
| Odor | Current noisy detection or intensity | Generated by the simplified sensor model. |
| Wind | Current discrete wind direction or wind-relative reading | Available only when enabled as a simulated sensor. |
| Previous action | The action selected on the preceding step | Reset to a documented start-of-episode value. |

The source location, agent absolute position, odor-field parameters that are not designated as sensor readings, and evaluation-only labels are never part of the policy observation. This separation is important: a useful score obtained by leaking the source position would not answer the research question.

### Reward

The common reward function gives a positive terminal reward when the source is reached and a small per-step cost otherwise. The numeric values are recorded in config.json. Using a shared reward makes the method comparison fair, but the reward remains a modelling choice and should be checked for unwanted shortcuts in later work.

## How observations become Q-learning states

The prototype uses a compact feed-forward DQN for the two primary baselines. Enabled sensor values are encoded into a fixed-length numeric feature vector.

- **Current-observation DQN baseline:** state = the latest permitted encoded observation vector.
- **Sensory-history DQN baseline:** state = a fixed-length ordered window of recent permitted observation vectors. Early in an episode, observations are padded with zeros to maintain a fixed input size.
- **Context candidate:** recent permitted observations are first transformed by a modular context-representation interface, then used by a policy. It is optional and explicitly not presented as ACIL.

This is a deep Q-network rather than a tabular Q-table. The history window expands feature dimensionality but keeps the policy architecture fixed, which is more appropriate for the present research prototype than a large tabular state table.

## Controlled experiment protocol

Training and evaluation are separate:

- Training uses the configured exploratory policy and updates Q-values.
- Evaluation disables exploration, does not update the learned policy, and resets policy history at every episode boundary.
- Each method receives comparable rewards, source-reaching conditions, episode limits, training budgets, and wind-change scenarios.
- Multiple independent training seeds and multiple evaluation episodes are used when configured. Matched evaluation scenarios/seeds are used where possible so that methods face comparable stochastic conditions.
- A no-wind-change control can be evaluated alongside the wind-change condition to separate general navigation from recovery after a disturbance.

Do not select hyperparameters based only on the final evaluation results of one method. Record selection decisions and all seeds in the run files.

## Recovery metric: an operational proxy

Recovery is not assumed to be a scientifically established tracking measure in this simulator. It is a configurable operational proxy based on odor reacquisition and, when enabled, a configured sustained-detection or tracking criterion after the wind event.

The evaluation data distinguish these events:

| Event | Meaning |
| --- | --- |
| Wind-change time | The step at which the configured wind change is applied. |
| First odor reacquisition | The first qualifying odor reading after that event. |
| Sustained-odor-detection proxy | The first step satisfying the configured recovery rule. This is a proxy, not validated tracking recovery. |
| Episode termination | Source success or maximum-step termination. |
| Source success | Arrival at the hidden source. |

Recovery time is measured from the wind change to the sustained-odor-detection proxy event. If the proxy rule is never met before termination, the episode is recorded as **no observed recovery**; it is not silently discarded. Every summary reports the denominator explicitly: the number of episodes where the wind change occurred, the number of episodes that ended before the change, the number of episodes eligible for the proxy calculation, and the observed vs. not-observed recovery counts. This does not establish genuine odor-tracking recovery without additional validation.

## Metrics

The evaluation summaries may include:

- source-finding success rate;
- mean and median steps to source, with treatment of failures stated;
- mean path length;
- first reacquisition and proxy-recovery times when observed;
- fraction/count of episodes without observed recovery;
- performance before versus after the wind change, where meaningful; and
- variation across independent training seeds, with uncertainty estimates when justified by the number and structure of runs.

Never interpret the best seed alone as the result. Read the episode-level CSV before interpreting an aggregate value, especially when many trials terminate without recovery.

## Outputs and reproducibility

Each invocation creates a separate run directory under runs/<run_id>/; prior runs are not silently overwritten. A typical run contains:

~~~text
runs/<run_id>/
├── config.json
├── run_log.txt
├── training_metrics.csv
├── evaluation_episodes.csv
├── summary.csv
├── odor_field.png
├── sample_trajectories.png
├── wind_change_timeline.png
├── success_rate_comparison.png
├── recovery_comparison.png
├── learning_progress.png
└── pre_post_comparison.png
~~~

| File | What it contains |
| --- | --- |
| config.json | Complete configuration, including seeds and sensor/recovery assumptions. |
| run_log.txt | Key training/evaluation settings plus progress/status messages. |
| training_metrics.csv | Per-episode or per-checkpoint training measurements; not final evaluation results. |
| evaluation_episodes.csv | One row per evaluated episode, including success and recovery-event fields. |
| summary.csv | Method- and condition-level aggregate metrics. |
| odor_field.png | Configured simplified detection field and wind annotation. |
| sample_trajectories.png | Illustrative evaluation paths with source and start markers. |
| wind_change_timeline.png | Controlled wind event and relevant episode event timing. |
| success_rate_comparison.png | Computed source-finding success comparison. |
| recovery_comparison.png | Proxy-recovery results, including unrecovered trials. |
| learning_progress.png | Training-progress measurements. |
| pre_post_comparison.png | Performance before and after the environmental change, where defined. |

Fixed seeds make pseudo-random sampling reproducible to the extent supported by the installed Python and NumPy versions. Preserve config.json, the code version, and generated CSVs when sharing a result.

## Configuration guide

main.py exposes a default experiment and the --quick smoke-test setting. The saved configuration is the authoritative record for a particular run.

| Category | Examples |
| --- | --- |
| Environment | Grid size, source/start locations or sampling policy, maximum episode length, action set. |
| Wind | Initial direction, wind-change step, change magnitude, whether a no-change control is evaluated. |
| Sensor | Odor/wind/previous-action availability, noise, intermittency, detection parameters. |
| Agents | Learning rate, discount factor, exploration schedule, history-window length, optional context candidate. |
| Recovery | Reacquisition/tracking rule and threshold/window. |
| Experiment | Training/evaluation episode counts, independent training seeds, evaluation seeds, run identifier. |

Change one factor at a time for a controlled comparison, then save and report the resulting configuration rather than describing settings from memory.

## Project structure

~~~text
OdorBot_RL_Project/
├── main.py                    # command-line entry point and experiment setup
├── src/
│   ├── environment.py          # grid, wind, odor sensing, rewards, termination
│   ├── agents.py               # Q-learning and policy/context interfaces
│   └── experiment.py           # training, evaluation, export, and plotting
├── tests/                      # unit tests for mechanics and evaluation safety
├── docs/
│   ├── RESEARCH_DESIGN.md      # controlled-study rationale and validity limits
│   └── PROFESSOR_DISCUSSION.md # concise discussion briefing and open questions
├── research_notes.md           # living hypothesis, settings, results, and decisions
├── requirements.txt
└── runs/                       # generated run directories (created when running)
~~~

## Current limitations

- The odor field is a configurable heuristic, not a validated turbulent plume.
- Discrete grid actions and simplified sensors do not represent a physical robot or animal.
- The recovery outcome is an operational proxy and needs domain validation.
- Tabular Q-learning can become data-inefficient as the observation history grows.
- Results can be sensitive to reward design, seed count, grid design, and chosen sensor assumptions.
- The generic context-representation interface is not an implementation of ACIL, and no ACIL-related conclusion is supported.

For the detailed design and validity discussion, read docs/RESEARCH_DESIGN.md. For a concise, honest briefing for a discussion with Professor Fujii, read docs/PROFESSOR_DISCUSSION.md.

