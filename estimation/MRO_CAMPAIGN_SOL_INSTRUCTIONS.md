# MRO orbit-fit campaign: handoff to Sol xhigh

## Status and authority

This is a prepared experiment, **not a completed campaign**. Do not invent results
from the old 12-hour runs or the synthetic unit-test output. Astra was asked to
prepare and test the code, then wait for Dominic to authorize the handoff/runs.
Once Dominic asks you to start, execute this plan, examine results after each
case, and adapt the later comparisons. Aim for **20–50 completed seven-arc cases**,
not 20–50 individual arcs. One case runs all seven arcs. The initial catalogue
contains 42 cases, including explicitly blocked cases; not all 42 are mandatory.

Do not push, compile C++, modify tracking/kernel input data, or modify files
outside the current tudatpy worktree. Keep the interactive example usable. Do not
silently change the observation model, epoch, data selection, frame, or weighting
to improve a score. If a genuine kernel defect or new API support is required,
report evidence and request direction instead of building an untested workaround.

## Files and commands

Worktree: `/home/dominic/Tudat/tudat-monorepo/tudatpy`.
Examples repository: `examples/tudatpy`; script directory: `estimation` within it.
Interpreter: `/home/dominic/miniconda3/envs/tudatpy-dev/bin/python`.

- `mro_tnf_estimation.py`: shared force/observation/estimation implementation;
  direct execution remains the interactive 12-hour example.
- `mro_tnf_estimation_test.py`: seven-arc campaign runner, configurations,
  metadata-based priors, measurements, logs, plots, memory retries.
- `test_mro_tnf_campaign.py`: targeted tests, including two-minute dynamics and
  variational-equation smoke tests. Does not perform a multi-day fit.
- `../mro_orbit_campaign/`: NEW results root, unrelated to old benchmark numbering.
- `../mro_orbit_campaign/validation/`: Astra's setup-check logs and diagnostics,
  **not fit results**.

From the examples directory, with the interpreter above:

```bash
python estimation/mro_tnf_estimation_test.py --list
python estimation/mro_tnf_estimation_test.py --check-inputs
python estimation/mro_tnf_estimation_test.py --write-plan
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
  python -m pytest estimation/test_mro_tnf_campaign.py -q
```

No arguments, `--list`, `--check-inputs` and `--write-plan` NEVER run a fit.
`--validate-setup 002 --arc 0` loads real data and constructs a full estimator and
partials but **does not propagate or fit**. It writes to `validation/`, not the
case register. Always cap native threads for these setup checks too.

Only after authorization:

```bash
python -u estimation/mro_tnf_estimation_test.py --run 001
python -u estimation/mro_tnf_estimation_test.py --run 002 --reference mro_orbit_campaign/case_001
python -u estimation/mro_tnf_estimation_test.py --run 003 --reference mro_orbit_campaign/case_002
```

For an adaptive case, copy the chosen leader's complete `settings.json` into a
new file under `mro_orbit_campaign/planned/`, change only the intended fields,
give it a clear description naming its parent case, and use a NEW number:

```bash
python -u estimation/mro_tnf_estimation_test.py --run 043 \
  --config mro_orbit_campaign/planned/case_043.json \
  --reference mro_orbit_campaign/case_002
```

Use `apply_patch` for source/settings edits. Never overwrite an existing case
directory. The runner refuses to do so. Do not execute the source snapshots
directly: they are audit copies, not a separately installed application.

## Objective and recommendation rule

The main objective is **agreement with the independent JPL/SPICE reference orbit**.
Doppler residuals must be acceptably small, but a slightly higher residual is
welcome if the orbit improves. Prefer fewer parameters when the orbit penalty is
minor. Do not rank by the smallest residual alone.

For each completed case report:

1. Pooled RMS Doppler residual, in mHz (sqrt(sum of squares / number of observations)).
2. R, T, N RMS position differences in metres, using the SPICE orbit's RTN basis.
3. Pooled 3D RMS and maximum position error, mean and worst per-arc 3D RMS.
4. Per-arc values and time histories, so one bad arc cannot hide in an average.
5. Total number of estimated scalar parameters, plus count per arc.
6. Actual end-to-end wall time, concurrency, retry count, best iteration and convergence.
7. Parameter plausibility, prior pulls, correlations, and MCD diagnostics below.

A practical selection rule, to be applied transparently rather than as a claim of
statistical significance:

- First discard failed, non-finite, misaligned or numerically unvalidated cases.
- Aim for pooled residual RMS around a few mHz; provisionally use 5 mHz as a
  review threshold, not a reason to discard observations. Inspect every arc/link.
- Find the best orbit on the same seven arcs and same data mask. Compare both
  the equal-arc mean and pooled 3D RMS; report any disagreement in ranking.
- Among cases within roughly 5% or 0.05 m (whichever is larger) of the best 3D
  RMS, prefer substantially fewer parameters and lower runtime, provided the
  worst arc and individual R/T/N components do not degrade substantially.
- The tie band is a decision aid, not a measured noise floor. Change it only
  with an explicit explanation. Show the best numerical case AND the practical
  recommendation if these differ.
- Require a clean repeat and numerical step-size check before making a final
  recommendation. Do not claim absolute orbit truth: this measures agreement
  with the reconstructed SPICE solution, which has its own errors.

## Fixed scientific setup and epochs

The exact seven UTC intervals are in `ARCS`, transcribed from the original
[GitHub notebook](https://github.com/tudat-team/tudatpy-examples/blob/master/estimation/mro_tnf_estimation.ipynb)
on 2026-09-26. They span January 1, 2012 03:18:01.965 through January 22,
2012 01:57:31.076, with small gaps between independent arcs. Each arc is about
71–72 hours; the old 12-hour benchmark is not a comparable timing baseline.

- State estimation epoch: midpoint of each requested interval AFTER conversion
  of both endpoints to TDB. Do not use first observation, a hard-coded offset,
  or a midpoint in a mixture of UTC and TDB.
- Propagate non-sequentially forward and backward from this epoch, from one hour
  before requested start through one hour after requested end.
- Environment covers an additional hour at each end. MRO's 10-second tabulated
  SPICE ephemeris has another six nodes of padding beyond the environment.
- The runner verifies that the midpoint nominal position agrees with direct
  SPICE to 2 mm, and that the selected iteration's propagated initial state
  equals its estimated state parameters.
- Score the entire nominal interval, including observation gaps, but **exclude
  propagation padding**. Evaluate on the same 60-second grid per arc in every
  case, including step-size comparisons. No favouring observation epochs only.
- Reference states: direct `spiceypy.spkezr('-74', t, 'J2000', 'NONE', '499')`,
  multiplied by 1000 (SPICE km/km/s -> m/m/s). Never compare against the MRO
  body ephemeris after propagation: that has been replaced by the fitted orbit.
- Define R from reference position, N from reference r cross v, T=N cross R.
  Check RTN norm equals inertial difference norm. Positive differences mean
  fitted minus SPICE position; residuals mean observed minus computed Doppler.
- Best residual column, parameter-history column, and trajectory must all use
  `best_iteration`. The last printed correction can be unpropagated and is NOT
  necessarily the state whose residual was printed. Save and verify histories.

Use degree/order-120 JGMRO120D, existing Konopliv Mars rotation implementation,
high-accuracy Earth GCRS/ITRS rotation, local reconstructed attitude/trajectory
kernels, transponder delay 1.4149 microseconds, antenna COM offset, troposphere,
ionosphere and relativistic corrections. Leave these alone in this campaign.
Keep self-shadowing **zero for both aerodynamic and radiation models**, including
panelled Mars radiation. No multiprocessing of cases: only arcs within one case.

The loader includes the preceding UTC day's TNF file and filters actual epochs.
The old notebook unconditionally dropped its first file and had an arc-number
special case. Neither is reproduced: this restores the requested time periods
without discarding valid data just because of a filename. Local file hashes and
selected file names are stored. Do not loosen the +/-8 mHz SPICE residual cutoff.

## Priors, weights, and parameter interpretation

Use `ParameterPlan`; do not recreate a concatenated list of guessed parameter
sigmas. It queries native block indices and sizes, checks complete nonoverlapping
coverage, and writes `parameters.csv` with every index, label, unit, nominal value,
sigma, estimate, formal error, and prior pull.

- State prior defaults: 1000 m position, 0.1 m/s velocity, at the midpoint.
- Scale-factor priors: nominal 1, sigma 2 by default; fixed scale means nominal
  force remains present, but no corresponding estimated parameter.
- Empirical priors: nominal zero, sigma 1e-6 m/s² by default; constant and periodic
  terms can have different sigmas via `constant_empirical_sigma_m_s2` and
  `empirical_sigma_m_s2`.
- Inverse prior covariance is diag(1/sigma²), not covariance. All values are SI.
- C++ empirical layout is **subarc -> functional shape -> R/T/N component**.
  Do not label or constrain the vector as R constant/sin/cos, then T, then N.
- Empirical subarc starts come from the nominal midpoint orbit's period. Drag/
  lift arc-wise scales use these EXACT SAME starts. Two-orbit variants coarsen
  both together. The last block remains active through the propagation end.
- With arc-wise drag or lift scaling, **do not estimate along-track empirical
  terms**. This is enforced in configuration validation.
- Dominic corrected the request: the extra empirical terms replacing aerodynamic
  scale estimation are RADIAL constant/sine/cosine, not additional along-track.
  Cases 016/017 have fixed aerodynamic scales and R/N empiricals; case 018 is a
  separate R/T/N comparison with fixed aerodynamic scales.
- Disabling lift removes both C_L and its estimated scale in the projected-area
  model. Storch computes its own transverse force; setting C_L=0 in another
  model does not disable Storch lift. The runner rejects this misleading mixture.
- Case 001 explicitly reproduces unit observation weights (equivalent to sigma
  1 Hz). Case 002 explicitly uses 3 mHz observation sigma. Changing this changes
  the relative strength of all priors; treat it as its own experiment. Compare
  physical residuals, not the unitless weighted objective.
- Keep the observation sigma fixed within the main model comparisons. Do not
  tune observation noise and empirical priors together without a paired control.

Use changes in estimated parameters, not only residuals, to choose prior tests.
If coefficients make huge, correlated compensating changes, test tighter priors
and/or remove redundant terms. If useful parameters are consistently prior-bound
and orbit error improves when loosened, test a factor 3–10 relaxation. Check
actual corrections/sigma and correlations; a large formal uncertainty alone is
not proof that a parameter should be freed. State-prior changes move the balance
relative to the midpoint SPICE seed and must be reported as such.

## Staged campaign, 20–50 cases

The catalogue is a starting point. Maintain `FINDINGS.md` after each completed
comparison: parent case, intentional differences, hypothesis, result, next choice.
Reserve later slots for combinations that the data justify. Do not run a giant
Cartesian product or spend ten cases on an obviously unsuccessful branch.

1. **Controls (001–004).** Run the unit-weight and physical-weight controls, then
   no lift and fixed lift scaling. Inspect pre-propagation PDFs BEFORE extending
   the series. The full-length arc-0 setup check retained 2298 points with
   2.475 mHz SPICE residual RMS; this is not a postfit result. Reject missing
   data/media/time-scale explanations before changing dynamics.
2. **Geometry/aerodynamics (005–007).** Compare reduced arrays and Storch to their
   matched controls. Reduced arrays now have four triangles per side, preserving
   the hexagonal face footprint EXACTLY (the prior three-triangle mesh lost
   0.2 m² per face). Total macromodel panels fall from 116 to 88. Only the thin
   30-mm side edges are omitted, 0.421878 m² per array; this approximation can
   matter at grazing incidence and must still be assessed in the fits. Retain
   actual array orientation, front/back separation,
   front/back reflectivity, bus, antenna and their rotations. Use the geometry
   validation test; do not create a crude new body shape. Storch uses the existing
   accommodation coefficients of one. Sentman is not a substitute: the local
   implementation's fixed terrestrial gas constant was inappropriate for Mars.
3. **Parameter economy (008–021).** No constant terms, constant-only, no empiricals,
   arc-wise drag/lift without T terms, optional R terms, fixed scales, and longer
   empirical subarcs. Prefer comparisons which isolate one change. Retain a
   simpler non-dominated case even if it has a slightly higher residual.
4. **MCD comparisons (035–042, before the slow radiation cases).** See detailed
   diagnostic rules below. Rebase onto the current physically reasonable
   global-scale and/or arc-wise-drag leader. Do not vary atmospheric settings
   and priors/geometry simultaneously in a claimed atmospheric comparison.
5. **Adaptive priors and combinations (022–025 and new IDs).** Rebase these
   planned settings onto the best few models rather than mechanically testing
   every prior on a losing setup. Keep 2–4 cases for useful combinations of
   independently successful changes. Record exactly which parent each changes.
6. **Validation (026–030 and new IDs).** Repeat the leader, reintegrate variational
   equations, increase iterations, and use 15/30/60-second integration steps on
   the SAME chosen model. Cases 029/030 contain placeholder no-lift configurations:
   replace them with leader-derived JSON configs before executing if another
   model won. Keep the same 60-second scoring grid and observation mask.
7. **Arc-wise Sun scaling (031/032).** Explicitly BLOCKED by current native API;
   do not skip this fact in the report. See below. Do not count blocked entries
   as completed experiments.
8. **Panelled Mars radiation LAST (033/034).** Run on a selected good force model,
   then optionally Storch if it remains competitive. It may take much longer
   than five minutes. Keep shadows off and change the radiation acceleration's
   target type consistently with the target settings. Do not merely create a
   panelled target while continuing to use a cannonball acceleration.

A reasonable allocation is 4 controls + 3 geometry cases + 10–14 parameter
cases + 4–8 MCD/prior cases + 4–6 validation cases + 1–2 Mars-radiation cases.
Adjust within 20–50 actual completed cases. The original roughly five-minute
runtime is a target, not a promise: seven ~3-day arcs with hundreds of parameters
must be timed, and concurrency/thermal throttling matter.

## MCD variations and atmospheric figure of merit

Installed data currently support scenario 1 (climatology/average EUV) and the
high-resolution topography toggle. Catalogue cases also include 2 (minimum EUV),
3 (maximum EUV), 31 (historical Mars year, verify date applicability before using),
7 (warm) and 8 (cold). The latter scenario directories were **not present** when
preparing this handoff. Files such as `MY31_all_var_eo.nc` in `clim_aveEUV` do NOT
provide the full `MY31/` seasonal/thermosphere dataset.

Check `Case.blocked_reason` and `third_parties/mcd/MCD.F90::opend` for required
directories. Search existing LOCAL resource directories first. If data are not
available, mark these cases blocked and tell Dominic what must be obtained; do
not silently use scenario 1, download large archives without discussing it, or
alter NetCDF data. If supplied later, set `mcd_data_path` explicitly. Directory
presence is only a preliminary check: make a small density query/short smoke
propagation before seven parallel fits to verify complete seasonal files.

Keep perturbation_key=0 for deterministic comparisons. Do not add random density
perturbations as a route to a better fit. High-resolution topography is a separate
MCD setting, not a replacement for testing different solar/dust scenarios.

For EACH matched MCD pair, report both orbit/residual metrics and:

- Duration-weighted mean absolute and RMS along-track empirical coefficients,
  separately for constant, sine and cosine, and per arc. If useful, also evaluate
  the actual T empirical acceleration c0+cs sin(f)+cc cos(f) along the trajectory;
  coefficient RMS alone is not the exact acceleration RMS for nonuniform f.
- Drag (and lift if fitted) signed mean, RMS |K|, maximum |K|, and RMS |K-1|.
  The physical baseline is K=1, so |K-1| measures the needed correction. A tiny
  K is NOT automatically a better atmosphere.
- Corresponding prior pulls, formal errors and scale/empirical correlations.
- Common subarc durations excluding propagation padding. The runner writes
  `parameter_diagnostics.json` per arc and globally with these weighted coefficient
  and scale magnitudes. Absent T terms remain absent, not reported as zero.

Reduced correction magnitude with stable/better orbit and residuals is evidence
that a scenario describes this period better. Do not call a reduction significant
solely because of a different prior, fewer coefficients, or nominal lift near
zero. For arc-wise drag cases, T empiricals are intentionally absent; assess drag
corrections, and compare against an otherwise-matched arc-wise drag control.
For global-drag cases inspect BOTH T coefficients and drag corrections, since
one can absorb the other's modelling error. Look for improvements over most
arcs, not a large gain in only one arc. Quantify the percent change and absolute
units, compare to repeat/numerical variability, and avoid an unwarranted p-value.

## Known API limitation: arc-wise panelled radiation scaling

The installed bindings expose `arcwise_drag_component_scaling` and
`arcwise_lift_component_scaling`, but not an arc-wise version of the panelled
Sun source-direction scale. `arcwise_radiation_pressure_coefficient` selects a
**cannonball** target in `createEstimatableParametersFactory.h`; in the existing
environment that is the MARS radiation target, not the panelled SUN target.
Using that function for the requested Sun comparison would be scientifically wrong.

Do not fake this with arc-wise empirical radial terms, a cannonball Sun, independent
short orbit fits, or a parameter disconnected from the force model. These are
different experiments. Keep 031/032 blocked unless an already implemented native
API is subsequently made available and its acceleration/variational partials are
verified. A new kernel implementation requires Dominic's direction/build authority.
Continue other experiments; disclose this outstanding request in the final report.

## Runtime, live logs and memory

The user's latest instruction makes condition numbers diagnostic-only. Campaign
workers set Tudat's warning threshold to one and request a warning every
iteration. The parent tails and final-scans every worker log, records all values
and flags values above the `5e15` reference ceiling, but does **not** interrupt,
reject, or retry a run because of conditioning. Every configured iteration and
every arc may finish and aggregate. Reports must never silently call a flagged
case well-conditioned. Actual estimator/propagation exceptions, non-finite
physical outputs, and OOM handling remain failures. The older archived
condition-limit attempts remain correctly labelled under the policy active when
they were run.

The proposed replacement case-001 keeps the recommended five-iteration forces,
unit observation weights, TN empirical terms with `1e-6 m/s^2` priors, and
global Sun scaling, but fixes the drag/lift scale factors at their nominal
values (the aerodynamic forces remain active) and uses two-orbit empirical
intervals. The user-set prior floors are state sigmas of `100 m` and `0.1 m/s`;
these are standard deviations and may not be tightened. Their inverse-prior
diagonal entries are `1e-4 m^-2` and `100 s^2/m^2`, respectively. The Sun-scale
sigma is `0.2`. No automatic prior tightening is allowed.

Completed workers save the full propagated residual history, explicitly
labelling iteration 0 as the initial propagation before any differential
correction and iteration 1 as the first post-correction evaluation. The SPICE
prefit residual stays separately labelled. Tudat exposes H and the normalized
normal matrix only for its selected `best_iteration`; diagnostics label that
iteration and save exact weights, normalization terms, the physical inverse
prior, `Hn.T W Hn`, the directly normalized prior, the reconstructed total, and
the native total. High-condition results use singular/eigen weak-mode loadings
and weighted-design column correlations rather than presenting a naive inverse
correlation matrix as trustworthy.

Cases run sequentially. Within a case the default is seven OS processes, one per
arc, each with BLAS/OpenMP/NumExpr limited to one native thread. This implements
the requested arc parallelism without sharing non-thread-safe SPICE/MCD state.
Every worker gets explicit MRO settings; inherited MRO_* environment variables
are cleared by the launcher. Full Python/C++/Fortran stdout/stderr goes directly
to `case_NNN/arcs/arc_XX/fit.log`; Python is unbuffered. Read that file to report
the current block/iteration. Never infer progress from wall time alone.

On memory failure the runner retries ONLY failed arcs, halving workers with
round-up: 7 -> 4 -> 2 -> 1. It keeps logs/results in `failed_attempts/`, records
`memory_retries.json`, and never changes scientific settings. SIGKILL/-9/137
alone is labelled *suspected* OOM; inspect system memory/logs to distinguish an
external kill. At one worker it stops rather than looping forever. Successful
arcs are not repeated. Carry the successful worker cap into later cases using
`--workers 4` (or 2/1); do not rediscover the same OOM every run.

All reported end-to-end wall times include worker startup, input manifests,
retries and figure generation. Also inspect per-arc wall times. Do not quote a
sum of parallel arc times as elapsed runtime, or confuse seconds with minutes.
Record cache/thermal effects when making close runtime comparisons. Retain at
least one repeated leader to measure reproducibility and timing variability.

## Output contract and reporting

```text
mro_orbit_campaign/
  CASES.md, cases.csv             all started cases, including failures
  FINDINGS.md                    your paired comparisons and decisions
  RECOMMENDATION.md               final reproducible recommendation
  planned/case_NNN.json           complete settings (editable before launch)
  validation/                    setup checks, never ranked as fits
  case_NNN/
    settings.json, runtime.json, arcs.json, input_manifest.json
    status.json, summary.json
    results.pdf                  all arcs; residuals, R/T/N, grouped parameters
    residuals.csv, orbit.csv, parameters.csv, parameter_diagnostics.json
    source/                      source/geometry snapshots
    memory_retries.json           if needed
    failed_attempts/              if needed, immutable failed-arc evidence
    arcs/arc_XX/
      fit.log, prefit.pdf, spice_residuals.csv, prefit_summary.json
      results.pdf, summary.json, residuals.csv, orbit.csv, parameters.csv
      parameter_diagnostics.json, iteration_history.npz
      prefit_states.npz, postfit_states.npz
      failure.json               if failed
```

PDF parameter curves show fitted coefficients over validity intervals, not
instantaneous accelerations. State correction curves are constant labels for the
MIDPOINT parameter, not a time-varying state correction. Global figures keep arc
segments separate across gaps. The orbit-error figure uses the actual comparison
grid; do not substitute parameter corrections for orbit errors.

Update FINDINGS.md after each case and give short chat progress updates at major
milestones. Do not flood chat with parameter vectors. Show the leading model's
residual/orbit plots when asked, with case ID, exact settings, units and time
interval. Always state whether metrics are SPICE-prefit, propagated-prefit or
postfit, and whether full-arc or observation-only (main metrics are full-arc).

At the end, RECOMMENDATION.md and the chat summary must include:

- Completed/failed/blocked counts and specific reasons for blocked requests.
- Table of the control, best orbit case, simpler recommended case, and numerical
  validation cases: residual, R/T/N/3D RMS, worst arc, parameter count, runtime.
- The tradeoff underlying the recommendation, not just a case number.
- MCD conclusions with actual empirical/scale magnitudes and |K-1| comparisons.
- Prior choices and their evidence, convergence and step-size sensitivity.
- Exact command and frozen settings JSON to reproduce it, plus links to PDFs.
- Limitations, including whether arc-wise Sun scaling and other MCD datasets
  were unavailable. No claims that an unrun case passed.

## Failure modes and required response

| Symptom | Check and action |
| --- | --- |
| Missing local data/scenario | Report exact missing paths; preserve case as blocked. Do not silently fall back or obtain large datasets without direction. |
| SPICE coverage error | Check actual CK/SPK coverage, time scale, filename boundaries and buffers; expand the correct local file selection, not observation rejection. |
| Empty or very small observation set | Check preceding-day TNF, source epochs, link ends and +/-8 mHz SPICE mask. Stop before propagation; do not loosen the cutoff to hide it. |
| Bad SPICE prefit | Dynamics/empirical changes cannot fix this diagnostic. Check transponder delay, antenna reference point, Earth rotation, media/ramp/time conventions first. |
| Observation-mask mismatch | Stop paired comparison; inspect count/order/epoch/link and input hashes. Do not compare changed data as if only dynamics changed. |
| NaN/propagation termination | First identify the earliest physical/partial error. An out-of-bounds Doppler ephemeris error can be downstream of truncated dynamics. Do not extrapolate to conceal it. |
| Prior-size/unassigned error | Correct the parameter metadata mapping; never pad/truncate the sigma vector. Update targeted tests before rerunning. |
| Rank deficiency/huge condition number | Inspect active parameters, dimensionless normalization, zero lift, sparse subarcs, correlations. Remove redundancy or test informative priors, documenting each change. |
| Low residual but bad orbit | Inspect unsupported components and compensating parameters; simplify, adjust priors and compare full-arc R/T/N. Do not select by residual alone. |
| Large parameter correction but tiny plotted error | Verify direct-SPICE seed, midpoint vs start, units, best-iteration association and independent reference RTN. Never compare fitted ephemeris to itself. |
| High step sensitivity | Retain smaller validated step; inspect variational/fixed-step crossings of empirical boundaries and interpolation. Do not smooth plotted states or enlarge acceptance to hide it. |
| Fewer stored iterations than expected | Read convergence output and use best_iteration, not a guessed final index. Hitting maximum iterations is not proof of convergence. |
| API missing | Inspect installed binding and source. Use only semantically equivalent native APIs; block unsupported science rather than fake it. |
| OOM/native kill | Preserve failed logs and halve concurrency rounding up. Keep native threads=1. At one worker stop and report; no force-model simplification disguised as same case. |
| A very short test environment allocates absurd memory | Earth rotation splines use hourly samples: supply several hours of environment even when propagation is only two minutes. This was caught in the setup tests; no broad C++ fix is part of this task. |
| Stalled log | Check worker CPU/memory and current block; C++/Fortran startup messages may differ from propagation prints. Do not launch duplicate cases while the original is still active. |
| Plot/report failure after fitting | Preserve raw arrays/CSV/logs; fix only postprocessing and regenerate without paying for another fit. Never mark complete before all seven validated arcs and global outputs exist. |

If making a small runner fix, retain failed evidence and use a new case ID when
scientific results could change. Re-run the narrow tests affected. Do not modify
unrelated broadly used Tudat functions. No tolerance changes without a numerical
or physical justification. Do not push anything.
