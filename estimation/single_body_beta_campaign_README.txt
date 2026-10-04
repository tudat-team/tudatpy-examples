Run the campaign with the tudatpy-dev interpreter:
  python -u mpc_radar_gaia_estimation_single_body_batch.py --campaign single_body_beta_campaign

Resume with the same command. Completed fits are reused. A fit saved before a
Horizons diagnostics failure is postprocessed without estimating again.
Do not edit the protected companion multi-body script during this campaign.

The manifest freezes scientific settings, source SHA256 and observation limits.
Each targets/<ID>/ directory contains raw terminal output, a summary, fit and
residual/parameter histories, full sparse weights, observation IDs and rejection
mask, Gaia scan data, final ephemerides, propagated 6x6 body covariance blocks,
and Horizons comparison arrays. States are not saved for each iteration.
Time-step, residual and orbit comparison PDFs are generated from these exports.
The PDFs can be regenerated offline:
  python mpc_radar_gaia_estimation_single_body_batch.py --mode plot --destination <result-directory>

beta_ranking.csv ranks returned formal beta uncertainty without residual scaling.
Iteration-limit and final position correction fields identify convergence limits.
The covariance is the estimator's returned covariance; orbit and residual samples
use the last evaluated iteration. All fits keep variational equations fixed after
the initial integration, as requested. Ranking does not discard fits simply for
reaching the existing 15-iteration limit.

selected_top20.json records the chosen targets. joint_top20/ contains the joint
fit using the frozen joint script with only its target list overridden in memory.
Validate exports without refitting:
  python validate_beta_campaign.py single_body_beta_campaign

Additional campaigns can use a different list without changing the frozen source:
  python mpc_radar_gaia_estimation_single_body_batch.py --campaign additional --workers 8 --skip-joint --targets <IDs>
The worker limit is 8; use 3 while a simultaneous large joint fit is running.
Each worker limits BLAS/OpenMP to one thread.

Combine completed campaigns with matching scientific settings and select 25:
  python rank_single_body_campaigns.py single_body_beta_campaign additional --count 25 --output combined_ranking
Run the JSON-selected targets together using the worker mode:
  python mpc_radar_gaia_estimation_single_body_batch.py --mode worker --campaign single_body_beta_campaign --destination single_body_beta_campaign/joint_top25 --targets <selected IDs>

Stations missing from the configured MPC Earth catalog are excluded. Spacecraft
observations with only one receiver position epoch are excluded for that target,
because a linear receiver ephemeris needs at least two points. Other spacecraft
observations remain available. Exclusions are recorded in summary.json and logs.

Optional --maximum-step (seconds) and --state-interpolation-order overrides are
for separate numerical convergence studies. They are recorded in the manifest;
combined rankings reject campaigns with different numerical settings. They do
not change the default scientific setup or the protected multi-body script.

Final joint runs can override the maximum step and state interpolation in their
new manifest's numerical_overrides; for the best-15 run these are maximum_step
43200 and state_interpolation_order 10. Retain the completed original campaign
manifest when studying different numerics in a separate campaign directory.

New fit exports also contain best parameters/residuals and the active flags for
each iteration. These map the best-fit design matrix to its actual observation
rows even when the rejection mask differs from the last iteration's mask.
The actual inverse prior covariance is exported from the same setup function
used by the estimator. No states are stored per iteration.

For separate diagnostic campaigns, --rejection-threshold, --recovery-threshold,
--first-rejection-iteration and --setup select controlled tests. Optional
--planetary-spk and --planetary-gm-kernel load planetary kernel overrides after
the default kernels; the original asteroid perturber masses are preserved.
All overrides are recorded and campaigns with different settings cannot be
combined in one uncertainty ranking. None of these options changes defaults.

Inspect an exported single-body fit with rejection flags:
  python diagnose_single_body_beta.py <result-directory>
This checks linearized source/transit/radar influence and predictive residuals
with the rejection mask held fixed; it does not replace nonlinear refitting.
  python check_rejected_covariance.py <result-directory>
This compares the active precision matrix against the precision obtained by
selecting retained rows of each full covariance matrix. It writes a separate
linearized comparison and preserves the original fit.

For a joint fit with saved best-iteration flags:
  python profile_joint_beta.py <result-directory>
This removes each body's observations and its initial-state/A2 columns in turn
and solves the remaining linearized nuisance parameters. The original fit is
preserved. Use the output to assess which body drives the joint beta result.

For the isolated 66391 radiation-pressure diagnostic, the separate
run_beta_radiation_pressure_diagnostic.py worker imports the frozen baseline
and estimates both its existing A2 and a cannonball solar-pressure Cr. Its
manifest supplies radiation_pressure_diagnostic with reference_radius_m,
reference_density_kg_m3, and initial_coefficient. The reference area is pi*r^2
and mass is 4*pi*density*r^3/3. Cr is free and signed, with no prior; the radius
and density only set its scale, so Cr is conditional on that assumed area/mass.
The summary and raw log also give the effective radial acceleration at 1 AU
and correlations with beta, A2, and solar C20. The original joint script,
solar C20 prior, observation/rejection settings, and fixed variational
equations are retained. The numerical comparison uses a six-hour maximum
step and ten-point state interpolation; manifests retain the exact settings.
