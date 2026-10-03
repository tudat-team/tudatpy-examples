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
