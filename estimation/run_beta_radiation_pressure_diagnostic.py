"""Isolated 66391 fit with free Yarkovsky A2 and cannonball solar pressure.

The protected multi-body script is imported unchanged. Radius and density set
only the scale of the unconstrained Cr; the measurable quantity is Cr * area/mass.
No positivity constraint or prior is imposed on Cr. Original solar C20 prior,
observations, rejection settings, and fixed variational equations are retained.
"""
from pathlib import Path
import numpy as np
import mpc_radar_gaia_estimation_single_body_batch as batch

original_baseline = batch.baseline
original_labels = batch.parameter_labels
original_export = batch.export_fit


def configure(manifest, targets):
    if list(targets) != ['66391']:
        raise ValueError('This isolated diagnostic is restricted to 66391')
    od = original_baseline(manifest, targets)
    od.ESTIMATE_YARKOVSKY = True
    model = manifest['radiation_pressure_diagnostic']
    radius = float(model['reference_radius_m'])
    density = float(model['reference_density_kg_m3'])
    area = np.pi * radius**2
    mass = 4 * np.pi * density * radius**3 / 3
    if not (radius > 0 and density > 0):
        raise ValueError('Reference radius and density must be positive')
    od.radiation_pressure_diagnostic = dict(model, reference_area_m2=area,
        reference_mass_kg=mass, area_to_mass_m2_kg=area/mass)

    original_system = od.environment_setup.create_system_of_bodies

    def radiation_bodies(settings):
        target = settings.get('66391')
        target.constant_mass = mass
        target.radiation_pressure_target_settings = (
            od.environment_setup.radiation_pressure.cannonball_radiation_target(
                area, float(model['initial_coefficient']), {}))
        try:
            return original_system(settings)
        finally:
            od.environment_setup.create_system_of_bodies = original_system

    od.environment_setup.create_system_of_bodies = radiation_bodies
    original_accelerations = od.acceleration_settings

    def accelerations(estimate_yarkovsky, estimated_bodies=None):
        settings = original_accelerations(estimate_yarkovsky, estimated_bodies)
        settings['66391']['Sun'] = list(settings['66391']['Sun']) + [
            od.propagation_setup.acceleration.radiation_pressure()]
        return settings

    od.acceleration_settings = accelerations
    original_parameters = od.parameters_setup.create_parameter_set

    def parameter_set(settings, *args, **kwargs):
        # One initial-state block, scalar A2, scalar beta, then vector C20.
        if len(settings) != 4 or not (od.ESTIMATE_BETA and od.ESTIMATE_SUN_J2):
            raise ValueError('Unexpected baseline parameter settings')
        settings = list(settings)
        settings.insert(2, od.parameters_setup.radiation_pressure_coefficient('66391'))
        try:
            result = original_parameters(settings, *args, **kwargs)
            if len(result.parameter_vector) != 10:
                raise ValueError('Expected 6 states, A2, Cr, beta, and C20')
            return result
        finally:
            od.parameters_setup.create_parameter_set = original_parameters

    od.parameters_setup.create_parameter_set = parameter_set
    od.global_parameter_indices = lambda: {'beta': 8, 'C20': 9}
    print('Solar radiation pressure diagnostic: ' + str(od.radiation_pressure_diagnostic), flush=True)
    return od


def labels(od):
    names, units = original_labels(od)
    index = names.index('beta')
    names.insert(index, '66391:Cr')
    units.insert(index, 'dimensionless')
    return names, units


def export_fit(od, output, dataset, estimator, destination, *args):
    original_export(od, output, dataset, estimator, destination, *args)
    covariance = np.asarray(output.covariance)
    values = od.last_iteration_parameters(output)
    cr = float(values[7])
    sigma = float(np.sqrt(covariance[7, 7]))
    model = od.radiation_pressure_diagnostic
    # Default Tudat solar luminosity (celestialBodyConstants.h), no occultations.
    a1_per_cr = 3.828e26 / (4 * np.pi * od.constants.ASTRONOMICAL_UNIT**2 *
        od.constants.SPEED_OF_LIGHT) * model['area_to_mass_m2_kg']
    correlations = {name: float(covariance[7, i] / np.sqrt(covariance[7, 7] * covariance[i, i]))
        for name, i in [('A2', 6), ('beta', 8), ('Sun:C20', 9)]}
    result = dict(model, coefficient=cr, coefficient_sigma=sigma,
        radial_acceleration_at_1au_m_s2=cr*a1_per_cr,
        radial_acceleration_sigma_m_s2=sigma*a1_per_cr, correlations=correlations,
        coefficient_iteration='last evaluated; covariance from best iteration',
        interpretation='Cr scale assumes the reference area and mass; free signed effective radial force')
    summary = batch.read_json(destination / 'summary.json')
    summary['radiation_pressure_diagnostic'] = result
    batch.write_json(destination / 'summary.json', summary)
    print(f'  Estimated radiation pressure Cr: {cr:.12g} +/- {sigma:.8g}', flush=True)
    print(f'  Effective radial acceleration at 1 AU: {cr*a1_per_cr:.12g} +/- {sigma*a1_per_cr:.8g} m/s^2', flush=True)
    for name, rho in correlations.items():
        print(f'  Correlation(Cr, {name}): {rho:.9g}', flush=True)


if __name__ == '__main__':
    batch.baseline = configure
    batch.parameter_labels = labels
    batch.export_fit = export_fit
    batch.RUNNER_SHA256 = batch.digest(Path(__file__))
    batch.main()
