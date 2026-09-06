"""Explicit intervention controller, independent of evaluation scores."""
from __future__ import annotations


class GraspAssistanceController:
    """Keep the frozen v1 admission rule; changing evaluators cannot activate it.

    A new contact-based intervention needs a separately versioned rule and new
    physical runs. This class deliberately never reads evaluator.pick/proxy.
    """
    rule_id = 'legacy-continuous-close-assistance-v1'

    def __init__(self, simulation, profile, record):
        if profile not in {'source_assisted','no_grasp_assist'}:
            raise ValueError('unknown formal physics profile')
        self.simulation, self.profile, self.record = simulation, profile, record
        self.active = self.created = False

    def before_command(self, *, opening):
        if self.active and opening:
            report = self.simulation.release_grasp_constraint(reason='formal_continuous_opening_target')
            if report.get('active') is not False:
                raise RuntimeError('grasp constraint release not confirmed')
            self.record('grasp_assistance_release', report)
            self.active = False

    def observe(self, *, lift, distance, speed, fraction, closed_command_seen):
        admit = closed_command_seen and fraction <= .5 and lift >= .04 and distance <= .08 and speed <= .30
        if self.profile == 'source_assisted' and admit and not self.created:
            report = self.simulation.create_verified_grasp_constraint()
            if report.get('active') is not True:
                raise RuntimeError('source grasp assistance did not activate')
            self.active = self.created = True
            self.record('grasp_assistance_attach', {'rule_id':self.rule_id, 'runtime':report,
                        'admission':{'lift_m':lift, 'distance_m':distance, 'speed_mps':speed,
                                     'measured_fraction':fraction, 'closed_command_seen':closed_command_seen}})
