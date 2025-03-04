from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import mwntr
from mwntr.network.controls import _ControlType
import mwntr.sim.hydraulics
from mwntr.sim.solvers import NewtonSolver
import mwntr.sim.results
import warnings
import logging
import scipy.sparse
import scipy.sparse.csr
from mwntr.network.base import LinkStatus
from mwntr.network.model import WaterNetworkModel
from mwntr.network.controls import ControlAction, Control
from mwntr.sim.models import constants, var, param, constraint

from mwntr.sim.core import _Diagnostics, _ValveSourceChecker, _solver_helper

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

class MWNTRInteractiveSimulator(mwntr.sim.WNTRSimulator):
    def __init__(self, wn: WaterNetworkModel):
        super().__init__(wn)
        self.initialized_simulation = False

    def init_simulation(self, solver=None, backup_solver=None, solver_options=None, backup_solver_options=None, convergence_error=False, HW_approx='default', diagnostics=False):
        logger.debug('creating hydraulic model')
        self.mode = self._wn.options.hydraulic.demand_model
        self._model, self._model_updater = mwntr.sim.hydraulics.create_hydraulic_model(wn=self._wn, HW_approx=HW_approx)

        self._hw_approx = HW_approx

        if solver is None:
            solver = NewtonSolver

        if diagnostics:
            self.diagnostics = _Diagnostics(self._wn, self._model, self.mode, enable=True)
        else:
            self.diagnostics = _Diagnostics(self._wn, self._model, self.mode, enable=False)

        self._setup_sim_options(solver=solver, backup_solver=backup_solver, solver_options=solver_options,
                                backup_solver_options=backup_solver_options, convergence_error=convergence_error)

        self._valve_source_checker = _ValveSourceChecker(self._wn)
        self._get_control_managers()
        self._register_controls_with_observers()

        node_res, link_res = mwntr.sim.hydraulics.initialize_results_dict(self._wn)
        
        self.node_res = node_res
        self.link_res = link_res
        
        self.results = mwntr.sim.results.SimulationResults()
        self.results.error_code = None
        self.results.time = []
        self.results.network_name = self._wn.name

        self._initialize_internal_graph()
        self._change_tracker.set_reference_point('graph')
        self._change_tracker.set_reference_point('model')

        
        self.trial = -1
        self.max_trials = self._wn.options.hydraulic.trials
        self.resolve = False
        self._rule_iter = 0  # this is used to determine the rule timestep

        mwntr.sim.hydraulics.update_network_previous_values(self._wn)
        self._wn._prev_sim_time = -1

        self._terminated = False
        self.initialized_simulation = True

        self.last_set_results_time = -1

        logger.debug('starting simulation')

        logger.info('{0:<10}{1:<10}{2:<10}{3:<15}{4:<15}'.format('Sim Time', 'Trial', 'Solver', '# isolated', '# isolated'))
        logger.info('{0:<10}{1:<10}{2:<10}{3:<15}{4:<15}'.format('', '', '# iter', 'junctions', 'links'))

    def hydraulic_timestep(self):
        return self._hydraulic_timestep

    def step_sim(self):
        if not self.initialized_simulation:
            raise RuntimeError('Simulation not initialized. Call init_simulation() before running the simulation.')

        if logger.getEffectiveLevel() <= logging.DEBUG:
                logger.debug('\n\n')

        if self._wn.sim_time == 0:
            first_step = True
        else:
            first_step = False

        if not self.resolve:
            if not first_step:
                """
                The tank levels/heads must be done before checking the controls because the TankLevelControls
                depend on the tank levels. These will be updated again after we determine the next actual timestep.
                """
                mwntr.sim.hydraulics.update_tank_heads(self._wn)
            self.trial = 0
            self._compute_next_timestep_and_run_presolve_controls_and_rules(first_step)

        self._run_feasibility_controls()

        # Prepare for solve
        self._update_internal_graph()
        num_isolated_junctions, num_isolated_links = self._get_isolated_junctions_and_links()
        if not first_step and not self.resolve:
            mwntr.sim.hydraulics.update_tank_heads(self._wn)
        mwntr.sim.hydraulics.update_model_for_controls(self._model, self._wn, self._model_updater, self._change_tracker)
        mwntr.sim.models.param.source_head_param(self._model, self._wn)
        mwntr.sim.models.param.expected_demand_param(self._model, self._wn)

        self.diagnostics.run(last_step='presolve controls, rules, and model updates', next_step='solve')

        solver_status, mesg, iter_count = _solver_helper(self._model, self._solver, self._solver_options)
        if solver_status == 0 and self._backup_solver is not None:
            solver_status, mesg, iter_count = _solver_helper(self._model, self._backup_solver, self._backup_solver_options)
        if solver_status == 0:
            if self._convergence_error:
                logger.error('Simulation did not converge at time ' + self._get_time() + '. ' + mesg) 
                raise RuntimeError('Simulation did not converge at time ' + self._get_time() + '. ' + mesg)
            warnings.warn('Simulation did not converge at time ' + self._get_time() + '. ' + mesg)
            logger.warning('Simulation did not converge at time ' + self._get_time() + '. ' + mesg)
            self.results.error_code = mwntr.sim.results.ResultsStatus.error
            self.diagnostics.run(last_step='solve', next_step='break')
            self._terminated = True
            self.get_results()
            return

        logger.info('{0:<10}{1:<10}{2:<10}{3:<15}{4:<15}'.format(self._get_time(), self.trial, iter_count, num_isolated_junctions, num_isolated_links))

        # Enter results in network and update previous inputs
        logger.debug('storing results in network')
        mwntr.sim.hydraulics.store_results_in_network(self._wn, self._model)

        self.diagnostics.run(last_step='solve and store results in network', next_step='postsolve controls')

        self._run_postsolve_controls()
        self._run_feasibility_controls()
        if self._change_tracker.changes_made(ref_point='graph'):
            self.resolve = True
            self._update_internal_graph()
            mwntr.sim.hydraulics.update_model_for_controls(self._model, self._wn, self._model_updater, self._change_tracker)
            self.diagnostics.run(last_step='postsolve controls and model updates', next_step='solve next trial')
            self.trial += 1
            if self.trial > self.max_trials:
                if self._convergence_error:
                    logger.error('Exceeded maximum number of trials at time ' + self._get_time() + '. ') 
                    raise RuntimeError('Exceeded maximum number of trials at time ' + self._get_time() + '. ' ) 
                self.results.error_code = mwntr.sim.results.ResultsStatus.error
                warnings.warn('Exceeded maximum number of trials at time ' + self._get_time() + '. ') 
                logger.warning('Exceeded maximum number of trials at time ' + self._get_time() + '. ' ) 
                self._terminated = True
                #self._set_results(self._wn, self.results, self.node_res, self.link_res)
                return
            self._terminated = False
            #self._set_results(self._wn, self.results, self.node_res, self.link_res)
            return

        self.diagnostics.run(last_step='postsolve controls and model updates', next_step='advance time')

        logger.debug('no changes made by postsolve controls; moving to next timestep')

        self.resolve = False
        if isinstance(self._report_timestep, (float, int)):
            if self._wn.sim_time % self._report_timestep == 0:
                mwntr.sim.hydraulics.save_results(self._wn, self.node_res, self.link_res)
                if len(self.results.time) > 0 and int(self._wn.sim_time) == self.results.time[-1]:
                    if int(self._wn.sim_time) != self._wn.sim_time:
                        raise RuntimeError('Time steps increments smaller than 1 second are forbidden.'+
                                            ' Keep time steps as an integer number of seconds.')
                    else:
                        raise RuntimeError('Simulation already solved this timestep')
                self.results.time.append(int(self._wn.sim_time))
        elif self._report_timestep.upper() == 'ALL':
            mwntr.sim.hydraulics.save_results(self._wn, self.node_res, self.link_res)
            if len(self.results.time) > 0 and int(self._wn.sim_time) == self.results.time[-1]:
                raise RuntimeError('Simulation already solved this timestep')
            self.results.time.append(int(self._wn.sim_time))
        mwntr.sim.hydraulics.update_network_previous_values(self._wn)
        first_step = False
        self._wn.sim_time += self._hydraulic_timestep
        overstep = float(self._wn.sim_time) % self._hydraulic_timestep
        self._wn.sim_time -= overstep

        if self._wn.sim_time > self._wn.options.time.duration:
            self._terminated = True
        #self._set_results(self._wn, self.results, self.node_res, self.link_res)
        return
        
    def full_run_sim(self):
        self.results = self.run_sim()
        return
    
    def is_terminated(self):
        return self._terminated
    
    def _set_results(self, wn, results: mwntr.sim.results.SimulationResults, node_res, link_res):
        """
        Parameters
        ----------
        wn: mwntr.network.WaterNetworkModel
        results: mwntr.sim.results.SimulationResults
        node_res: OrderedDict
        link_res: OrderedDict
        """
        node_names = wn.junction_name_list + wn.tank_name_list + wn.reservoir_name_list
        link_names = wn.pipe_name_list + wn.head_pump_name_list + wn.power_pump_name_list + wn.valve_name_list

        self.last_set_results_time = wn.sim_time

        results.node = {}
        results.link = {}

        for key, _ in node_res.items():
            results.node[key] = pd.DataFrame(data=np.array([node_res[key][name] for name in node_names]).transpose(), index=results.time,
                                        columns=node_names)

        for key, _ in link_res.items():
            results.link[key] = pd.DataFrame(data=np.array([link_res[key][name] for name in link_names]).transpose(), index=results.time,
                                                columns=link_names)
        self.results = results
    
    def get_results(self):
        if self.last_set_results_time != self._wn.sim_time:
            self._set_results(self._wn, self.results, self.node_res, self.link_res)
        return self.results
    
    def get_sim_time(self):
        return self._wn.sim_time
    
    def start_leak(self, node_name, leak_area=0.1, leak_discharge_coefficient=0.75):
        junction = self._wn.get_node(node_name)
        junction._leak_status = True
        junction._leak = True
        junction._leak_area = leak_area
        junction._leak_discharge_coeff = leak_discharge_coefficient
        self.update_wn_model()
        

    def update_wn_model(self):
        self._model, self._model_updater = mwntr.sim.hydraulics.create_hydraulic_model(wn=self._wn, HW_approx=self._hw_approx)

    def stop_leak(self, node_name):
        junction = self._wn.get_node(node_name)
        junction._leak_status = False
        junction._leak = False
        self.update_wn_model()

    def _add_control(self, control):
        if control.epanet_control_type in {_ControlType.presolve, _ControlType.pre_and_postsolve}:
            self._presolve_controls.register_control(control)
        if control.epanet_control_type in {_ControlType.postsolve, _ControlType.pre_and_postsolve}:
            self._postsolve_controls.register_control(control)
        if control.epanet_control_type == _ControlType.rule:
            self._rules.register_control(control)
        if control.epanet_control_type == _ControlType.feasibility:
            self._feasibility_controls.register_control(control)


    def close_pipe(self, pipe_name) -> None:
        pipe = self._wn.get_link(pipe_name)
        c1 = mwntr.network.controls.ControlAction(pipe, "status", LinkStatus.Closed)
        condition = mwntr.network.controls.SimTimeCondition(self._wn, "=", self.get_sim_time() + self._wn.options.time.hydraulic_timestep)
        c = mwntr.network.controls.Control(condition=condition, then_action=c1) 
        self._add_control(c)
        self._register_controls_with_observers()
        
    def open_pipe(self, pipe_name):
        pipe = self._wn.get_link(pipe_name)
        c1 = mwntr.network.controls.ControlAction(pipe, "status", LinkStatus.Open)
        condition = mwntr.network.controls.SimTimeCondition(self._wn, "=", self.get_sim_time() + self._wn.options.time.hydraulic_timestep)
        c = mwntr.network.controls.Control(condition=condition, then_action=c1)
        self._add_control(c)
        self._register_controls_with_observers()
        
    def plot_network(self, title='Water Network', node_labels=False, link_labels=False, show_plot=False):
        mwntr.graphics.plot_interactive_network(self._wn, title=title, node_labels=node_labels)