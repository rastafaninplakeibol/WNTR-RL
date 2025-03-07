from uuid import uuid4
from matplotlib import pyplot as plt
from matplotlib.pylab import f
import numpy as np
import pandas as pd
import mwntr
from mwntr.network.controls import _ControlType
import mwntr.sim.hydraulics
from mwntr.sim.solvers import NewtonSolver
import mwntr.sim.results
import warnings
import logging
from mwntr.network.base import LinkStatus
from mwntr.network.model import WaterNetworkModel
from copy import deepcopy
import plotly.express as px


from mwntr.sim.core import _Diagnostics, _ValveSourceChecker, _solver_helper

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

class MWNTRInteractiveSimulator(mwntr.sim.WNTRSimulator):
    def __init__(self, wn: WaterNetworkModel):
        super().__init__(wn)
        self.initialized_simulation = False
        self._sim_id = f"{uuid4()}"
        self.events_history = []

    def init_simulation(self, solver=None, backup_solver=None, solver_options=None, backup_solver_options=None, convergence_error=False, HW_approx='default', diagnostics=False):
        logger.debug('creating hydraulic model')
        self.mode = self._wn.options.hydraulic.demand_model
        self._model, self._model_updater = mwntr.sim.hydraulics.create_hydraulic_model(wn=self._wn, HW_approx=HW_approx)

        self._hw_approx = HW_approx

        if solver is None:
            solver = NewtonSolver

        self.diagnostics_enabled = diagnostics

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
        
        self._wn.add_pattern('interactive_pattern', [0.1]*24)

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

        self._wn.sim_time += self._hydraulic_timestep
        overstep = float(self._wn.sim_time) % self._hydraulic_timestep
        self._wn.sim_time -= overstep

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
        self.events_history.append((self.get_sim_time(), 'start_leak', (node_name, leak_area, leak_discharge_coefficient)))

        junction = self._wn.get_node(node_name)
        junction._leak_status = True
        junction._leak = True
        junction._leak_area = leak_area
        junction._leak_discharge_coeff = leak_discharge_coefficient
        self.update_wn_model()
        

    def update_wn_model(self):
        self._model, self._model_updater = mwntr.sim.hydraulics.create_hydraulic_model(wn=self._wn, HW_approx=self._hw_approx)

    def stop_leak(self, node_name):
        self.events_history.append((self.get_sim_time(), 'stop_leak', (node_name)))

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
        self.events_history.append((self.get_sim_time(), 'close_pipe', (pipe_name)))

        pipe = self._wn.get_link(pipe_name)
        c1 = mwntr.network.controls.ControlAction(pipe, "status", LinkStatus.Closed)
        condition = mwntr.network.controls.SimTimeCondition(self._wn, "=", self.get_sim_time())
        c = mwntr.network.controls.Control(condition=condition, then_action=c1) 
        self._add_control(c)
        self._register_controls_with_observers()
        
    def open_pipe(self, pipe_name):
        self.events_history.append((self.get_sim_time(), 'open_pipe', (pipe_name)))

        pipe = self._wn.get_link(pipe_name)
        c1 = mwntr.network.controls.ControlAction(pipe, "status", LinkStatus.Open)
        condition = mwntr.network.controls.SimTimeCondition(self._wn, "=", self.get_sim_time())
        c = mwntr.network.controls.Control(condition=condition, then_action=c1)
        self._add_control(c)
        self._register_controls_with_observers()
        
    def plot_network(self, title='Water Network Map', node_labels=False, link_labels=False, show_plot=False):
        mwntr.graphics.plot_interactive_network(self._wn, title=f"{title} - {self._sim_id}", node_labels=node_labels)


    def add_demand(self, node_name, base_demand, category=None):
        self.events_history.append((self.get_sim_time(), 'add_demand', (node_name, base_demand, category)))

        node = self._wn.get_node(node_name)
        node._pattern_reg.add_usage('interactive_pattern', (node.name, 'Junction'))
        node.demand_timeseries_list.append((base_demand, 'interactive_pattern', category))

    def remove_demand(self, node_name):
        self.events_history.append((self.get_sim_time(), 'remove_demand', (node_name)))
        
        node = self._wn.get_node(node_name)
        node._pattern_reg.remove_usage('interactive_pattern', (node.name, 'Junction'))
        base = node.demand_timeseries_list.pop(0)
        node.demand_timeseries_list.clear()
        node.demand_timeseries_list.append(base)

    def toggle_demand(self, node_name, base_demand=0.0, category=None):
        node = self._wn.get_node(node_name)
        if len(node.demand_timeseries_list) == 1:
            self.add_demand(node_name, base_demand, category)
        else:
            self.remove_demand(node_name)

    def sim_id(self):
        return self._sim_id


    def branch(self):
        """
        Creates a branch (copy) of the current simulator instance.

        This method appends a 'create_branch' event to the events history with the current simulation time.
        It then creates a deep copy of the current instance, assigns a new unique simulation ID to the branch,
        and returns the branched instance.

        Returns:
            InteractiveNetworkSimulator: A deep copy of the current simulator instance with a new simulation ID.
        """
        self.events_history.append((self.get_sim_time(), 'create_branch', ()))

        branch = deepcopy(self)
        branch._sim_id = f"{self._sim_id}#branch_{uuid4()}"
        return branch   

    def __deepcopy__(self, memo):
        cls = self.__class__
        # Create a new instance without calling __init__
        result = cls.__new__(cls)
        memo[id(self)] = result
        # Iterate over all attributes in __dict__
        model, model_updater = mwntr.sim.hydraulics.create_hydraulic_model(wn=self._wn, HW_approx=self._hw_approx)

        diagnostics = None
        if self.diagnostics_enabled:
            diagnostics = _Diagnostics(self._wn, model, self.mode, enable=True)
        else:
            diagnostics = _Diagnostics(self._wn, model, self.mode, enable=False)

        for key, value in self.__dict__.items():
            if key == "_model":
                setattr(result, key, model)
            elif key == "_model_updater":
                setattr(result, key, model_updater)
            elif key == "diagnostics":
                setattr(result, key, diagnostics)
            else:
                setattr(result, key, deepcopy(value, memo))
        return result

    

    def plot_results(self, nodeOrLink, key, node_list=None, show_events=True):
        """
        Plots the results of the simulation for a specified key and nodes.
        Parameters:
            nodeOrLink (str): The type of object to plot, either 'node' or 'link'.
            key (str): The key to identify the data to be plotted, can be ['head', 'demand', 'pressure', 'leak_demand'] for nodes, ['flowrate', 'velocity', 'status', 'setting'] for link.
            nodes (list, optional): A list of nodes to filter the data. If None, data for all nodes will be plotted. Defaults to None.
            show_events (bool, optional): If True, events from the events history will be shown on the plot. Defaults to True.
        
        Returns:
            None: This function does not return any value. It displays a plot.
        Notes:
        - The function retrieves the results using the `get_results` method.
        - If no data is available for the specified key and nodes, a message is printed and the function returns.
        - The plot is created using Plotly Express and shows the data over time.
        - If `show_events` is True, vertical lines representing events are added to the plot with different colors and styles based on the event type.
        """


        results = self.get_results()
        data = results.node[key] if nodeOrLink == 'node' else results.link[key]
        data = data[node_list] if node_list is not None else data
        if len(data) == 0:
            print('No results to plot')
            return
        fig = px.line(data, x=data.index, y=data.columns, title=f'{key.capitalize()} at nodes over time - {self._sim_id}')
        if show_events:
            for time, action, args in self.events_history:
                color = '#'+str(hex(hash(action+str(args)) % 16777215)[2:].zfill(6))
                line_style = 'solid' if 'create_branch' in action else 'dash'
                fig.add_vline(x=time, line_dash=line_style, line_color=color, annotation_text='  ', annotation_hovertext=f'{action}({args})')           
        fig.show()