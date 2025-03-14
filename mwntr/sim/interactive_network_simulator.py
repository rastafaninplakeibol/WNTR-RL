import json
import math
import time
from uuid import uuid4
from networkx import diameter
import numpy as np
import pandas as pd
from sympy import Q
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
import plotly.graph_objs as go

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

    def hydraulic_timestep(self):
        return self._hydraulic_timestep

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

        self.rebuild_hydraulic_model = False
        self.demand_modifications = []

        logger.debug('starting simulation')

        logger.info('{0:<10}{1:<10}{2:<10}{3:<15}{4:<15}'.format('Sim Time', 'Trial', 'Solver', '# isolated', '# isolated'))
        logger.info('{0:<10}{1:<10}{2:<10}{3:<15}{4:<15}'.format('', '', '# iter', 'junctions', 'links'))

    def save_expected_demand(self):
        expected_demand = self._model.expected_demand
        for node_name, demand in expected_demand.items():
            v = demand.value
            self.node_res['expected_demand'][node_name].append(v)
            if v == 0:
                self.node_res['satisfied_demand'][node_name].append(1)
            else:
                self.node_res['satisfied_demand'][node_name].append(self.node_res['demand'][node_name][-1] / v)
        
        for node_name, node in self._wn.reservoirs():
            self.node_res['expected_demand'][node_name].append(0.0)
            self.node_res['satisfied_demand'][node_name].append(1.0)
        for node_name, node in self._wn.tanks():
            self.node_res['expected_demand'][node_name].append(0.0)
            self.node_res['satisfied_demand'][node_name].append(1.0)

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
                self.save_expected_demand()

                if len(self.results.time) > 0 and int(self._wn.sim_time) == self.results.time[-1]:
                    if int(self._wn.sim_time) != self._wn.sim_time:
                        raise RuntimeError('Time steps increments smaller than 1 second are forbidden.'+
                                            ' Keep time steps as an integer number of seconds.')
                    else:
                        raise RuntimeError('Simulation already solved this timestep')
                self.results.time.append(int(self._wn.sim_time))
        elif self._report_timestep.upper() == 'ALL':
            mwntr.sim.hydraulics.save_results(self._wn, self.node_res, self.link_res)
            self.save_expected_demand()

            if len(self.results.time) > 0 and int(self._wn.sim_time) == self.results.time[-1]:
                raise RuntimeError('Simulation already solved this timestep')
            self.results.time.append(int(self._wn.sim_time))
        mwntr.sim.hydraulics.update_network_previous_values(self._wn)
        
        self._wn.sim_time += self._hydraulic_timestep
        #overstep = float(self._wn.sim_time) % self._hydraulic_timestep
        #self._wn.sim_time -= overstep

        if self.rebuild_hydraulic_model:
            self._model, self._model_updater = mwntr.sim.hydraulics.create_hydraulic_model(wn=self._wn, HW_approx=self._hw_approx)
            self.rebuild_hydraulic_model = False
        
        if len(self.demand_modifications) > 0:
            self._apply_demand_modifications()

        if self._wn.sim_time > self._wn.options.time.duration:
            self._terminated = True
        return
        
    def full_run_sim(self):
        self.results = self.run_sim()
        self.last_set_results_time = self._wn.sim_time
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
            data = [node_res[key][name] for name in node_names]
            results.node[key] = pd.DataFrame(data=np.array(data).transpose(), index=results.time,
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

        node = self._wn.get_node(node_name)
        node._leak_status = True
        node._leak = True
        node._leak_area = leak_area
        node._leak_discharge_coeff = leak_discharge_coefficient
        self.rebuild_hydraulic_model = True
        
    def stop_leak(self, node_name):
        self.events_history.append((self.get_sim_time(), 'stop_leak', (node_name)))

        node = self._wn.get_node(node_name)
        node._leak_status = False
        node._leak = False
        self.rebuild_hydraulic_model = True

    def _add_control(self, control):
        if control.epanet_control_type in {_ControlType.presolve, _ControlType.pre_and_postsolve}:
            self._presolve_controls.register_control(control)
        if control.epanet_control_type in {_ControlType.postsolve, _ControlType.pre_and_postsolve}:
            self._postsolve_controls.register_control(control)
        if control.epanet_control_type == _ControlType.rule:
            self._rules.register_control(control)
        if control.epanet_control_type == _ControlType.feasibility:
            self._feasibility_controls.register_control(control)

    def _close_link(self, link_name) -> None:
        link = self._wn.get_link(link_name)
        c1 = mwntr.network.controls.ControlAction(link, "status", LinkStatus.Closed)
        condition = mwntr.network.controls.SimTimeCondition(self._wn, "=", self.get_sim_time() + self.hydraulic_timestep())
        c = mwntr.network.controls.Control(condition=condition, then_action=c1) 
        self._add_control(c)
        self._register_controls_with_observers()
        self.rebuild_hydraulic_model = True

    def _open_link(self, link_name):
        link = self._wn.get_link(link_name)
        c1 = mwntr.network.controls.ControlAction(link, "status", LinkStatus.Open)
        condition = mwntr.network.controls.SimTimeCondition(self._wn, "=", self.get_sim_time()  + self.hydraulic_timestep())
        c = mwntr.network.controls.Control(condition=condition, then_action=c1)
        self._add_control(c)
        self._register_controls_with_observers()
        self.rebuild_hydraulic_model = True

    def close_pipe(self, pipe_name) -> None:
        self.events_history.append((self.get_sim_time(), 'close_pipe', (pipe_name)))
        self._close_link(pipe_name)
        
    def open_pipe(self, pipe_name):
        self.events_history.append((self.get_sim_time(), 'open_pipe', (pipe_name)))
        self._open_link(pipe_name)
    
    def close_valve(self, valve_name) -> None:
        self.events_history.append((self.get_sim_time(), 'close_valve', (valve_name)))
        self._close_link(valve_name)
        
    def open_valve(self, valve_name):
        self.events_history.append((self.get_sim_time(), 'open_valve', (valve_name)))
        self._open_link(valve_name)
    
    def close_pump(self, pump_name) -> None:
        self.events_history.append((self.get_sim_time(), 'close_pump', (pump_name)))
        self._close_link(pump_name)
           
    def open_pump(self, pump_name):
        self.events_history.append((self.get_sim_time(), 'open_pump', (pump_name)))
        self._open_link(pump_name)

    def plot_network(self, title='Water Network Map'):
        mwntr.graphics.plot_interactive_network(self._wn, title=f"{title} - {self._sim_id}", node_labels=True)    

    def _create_base_figure(self, node_positions, edge_list, node_color_0, edge_color_0,
                        node_hover_0, edge_hover_0, edge_names, key, max_value, min_value, node_labels=True, link_labels=True):
        """
        node_positions: dict of node_name -> (x, y)
        edge_list: list of edges, each a tuple (start_node, end_node)
        node_color_0: array of node colors for time 0
        edge_color_0: single color or array for edges at time 0
        node_hover_0: array of hover text strings for nodes at time 0
        edge_hover_0: array of hover text strings for edges at time 0
        edge_names: initial list of annotations for edges
        """

        # Build node trace (single scatter for all nodes)
        node_x = []
        node_y = []
        node_text = []
        for node, (x, y) in node_positions.items():
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text' if node_labels else 'markers',
            marker=dict(
                size=30,
                color=node_color_0,
                colorscale='Viridis',
                colorbar=dict(title=f"{key.capitalize()}"),
                cmax=max_value,
                cmin=min_value,
            ),
            text=node_text,
            hovertext=node_hover_0,
            hoverinfo='text',
            name='Nodes',
            showlegend=False   
        )


        edge_traces = []
        for i, (start, end) in enumerate(edge_list):
            sx, sy = node_positions[start]
            ex, ey = node_positions[end]
            edge_x = [sx, ex]
            edge_y = [sy, ey]
            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(color=edge_color_0[i], width=1),
                # If you have a single color for all edges:
                # or if edge_color_0 is e.g. 'black' or a single hex
                hoverinfo='none',
                name='Edges',
                showlegend=False
            )
            edge_traces.append(edge_trace)

        # (Optional) If you want hover text at edge midpoints:
        # build a small marker trace
        mid_x = []
        mid_y = []
        mid_hover = []
        annotations = []
        for i, (start, end) in enumerate(edge_list):
            sx, sy = node_positions[start]
            ex, ey = node_positions[end]
            mx, my = (sx+ex)/2, (sy+ey)/2
            mid_x.append(mx)
            mid_y.append(my)
            mid_hover.append(edge_hover_0[i])

            if link_labels:
                annotations.append(dict(
                    x=mx,
                    y=my,
                    xref="x",
                    yref="y",
                    text=edge_names[i],
                    showarrow=False,
                    font=dict(size=10, color='black'),
                    bgcolor="#e5ecf6",
                    align="center"
                ))

        edge_midpoints_trace = go.Scatter(
            x=mid_x,
            y=mid_y,
            mode='markers',
            marker=dict(size=20, color='rgba(0,0,0,0)'),  # invisible
            hoverinfo='text',
            hovertext=mid_hover,
            showlegend=False,
            name='EdgeMidpoints'
        )

        # Build the base layout with the initial annotations
        layout = go.Layout(
            title=f"{key.capitalize()} Over Time",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            hovermode='closest',
            annotations=annotations
        )

        # Our base figure data has these 3 traces
        data = []
        data.extend(edge_traces)
        data.append(edge_midpoints_trace)
        data.append(node_trace)

        fig = go.Figure(data=data, layout=layout)
        return fig, annotations

    def _build_frames(self, timesteps, node_color_map, node_hover_map,
                 edge_color_map, edge_hover_map, annotations,
                 num_edges, link_labels=True):
        """
        timesteps: list of times
            e.g. sorted(data_over_time.keys())
        node_color_map[t]: array of node colors at time t
        node_hover_map[t]: array of node hover strings at time t
        edge_color_map[t]: array of edge colors at time t, length = num_edges
        edge_hover_map[t]: array of edge midpoint hover text at time t, length = num_edges
        annotations_map[t]: list of updated annotations for time t
        num_edges: int
            The number of edges (used to slice partial updates)
        """

        frames = []
        for t in timesteps:
            # We'll build a partial data update for each trace.
            # The final data array for each frame must have (num_edges + 2) dicts:
            #   [update_edge_0, update_edge_1, ..., update_edge_(num_edges-1),
            #    update_midpoints, update_nodes]

            frame_data = []

            # 1) Edge traces partial updates
            #    Each edge_color_map[t] is an array of length = num_edges
            #    so we update line.color for each edge trace
            for edge_idx in range(num_edges):
                edge_partial = dict(
                    type='scatter',
                    name='Edges',
                    line=dict(color=edge_color_map[t][edge_idx])
                )
                frame_data.append(edge_partial)

            # 2) Edge midpoints partial update (the next trace after the edges)
            #    We'll set the hovertext for the midpoints to edge_hover_map[t]
            midpoint_partial = dict(
                type='scatter',
                name='EdgeMidpoints',
                hovertext=edge_hover_map[t]
            )
            frame_data.append(midpoint_partial)

            # 3) Node trace partial update (the last trace)
            #    We'll set marker.color and hovertext to node_color_map[t], node_hover_map[t]
            node_partial = dict(
                name='Nodes',
                marker=dict(
                    color=node_color_map[t],
                ),
                hovertext=node_hover_map[t],
            )
            frame_data.append(node_partial)

            # 4) Updated layout with annotations (if your annotations change each frame)
            frame = go.Frame(
                name=str(t),
                data=frame_data,
                layout=go.Layout(
                    annotations=annotations if link_labels else []
                )
            )
            frames.append(frame)

        return frames

    def plot_network_over_time(self, data_key, node_labels=True, link_labels=True):

        #before = time.time()

        results = self.get_results()
        data_over_time = {}
        node_hover_map = {}
        edge_hover_map = {}
        edge_color_map = {}

        global_maximum = -np.inf
        global_minimum = np.inf

        for t in results.time:
            data_over_time[t] = []
            node_hover_map[t] = []
            edge_hover_map[t] = []
            edge_color_map[t] = []

            for node in self._wn.nodes():
                v = results.node[data_key][node[0]][t]
                if v > global_maximum:
                    global_maximum = v
                if v < global_minimum:
                    global_minimum = v
                data_over_time[t].append(v)
                info = f"Node: {node[0]}"
                for k in results.node.keys():
                    info += f"<br>{k}: {results.node[k][node[0]][t]:.4f}"
                node_hover_map[t].append(info)

            for (name, link) in self._wn.links():
                start = link.start_node_name
                end = link.end_node_name
                info = f"Edge {name}: {start} - {end}"
                for k in results.link.keys():
                    info += f"<br>{k}: {results.link[k][name][t]:.4f}"
                edge_hover_map[t].append(info)
                edge_color_map[t].append('black')

        node_positions = {}
        for node in self._wn.nodes():
            node_positions[node[0]] = (node[1].coordinates[0], node[1].coordinates[1])

        edges = [(link[1].start_node_name, link[1].end_node_name) for link in self._wn.links()]
        edges_names = [link[0] for link in self._wn.links()]
        times = sorted([t for t in results.time])

        # Build base figure
        fig, annotations = self._create_base_figure(
            node_positions=node_positions,
            edge_list=edges,
            node_color_0=data_over_time[times[0]],
            edge_color_0=edge_color_map[times[0]],
            node_hover_0=node_hover_map[times[0]],
            edge_hover_0=edge_hover_map[times[0]],
            edge_names=edges_names,
            key=data_key,
            max_value=global_maximum,
            min_value=global_minimum,
            node_labels=node_labels,
            link_labels=link_labels,
        )

        # Build frames
        frames = self._build_frames(
            timesteps=times,
            node_color_map=data_over_time,
            node_hover_map=node_hover_map,
            edge_color_map=edge_color_map,
            edge_hover_map=edge_hover_map,
            annotations=annotations,
            num_edges=len(edges),
            link_labels=link_labels,
        )

        fig.frames = frames

        # Add standard slider or play button
        fig.update_layout(
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    y=1,
                    x=1.2,
                    xanchor='right',
                    yanchor='top',
                    pad=dict(t=0, r=10),
                    buttons=[
                        dict(
                            label='Play',
                            method='animate',
                            args=[None, dict(
                                frame=dict(duration=250, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode='immediate'
                            )]
                        ),
                        dict(
                            label='Pause',
                            method='animate',
                            args=[[None], dict(
                                frame=dict(duration=0, redraw=True),
                                transition=dict(duration=0),
                                mode='immediate'
                            )]
                        )
                    ]
                )
            ],
            sliders=[dict(
                steps=[
                    dict(
                        label=str(t),
                        method='animate',
                        args=[[str(t)], dict(
                            frame=dict(duration=0, redraw=True),
                            transition=dict(duration=0),
                            mode='immediate'
                        )]
                    ) for t in times
                ]
            )]
        )

        #after = time.time()
        #print(f"Time to build figure: {after - before:.2f} seconds")
        fig.show()

    def add_demand(self, node_name, base_demand, name=None, category=None):
        self.events_history.append((self.get_sim_time(), 'add_demand', (node_name, base_demand, name, category)))
        self.demand_modifications.append(('add', node_name, base_demand, name, category))

    def _apply_demand_modifications(self):
        for action, *args in self.demand_modifications:
            if action == 'add':
                node_name, base_demand, name, category = args
                if name is None:
                    name = 'interactive_pattern'
                node = self._wn.get_node(node_name)
                node._pattern_reg.add_usage(name, (node.name, 'Junction'))
                node.demand_timeseries_list.append((base_demand, name, category))
            elif action == 'remove':
                node_name, name = args
                if name is None:
                    name = 'interactive_pattern'
                node = self._wn.get_node(node_name)
                node._pattern_reg.remove_usage(name, (node.name, 'Junction'))
                for ts in node.demand_timeseries_list._list:
                    if ts.pattern_name == name:
                        node.demand_timeseries_list.remove(ts)
        self.demand_modifications.clear()

    def remove_demand(self, node_name, name=None):
        self.events_history.append((self.get_sim_time(), 'remove_demand', (node_name, name)))
        self.demand_modifications.append(('remove', node_name, name))

    def toggle_demand(self, node_name, base_demand=0.0, name=None, category=None):
        node = self._wn.get_node(node_name)
        for ts in node.demand_timeseries_list._list:
            if ts.pattern_name == name:
                self.remove_demand(node_name, name)
                return
        self.add_demand(node_name, base_demand, name, category)    

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

    def _compute_feature_ranges(self, nodes_features, edges_features):
        """Computes min/max for all numerical features before scaling."""
        feature_ranges = {}
        nodes = self._wn.nodes._data.values()
        edges = self._wn.links._data.values()

        for node in nodes:
            for feature in nodes_features:
                value = getattr(node, feature, -1)
                if value is not None and value != -1:
                    if feature not in feature_ranges:
                        feature_ranges[feature] = [value, value]
                    else:
                        feature_ranges[feature][0] = min(feature_ranges[feature][0], value)
                        feature_ranges[feature][1] = max(feature_ranges[feature][1], value)

        for edge in edges:
            for feature in edges_features:
                value = getattr(edge, feature, -1)
                if value is not None and value != -1:
                    if feature not in feature_ranges:
                        feature_ranges[feature] = [value, value]
                    else:
                        feature_ranges[feature][0] = min(feature_ranges[feature][0], value)
                        feature_ranges[feature][1] = max(feature_ranges[feature][1], value)

                    if feature == 'diameter':
                        value = getattr(edge, 'flow', -1)
                        if value is not None and value != -1 and value != 0:
                            velocity = abs(edge.flow)*4.0 / (math.pi*edge.diameter**2)
                            feature = 'velocity'
                            if feature not in feature_ranges:
                                feature_ranges[feature] = [velocity, velocity]
                            else:
                                feature_ranges[feature][0] = min(feature_ranges[feature][0], velocity)
                                feature_ranges[feature][1] = max(feature_ranges[feature][1], velocity)
                
        
        return feature_ranges

    def _scale(self, value, feature, feature_ranges):
        """Scales a feature dynamically based on dataset min/max"""
        has_feature = 0
        if value is None:
            value = -1
        if feature in feature_ranges and value != -1:
            has_feature = 1
            min_val, max_val = feature_ranges[feature]
            if min_val != max_val:  # Avoid division by zero
                value = (value - min_val) / (max_val - min_val)
        
        if abs(value) < 1e-5:
            value = 0
 
        return value, has_feature

    def extract_snapshot(self, filename=None):

        nodes = self._wn.nodes._data.values()
        edges = self._wn.links._data.values()

        snapshot = {
            'time': self.get_sim_time(),
            'nodes': {},
            'edges': {},
        }

        type_map = {
            'Junction':  [1, 0, 0, 0, 0, 0],
            'Tank':      [0, 1, 0, 0, 0, 0],
            'Reservoir': [0, 0, 1, 0, 0, 0],
            'Pipe':      [0, 0, 0, 1, 0, 0],
            'Pump':      [0, 0, 0, 0, 1, 0],
            'Valve':     [0, 0, 0, 0, 0, 1],
        }

        

        nodes_features = ['demand', 'elevation', 'head', 'leak_status', 'leak_area',
                        'leak_discharge_coeff', 'leak_demand', 'pressure', 'diameter',
                        'level', 'max_level', 'min_level', 'overflow']
        edges_features = ['base_speed', 'flow', 'headloss', 'roughness', 'velocity', 'diameter']

        nodes_always_valued_features = ["demand", "head", "leak_area", "leak_demand", "leak_discharge_coeff", "leak_status", "pressure"]
        edges_always_valued_features = ['flow']

        feature_ranges = self._compute_feature_ranges(nodes_features, edges_features)
        print(feature_ranges)

        for n in nodes:
            node_data = {}
        
            for feature in nodes_features:
                value = getattr(n, feature, -1)
                scaled_value, has_feature = self._scale(value, feature, feature_ranges)
                node_data[feature] = scaled_value
                if feature not in nodes_always_valued_features:
                    node_data[f'has_{feature}'] = has_feature
                    
            setting = getattr(n, 'setting', -1)
            if isinstance(setting, mwntr.network.TimeSeries):
                setting = setting.at(self.get_sim_time())
            if setting is not None and setting != -1:
                node_data['setting'] = setting
                node_data['has_setting'] = 1
            else:
                node_data['setting'] = -1
                node_data['has_setting'] = 0
            
            node_data['node_type'] = type_map[n.node_type]

            snapshot['nodes'][n.name] = node_data

        for l in edges:
            edge_data = {}

            for feature in edges_features:
                value = getattr(l, feature, -1)
                scaled_value, has_feature = self._scale(value, feature, feature_ranges)
                edge_data[feature] = scaled_value
                if feature not in edges_always_valued_features:
                    edge_data[f'has_{feature}'] = has_feature

            setting = getattr(l, 'setting', -1)
            if isinstance(setting, mwntr.network.TimeSeries):
                setting = setting.at(self.get_sim_time())
            if setting is not None and setting != -1:
                edge_data['setting'] = setting
                edge_data['has_setting'] = 1
            else:
                edge_data['setting'] = -1
                edge_data['has_setting'] = 0

            if hasattr(l,"diameter") and l.diameter is not None:
                velocity, has_velocity = self._scale(abs(l.flow)*4.0 / (math.pi*l.diameter**2), 'velocity', feature_ranges)
                edge_data["velocity"] = velocity
                edge_data["has_velocity"] = has_velocity
            else:
                edge_data["velocity"] = -1
                edge_data["has_velocity"] = 0
            
            if l.status == LinkStatus.Closed:
                edge_data['status'] = 0
            else:
                edge_data['status'] = 1
            

            edge_data['link_type'] = type_map[l.link_type]  
            edge_data['start'] = l.start_node_name
            edge_data['end'] = l.end_node_name

            snapshot['edges'][l.name] = edge_data

        if filename is not None:
            with open(filename, 'w') as f:
                json.dump(snapshot, f, indent=4, sort_keys=True)

        return snapshot

        #def _set_active_valve(self, valve):
    #    #c1 = _InternalControlAction(valve, '_user_status', LinkStatus.Active, 'status')
    #    c1 = _InternalControlAction(valve, '_internal_status', LinkStatus.Active, 'status')
    #    condition = mwntr.network.controls.SimTimeCondition(self._wn, "=", self.get_sim_time()  + self.hydraulic_timestep())
    #    c = mwntr.network.controls.Control(condition=condition, then_action=c1, priority=ControlPriority.very_high)
    #    self._add_control(c)
    #    self._register_controls_with_observers()
    #    self.rebuild_hydraulic_model = True


    #def set_valve_opening(self, valve_name, setting):
    #    if setting < 0 or setting > 1:
    #        raise ValueError('Valve setting must be between 0 and 1')
    #    elif setting == 0:
    #        self.close_valve(valve_name)
    #    elif setting == 1:
    #        self.open_valve(valve_name)
    #    else:
    #        self.events_history.append((self.get_sim_time(), 'set_valve_opening', (valve_name, setting)))
    #        valve = self._wn.get_link(valve_name)
    #        #valve.initial_setting = setting
    #        #valve._setting = setting
    #        self._set_active_valve(valve)
    #        c1 = mwntr.network.controls.ControlAction(valve, "setting", setting)
    #        condition = mwntr.network.controls.SimTimeCondition(self._wn, "=", self.get_sim_time() + self.hydraulic_timestep())
    #        c = mwntr.network.controls.Control(condition=condition, then_action=c1) 
    #        self._add_control(c)
    #        self._register_controls_with_observers()
    #        self.rebuild_hydraulic_model = True
        
    '''
    
    def plot_network_over_time_old(self, data_key, node_labels=True, link_labels=True):
        start_time = time.time()

        node_positions = {}
        for node in self._wn.nodes():
            node_positions[node[0]] = (node[1].coordinates[0], node[1].coordinates[1])

        edges = [(link[0], (link[1].start_node_name, link[1].end_node_name)) for link in self._wn.links()]

        data_over_time = {}
        results = self.get_results()
        node_hover_text = {}
        edge_hover_text = {}

        for t in results.time:
            data_over_time[t] = {}
            node_hover_text[t] = []
            edge_hover_text[t] = []

            for node in self._wn.nodes():
                data_over_time[t][node[0]] = results.node[data_key][node[0]][t]
                info = f"Node: {node[0]}"
                for k in results.node.keys():
                    info += f"<br>{k}: {results.node[k][node[0]][t]:.4f}"
                node_hover_text[t].append(info)

            for (name, link) in self._wn.links():
                start = link.start_node_name
                end = link.end_node_name
                info = f"Edge {name}: {start} - {end}"
                for k in results.link.keys():
                    info += f"<br>{k}: {results.link[k][name][t]:.4f}"
                edge_hover_text[t].append(info)



        # Base node trace (for initial frame)
        node_x = [node_positions[n][0] for n in node_positions]
        node_y = [node_positions[n][1] for n in node_positions]
        initial_data = [data_over_time[results.time[0]][n] for n in node_positions]

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text' if node_labels else 'markers',
            text=list(node_positions.keys()),
            hovertext=node_hover_text[0],
            hoverinfo='text',
            marker=dict(
                size=20,
                color=initial_data,  # Use data as color
                colorscale='Viridis',
                colorbar=dict(title=data_key.capitalize())
            ),
            name='Nodes',
            showlegend=False  # Hides legend for nodes
        )

        edge_traces = []
        initial_annotations = []
        for i,(name, (start, end)) in enumerate(edges):
            x0, y0 = node_positions[start]
            x1, y1 = node_positions[end]

            mx, my = (x0 + x1)/2, (y0 + y1)/2
            edge_trace = go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode='lines',
                line=dict(color='black', width=1),
                name='Edges',
                showlegend=False  # Hides legend for edges
            )

            midpoint_marker = go.Scatter(
                x=[mx],
                y=[my],
                mode='markers',
                marker=dict(size=20, color='rgba(0, 0, 0, 0)'),  # invisible
                hoverinfo='text',
                hovertext=[edge_hover_text[0][i]],
                showlegend=False
            )

            edge_traces += [edge_trace, midpoint_marker]


            if link_labels:
                initial_annotations.append(dict(
                    x=mx,
                    y=my,
                    xref="x",
                    yref="y",
                    text=name,
                    showarrow=False,
                    font=dict(size=12, color='black'),
                    bgcolor="#e5ecf6",
                    align="center"
                ))

                

        # Create frames for each timestep
        frames = []
        for t in sorted(data_over_time.keys()):
            data_at_t = [data_over_time[t][n] for n in node_positions]
            # Update hover text for nodes per frame if desired.
            # For a starting point, we'll reuse the same hover text;
            # you can modify this to include time-dependent information.
            frame_node = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text' if node_labels else 'markers',
                text=list(node_positions.keys()),
                hovertext=node_hover_text[t],
                hoverinfo='text',
                marker=dict(
                    size=20,
                    color=data_at_t,
                    colorscale='Viridis',
                    colorbar=dict(title=data_key.capitalize())
                ),
                name="Nodes",
                showlegend=False
            )
            frame_edges_traces = []
            annotations = []

            for i,(name, (start, end)) in enumerate(edges):
                x0, y0 = node_positions[start]
                x1, y1 = node_positions[end]

                edge_trace = go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode='lines',
                    line=dict(color='black', width=1),
                    name='Edges',
                    showlegend=False  # Hides legend for edges
                )
                midpoint_marker = go.Scatter(
                        x=[mx],
                        y=[my],
                        mode='markers',
                        marker=dict(size=20, color='rgba(0, 0, 0, 0)'),  # invisible
                        hoverinfo='text',
                        hovertext=[edge_hover_text[t][i]],
                        showlegend=False
                    )

                frame_edges_traces += [edge_trace, midpoint_marker]


                if link_labels:
                    mx, my = (x0 + x1)/2, (y0 + y1)/2
                    annotations.append(dict(
                        x=mx,
                        y=my,
                        xref="x",
                        yref="y",
                        text=name,
                        showarrow=False,
                        font=dict(size=12, color='black'),
                        bgcolor="#e5ecf6",          # background color
                        align="center"
                    ))
                
            frame = go.Frame(
                data= frame_edges_traces + [frame_node],
                name=str(t),
                layout=go.Layout(annotations=annotations)
            )
            frames.append(frame)

        # Layout with slider settings; remove the legend config to hide trace labels.
        layout = go.Layout(
            title=f"{data_key.capitalize()} Over Time",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            hovermode='closest',
            annotations=initial_annotations,
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    y=1,
                    x=1.2,
                    xanchor='right',
                    yanchor='top',
                    pad=dict(t=0, r=10),
                    buttons=[
                        dict(
                            label='Play',
                            method='animate',
                            args=[None, dict(
                                frame=dict(duration=250, redraw=False),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode='immediate'
                            )]
                        ),
                        dict(
                            label='Pause',
                            method='animate',
                            args=[[None], dict(
                                frame=dict(duration=0, redraw=False),
                                transition=dict(duration=0),
                                mode='immediate'
                            )]
                        )
                    ]
                )
            ],
            sliders=[dict(
                active=0,
                currentvalue=dict(prefix="Time: "),
                pad=dict(t=50),
                steps=[
                    dict(
                        label=str(t),
                        method='animate',
                        args=[[str(t)], dict(
                            frame=dict(duration=250, redraw=True),
                            transition=dict(duration=0),
                            mode='immediate'
                        )]
                    ) for t in sorted(data_over_time.keys())
                ]
            )]
        )

        data = edge_traces + [node_trace]
        fig = go.Figure(data=data, layout=layout, frames=frames)

        end_time = time.time()
        print(f"Time to create plot: {end_time - start_time} seconds")
        fig.show()

    '''