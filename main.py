from calendar import c
from turtle import color
import mwntr
import matplotlib.pyplot as plt
import mwntr.sim.results

from mwntr.network.elements import Junction, Pipe, Reservoir, Valve
from mwntr.sim.interactive_network_simulator import MWNTRInteractiveSimulator

def plot_results(results: mwntr.sim.results.SimulationResults, scheduling=None):
    #take only pressure of H1, H2 and H3
    #pressure = results.node['pressure'][['H1', 'H2', 'H3']]
    pressure = results.node['pressure']

    num_series = len(pressure.columns)
    cmap = plt.colormaps.get_cmap('tab20')
    color_list = [cmap(i) for i in range(num_series)]
    pressure.plot(title='Pressure at Nodes Over Time', figsize=(10, 6), color=color_list)

    if scheduling:
        for time, action, args in scheduling:
            #use a random color for each action
            color = plt.cm.tab20(hash(f'{action.__name__}({args})') % len(scheduling))
            plt.axvline(x=time, color=color, linestyle='--', label=f'{action.__name__}({args})')

    plt.ylabel('Pressure (m)')
    plt.legend()
    plt.show()

def create_water_network_model():
    # 1. Create a new water network model
    wn = mwntr.network.WaterNetworkModel()

    # --- Simulation options ---
    wn.options.time.duration = 86400 * 2       # 2 days
    wn.options.time.hydraulic_timestep = 2     # 2 seconds per step (just an example)
    wn.options.time.report_timestep = 60       # report every 60 seconds
    wn.options.time.pattern_timestep = 3600    # 1 hour per pattern step
    wn.options.hydraulic.demand_model = 'PDD'

    # -------------------------------
    # Define demand patterns for houses
    # -------------------------------
    pattern_house1 = [1.0]*8 + [2.0]*12 + [1.0]*4    # House1
    pattern_house2 = [0.5]*7 + [1.5]*4 + [0.5]*6 + [1.5]*4 + [0.5]*3  # House2
    pattern_house3 = [2.5]*8 + [0.0]*12 + [2.5]*4    # Reusing your "house4" pattern as House3

    wn.add_pattern('house1_pattern', pattern_house1)
    wn.add_pattern('house2_pattern', pattern_house2)
    wn.add_pattern('house3_pattern', pattern_house3)

    # 2. Add a Reservoir (Tank1) on the left
    wn.add_reservoir('R1', base_head=100.0, head_pattern=None, coordinates=(-50, 50))

    # 3. Build a rectangular loop (8 junctions: J0–J7)
    #    Coordinates are just an example. Adjust as you see fit.
    wn.add_junction('J0', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(0, 100))
    wn.add_junction('J1', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(50, 100))
    wn.add_junction('J2', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(100, 100))
    wn.add_junction('J3', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(100, 50))
    wn.add_junction('J4', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(100, 0))
    wn.add_junction('J5', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(50, 0))
    wn.add_junction('J6', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(0, 0))
    wn.add_junction('J7', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(0, 50))
    wn.add_junction('J8', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(50, 50))

    # 4. Connect the reservoir (Tank1) to the loop at J7
    wn.add_pipe('P_R1_J7', 'R1', 'J7', length=50, diameter=0.3, roughness=100, minor_loss=0)

    # 5. Connect the 8 junctions in a loop (rectangle)
    wn.add_pipe('PR0', 'J0', 'J1', length=50, diameter=0.3, roughness=100, minor_loss=0)
    wn.add_pipe('PR1', 'J1', 'J2', length=50, diameter=0.3, roughness=100, minor_loss=0, initial_status='CLOSED')
    wn.add_pipe('PR2', 'J2', 'J3', length=50, diameter=0.3, roughness=100, minor_loss=0)
    wn.add_pipe('PR3', 'J3', 'J4', length=50, diameter=0.3, roughness=100, minor_loss=0)
    wn.add_pipe('PR4', 'J4', 'J5', length=50, diameter=0.3, roughness=100, minor_loss=0, initial_status='CLOSED')
    wn.add_pipe('PR5', 'J5', 'J6', length=50, diameter=0.3, roughness=100, minor_loss=0)
    wn.add_pipe('PR6', 'J6', 'J7', length=50, diameter=0.3, roughness=100, minor_loss=0, initial_status='CLOSED')
    wn.add_pipe('PR7', 'J7', 'J0', length=50, diameter=0.3, roughness=100, minor_loss=0, initial_status='CLOSED')

    wn.add_pipe('PR8', 'J7', 'J8', length=50, diameter=0.3, roughness=100, minor_loss=0)
    wn.add_pipe('PR9', 'J8', 'J3', length=50, diameter=0.3, roughness=100, minor_loss=0)

    # 6. Add three houses (H1, H2, H3) branching from the right side
    #    and assign the previously defined demand patterns
    wn.add_junction('H1', base_demand=0.5, elevation=10.0, demand_pattern='house1_pattern', coordinates=(120, 100))
    wn.add_junction('H2', base_demand=0.5, elevation=10.0, demand_pattern='house2_pattern', coordinates=(120, 50))
    wn.add_junction('H3', base_demand=0.5, elevation=10.0, demand_pattern='house3_pattern', coordinates=(120, 0))

    # Connect them to J2, J3, J4 (top-right corner, mid-right, bottom-right corner)
    wn.add_pipe('PH1', 'J2', 'H1', length=20, diameter=0.3, roughness=100, minor_loss=0)
    wn.add_pipe('PH2', 'J3', 'H2', length=20, diameter=0.3, roughness=100, minor_loss=0)
    wn.add_pipe('PH3', 'J4', 'H3', length=20, diameter=0.3, roughness=100, minor_loss=0)

    return wn

wn = mwntr.network.WaterNetworkModel('NET_2.inp')

sim = MWNTRInteractiveSimulator(wn)
sim.init_simulation()

sim.plot_network(link_labels=True, node_labels=True, show_plot=True)


scheduling = [
    #(20000, sim.start_leak, 'J3'),
    #(21000, sim.start_leak, 'J2'),
    #(40000, sim.open_pipe, 'PR1'),
    #(40000, sim.open_pipe, 'PR7'),
    #(40000, sim.open_pipe, 'PR6'),
    #(40000, sim.open_pipe, 'PR4'),
    #(50000, sim.stop_leak, 'J3'),
    #(51000, sim.stop_leak, 'J2'),
]

while not sim.is_terminated():
    current_time = sim.get_sim_time()

    for time, action, args in scheduling:
        if current_time == time:
            action(args)
            
    sim.step_sim()

# Plot final results
plot_results(sim.get_results(), scheduling=scheduling)
