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

    import plotly.express as px
    key = 'demand'

    pressure = results.node[key]
    num_series = len(pressure.columns)
    cmap = plt.colormaps.get_cmap('tab20')

    #color_list = [cmap(i % 20) for i in range(num_series)]
    if len(pressure) == 0:
        print('No results to plot')
        return
    fig = px.line(pressure, x=pressure.index, y=pressure.columns, title=f'{key.capitalize()} at nodes over time')

    if scheduling:
        for time, action, args in scheduling:
            color = '#'+str(hex(hash(action.__name__+str(args)) % 16777215)[2:].zfill(6))
            fig.add_vline(x=time, line_dash='dash', line_color=color, annotation_text='  ', annotation_hovertext=f'{action.__name__}({args})')
            
    fig.show()

def create_water_network_model():
    # 1. Create a new water network model
    wn = mwntr.network.WaterNetworkModel()

    # --- Simulation options ---
    wn.options.time.duration = 86400 * 10       # 2 days
    wn.options.time.hydraulic_timestep = 60     
    wn.options.time.report_timestep = 60       
    wn.options.time.pattern_timestep = 3600    
    wn.options.hydraulic.demand_model = 'PDD'

    # -------------------------------
    # Define demand patterns for houses
    # -------------------------------
    pattern_house1 = [1.0]*8 + [2.0]*12 + [1.0]*4    
    pattern_house2 = [0.5]*7 + [1.5]*4 + [0.5]*6 + [1.5]*4 + [0.5]*3  
    pattern_house3 = [2.5]*8 + [0.0]*12 + [2.5]*4    

    pump_speed_pattern = [5.0]*24

    wn.add_pattern('house1_pattern', pattern_house1)
    wn.add_pattern('house2_pattern', pattern_house2)
    wn.add_pattern('house3_pattern', pattern_house3)
    
    wn.add_pattern('pump_speed_pattern', pump_speed_pattern)

    # 2. Add a Reservoir (Tank1) on the left
    wn.add_reservoir('R1', base_head=500.0, head_pattern=None, coordinates=(-50, 50))

    # 3. Build a rectangular loop (9 junctions: J0–J8)
    wn.add_junction('J0', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(0, 100))
    wn.add_junction('J1', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(50, 100))
    wn.add_junction('J2', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(100, 100))
    wn.add_junction('J3', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(100, 50))
    wn.add_junction('J4', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(100, 0))
    wn.add_junction('J5', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(50, 0))
    wn.add_junction('J6', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(0, 0))
    wn.add_junction('J7', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(0, 50))
    wn.add_junction('J8', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(50, 50))

    # 4. Replace the reservoir connection with a pump:
    wn.add_pipe('P_R1_J7', 'R1', 'J7', length=50, diameter=0.3, roughness=100, minor_loss=0)
    #wn.add_pump('Pump1', 'R1', 'J7', pump_parameter=1000.0)

    # 5. Connect the 8 junctions in a loop (rectangle)
    wn.add_pipe('PR0', 'J0', 'J1', length=50, diameter=0.3, roughness=100, minor_loss=0)
    wn.add_pipe('PR1', 'J1', 'J2', length=50, diameter=0.3, roughness=100, minor_loss=0)
    wn.add_pipe('PR2', 'J2', 'J3', length=50, diameter=0.3, roughness=100, minor_loss=0)
    wn.add_pipe('PR3', 'J3', 'J4', length=50, diameter=0.3, roughness=100, minor_loss=0)
    wn.add_pipe('PR4', 'J4', 'J5', length=50, diameter=0.3, roughness=100, minor_loss=0)
    wn.add_pipe('PR5', 'J5', 'J6', length=50, diameter=0.3, roughness=100, minor_loss=0)
    wn.add_pipe('PR6', 'J6', 'J7', length=50, diameter=0.3, roughness=100, minor_loss=0)
    wn.add_pipe('PR7', 'J7', 'J0', length=50, diameter=0.3, roughness=100, minor_loss=0)
    wn.add_pipe('PR8', 'J7', 'J8', length=50, diameter=0.3, roughness=100, minor_loss=0)
    wn.add_pipe('PR9', 'J8', 'J3', length=50, diameter=0.3, roughness=100, minor_loss=0)

    # 6. Add three houses (H1, H2, H3) branching from the right side
    #wn.add_junction('H1', base_demand=0.5, elevation=10.0, demand_pattern='house1_pattern', coordinates=(120, 100))
    #wn.add_junction('H2', base_demand=0.5, elevation=10.0, demand_pattern='house2_pattern', coordinates=(120, 50))
    #wn.add_junction('H3', base_demand=0.5, elevation=10.0, demand_pattern='house3_pattern', coordinates=(120, 0))
    wn.add_junction('H1', base_demand=0.5, elevation=10.0, coordinates=(120, 100))
    wn.add_junction('H2', base_demand=0.5, elevation=10.0, coordinates=(120, 50))
    wn.add_junction('H3', base_demand=0.5, elevation=10.0, coordinates=(120, 0))

    # 7. Connect houses to the loop.
    wn.add_pipe('PH1', 'J2', 'H1', length=20, diameter=0.3, roughness=100, minor_loss=0)
    wn.add_pipe('PH3', 'J4', 'H3', length=20, diameter=0.3, roughness=100, minor_loss=0)

    # For H2, remove the existing pipe and add a valve instead:
    #wn.add_pipe('PH2', 'J3', 'H2', length=20, diameter=0.3, roughness=100, minor_loss=0)
    wn.add_valve('Valve1', 'J3', 'H2')

    return wn

#wn = mwntr.network.WaterNetworkModel('NET_2.inp')
wn = create_water_network_model()

sim = MWNTRInteractiveSimulator(wn)
sim.init_simulation()

sim.plot_network(link_labels=True, node_labels=True, show_plot=True)

scheduling = [
    #(sim.hydraulic_timestep()*1000, sim.start_leak, ('J7', 0.01)),
    (sim.hydraulic_timestep()*1000, sim.toggle_demand, ('H1', 1)),
    (sim.hydraulic_timestep()*2000, sim.toggle_demand, ('H1',)),
    (sim.hydraulic_timestep()*3000, sim.toggle_demand, ('H1', 10)),
    (sim.hydraulic_timestep()*4000, sim.toggle_demand, ('H1',)),
    (sim.hydraulic_timestep()*5000, sim.toggle_demand, ('H1', 100)),
]

while not sim.is_terminated():
    current_time = sim.get_sim_time()

    for time, action, args in scheduling:
        if current_time == time:
            action(*args)

    
            
    sim.step_sim()

# Plot final results
plot_results(sim.get_results(), scheduling=scheduling)
