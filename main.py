from calendar import c
import sys
import time
from turtle import color
from uuid import UUID
import mwntr
import matplotlib.pyplot as plt
import mwntr.sim.results

from mwntr.network.elements import Junction, Pipe, Reservoir, Valve
from mwntr.sim.interactive_network_simulator import MWNTRInteractiveSimulator



def create_water_network_model():
    # 1. Create a new water network model
    wn = mwntr.network.WaterNetworkModel()

    # --- Simulation options ---
    wn.options.time.duration = 86400 / 12     # 10 days
    wn.options.time.hydraulic_timestep = 30
    wn.options.time.pattern_timestep = 30 
    wn.options.time.rule_timestep = 30
    wn.options.time.report_timestep = 30
    wn.options.time.quality_timestep = 30   
    wn.options.hydraulic.demand_model = 'PDD'

    # -------------------------------
    # Define demand patterns for houses
    # -------------------------------
    pattern_house1 = [1.0] + [2.0] + [3.0] + [4.0] + [5.0] + [6.0] + [1.0] + [2.0] + [3.0] + [4.0] + [5.0] + [6.0] + [1.0] + [2.0] + [3.0] + [4.0] + [5.0] + [6.0] + [1.0] + [2.0] + [3.0] + [4.0] + [5.0] + [6.0]  
    pattern_house2 = [0.5]*7 + [1.5]*4 + [0.5]*6 + [1.5]*4 + [0.5]*3  
    pattern_house3 = [2.5]*8 + [0.0]*12 + [2.5]*4    

    pump_speed_pattern = [1.0]*24

    wn.add_pattern('house1_pattern', pattern_house1)
    wn.add_pattern('house2_pattern', pattern_house2)
    wn.add_pattern('house3_pattern', pattern_house3)
    
    wn.add_pattern('pump_speed_pattern', pump_speed_pattern)
    wn.add_curve('pump_curve', 'HEAD', [(100,100)])

    # 2. Add a Reservoir (Tank1) on the left
    #wn.add_reservoir('R1', base_head=500.0, head_pattern=None, coordinates=(-50, 50))
    wn.add_tank('R1', elevation=100, init_level=100, max_level=100, coordinates=(-50, 50))

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
    #wn.add_pipe('P_R1_J7', 'R1', 'J7', length=50, diameter=0.3, roughness=100, minor_loss=0)
    wn.add_pump('P1', 'R1', 'J7', initial_status='OPEN', pump_type='HEAD', pump_parameter='pump_curve')

    # 5. Connect the 8 junctions in a loop (rectangle)
    wn.add_pipe('PR0', 'J0', 'J1', length=50, diameter=0.3, roughness=99 , minor_loss=0)
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
    wn.add_junction('H2', base_demand=1.0, elevation=10.0, coordinates=(120, 50))
    wn.add_junction('H3', base_demand=1.5, elevation=10.0, coordinates=(120, 0))

    # 7. Connect houses to the loop.
    wn.add_pipe('PH1', 'J2', 'H1', length=20, diameter=0.3, roughness=100, minor_loss=0)
    wn.add_pipe('PH3', 'J4', 'H3', length=20, diameter=0.3, roughness=100, minor_loss=0)

    # For H2, remove the existing pipe and add a valve instead:
    #wn.add_pipe('PH2', 'J3', 'H2', length=20, diameter=0.3, roughness=100, minor_loss=0)
    wn.add_valve('V1', 'J3', 'H2', diameter=0.3, valve_type="FCV", initial_status='Active')

    return wn

#wn = mwntr.network.WaterNetworkModel('NET_2.inp')
wn = create_water_network_model()

one_day_in_seconds = 86400 / 24
wn.options.time.duration = one_day_in_seconds  # 2 hours
wn.options.time.hydraulic_timestep = 30
wn.options.time.pattern_timestep = 30 
wn.options.time.rule_timestep = 30
wn.options.time.report_timestep = 30
wn.options.time.quality_timestep = 30


t = wn.get_node('R1')
t.add_leak(wn, 0.1, start_time=30*5, end_time=30*37)
#t.add_demand(0.1, 'house1_pattern')

sim = MWNTRInteractiveSimulator(wn)

sim.init_simulation()

#sim.plot_network()

branched_sim_1 = None
branched_sim_2 = None

sims = [sim]

start = time.time()
while not sim.is_terminated():
    #print(f"Current time: {current_time} {current_time / sim.hydraulic_timestep()}")
    current_time = sim.get_sim_time()

    #if current_time == sim.hydraulic_timestep() * 2:
    #    #sim.toggle_demand('H1', 0.1, name='house1_pattern')
    #    #sim.start_leak('J3', 0.01)

    #if current_time == sim.hydraulic_timestep() * 0:
    #   #sim.toggle_demand('H1', 0.1, name='house1_pattern')
    #   sim.start_leak('R1', 0.1)
    
    
    #elif current_time == sim.hydraulic_timestep() * 26:
    #    sim.toggle_demand('H1', 0.2, name='house2_pattern')
    #
    #elif current_time == sim.hydraulic_timestep() * 37:
    #    sim.stop_leak('J1')
    #
    #if current_time == sim.hydraulic_timestep() * 145:
    #    b = time.time()
    #    sim.extract_snapshot(filename='snapshot_2.json')
    #    e = time.time()
    #    print(f"Elapsed time: {e - b}")
    #    #print(sim.extract_snapshot(filename=None))
    #
    #
    #    sim.toggle_demand('H1', name='house1_pattern')
    #    sim.toggle_demand('H1', name='house2_pattern')
    #    sim.toggle_demand('H1', name='house3_pattern')
    #
    #elif current_time == sim.hydraulic_timestep() * 178:
    #    sim.close_pump('P1')
    #
    #elif current_time == sim.hydraulic_timestep() * 211:
    #    sim.open_pump('P1')

    #elif current_time == sim.hydraulic_timestep() * 60:
    #    #branched_sim_1 = sim.branch()
    #    #branched_sim_2 = sim.branch()
    #    #sims.append(branched_sim_1)
    #    #sims.append(branched_sim_2)
    #    #branched_sim_1.start_leak('J1', 0.1)
    #
    #
    #elif current_time == sim.hydraulic_timestep() * 100:
    #    branched_sim_1.stop_leak('J1')
    #    branched_sim_2.close_pipe('PR0')
    #    branched_sim_2.close_pipe('PR1')
    #    branched_sim_2.close_valve('V1')
    #
    #    
    #
    #elif current_time == sim.hydraulic_timestep() * 80:
    #    branched_sim_2.stop_leak('J1')

    for s in sims:
        s.step_sim()


end = time.time()
print(f"Elapsed time: {end - start}")

for s in sims:
    s.plot_results('node','pressure')

#sim.plot_results('node','demand', ['H1', 'H2', 'H3'])
#branched_sim_1.plot_results('node','pressure', ['H1', 'H2', 'H3'])
#branched_sim_2.plot_results('node','pressure', ['H1', 'H2', 'H3'])

#sim.plot_results('link','flowrate', ['PH3', 'V1', 'PH1'])
#sim.plot_network_over_time('pressure', node_labels=True, link_labels=True)
#sim.plot_results('node','expected_demand', ['R1', 'H1', 'H2', 'H3'])
#sim.plot_results('node','satisfied_demand', ['R1', 'H1', 'H2', 'H3'])
