"""
The wntr.graphics package contains graphic functions
"""
from mwntr.graphics.network import plot_network, plot_interactive_network, plot_leaflet_network, network_animation
from mwntr.graphics.layer import plot_valve_layer
from mwntr.graphics.curve import plot_fragility_curve, plot_pump_curve, plot_tank_volume_curve
from mwntr.graphics.color import custom_colormap, random_colormap

