"""
geoplot.py
----------

This visualization renders a 3D time-series plot of agent-based simulation data
using CesiumJS and GeoJSON. It maps agent positions and a tracked property
(e.g., money spent) over time, outputting a Cesium-compatible HTML viewer.

Outputs:
- A `.geojson` file with feature data over time
- A `.html` file that visualizes this data with CesiumJS

Used in: agent_torch.visualize.GeoPlot
"""

import re
import json
import pandas as pd
import numpy as np

from string import Template
from agent_torch.core.helpers import get_by_path  # Utility to access nested values in state dicts

# HTML + JS template used to generate the interactive 3D plot using CesiumJS
# It embeds a Cesium viewer and loads time-series data from the GeoJSON provided
geoplot_template = """ ... (same content, unchanged for clarity) ... """

# Helper function to resolve a slash-separated variable path (e.g., "agents/consumer/location")
# from a nested simulation state dictionary
def read_var(state, var):
    return get_by_path(state, re.split("/", var))

class GeoPlot:
    def __init__(self, config, options):
        """
        Initialize the GeoPlot renderer.

        Args:
            config (dict): Global simulation configuration, including metadata
            options (dict): Visualization options:
                - cesium_token (str): Access token for CesiumJS API
                - step_time (int): Duration (in seconds) between simulation steps
                - coordinates (str): Path to the agent coordinates in the state
                - feature (str): Path to the property to visualize (e.g., energy used)
                - visualization_type (str): 'color' or 'size' (affects Cesium rendering)
        """
        self.config = config
        (
            self.cesium_token,
            self.step_time,
            self.entity_position,
            self.entity_property,
            self.visualization_type,
        ) = (
            options["cesium_token"],
            options["step_time"],
            options["coordinates"],
            options["feature"],
            options["visualization_type"],
        )

    def render(self, state_trajectory):
        """
        Render the visualization for the provided simulation trajectory.

        Args:
            state_trajectory (List[List[Dict]]): A list of episodes, where each
            episode is a list of state dictionaries recorded at each step.
        """
        coords, values = [], []

        # Define output file paths based on the simulation name
        name = self.config["simulation_metadata"]["name"]
        geodata_path, geoplot_path = f"{name}.geojson", f"{name}.html"

        # Loop through each episode (except the final one) to extract data
        for i in range(0, len(state_trajectory) - 1):
            final_state = state_trajectory[i][-1]  # Take last step from each episode

            # Extract positions and property values for all agents at this final step
            coords = np.array(read_var(final_state, self.entity_position)).tolist()
            values.append(
                np.array(read_var(final_state, self.entity_property)).flatten().tolist()
            )

        # Start time for simulation (using current UTC time as baseline)
        start_time = pd.Timestamp.utcnow()

        # Generate a list of timestamps spaced by step_time, one for each simulation step
        timestamps = [
            start_time + pd.Timedelta(seconds=i * self.step_time)
            for i in range(
                self.config["simulation_metadata"]["num_episodes"]
                * self.config["simulation_metadata"]["num_steps_per_episode"]
            )
        ]

        geojsons = []
        # For each agent coordinate, create a separate GeoJSON time series
        for i, coord in enumerate(coords):
            features = []
            # Loop over each timestep and extract the corresponding value for this agent
            for time, value_list in zip(timestamps, values):
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [coord[1], coord[0]],  # GeoJSON expects [lon, lat]
                        },
                        "properties": {
                            "value": value_list[i],              # Property value at this time
                            "time": time.isoformat(),            # Timestamp in ISO format
                        },
                    }
                )
            geojsons.append({
                "type": "FeatureCollection",
                "features": features
            })

        # Save GeoJSON time-series data to disk
        with open(geodata_path, "w", encoding="utf-8") as f:
            json.dump(geojsons, f, ensure_ascii=False, indent=2)

        # Generate HTML output by substituting the template with real values
        tmpl = Template(geoplot_template)
        with open(geoplot_path, "w", encoding="utf-8") as f:
            f.write(
                tmpl.substitute(
                    {
                        "accessToken": self.cesium_token,
                        "startTime": timestamps[0].isoformat(),
                        "stopTime": timestamps[-1].isoformat(),
                        "data": json.dumps(geojsons),
                        "visualType": self.visualization_type,
                    }
                )
            )
