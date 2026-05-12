import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt

# Airport data
airports_url = "https://davidmegginson.github.io/ourairports-data/airports.csv"

# World map polygons from Natural Earth
world_url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"

# Load data
airports = pd.read_csv(airports_url)
world = gpd.read_file(world_url)

# Filter out NULL coordinates
airports = airports.dropna(subset=["latitude_deg", "longitude_deg"])

# Focus dataset on large airpots
airports = airports[airports["type"] != "closed_airport"]
airports = airports[airports["type"] != "heliport"]
airports = airports[airports["type"] != "balloonport"]
airports = airports[airports["type"] != "seaplane_base"]
# Optional: Can remove this to include med and small airports
airports = airports[airports["type"] != "medium_airport"]
airports = airports[airports["type"] != "small_airport"]

G = nx.Graph()

for _, row in airports.iterrows():
    G.add_node(
        row["ident"],
        name=row["name"],
        airport_type=row["type"],
        lat=row["latitude_deg"],
        lon=row["longitude_deg"],
        iso_country=row["iso_country"],
        iata=row.get("iata_code"),
    )

# NetworkX position dictionary: node -> (x, y)
# x = longitude, y = latitude
pos = {
    node: (data["lon"], data["lat"])
    for node, data in G.nodes(data=True)
}

# Plot world map
fig, ax = plt.subplots(figsize=(16, 8))

world.plot(
    ax=ax,
    color="whitesmoke",
    edgecolor="gray",
    linewidth=0.4
)

# Draw airport nodes
nx.draw_networkx_nodes(
    G,
    pos,
    ax=ax,
    node_size=3,
    node_color="crimson",
    alpha=1,
    linewidths=0
)

ax.set_title("Airports as NetworkX Nodes")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_xlim(-180, 180)
ax.set_ylim(-60, 85)

plt.tight_layout()
plt.show()