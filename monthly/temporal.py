import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


ROUTES_CSV = "T100.csv"
OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"


def load_us_airports() -> pd.DataFrame:
    airports = pd.read_csv(OURAIRPORTS_URL)

    # Filter out non US airports 
    airports = airports[
        (airports["iso_country"] == "US")
        & airports["iata_code"].notna()
        & (airports["iata_code"].str.len() == 3)
    ].copy()
    
    airports["iata_code"] = airports["iata_code"].str.upper()

    return airports


def load_routes(routes_csv: str, month: int = None) -> pd.DataFrame:
    routes = pd.read_csv(routes_csv, low_memory=False)

    print(routes.head())
    print(routes.columns)

    routes.columns = (
        routes.columns
        .str.strip()
        .str.upper()
        .str.replace(" ", "_")
    )

    routes = routes[["ORIGIN", "DEST", "YEAR", "MONTH"]].dropna()

    routes["ORIGIN"] = routes["ORIGIN"].astype(str).str.strip().str.upper()
    routes["DEST"] = routes["DEST"].astype(str).str.strip().str.upper()

    routes = routes[routes["ORIGIN"] != routes["DEST"]]

    print(routes["YEAR"].unique())
    print(routes["MONTH"].unique())

    if month is not None:
        routes = routes[routes["MONTH"] == month]


    return routes


def build_graph(routes_csv: str, month: int = None) -> nx.DiGraph:
    airports = load_us_airports()
    routes = load_routes(routes_csv, month)

    airport_lookup = airports.set_index("iata_code").to_dict(orient="index")
    valid_airports = set(airport_lookup.keys())

    # Keep only routes where both airports exist in OurAirports
    routes = routes[
        routes["ORIGIN"].isin(valid_airports)
        & routes["DEST"].isin(valid_airports)
    ]

    G = nx.DiGraph()

    for iata, attrs in airport_lookup.items():
        G.add_node(
            iata,
            name=attrs.get("name"),
            city=attrs.get("municipality"),
            airport_type=attrs.get("type"),
            latitude=attrs.get("latitude_deg"),
            longitude=attrs.get("longitude_deg"),
        )

    for _, row in routes.iterrows():
        G.add_edge(row["ORIGIN"], row["DEST"])

    # Remove airports with no routes
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    return G


def display_graph(G: nx.DiGraph, year: int, month: int) -> None:
    pos = {
        airport: (
            data["longitude"],
            data["latitude"],
        )
        for airport, data in G.nodes(data=True)
        if pd.notna(data.get("longitude")) and pd.notna(data.get("latitude"))
    }

    # Keep only nodes with valid coordinates
    H = G.subgraph(pos.keys()).copy()

    plt.figure(figsize=(14, 9))

    nx.draw_networkx_edges(
        H,
        pos,
        arrows=True,
        arrowsize=6,
        alpha=0.2,
        width=0.6,
    )

    nx.draw_networkx_nodes(
        H,
        pos,
        node_size=75,
        node_color='#5DBB63',
    )

    nx.draw_networkx_labels(
        H,
        pos,
        font_size=10,
    )

    plt.title(f"U.S. Airport Route Graph — {year}-{month:02d}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")
    plt.tight_layout()
    plt.style.use('fast')
    
    plt.show()

def generate_monthly_plots():

    files = [
    r"month by month\T_T100 2018.csv",
    r"month by month\T_T100 2019.csv",
    r"month by month\T_T100 2020.csv",
    ]

    for file in files:

        print(f"\nProcessing {file}")

        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()

        months = sorted(df["MONTH"].unique())

        for month in months:

            print(f"  → Month {month}")

            G = build_graph(file, month=month)

            year = int(df["YEAR"].iloc[0])

            display_graph(G, year, month)

            filename = f"snapshot_{year}_{month:02d}.png"

            plt.savefig(filename, dpi=300)
            plt.close()

            print(f"    saved {filename}")


if __name__ == "__main__":
    # graph = build_graph(ROUTES_CSV, year=2026, month=1)
    # print(f"Airports: {graph.number_of_nodes()}")
    # print(f"Routes: {graph.number_of_edges()}")

    # display_graph(graph)
    # df = pd.read_csv("month by month/T_T100 2018.csv")
    # print(df["MONTH"].unique())
    # print(df["YEAR"].unique())

    G = build_graph("month by month/T_T100 2018.csv", month=1)
    display_graph(G, 2018, 1)
    plt.show()