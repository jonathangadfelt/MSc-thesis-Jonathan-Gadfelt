import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pypsa
import sys
import logging
from contextlib import redirect_stdout
import tqdm
from pathlib import Path
np.set_printoptions(suppress=True)


"  ____________________ LOAD ALL DATA ____________________ "

def load_all_data():
    data = {
        "solar": pd.read_csv("Data/pv_optimal_NOR_ESP.csv", sep=";", index_col=0, parse_dates=True),
        "onwind": pd.read_csv("Data/onshore_wind_1979-2017_NOR_ESP.csv", sep=";", index_col=0, parse_dates=True),
        "offwind": pd.read_csv("Data/offshore_wind_1979-2017_NOR.csv", sep=";", index_col=0, parse_dates=True),
        "demand": pd.read_csv("Data/load_data_actual_NOR_ES.csv", sep=";", index_col=0, parse_dates=True)
    }

    # Rename demand columns to standard naming
    data["demand"] = data["demand"].rename(columns={
        "NO_load_actual_entsoe_transparency": "NOR",
        "ES_load_actual_entsoe_transparency": "ESP"
    })
    data["demand"] = data["demand"].ffill().bfill()

    # Load and convert hydro inflow data to hourly MW
    def convert_hydro_to_hourly(path, label):
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
        df = df[df['date'].dt.strftime('%m-%d') != '02-29']
        df['Hourly_MW'] = round(df['Inflow [GWh]'] * 1000 / 24)  # Convert GWh to MW and distribute evenly across 24 hours
        df_hourly = pd.DataFrame({
            'datetime': df['date'].repeat(24) + pd.to_timedelta(list(range(24)) * len(df), unit='h'),
            label: df['Hourly_MW'].repeat(24).values
        })
        df_hourly = df_hourly.set_index('datetime')
        return df_hourly

    es_hourly = convert_hydro_to_hourly("Data/Hydro_Inflow_ES.csv", 'ESP')
    no_hourly = convert_hydro_to_hourly("Data/Hydro_Inflow_NO.csv", 'NOR')

    hydro_inflow = pd.concat([es_hourly, no_hourly], axis=1)
    data["hydro_inflow"] = hydro_inflow

    return data

" ____________________ COST DATA CLASS ____________________ "

class CostGeneration:
    def __init__(self, year: int = 2020):
        self.year = year
        self.costs, self.units = self.cost_data()

    def cost_data(self):
        url = f"https://raw.githubusercontent.com/PyPSA/technology-data/master/outputs/costs_{self.year}.csv"
        df = pd.read_csv(url, index_col=[0, 1])

        df.loc[df.unit.str.contains("/kW"), "value"] *= 1e3
        df.unit = df.unit.str.replace("/kW", "/MW")

        # Save units before dropping
        unit_df = df["unit"].copy()
        
        defaults = {
            "FOM": 0,
            "VOM": 0,
            "efficiency": 1,
            "fuel": 0,
            "investment": 0,
            "lifetime": 25,
            "CO2 intensity": 0,
            "discount rate": 0.07,
        }

        costs = df.value.unstack().fillna(defaults)

        costs.at["OCGT", "fuel"] = costs.at["gas", "fuel"]
        costs.at["CCGT", "fuel"] = costs.at["gas", "fuel"]
        costs.at["OCGT", "CO2 intensity"] = costs.at["gas", "CO2 intensity"]
        costs.at["CCGT", "CO2 intensity"] = costs.at["gas", "CO2 intensity"]

        #costs.at["onwind", "VOM"] = 2.06
        #costs.at["offwind", "VOM"] = 4.41

        costs["marginal_cost"] = costs["VOM"] + costs["fuel"] / costs["efficiency"]
        annuity = costs.apply(lambda x: self.annuity(x["discount rate"], x["lifetime"]), axis=1)
        costs["capital_cost"] = (annuity + costs["FOM"] / 100) * costs["investment"]


        return costs, unit_df

    @staticmethod
    def annuity(r, n):
        """ Calculate the annuity factor for an asset with lifetime n years and
        discount rate r """
        return r / (1.0 - 1.0 / (1.0 + r) ** n)

All_data = load_all_data()
Cost = CostGeneration(year=2025)

" ____________________ PRINT RESULTS FUNCTION ____________________ "

def print_Results(N):
    print("\nObjective value (MEUR):", round(N.objective / 1e6))
    print("\nInstalled generator capacities (MW):")
    print(N.generators.p_nom_opt.round(0).to_string())
    print("\nInstalled store energy capacities (MWh):")
    print(N.stores.e_nom_opt.round().to_string())
    print("\nInstalled hydro power capacity (MW):")
    print(N.storage_units.p_nom_opt.round().to_string())
    print("\nInstalled link power capacities (MW):")
    print(N.links.p_nom_opt.round().to_string())

    return



" ____________________ SILENT OPTIMIZE ____________________ "

def silent_optimize(network, solver_name="gurobi", solver_options=None):
    """
    Optimize a PyPSA network silently: suppress logs, progress bars, and Gurobi messages.
    
    Parameters:
    - network: PyPSA Network object
    - solver_name: Solver name (default: "gurobi")
    - solver_options: Dictionary of solver options (default: {"OutputFlag": 0})
    """
    if solver_options is None:
        solver_options = {"OutputFlag": 0}

    # Suppress PyPSA, Linopy, Gurobi logging output
    for name in ["pypsa", "linopy", "gurobipy"]:
        logging.getLogger(name).setLevel(logging.CRITICAL)

    # Monkey-patch tqdm to disable progress bars
    tqdm.tqdm = lambda *args, **kwargs: iter(args[0]) if args else iter([])

    # Redirect stdout to suppress Gurobi messages
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull):
            network.optimize(solver_name=solver_name, solver_options=solver_options)
            #print("Objective value:", network.objective)


" ____________________ PLOTTING FUNCTIONS ____________________ "

def plot_electricity_mix(network, save_plot=False, plot_title='Electricity mix', min_cap=10):
    # generator capacity by carrier
    gen_caps = (network.generators[["carrier","p_nom_opt"]]
                .groupby("carrier")["p_nom_opt"].sum())

    # add hydro StorageUnit power capacity (if present)
    if not network.storage_units.empty:
        hydro_cap = network.storage_units.loc[
            network.storage_units.carrier == "hydro", "p_nom_opt"
        ].sum()
        if hydro_cap > 0:
            gen_caps = gen_caps.add(pd.Series({"hydro": hydro_cap}), fill_value=0)

    # drop load shedding and tiny slices
    ls_names = {c for c in gen_caps.index if c.lower().replace("_"," ") == "load shedding"}
    gen_caps = gen_caps.drop(labels=list(ls_names), errors="ignore")
    gen_caps = gen_caps[gen_caps > min_cap]

    if gen_caps.empty:
        print("Nothing to plot.")
        return

    labels = gen_caps.index.tolist()
    sizes  = gen_caps.values.tolist()

    # use carrier colors from the network (keeps order aligned with labels)
    pie_colors = (network.carriers.reindex(labels)["color"].tolist()
                  if "color" in network.carriers else None)

    plt.figure()
    plt.pie(sizes, labels=labels, autopct='%1.1f%%',
            colors=pie_colors, wedgeprops={'linewidth': 0})
    plt.axis('equal')
    plt.title(plot_title, y=1.07)
    plt.tight_layout()
    if save_plot:
        Path("Plots").mkdir(exist_ok=True)
        plt.savefig("Plots/electricity_mix.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_generator_capacity(results_df, generator, type, region, d_year_exp, h_year_exp):
    """
    Plot capacity of a selected generator over years as bar and line plots.

    Parameters
    ----------
    results_df : pd.DataFrame
        MultiIndex DataFrame with (component, technology) columns, years as index.
    generator : str
        Name of the generator technology, e.g. 'onwind', 'solar'.
    region : str
        Region string for plot title.
    d_year_exp : int
        Demand year used in the experiment.
    h_year_exp : int
        Hydro year used in the experiment.
    """
    # Bar plot
    results_df[(type, generator)].plot(
        kind='bar', figsize=(10, 5),
        title=f'{generator.capitalize()} Capacity Over Years - r: {region}, d: {d_year_exp}, h: {h_year_exp}'
    )
    plt.xlabel('Year')
    plt.ylabel('Capacity (MW)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Line plot
    plt.figure(figsize=(10, 5))
    plt.plot(results_df.index, results_df[(type, generator)].values, marker='o')
    plt.xlabel("Year")
    plt.ylabel(f"{generator.capitalize()} Capacity (MW)")
    plt.title(f"{generator.capitalize()} Capacity Over Years - r: {region}, d: {d_year_exp}, h: {h_year_exp}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


" ____________________ LOAD NETWORK RESULTS ____________________ "

def load_networks(folder_name: str, weather_years: list[int], region: str = "ESP", ext: str = ".nc"):
    """
    Load PyPSA networks for given weather_years from Network_results/<folder_name>.
    Keys in the dict are the years (int). Returns dict {year: Network}.
    """
    path = Path.cwd() / "Network_results" / folder_name
    if not path.exists():
        raise FileNotFoundError(f"Folder not found: {path}")

    files = sorted(path.glob(f"*{ext}"))
    if not files:
        print(f"No {ext} files found in {path}")
        return {}

    if len(files) != len(weather_years):
        print(f"Warning: {len(files)} files but {len(weather_years)} years — check alignment")

    networks = {}
    for year, f in zip(weather_years, files):
        n = pypsa.Network(str(f))
        n.name = f.stem       # file name without .nc
        networks[year] = n
        print(f"Loaded N_{year} from {f.name}")

    return networks



