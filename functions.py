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
from typing import Sequence, Any
from pypsa import Network
import logging
from types import MethodType
logger = logging.getLogger(__name__)


np.set_printoptions(suppress=True)

region = "ESP"          # Region for hydro inflow data


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

" ____________________ ROLLING HORIZON ____________________ "
# NEW VERSION: 
def optimize_with_rolling_horizon_collect(self, snapshots=None, horizon=100, overlap=0, **kwargs):
    """
    Custom rolling horizon optimization that collects objectives
    and stores them in n.attrs.
    """
    n = self  

    if snapshots is None:
        snapshots = n.snapshots

    if horizon <= overlap:
        raise ValueError("overlap must be smaller than horizon")

    objs, runs = [], []

    for i in range(0, len(snapshots), horizon - overlap):
        start = i
        end = min(len(snapshots), i + horizon)
        sns = snapshots[start:end]

        if i:
            if not n.stores.empty:
                n.stores.e_initial = n.stores_t.e.loc[snapshots[start - 1]]
            if not n.storage_units.empty:
                n.storage_units.state_of_charge_initial = (
                    n.storage_units_t.state_of_charge.loc[snapshots[start - 1]]
                )

        status, condition = n.optimize(sns, **kwargs)

        if status != "ok":
            logger.warning(
                "Optimization failed with status %s and condition %s",
                status, condition
            )

        if hasattr(n, "objective"):
            objs.append(n.objective)
            runs.append({"run": i + 1, "start": sns[0], "end": sns[-1]})

    # Store results in network attributes
    n.generators.attrs["rolling_objectives"] = objs
    n.generators.attrs["rolling_runs"] = runs

    return n


def rh_pf_test_yearly(test_year):    
    N_dispatch_class = Build_dispatch_network(
        opt_capacities_df=opt_capacities_df.loc["75%"],
        weather_year=test_year, hydro_year=h_year_dispatch, demand_year=d_year_dispatch,
        data=All_data, cost_data=Cost, setup=setup_dispatch
    )
    N_dispatch = N_dispatch_class.network
    silent_optimize(N_dispatch)

    N_rlh_class = Build_dispatch_network(
        opt_capacities_df=opt_capacities_df.loc["75%"],
        weather_year=test_year, hydro_year=h_year_dispatch, demand_year=d_year_dispatch,
        data=All_data, cost_data=Cost, setup=setup_dispatch
    )
    N_rlh = N_rlh_class.network

    # Patch onto the Network itself
    N_rlh.optimize_with_rolling_horizon = MethodType(optimize_with_rolling_horizon_collect, N_rlh)

    # Run the rolling-horizon optimization
    N_rlh.optimize_with_rolling_horizon(
        snapshots=N_rlh.snapshots,
        horizon=24 * 7,
        overlap=0,
        solver_name="gurobi",
        solver_options={"OutputFlag": 0},
        assign_all_duals=True
    )

    return N_dispatch, N_rlh, N_rlh.generators.attrs["rolling_objectives"]


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

def extract_total_dispatch_profiles(network_pf, network_rh):    # Energy Dipatch pr carrier 
    records = []

    for model_label, network in zip(["Perfect Foresight", "Rolling Horizon"], [network_pf, network_rh]):
        # Generators
        for gen in network.generators.index:
            total_dispatch = round(network.generators_t.p[gen].sum() / 1e3, 1)  # Convert MWh to GWh
            records.append({
                "Model": model_label,
                "Component": gen,
                "Carrier": network.generators.at[gen, "carrier"],
                "Type": "generator",
                "Total Dispatch [GWh]": total_dispatch
            })

        # Hydro StorageUnits
        for hydro in network.storage_units.index:
            if network.storage_units.at[hydro, "carrier"] == "hydro":
                total_dispatch = round(network.storage_units_t.p_dispatch[hydro].sum() / 1e3, 1)  # Convert MWh to GWh
                records.append({
                    "Model": model_label,
                    "Component": hydro,
                    "Carrier": "hydro",
                    "Type": "hydro storage dispatch",
                    "Total Dispatch [GWh]": total_dispatch
                })

        # Battery Links: charge (p0), discharge (p1)
        if {"battery charge", "battery discharge"}.issubset(network.links.index):
            total_charge = round(network.links_t.p1["battery charge"].sum() / 1e3, 1)
            total_discharge = round(network.links_t.p0["battery discharge"].sum() / 1e3, 1)

            records.append({
                "Model": model_label,
                "Component": "battery charge",
                "Carrier": "battery",
                "Type": "charge",
                "Total Dispatch [GWh]": total_charge
            })
            records.append({
                "Model": model_label,
                "Component": "battery discharge",
                "Carrier": "battery",
                "Type": "discharge",
                "Total Dispatch [GWh]": total_discharge
            })

    dispatch_df = pd.DataFrame(records)
    # Add total sum for each model
    total_rows = []
    for model in dispatch_df["Model"].unique():
        total = dispatch_df[dispatch_df["Model"] == model]["Total Dispatch [GWh]"].sum()
        total_rows.append({
            "Model": model,
            "Component": "Total",
            "Carrier": "",
            "Type": "",
            "Total Dispatch [GWh]": round(total, 1)
        })
    dispatch_df = pd.concat([dispatch_df, pd.DataFrame(total_rows)], ignore_index=True)
    return dispatch_df.sort_values(by=["Component", "Carrier"])

def make_dispatch_diff_table(dispatch_df):         # Energi difference between PF and RH from above function
    """
    Create PF vs RH comparison from extract_total_dispatch_profiles output.
    Keeps each Component separate; no premature rounding.
    """
    # Remove any total rows if present
    df = dispatch_df[dispatch_df["Component"] != "Total"].copy()
    df = dispatch_df[dispatch_df["Component"] != "Total"].copy()

    # Pivot so PF and RH are side by side per Component
    pivot = df.pivot_table(index=["Component", "Carrier", "Type"],
                           columns="Model",
                           values="Total Dispatch [GWh]",
                           aggfunc="sum").reset_index()

    # Ensure both PF and RH columns exist
    if "Perfect Foresight" not in pivot.columns:
        pivot["Perfect Foresight"] = 0.0
    if "Rolling Horizon" not in pivot.columns:
        pivot["Rolling Horizon"] = 0.0

    # Rename
    pivot = pivot.rename(columns={
        "Perfect Foresight": "PF [GWh]",
        "Rolling Horizon": "RH [GWh]"
    })

    # Differences
    pivot["Diff [GWh]"] = pivot["RH [GWh]"] - pivot["PF [GWh]"]
    pivot["Percentage Difference [%]"] = pivot.apply(
        lambda r: 0.0 if r["PF [GWh]"] == 0 and r["RH [GWh]"] == 0
        else float("inf") if r["PF [GWh]"] == 0
        else 100 * (r["RH [GWh]"] - r["PF [GWh]"]) / r["PF [GWh]"],
        axis=1
    )

    # Round for display
    for col in ["PF [GWh]", "RH [GWh]", "Diff [GWh]"]:
        pivot[col] = pivot[col].round(1)
    pivot["Percentage Difference [%]"] = pivot["Percentage Difference [%]"].round(1)

    return pivot.sort_values(["Component", "Carrier"]).reset_index(drop=True)

def tot_cost_N(N):
    return (N.buses_t['marginal_price'].iloc[:, 0] * 
               N.loads_t.p_set['load']).sum() 

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
            network.optimize(solver_name=solver_name, assign_all_duals=True, solver_options=solver_options)
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

def plot_dispatch_old(network, colors=None, save_plots=False, start_hour=0, duration_hours=7 * 24, title="Dispatch"):
    import matplotlib.pyplot as plt

    generators = network.generators.index
    storage_units = network.storage_units

    end_hour = start_hour + duration_hours

    plt.figure(figsize=(10, 5))

    # Plot generator dispatch
    for generator in generators:
        name = generator
        if network.generators.p_nom_opt.get(generator, 0) > 10:
            plt.plot(
                network.generators_t.p[generator][start_hour:end_hour],
                label=name,
                color=colors.get(name, None) if colors else None
            )

    # Plot hydro dispatch
    hydro_mask = (storage_units.carrier == "hydro") & (storage_units.bus == "electricity bus")
    if hydro_mask.any():
        for idx in storage_units[hydro_mask].index:
            plt.plot(
                network.storage_units_t.p_dispatch[idx][start_hour:end_hour],
                label="hydro dispatch",
                color=colors.get("hydro", "green") if colors else "green",
                linestyle="--"
            )

    # Plot load
    if "load" in network.loads.index:
        plt.plot(
            network.loads_t.p_set["load"][start_hour:end_hour],
            label="load",
            color="black",
            linestyle=":"
        )

    # X-axis formatting
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Labels and title
    plt.title(f'Dispatch {title}', y=1.07)
    plt.ylabel('Generation in MWh')
    plt.grid(True, which='major', alpha=0.25)
    plt.legend()
    if save_plots:
        plt.savefig(f'./Plots/dispatch_{start_hour}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_dispatch(network, colors=None, save_plots=False,
                  start_hour=0, duration_hours=7 * 24, interval=None,
                  title="Dispatch"):
    """
    Plot dispatch for a network.

    Parameters
    ----------
    network : pypsa.Network-like
    colors : dict or None
        mapping from component name -> matplotlib color
    save_plots : bool
    start_hour : int
        only used when `interval` is None (iloc-based slicing)
    duration_hours : int
        only used when `interval` is None (iloc-based slicing)
    interval : tuple (start, end) or None
        If provided, slices all time series with .loc[start:end] using
        the network time index. start/end can be pd.Timestamp, string, etc.
    title : str
    """
    import matplotlib.pyplot as plt

    # Determine slicing mode
    use_label_slice = interval is not None
    if use_label_slice:
        start, end = interval

    end_hour = start_hour + duration_hours

    plt.figure(figsize=(12, 5))

    # Plot generator dispatch
    for generator in network.generators.index:
        # only plot sizable optimized capacities (keeps behaviour similar to your original)
        p_nom_opt = network.generators.p_nom_opt.get(generator, 0)
        if p_nom_opt <= 10:
            continue

        series = network.generators_t.p[generator]

        if use_label_slice:
            series_slice = series.loc[start:end]
        else:
            series_slice = series.iloc[start_hour:end_hour]

        # defensive: skip empty slices
        if series_slice.empty:
            continue

        plt.plot(series_slice.index, series_slice.values,
                 label=generator,
                 color=(colors.get(generator) if colors else None))

    # Plot hydro dispatch (storage_units_t.p_dispatch)
    storage_units = network.storage_units
    hydro_mask = (storage_units.carrier == "hydro") & (storage_units.bus == "electricity bus")
    if hydro_mask.any():
        for idx in storage_units[hydro_mask].index:
            # some pypsa versions: storage_units_t.p_dispatch or storage_units_t.p
            # prefer p_dispatch if available
            if "p_dispatch" in getattr(network.storage_units_t, "columns", []):
                series = network.storage_units_t.p_dispatch[idx]
            else:
                # fallback if different naming
                series = network.storage_units_t.p[idx]

            if use_label_slice:
                series_slice = series.loc[start:end]
            else:
                series_slice = series.iloc[start_hour:end_hour]

            if series_slice.empty:
                continue

            plt.plot(series_slice.index, series_slice.values,
                     label=f"{idx} (hydro dispatch)",
                     color=(colors.get("hydro") if colors else "green"),
                     linestyle="--")

    # Plot load (if present)
    # loads_t.p_set is usually a DataFrame with columns being load names; we used "load"
    try:
        load_series = network.loads_t.p_set["load"]
    except Exception:
        # try the case where loads_t.p_set has a different structure
        # pick the first column if "load" not present
        if hasattr(network.loads_t.p_set, "columns") and len(network.loads_t.p_set.columns) > 0:
            load_series = network.loads_t.p_set.iloc[:, 0]
        else:
            load_series = None

    if load_series is not None:
        if use_label_slice:
            load_slice = load_series.loc[start:end]
        else:
            load_slice = load_series.iloc[start_hour:end_hour]

        if not load_slice.empty:
            plt.plot(load_slice.index, load_slice.values,
                     label="load", color="black", linestyle=":")

    # Formatting
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.title(f'{title}', y=1.07)
    plt.ylabel('Generation in MWh')
    plt.grid(True, which='major', alpha=0.25)
    plt.legend()
    if save_plots:
        # derive a simple filename from interval or start_hour
        if use_label_slice:
            s = str(start).replace(":", "-")
            e = str(end).replace(":", "-")
            fname = f'./Plots/dispatch_{s}_to_{e}.png'
        else:
            fname = f'./Plots/dispatch_{start_hour}_{end_hour}.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()

def plot_dispatch_bat(network, colors=None, save_plots=False, start_hour=0, duration_hours=7 * 24, title="Dispatch"):
    import matplotlib.pyplot as plt

    generators = network.generators.index
    storage_units = network.storage_units
    links = network.links

    end_hour = start_hour + duration_hours

    plt.figure(figsize=(10, 5))

    # Plot generator dispatch
    for generator in generators:
        if getattr(network.generators, "p_nom_opt", network.generators.p_nom).get(generator, 0) > 10:
            plt.plot(
                network.generators_t.p[generator][start_hour:end_hour],
                label=generator,
                color=colors.get(generator, None) if colors else None
            )

    # Plot hydro dispatch
    hydro_mask = (storage_units.carrier == "hydro") & (storage_units.bus == "electricity bus")
    if hydro_mask.any():
        for idx in storage_units[hydro_mask].index:
            plt.plot(
                network.storage_units_t.p_dispatch[idx][start_hour:end_hour],
                label="hydro dispatch",
                color=colors.get("hydro", "green") if colors else "green",
                linestyle="--"
            )

    # Plot battery links (charge/discharge)
    if {"battery charge", "battery discharge"}.issubset(links.index):
        plt.plot(
            network.links_t.p0["battery charge"][start_hour:end_hour],
            label="battery charge (p0)",
            color=colors.get("battery charge", "blue") if colors else "blue",
            linestyle=":"
        )
        plt.plot(
            network.links_t.p1["battery discharge"][start_hour:end_hour],
            label="battery discharge (p1)",
            color=colors.get("battery discharge", "red") if colors else "red",
            linestyle=":"
        )

    # Plot load
    if "load" in network.loads.index:
        plt.plot(
            network.loads_t.p_set["load"][start_hour:end_hour],
            label="load",
            color="black",
            linestyle=":"
        )

    # X-axis formatting
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Labels and title
    plt.title(f'Dispatch {title}', y=1.07)
    plt.ylabel('Generation in MWh')
    plt.grid(True, which='major', alpha=0.25)
    plt.legend()
    if save_plots:
        plt.savefig(f'./Plots/dispatch_{start_hour}.png', dpi=300, bbox_inches='tight')
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

def plot_normalized_hydro(network_pf, network_rh, interval=None):
    # years
    year_pf = network_pf.snapshots[0].year
    year_rh = network_rh.snapshots[0].year

    # inflow year to use (e.g. 2007)
    inflow_src = All_data["hydro_inflow"][region]
    inflow_year = inflow_src[inflow_src.index.year == h_year_dispatch].copy()

    # remap inflow calendar year to the networks' years and align to snapshots
    def align_inflow_to_network(inflow, net_year, snaps):
        s = inflow.copy()
        s.index = s.index.map(lambda t: t.replace(year=net_year))
        # align to snapshots; allow small gaps
        s = s.reindex(snaps).interpolate(limit_direction="both")
        return s

    hydro_inflow_pf = align_inflow_to_network(inflow_year, year_pf, network_pf.snapshots)
    hydro_inflow_rh = align_inflow_to_network(inflow_year, year_rh, network_rh.snapshots)

    # dispatch & SOC (already on snapshot index)
    dispatch_pf = network_pf.storage_units_t.p["Reservoir hydro storage"]
    soc_pf = network_pf.storage_units_t.state_of_charge["Reservoir hydro storage"]
    dispatch_rh = network_rh.storage_units_t.p["Reservoir hydro storage"]
    soc_rh = network_rh.storage_units_t.state_of_charge["Reservoir hydro storage"]

    # optional date interval
    if interval is not None:
        start, end = interval
        hydro_inflow_pf = hydro_inflow_pf.loc[start:end]
        dispatch_pf = dispatch_pf.loc[start:end]
        soc_pf = soc_pf.loc[start:end]

        hydro_inflow_rh = hydro_inflow_rh.loc[start:end]
        dispatch_rh = dispatch_rh.loc[start:end]
        soc_rh = soc_rh.loc[start:end]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # RH
    axes[0].plot(dispatch_rh / dispatch_rh.max(), label="Hydro Dispatch", color='#6baed6')
    axes[0].plot(hydro_inflow_rh / hydro_inflow_rh.max(), label=f"Hydro Inflow {region} (year {h_year_dispatch})", color='tab:blue')
    axes[0].plot(soc_rh / (12700*1300), label="State of Charge", color='tab:green')
    axes[0].set_title(f"Normalized Hydro - Rolling Horizon - {region} {year_rh}")
    axes[0].set_ylabel("Normalized"); axes[0].set_ylim(0, 1.1); axes[0].grid(True); axes[0].legend(loc='upper right')

    # PF
    axes[1].plot(dispatch_pf / dispatch_pf.max(), label="Hydro Dispatch", color='#6baed6')
    axes[1].plot(hydro_inflow_pf / hydro_inflow_pf.max(), label=f"Hydro Inflow {region} (year {h_year_dispatch})", color='tab:blue')
    axes[1].plot(soc_pf / (12700*1300), label="State of Charge", color='tab:green')
    axes[1].set_title(f"Normalized Hydro - Perfect Foresight - {region} {year_pf}")
    axes[1].set_xlabel("Date"); axes[1].set_ylabel("Normalized"); axes[1].set_ylim(0, 1.1); axes[1].grid(True); axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

import matplotlib.dates as mdates

def plot_hydro_old(network_pf, network_rh, interval=None, normalized=True, show_inflow=True, show_dispatch=True, show_soc=True, same_axes=False):
    # years
    year_pf = network_pf.snapshots[0].year
    year_rh = network_rh.snapshots[0].year

    # inflow source year -> remap to network years and align to snapshots
    inflow_src = All_data["hydro_inflow"][region]
    inflow_year = inflow_src[inflow_src.index.year == h_year_dispatch].copy()

    def align_inflow_to_network(inflow, net_year, snaps):
        s = inflow.copy()
        s.index = s.index.map(lambda t: t.replace(year=net_year))
        return s.reindex(snaps).interpolate(limit_direction="both")

    inflow_pf = network_pf.storage_units_t.inflow["Reservoir hydro storage"]
    inflow_rh = network_rh.storage_units_t.inflow["Reservoir hydro storage"]

    # dispatch & SOC
    disp_pf = network_pf.storage_units_t.p["Reservoir hydro storage"]
    soc_pf  = network_pf.storage_units_t.state_of_charge["Reservoir hydro storage"]
    disp_rh = network_rh.storage_units_t.p["Reservoir hydro storage"]
    soc_rh  = network_rh.storage_units_t.state_of_charge["Reservoir hydro storage"]

    # optional interval (use datetime strings or Timestamps)
    if interval is not None:
        start, end = interval
        inflow_pf = inflow_pf.loc[start:end];    disp_pf = disp_pf.loc[start:end];    soc_pf = soc_pf.loc[start:end]
        inflow_rh = inflow_rh.loc[start:end];    disp_rh = disp_rh.loc[start:end];    soc_rh = soc_rh.loc[start:end]

    # normalization helpers
    def safe_norm(s):
        m = float(s.max()) if len(s) else 0.0
        return s / m if m > 0 else s*0.0

    def soc_norm(s):
        denom = 12700 * 1300
        return s / denom if denom else s

    # choose data as normalized or raw
    disp_pf_plot = safe_norm(disp_pf) if normalized else disp_pf
    disp_rh_plot = safe_norm(disp_rh) if normalized else disp_rh
    inflow_pf_plot = safe_norm(inflow_pf) if normalized else inflow_pf
    inflow_rh_plot = safe_norm(inflow_rh) if normalized else inflow_rh
    soc_pf_plot  = soc_norm(soc_pf) if normalized else soc_pf
    soc_rh_plot  = soc_norm(soc_rh) if normalized else soc_rh

    # plotting
    if same_axes:
        fig, ax = plt.subplots(figsize=(14, 5))

        if show_dispatch:
            ax.plot(disp_pf_plot, label="Hydro Dispatch PF", color='#6baed6', linestyle='-')
            ax.plot(disp_rh_plot, label="Hydro Dispatch RH", color='#6baed6', linestyle='--')
        if show_inflow:
            ax.plot(inflow_pf_plot, label=f"Inflow PF (yr {h_year_dispatch})", color='tab:blue', linestyle='-')
            ax.plot(inflow_rh_plot, label=f"Inflow RH (yr {h_year_dispatch})", color='tab:blue', linestyle='--')
        if show_soc:
            ax.plot(soc_pf_plot, label="SOC PF", color='tab:green', linestyle='-')
            ax.plot(soc_rh_plot, label="SOC RH", color='tab:green', linestyle='--')

        ax.set_title(f"{'Normalized ' if normalized else ''}Hydro - PF vs RH - {region} ({year_pf} / {year_rh})")
        ax.set_ylabel("Normalized" if normalized else "MWh")
        ax.set_xlabel("Date")
        if normalized:
            ax.set_ylim(0, 1.1)
        ax.grid(True)
        ax.legend(loc='upper right')
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
        plt.xticks(rotation=30)
        plt.tight_layout()
        return fig, ax

    else:
        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

        # RH (top)
        if show_dispatch: axes[0].plot(disp_rh_plot, label="Hydro Dispatch RH", color='#6baed6')
        if show_inflow:   axes[0].plot(inflow_rh_plot, label=f"Hydro Inflow RH (yr {h_year_dispatch})", color='tab:blue')
        if show_soc:      axes[0].plot(soc_rh_plot, label="State of Charge RH", color='tab:green')
        axes[0].set_title(f"{'Normalized ' if normalized else ''}Hydro - Rolling Horizon - {region} {year_rh}")
        axes[0].set_ylabel("Normalized" if normalized else "MWh")
        if normalized: axes[0].set_ylim(0, 1.1)
        axes[0].grid(True); axes[0].legend(loc='upper right')

        # PF (bottom)
        if show_dispatch: axes[1].plot(disp_pf_plot, label="Hydro Dispatch PF", color='#6baed6')
        if show_inflow:   axes[1].plot(inflow_pf_plot, label=f"Hydro Inflow PF (yr {h_year_dispatch})", color='tab:blue')
        if show_soc:      axes[1].plot(soc_pf_plot, label="State of Charge PF", color='tab:green')
        axes[1].set_title(f"{'Normalized ' if normalized else ''}Hydro - Perfect Foresight - {region} {year_pf}")
        axes[1].set_xlabel("Date"); axes[1].set_ylabel("Normalized" if normalized else "MWh")
        if normalized: axes[1].set_ylim(0, 1.1)
        axes[1].grid(True); axes[1].legend(loc='upper right')

        axes[1].xaxis.set_major_locator(mdates.AutoDateLocator())
        axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
        plt.xticks(rotation=30)
        plt.tight_layout()
        return fig, axes

def plot_hydro(network_pf, network_rh, interval=None, normalized=True, show_inflow=True, show_dispatch=True, show_soc=True, same_axes=False):
    # years
    year_pf = network_pf.snapshots[0].year
    year_rh = network_rh.snapshots[0].year

    # inflow source year -> remap to network years and align to snapshots
    inflow_src = All_data["hydro_inflow"][region]
    inflow_year = inflow_src[inflow_src.index.year == h_year_dispatch].copy()

    def align_inflow_to_network(inflow, net_year, snaps):
        s = inflow.copy()
        s.index = s.index.map(lambda t: t.replace(year=net_year))
        return s.reindex(snaps).interpolate(limit_direction="both")

    inflow_pf = network_pf.storage_units_t.inflow["Reservoir hydro storage"]
    inflow_rh = network_rh.storage_units_t.inflow["Reservoir hydro storage"]

    # dispatch & SOC
    disp_pf = network_pf.storage_units_t.p["Reservoir hydro storage"]
    soc_pf  = network_pf.storage_units_t.state_of_charge["Reservoir hydro storage"]
    disp_rh = network_rh.storage_units_t.p["Reservoir hydro storage"]
    soc_rh  = network_rh.storage_units_t.state_of_charge["Reservoir hydro storage"]

    # optional interval (use datetime strings or Timestamps)
    if interval is not None:
        start, end = interval
        inflow_pf = inflow_pf.loc[start:end];    disp_pf = disp_pf.loc[start:end];    soc_pf = soc_pf.loc[start:end]
        inflow_rh = inflow_rh.loc[start:end];    disp_rh = disp_rh.loc[start:end];    soc_rh = soc_rh.loc[start:end]

    # normalization helpers
    def safe_norm(s):
        m = float(s.max()) if len(s) else 0.0
        return s / m if m > 0 else s*0.0

    def soc_norm(s):
        denom = 12700 * 1300
        return s / denom if denom else s

    # choose data as normalized or raw
    disp_pf_plot = safe_norm(disp_pf) if normalized else disp_pf
    disp_rh_plot = safe_norm(disp_rh) if normalized else disp_rh
    inflow_pf_plot = safe_norm(inflow_pf) if normalized else inflow_pf
    inflow_rh_plot = safe_norm(inflow_rh) if normalized else inflow_rh
    soc_pf_plot  = soc_norm(soc_pf) if normalized else soc_pf
    soc_rh_plot  = soc_norm(soc_rh) if normalized else soc_rh

    # plotting
    if same_axes:
        fig, ax = plt.subplots(figsize=(12, 5))

        if show_dispatch:
            ax.plot(disp_pf_plot, label="Hydro Dispatch PF", color='#6baed6', linestyle='-')
            ax.plot(disp_rh_plot, label="Hydro Dispatch RH", color='#6baed6', linestyle='--')
        if show_inflow:
            ax.plot(inflow_pf_plot, label=f"Inflow PF (yr {h_year_dispatch})", color='tab:blue', linestyle='-')
            ax.plot(inflow_rh_plot, label=f"Inflow RH (yr {h_year_dispatch})", color='tab:blue', linestyle='--')
        if show_soc:
            ax.plot(soc_pf_plot, label="SOC PF", color='tab:green', linestyle='-')
            ax.plot(soc_rh_plot, label="SOC RH", color='tab:green', linestyle='--')

        ax.set_title(f"{'Normalized ' if normalized else ''}Hydro - PF vs RH - {region} ({year_pf} / {year_rh})")
        ax.set_ylabel("Normalized" if normalized else "MWh")
        ax.set_xlabel("Date")
        if normalized:
            ax.set_ylim(0, 1.1)
        ax.grid(True)
        ax.legend(loc='upper right')
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
        plt.xticks(rotation=30)
        plt.tight_layout()
        return fig, ax

    else:
        fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

        # RH (top)
        if show_dispatch: axes[0].plot(disp_rh_plot, label="Hydro Dispatch RH", color='#6baed6')
        if show_inflow:   axes[0].plot(inflow_rh_plot, label=f"Hydro Inflow RH (yr {h_year_dispatch})", color='tab:blue')
        if show_soc:      axes[0].plot(soc_rh_plot, label="State of Charge RH", color='tab:green')
        axes[0].set_title(f"{'Normalized ' if normalized else ''}Hydro - Rolling Horizon - {region} {year_rh}")
        axes[0].set_ylabel("Normalized" if normalized else "MWh")
        if normalized: axes[0].set_ylim(0, 1.1)
        axes[0].grid(True); axes[0].legend(loc='upper right')

        # PF (bottom)
        if show_dispatch: axes[1].plot(disp_pf_plot, label="Hydro Dispatch PF", color='#6baed6')
        if show_inflow:   axes[1].plot(inflow_pf_plot, label=f"Hydro Inflow PF (yr {h_year_dispatch})", color='tab:blue')
        if show_soc:      axes[1].plot(soc_pf_plot, label="State of Charge PF", color='tab:green')
        axes[1].set_title(f"{'Normalized ' if normalized else ''}Hydro - Perfect Foresight - {region} {year_pf}")
        axes[1].set_xlabel("Date"); axes[1].set_ylabel("Normalized" if normalized else "MWh")
        if normalized: axes[1].set_ylim(0, 1.1)
        axes[1].grid(True); axes[1].legend(loc='upper right')

        axes[1].xaxis.set_major_locator(mdates.AutoDateLocator())
        axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
        plt.xticks(rotation=30)
        plt.tight_layout()
        return fig, axes

def plot_extreme_period(networks_by_year,  # {year: Network}
                                                 extreme_periods_by_year,  # {year: [objects with .period.left/.right]}
                                                 region,
                                                 bus_name="electricity bus",
                                                 standard_year=2018,
                                                 periods_per_year=1):
    # 1) compute max price & time per year (dicts)
    max_price = {}
    max_time  = {}
    for y, n in networks_by_year.items():
        if bus_name not in n.buses_t.marginal_price.columns: 
            continue
        mp = n.buses_t.marginal_price[bus_name]
        if mp.empty: 
            continue
        idx = mp.values.argmax()
        max_price[y] = float(mp.iloc[idx])
        t = mp.index[idx]
        try:
            max_time[y] = t.replace(year=standard_year)
        except ValueError:
            # Feb 29 guard
            max_time[y] = t.replace(year=standard_year, day=28)

    # 2) build plot data
    lines, dots = [], []
    for y in sorted(extreme_periods_by_year.keys()):
        periods = extreme_periods_by_year.get(y, [])
        if not periods or y not in max_price:   # skip years without a period
            continue

        # dot at yearly max
        dots.append((max_time[y], max_price[y]))

        # lines for the year's periods at that y-level
        for p in periods[:periods_per_year]:
            pobj = getattr(p, "period", p)
            s, e = pobj.left, pobj.right
            try:
                s = s.replace(year=standard_year)
                e = e.replace(year=standard_year)
            except ValueError:
                if s.month == 2 and s.day == 29: s = s.replace(year=standard_year, day=28)
                if e.month == 2 and e.day == 29: e = e.replace(year=standard_year, day=28)
            lines.append((s, e, max_price[y]))

    # 3) plot
    plt.figure(figsize=(12, 6))
    first = True
    for s, e, price in lines:
        plt.plot([s, e], [price, price], color='grey', linewidth=2,
                 label='Extreme Periods' if first else "")
        first = False

    first = True
    for t, price in dots:
        plt.scatter(t, price, color='red', zorder=5,
                    label='Max Marginal Prices' if first else "")
        first = False

    plt.title(f"Extreme Periods and Max Marginal Prices ({region}) weather years 1979–2017")
    plt.xlabel(f"Time of Year (standardized to {standard_year})")
    plt.ylabel("Marginal Price (EUR/MWh)")
    plt.grid(True, alpha=0.3)
    if lines or dots:
        plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_marginal_prices(network_pf, network_rh, month=None, interval=None):
    # Extract marginal prices (first bus column)
    mc_pf = network_pf.buses_t['marginal_price'].iloc[:, 0]
    mc_rh = network_rh.buses_t['marginal_price'].iloc[:, 0]

    # Apply month filter
    if month is not None:
        mc_pf = mc_pf[mc_pf.index.month == month]
        mc_rh = mc_rh[mc_rh.index.month == month]

    # Apply interval filter
    if interval is not None:
        start, end = interval
        mc_pf = mc_pf.loc[start:end]
        mc_rh = mc_rh.loc[start:end]

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(mc_pf, label="Marginal prices PF", linestyle="--")
    plt.plot(mc_rh, label="Marginal prices RH", linestyle=":")
    plt.xlabel("Time")
    plt.ylabel("Marginal Price [EUR/MWh]")
    plt.title("Marginal Prices Comparison PF vs RH")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Cost calculations
    cost_pf = (network_pf.buses_t['marginal_price'].iloc[:, 0] * 
               network_pf.loads_t.p_set['load']).sum() / 1e6
    cost_rh = (network_rh.buses_t['marginal_price'].iloc[:, 0] * 
               network_rh.loads_t.p_set['load']).sum() / 1e6

    print(f"Sum of marginal prices (PF): {mc_pf.sum():.1f}")
    print(f"Sum of marginal prices (RH): {mc_rh.sum():.1f}")
    print(f"Total cost (PF) [MEUR]: {cost_pf:.2f}")
    print(f"Total cost (RH) [MEUR]: {cost_rh:.2f}")
    print(f"Total cost difference PF minus RH [MEUR]: {cost_rh - cost_pf:.2f}")



" ____________________ Unique prices ____________________ "
def unique_prices(network):
    prices = network.buses_t.marginal_price["electricity bus"].unique().round(1)
    return sorted(float(p) for p in prices)


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

" ____________________ COST RECOVERY ____________________ "
# big L is because Load shedding has been renamed to load shedding in the newer version
def cost_recovery_yearly_big_L(network_pf, network_rh):
    """
    Compute yearly cost recovery for generators, hydro (StorageUnit), and battery.
    """
    results = []

    def capex_lookup(name):
        # Safe lookup in Cost.costs; return 0 if missing
        try:
            return float(Cost.costs.at[name, "capital_cost"])
        except Exception:
            return 0.0

    for model_label, network in (("Perfect Foresight", network_pf), ("Rolling Horizon", network_rh)):

        # ---------- Generators ----------
        for gen in network.generators.index:
            carrier = network.generators.at[gen, "carrier"]
            bus = network.generators.at[gen, "bus"]
            p_nom = float(network.generators.at[gen, "p_nom"])
            marginal_cost = float(network.generators.at[gen, "marginal_cost"])

            # 'Load shedding' has no CAPEX; otherwise read from Cost.costs
            if gen == "Load shedding":
                capital_cost = 0.0
            else:
                capital_cost = capex_lookup(gen)

            dispatch = network.generators_t.p[gen]
            prices = network.buses_t.marginal_price[bus]

            revenue = (dispatch * prices).sum()
            production_cost = (dispatch * marginal_cost).sum()
            capex = p_nom * capital_cost
            profit = revenue - (capex + production_cost)

            results.append({
                "Model": model_label,
                "name": gen,
                "carrier": carrier,
                "revenue [MEUR]": round(revenue / 1e6, 1),
                "production cost [MEUR]": round(production_cost / 1e6, 1),
                "capital cost [MEUR]": round(capex / 1e6, 1),
                "total cost [MEUR]": round((production_cost + capex) / 1e6, 1),
                "profit [MEUR]": round(profit / 1e6, 1),
            })

        # ---------- Hydro (StorageUnit) ----------
        if hasattr(network, "storage_units") and len(network.storage_units.index) > 0:
            for su in network.storage_units.index:
                if network.storage_units.at[su, "carrier"] == "hydro":
                    bus = network.storage_units.at[su, "bus"]
                    # Positive when discharging to the bus
                    dispatch = network.storage_units_t.p_dispatch[su]
                    prices = network.buses_t.marginal_price[bus]
                    revenue = (dispatch * prices).sum()

                    results.append({
                        "Model": model_label,
                        "name": su,
                        "carrier": "hydro",
                        "revenue [MEUR]": round(revenue / 1e6, 1),
                        "production cost [MEUR]": 0.0,
                        "capital cost [MEUR]": 0.0,
                        "total cost [MEUR]": 0.0,
                        "profit [MEUR]": round(revenue / 1e6, 1),
                    })

        # ---------- Battery (links + store) ----------
        # Expect names: "battery charge", "battery discharge", "battery storage"
        link_names = {"battery charge", "battery discharge"}
        store_name = "battery storage"

        has_links = link_names.issubset(set(network.links.index))
        has_store = (store_name in getattr(network, "stores", pd.DataFrame()).index) if hasattr(network, "stores") else False

        if has_links and has_store:
            discharge = network.links_t.p1["battery discharge"]    # + when exporting to bus1
            charge = network.links_t.p0["battery charge"]          # + when importing from bus0
            discharge_bus = network.links.at["battery discharge", "bus1"]
            charge_bus = network.links.at["battery charge", "bus0"]

            prices_discharge = network.buses_t.marginal_price[discharge_bus]
            prices_charge = network.buses_t.marginal_price[charge_bus]

            revenue = (discharge * prices_discharge).sum()
            production_cost = (charge * prices_charge).sum()

            # CAPEX from Cost.costs
            inv_cost_charge = capex_lookup("battery charge")
            inv_cost_discharge = capex_lookup("battery discharge")
            p_nom_charge = float(network.links.at["battery charge", "p_nom"])
            p_nom_discharge = float(network.links.at["battery discharge", "p_nom"])

            # If you previously split equally, keep symmetry:
            capex_links = 0.5 * p_nom_charge * inv_cost_charge + 0.5 * p_nom_discharge * inv_cost_discharge

            store_capital_cost = capex_lookup(store_name)
            e_nom_store = float(network.stores.at[store_name, "e_nom"])
            capex_store = e_nom_store * store_capital_cost

            total_capex = capex_links + capex_store
            profit = revenue - (production_cost + total_capex)

            results.append({
                "Model": model_label,
                "name": store_name,
                "carrier": "battery",
                "revenue [MEUR]": round(revenue / 1e6, 1),
                "production cost [MEUR]": round(production_cost / 1e6, 1),
                "capital cost [MEUR]": round(total_capex / 1e6, 1),
                "total cost [MEUR]": round((production_cost + total_capex) / 1e6, 1),
                "profit [MEUR]": round(profit / 1e6, 1),
            })

    df = pd.DataFrame(results)

    # Add total row per model
    if not df.empty:
        totals = (df
                  .groupby("Model")[["revenue [MEUR]", "production cost [MEUR]", "capital cost [MEUR]", "total cost [MEUR]", "profit [MEUR]"]]
                  .sum()
                  .round(2)
                  .reset_index())
        totals.insert(1, "name", totals["Model"].apply(lambda m: f"Total {m}"))
        totals.insert(2, "carrier", "")
        df = pd.concat([df, totals], ignore_index=True)

    return df

# With small l in load
def cost_recovery_yearly(network_pf, network_rh):
    """
    Compute yearly cost recovery for generators, hydro (StorageUnit), and battery.
    """
    results = []

    def capex_lookup(name):
        # Safe lookup in Cost.costs; return 0 if missing
        try:
            return float(Cost.costs.at[name, "capital_cost"])
        except Exception:
            return 0.0

    for model_label, network in (("Perfect Foresight", network_pf), ("Rolling Horizon", network_rh)):

        # ---------- Generators ----------
        for gen in network.generators.index:
            carrier = network.generators.at[gen, "carrier"]
            bus = network.generators.at[gen, "bus"]
            p_nom = float(network.generators.at[gen, "p_nom"])
            marginal_cost = float(network.generators.at[gen, "marginal_cost"])

            # 'load shedding' has no CAPEX; otherwise read from Cost.costs
            if gen == "load shedding":
                capital_cost = 0.0
            else:
                capital_cost = capex_lookup(gen)

            dispatch = network.generators_t.p[gen]
            prices = network.buses_t.marginal_price[bus]

            revenue = (dispatch * prices).sum()
            production_cost = (dispatch * marginal_cost).sum()
            capex = p_nom * capital_cost
            profit = revenue - (capex + production_cost)

            results.append({
                "Model": model_label,
                "name": gen,
                "carrier": carrier,
                "revenue [MEUR]": round(revenue / 1e6, 1),
                "production cost [MEUR]": round(production_cost / 1e6, 1),
                "capital cost [MEUR]": round(capex / 1e6, 1),
                "total cost [MEUR]": round((production_cost + capex) / 1e6, 1),
                "profit [MEUR]": round(profit / 1e6, 1),
            })

        # ---------- Hydro (StorageUnit) ----------
        if hasattr(network, "storage_units") and len(network.storage_units.index) > 0:
            for su in network.storage_units.index:
                if network.storage_units.at[su, "carrier"] == "hydro":
                    bus = network.storage_units.at[su, "bus"]
                    # Positive when discharging to the bus
                    dispatch = network.storage_units_t.p_dispatch[su]
                    prices = network.buses_t.marginal_price[bus]
                    revenue = (dispatch * prices).sum()

                    results.append({
                        "Model": model_label,
                        "name": su,
                        "carrier": "hydro",
                        "revenue [MEUR]": round(revenue / 1e6, 1),
                        "production cost [MEUR]": 0.0,
                        "capital cost [MEUR]": 0.0,
                        "total cost [MEUR]": 0.0,
                        "profit [MEUR]": round(revenue / 1e6, 1),
                    })

        # ---------- Battery (links + store) ----------
        # Expect names: "battery charge", "battery discharge", "battery storage"
        link_names = {"battery charge", "battery discharge"}
        store_name = "battery storage"

        has_links = link_names.issubset(set(network.links.index))
        has_store = (store_name in getattr(network, "stores", pd.DataFrame()).index) if hasattr(network, "stores") else False

        if has_links and has_store:
            discharge = network.links_t.p1["battery discharge"]    # + when exporting to bus1
            charge = network.links_t.p0["battery charge"]          # + when importing from bus0
            discharge_bus = network.links.at["battery discharge", "bus1"]
            charge_bus = network.links.at["battery charge", "bus0"]

            prices_discharge = network.buses_t.marginal_price[discharge_bus]
            prices_charge = network.buses_t.marginal_price[charge_bus]

            revenue = (discharge * prices_discharge).sum()
            production_cost = (charge * prices_charge).sum()

            # CAPEX from Cost.costs
            inv_cost_charge = capex_lookup("battery charge")
            inv_cost_discharge = capex_lookup("battery discharge")
            p_nom_charge = float(network.links.at["battery charge", "p_nom"])
            p_nom_discharge = float(network.links.at["battery discharge", "p_nom"])

            # If you previously split equally, keep symmetry:
            capex_links = 0.5 * p_nom_charge * inv_cost_charge + 0.5 * p_nom_discharge * inv_cost_discharge

            store_capital_cost = capex_lookup(store_name)
            e_nom_store = float(network.stores.at[store_name, "e_nom"])
            capex_store = e_nom_store * store_capital_cost

            total_capex = capex_links + capex_store
            profit = revenue - (production_cost + total_capex)

            results.append({
                "Model": model_label,
                "name": store_name,
                "carrier": "battery",
                "revenue [MEUR]": round(revenue / 1e6, 1),
                "production cost [MEUR]": round(production_cost / 1e6, 1),
                "capital cost [MEUR]": round(total_capex / 1e6, 1),
                "total cost [MEUR]": round((production_cost + total_capex) / 1e6, 1),
                "profit [MEUR]": round(profit / 1e6, 1),
            })

    df = pd.DataFrame(results)

    # Add total row per model
    if not df.empty:
        totals = (df
                  .groupby("Model")[["revenue [MEUR]", "production cost [MEUR]", "capital cost [MEUR]", "total cost [MEUR]", "profit [MEUR]"]]
                  .sum()
                  .round(2)
                  .reset_index())
        totals.insert(1, "name", totals["Model"].apply(lambda m: f"Total {m}"))
        totals.insert(2, "carrier", "")
        df = pd.concat([df, totals], ignore_index=True)

    return df


