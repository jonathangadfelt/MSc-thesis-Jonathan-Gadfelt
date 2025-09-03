import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

All_data = load_all_data()
Cost = CostGeneration(year=2025)

# PLOTTTING FUNCTIONS
def plot_electricity_mix(network, colors=None, save_plot=False, plot_title='Electricity mix'):
    """
    Plots the electricity mix as a pie chart for a given PyPSA network.
    """
    labels = []
    sizes = []

    # Add generator capacities
    for generator in network.generators.index:
        cap = network.generators.p_nom_opt[generator]
        if cap > 10:
            labels.append(generator)
            sizes.append(cap)

    # Add hydro storage unit capacity (if present)
    if hasattr(network, "storage_units") and not network.storage_units.empty:
        hydro_mask = (network.storage_units.carrier == "hydro")
        if hydro_mask.any():
            hydro_capacity = network.storage_units.loc[hydro_mask, "p_nom_opt"].sum()
            if hydro_capacity > 10:
                labels.append("hydro")
                sizes.append(hydro_capacity)

    # Map each label to its color if provided
    if colors is not None:
        pie_colors = [colors.get(label, None) for label in labels]
    else:
        pie_colors = None

    plt.pie(
        sizes,
        labels=labels,
        wedgeprops={'linewidth': 0},
        autopct='%1.1f%%',
        colors=pie_colors
    )
    plt.axis('equal')
    plt.title(plot_title, y=1.07)
    plt.tight_layout()

    if save_plot:
        plt.savefig(f'./Plots/electricity_mix.png', dpi=300, bbox_inches='tight')
    plt.show()
