import pandas as pd
import numpy as np
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

create_new_files = False

if create_new_files:
    # Read input CSV files
    OffWind_all_df = pd.read_csv("Data/offshore_wind_1979-2017.csv", sep=';')
    OnWind_all_df = pd.read_csv("Data/onshore_wind_1979-2017.csv", sep=';')
    Solar_all_df = pd.read_csv("Data/pv_optimal.csv", sep=';')
    Load_data_all_df = pd.read_csv("Data/time_series_60min_singleindex_filtered_2015-2019.csv", sep=',')

    # Slice relevant columns
    OffWind_NOR = OffWind_all_df[['utc_time', 'NOR']]
    OnWind_NOR_ESP = OnWind_all_df[['utc_time', 'NOR', 'ESP']]
    Solar_NOR_ESP = Solar_all_df[['utc_time', 'NOR', 'ESP']]
    Load_data_actual_df = Load_data_all_df[['utc_timestamp', 'ES_load_actual_entsoe_transparency','NO_load_actual_entsoe_transparency']]

    # Export to new CSV files
    OffWind_NOR.to_csv("Data/offshore_wind_1979-2017_NOR.csv", sep=';', index=False)
    OnWind_NOR_ESP.to_csv("Data/onshore_wind_1979-2017_NOR_ESP.csv", sep=';', index=False)
    Solar_NOR_ESP.to_csv("Data/pv_optimal_NOR_ESP.csv", sep=';', index=False)
    Load_data_actual_df.to_csv("Data/load_data_actual_NOR_ES.csv", sep=';', index=False)

# Load offshore wind for spain. 
 

# # Read the sliced files
# OffWind_df = pd.read_csv("Data/offshore_wind_1979-2017_NOR.csv", sep=';', index_col=0)
# OnWind_df = pd.read_csv("Data/onshore_wind_1979-2017_NOR_ESP.csv", sep=';', index_col=0)
# Solar_df = pd.read_csv("Data/pv_optimal_NOR_ESP.csv", sep=';', index_col=0)
# Load_df = pd.read_csv("Data/load_data_actual_NOR_ES.csv", sep=';', index_col=0)

# # Convert index to datetime
# OffWind_df.index = pd.to_datetime(OffWind_df.index)
# OnWind_df.index = pd.to_datetime(OnWind_df.index)
# Solar_df.index = pd.to_datetime(Solar_df.index)
# Load_df.index = pd.to_datetime(Load_df.index)


class DataGeneration:
    def __init__(self, year: int, demand_year: int, region: str):
        self.year = year
        self.demand_year = demand_year
        self.region = region

        self.OffWind_df, self.OnWind_df, self.Solar_df, self.Load_df = self.load_all_data()

        self.data = self.combine_data()

    def load_all_data(self):
        OffWind_df = pd.read_csv("Data/offshore_wind_1979-2017_NOR.csv", sep=';', index_col=0)
        OnWind_df = pd.read_csv("Data/onshore_wind_1979-2017_NOR_ESP.csv", sep=';', index_col=0)
        Solar_df = pd.read_csv("Data/pv_optimal_NOR_ESP.csv", sep=';', index_col=0)
        Load_df = pd.read_csv("Data/load_data_actual_NOR_ES.csv", sep=';', index_col=0)

        # Convert index to datetime
        OffWind_df.index = pd.to_datetime(OffWind_df.index)
        OnWind_df.index = pd.to_datetime(OnWind_df.index)
        Solar_df.index = pd.to_datetime(Solar_df.index)
        Load_df.index = pd.to_datetime(Load_df.index)

        return OffWind_df, OnWind_df, Solar_df, Load_df

    def load_demand(self):
        df = self.Load_df.copy()
        df = df[df.index.year == self.demand_year]
        df = df.rename(columns={
            'ES_load_actual_entsoe_transparency': 'ESP_demand',
            'NO_load_actual_entsoe_transparency': 'NOR_demand'
        })
        return df[[f"{self.region}_demand"]]

    def load_solar(self):
        df = self.Solar_df.copy()
        df = df[df.index.year == self.year]
        return df[[self.region]].rename(columns={self.region: "solar"})

    def load_onshore_wind(self):
        df = self.OnWind_df.copy()
        df = df[df.index.year == self.year]
        return df[[self.region]].rename(columns={self.region: "onshore_wind"})

    def load_offshore_wind(self):
        df = self.OffWind_df.copy()
        df = df[df.index.year == self.year]
        return df[[self.region]].rename(columns={self.region: "offshore_wind"})

    def combine_data(self):
        demand = self.load_demand()
        solar = self.load_solar()
        onshore = self.load_onshore_wind()
        offshore = self.load_offshore_wind()
        return pd.concat([demand, solar, onshore, offshore], axis=1).dropna()


class DataGeneration_2:
    def __init__(self, year: int, demand_year: int, regions: list):
        self.year = year
        self.demand_year = demand_year
        self.regions = regions

        self.OffWind_df, self.OnWind_df, self.Solar_df, self.Load_df = self.load_all_data()

        self.data = self.combine_data()

    def load_all_data(self):
        OffWind_df = pd.read_csv("Data/offshore_wind_1979-2017_NOR.csv", sep=';', index_col=0)
        OnWind_df = pd.read_csv("Data/onshore_wind_1979-2017_NOR_ESP.csv", sep=';', index_col=0)
        Solar_df = pd.read_csv("Data/pv_optimal_NOR_ESP.csv", sep=';', index_col=0)
        Load_df = pd.read_csv("Data/load_data_actual_NOR_ES.csv", sep=';', index_col=0)

        # Convert index to datetime
        OffWind_df.index = pd.to_datetime(OffWind_df.index)
        OnWind_df.index = pd.to_datetime(OnWind_df.index)
        Solar_df.index = pd.to_datetime(Solar_df.index)
        Load_df.index = pd.to_datetime(Load_df.index)

        return OffWind_df, OnWind_df, Solar_df, Load_df

    def load_demand(self):
        df = self.Load_df.copy()
        df = df[df.index.year == self.demand_year]
        df = df.rename(columns={
            'ES_load_actual_entsoe_transparency': 'ESP_demand',
            'NO_load_actual_entsoe_transparency': 'NOR_demand'
        })
        return df[[f"{region}_demand" for region in self.regions if f"{region}_demand" in df.columns]]

    def load_solar(self):
        df = self.Solar_df.copy()
        df = df[df.index.year == self.year]
        return df[self.regions].add_suffix("_solar")

    def load_onshore_wind(self):
        df = self.OnWind_df.copy()
        df = df[df.index.year == self.year]
        return df[self.regions].add_suffix("_onshore_wind")

    def load_offshore_wind(self):
        df = self.OffWind_df.copy()
        df = df[df.index.year == self.year]
        return df[self.regions].add_suffix("_offshore_wind")

    def combine_data(self):
        demand = self.load_demand()
        solar = self.load_solar()
        onshore = self.load_onshore_wind()
        offshore = self.load_offshore_wind()

        dfs = [d.reindex(demand.index) for d in [solar, onshore, offshore]]
        return pd.concat([demand] + dfs, axis=1).dropna()


if __name__ == "__main__":
    print("Running the main script...")

    data_gen = DataGeneration_2(year=1980, demand_year=2016, regions=('NOR', 'ESP'))
    OffWind_df = data_gen.OffWind_df
    OnWind_df = data_gen.OnWind_df
    Solar_df = data_gen.Solar_df
    Load_df = data_gen.Load_df
    all_data = data_gen.data

    # print the first few rows of each DataFrame
    print("Offshore Wind Data (NOR):")
    print(OffWind_df.head())
    print("\nOnshore Wind Data (NOR):")
    print(OnWind_df.head())
    print("\nSolar Data (NOR):")
    print(Solar_df.head())
    print("\nLoad Data (NOR):")
    print(Load_df.head())
    print("\nAll Data (NOR):")
    print(all_data.head())

    # print(Load_df[Load_df.index.year == 2016])
