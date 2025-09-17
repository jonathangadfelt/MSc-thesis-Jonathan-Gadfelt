from functions import *
np.random.seed(1) 

class Build_network_capacity_exp:
    def __init__(
        self,
        weather_year: int = 2011,
        hydro_year: int = 2011,
        demand_year: int = 2018,
        data: dict = None,
        cost_data: tuple = None,
        setup: dict = None
    ):
        if setup is None:
            setup = {
                'NOR': {
                    'OCGT': True,
                    'CCGT': False,
                    'battery storage': True,
                    'onwind': True,
                    'offwind': False,
                    'solar': True,
                    'electrolysis': True,
                    'fuel cell': True,
                    'Hydrogen storage': True,
                    'Reservoir hydro storage': True,
                    'load shedding': True
                }
            }

        self.weather_year = weather_year
        self.hydro_year = hydro_year
        self.demand_year = demand_year
        self.setup = setup
        self.region = list(setup.keys())[0]  # Single region expected

        self.costs = cost_data.costs
        self.cost_units = cost_data.units

        self.all_data = data if data is not None else load_all_data()
        self.data_dict = {self.region: self.extract_data(self.region, self.weather_year, self.hydro_year, self.demand_year)}
        
        self.network = pypsa.Network()
        self.hours_in_year = pd.date_range(f'{weather_year}-01-01 00:00', f'{weather_year}-12-31 23:00', freq='h')
        if len(self.hours_in_year) > 8760:
            self.hours_in_year = self.hours_in_year[self.hours_in_year.strftime('%m-%d') != '02-29']
        self.network.set_snapshots(self.hours_in_year.values)

        self.carriers = [
            'gas', 'onwind', 'offwind', 'solar',
            'battery charge', 'battery discharge', 'hydro',
            'electrolysis', 'fuel cell', 'hydrogen', 'load shedding'
        ]

        self.colors = {
            'gas': 'gray', 'onwind': 'lightblue', 'offwind': 'dodgerblue', 'load shedding': 'red',
            'solar': 'orange', 'battery charge': 'gold', 'battery discharge': 'darkorange',
            'electrolysis': 'green', 'fuel cell': 'limegreen', 'hydrogen': 'deepskyblue', 'hydro': 'slateblue'
        }

        self.network.add("Carrier",
            self.carriers,
            color=[self.colors[c] for c in self.carriers],
            co2_emissions=[self.costs.at[c, "CO2 intensity"] if c in self.costs.index else 0.0 for c in self.carriers]
        )


        self.network.add("Bus", 'electricity bus')
        self.network.add("Bus", 'hydrogen bus')
        self.network.add("Load", 'load',
                         bus='electricity bus',
                         p_set=self.data_dict[self.region]['demand'].values.flatten()
                         )


        technologies = self.setup[self.region].keys()
        for tech in technologies:
            if not self.setup[self.region][tech]:
                continue

            if tech in ['OCGT', 'CCGT']:
                self.network.add("Generator", tech,
                    bus='electricity bus',
                    p_nom_extendable=True,
                    carrier='gas',
                    capital_cost=self.costs.at[tech, "capital_cost"],
                    marginal_cost=self.costs.at[tech, "marginal_cost"]
                    )

            elif tech == 'load shedding':
                self.network.add("Generator", tech,
                    bus="electricity bus",
                    p_nom_extendable=True,
                    marginal_cost=2000,   # €/MWh, can adjust based on VoLL
                    capital_cost=0,
                    carrier="load shedding"
                    )
                
            elif tech == 'solar':
                self.network.add("Generator", tech,
                    bus='electricity bus',
                    p_nom_extendable=True,
                    carrier='solar',
                    capital_cost=self.costs.at[tech, "capital_cost"],
                    marginal_cost=self.costs.at[tech, "marginal_cost"],
                    p_max_pu=self.data_dict[self.region]['solar'].values.flatten()
                    )

            elif tech == 'onwind':
                self.network.add("Generator", tech,
                    bus='electricity bus',
                    p_nom_extendable=True,
                    carrier='onwind',
                    capital_cost=self.costs.at[tech, "capital_cost"],
                    marginal_cost=self.costs.at[tech, "marginal_cost"],
                    p_max_pu=self.data_dict[self.region]['onwind'].values.flatten()
                    )

            elif tech == 'offwind':
                self.network.add("Generator", tech,
                    bus='electricity bus',
                    p_nom_extendable=True,
                    carrier='offwind',
                    capital_cost=self.costs.at[tech, "capital_cost"],
                    marginal_cost=self.costs.at[tech, "marginal_cost"],
                    p_max_pu=self.data_dict[self.region]['offwind'].values.flatten()
                    )

            elif tech == 'battery storage':
                self.network.add("Bus", 'battery bus')

                self.network.add("Link", 'battery charge',
                    bus0='electricity bus',
                    bus1='battery bus',
                    carrier='battery charge',
                    p_nom_extendable=True,
                    capital_cost=self.costs.at["battery inverter", "capital_cost"]/2,    # Divide by two as only one inverter will be baught in reality
                    efficiency=self.costs.at["battery inverter", "efficiency"]
                    )

                self.network.add("Link", 'battery discharge',
                    bus0='battery bus',
                    bus1='electricity bus',
                    carrier='battery discharge',
                    p_nom_extendable=True,
                    capital_cost=self.costs.at["battery inverter", "capital_cost"]/2,     # Divide by two as only one inverter will be baught in reality
                    efficiency=self.costs.at["battery inverter", "efficiency"]
                    )        

                self.network.add("Store", tech,
                    bus='battery bus',
                    e_nom_extendable=True,
                    e_cyclic=False,
                    capital_cost=self.costs.at[tech, "capital_cost"]
                    )

            elif tech == 'electrolysis':
                self.network.add("Link", tech,
                    bus0='electricity bus',
                    bus1='hydrogen bus',
                    carrier='electrolysis',
                    p_nom_extendable=True,
                    capital_cost=self.costs.at[tech, "capital_cost"],
                    efficiency=self.costs.at[tech, "efficiency"]
                    )

            elif tech == 'fuel cell':
                self.network.add("Link", tech,
                    bus0='hydrogen bus',
                    bus1='electricity bus',
                    carrier='fuel cell',
                    p_nom_extendable=True,
                    capital_cost=self.costs.at[tech, "capital_cost"],
                    efficiency=self.costs.at[tech, "efficiency"]
                    )

            elif tech == 'Hydrogen storage':
                self.network.add("Store", tech,
                    bus='hydrogen bus',
                    e_nom_extendable=True,
                    e_cyclic=False,
                    capital_cost=self.costs.at["H2 (l) storage tank", "capital_cost"],
                    carrier='hydrogen storage'
                    )
                
            elif tech == 'Reservoir hydro storage':
                self.network.add("StorageUnit", tech,
                    bus='electricity bus',
                    carrier='hydro',
                    p_nom_extendable=False,
                    p_nom = 12700,  # 12 GW
                    max_hours=1300,
                    efficiency_store=0,
                    efficiency_dispatch=self.costs.at["Pumped-Storage-Hydro-bicharger", "efficiency"],
                    cyclic_state_of_charge=False,
                    state_of_charge_initial= (12700 * 1300)*0.3 ,  # Initial storage capacity in MWh
                    inflow=self.data_dict[self.region]['hydro'].values.flatten(),
                    marginal_cost=self.costs.at["onwind", "marginal_cost"]*1.2,  # higher than wind to prioritize wind usage
                    capital_cost=0
                    )

    def extract_data(self, region: str, weather_year: int, hydro_year: int, demand_year: int):
        extracted = {}
        if demand_year not in self.all_data["demand"].index.year:
            print(f"Demand year {demand_year} not in data. Using 2017 instead.")
            demand_year = 2017
        if weather_year not in self.all_data["solar"].index.year:
            print(f"Weather year {weather_year} not in data. Using 2012 instead.")
            weather_year = 2012
        if hydro_year not in self.all_data["hydro_inflow"].index.year:
            print(f"Hydro year {hydro_year} not in data. Using 2011 instead.")
            hydro_year = 2011

        if region in self.all_data["demand"].columns:
            demand_series = self.all_data["demand"].loc[self.all_data["demand"].index.year == demand_year, region]
            extracted["demand"] = demand_series[demand_series.index.strftime('%m-%d') != '02-29'][:8760]

        for carrier in ["solar", "onwind", "offwind"]:
            if region in self.all_data[carrier].columns:
                weather_series = self.all_data[carrier].loc[self.all_data[carrier].index.year == weather_year, region]
                extracted[carrier] = weather_series[weather_series.index.strftime('%m-%d') != '02-29'][:8760]

        if region in self.all_data["hydro_inflow"].columns:
            hydro_series = self.all_data["hydro_inflow"].loc[self.all_data["hydro_inflow"].index.year == hydro_year, region]
            extracted["hydro"] = hydro_series[hydro_series.index.strftime('%m-%d') != '02-29'][:8760]

        return extracted  

class Build_network_capacity_exp_gas:
    def __init__(
        self,
        weather_year: int = 2011,
        hydro_year: int = 2011,
        demand_year: int = 2018,
        data: dict = None,
        cost_data: tuple = None,
        setup: dict = None
    ):
        if setup is None:
            setup = {
                'NOR': {
                    'OCGT': True,
                    'CCGT': False,
                    'battery storage': True,
                    'onwind': True,
                    'offwind': False,
                    'solar': True,
                    'electrolysis': True,
                    'fuel cell': True,
                    'Hydrogen storage': True,
                    'Reservoir hydro storage': True,
                    'load shedding': True
                }
            }

        self.weather_year = weather_year
        self.hydro_year = hydro_year
        self.demand_year = demand_year
        self.setup = setup
        self.region = list(setup.keys())[0]  # Single region expected

        self.costs = cost_data.costs
        self.cost_units = cost_data.units

        self.all_data = data if data is not None else load_all_data()
        self.data_dict = {self.region: self.extract_data(self.region, self.weather_year, self.hydro_year, self.demand_year)}
        
        self.network = pypsa.Network()
        self.hours_in_year = pd.date_range(f'{weather_year}-01-01 00:00', f'{weather_year}-12-31 23:00', freq='h')
        if len(self.hours_in_year) > 8760:
            self.hours_in_year = self.hours_in_year[self.hours_in_year.strftime('%m-%d') != '02-29']
        self.network.set_snapshots(self.hours_in_year.values)

        self.carriers = [
            'gas', 'onwind', 'offwind', 'solar',
            'battery charge', 'battery discharge', 'hydro',
            'electrolysis', 'fuel cell', 'hydrogen', 'load shedding'
        ]

        self.colors = {
            'gas': 'gray', 'onwind': 'lightblue', 'offwind': 'dodgerblue', 'load shedding': 'red',
            'solar': 'orange', 'battery charge': 'gold', 'battery discharge': 'darkorange',
            'electrolysis': 'green', 'fuel cell': 'limegreen', 'hydrogen': 'deepskyblue', 'hydro': 'slateblue'
        }

        self.network.add("Carrier",
            self.carriers,
            color=[self.colors[c] for c in self.carriers],
            co2_emissions=[self.costs.at[c, "CO2 intensity"] if c in self.costs.index else 0.0 for c in self.carriers]
        )


        self.network.add("Bus", 'electricity bus')
        self.network.add("Bus", 'hydrogen bus')
        self.network.add("Load", 'load',
                         bus='electricity bus',
                         p_set=self.data_dict[self.region]['demand'].values.flatten()
                         )


        technologies = self.setup[self.region].keys()
        for tech in technologies:
            if not self.setup[self.region][tech]:
                continue

            if tech in ['OCGT', 'CCGT']:
                self.network.add("Generator", tech,
                    bus='electricity bus',
                    p_nom_extendable=True,
                    carrier='gas',
                    capital_cost=self.costs.at[tech, "capital_cost"],
                    marginal_cost=self.costs.at[tech, "marginal_cost"] + 0.1 * np.random.uniform(1, 10) # Adding small random cost to break symmetry
                    )

            elif tech == 'load shedding':
                self.network.add("Generator", tech,
                    bus="electricity bus",
                    p_nom_extendable=True,
                    marginal_cost=2000,   # €/MWh, can adjust based on VoLL
                    capital_cost=0,
                    carrier="load shedding"
                    )
                
            elif tech == 'solar':
                self.network.add("Generator", tech,
                    bus='electricity bus',
                    p_nom_extendable=True,
                    carrier='solar',
                    capital_cost=self.costs.at[tech, "capital_cost"],
                    marginal_cost=self.costs.at[tech, "marginal_cost"],
                    p_max_pu=self.data_dict[self.region]['solar'].values.flatten()
                    )

            elif tech == 'onwind':
                self.network.add("Generator", tech,
                    bus='electricity bus',
                    p_nom_extendable=True,
                    carrier='onwind',
                    capital_cost=self.costs.at[tech, "capital_cost"],
                    marginal_cost=self.costs.at[tech, "marginal_cost"],
                    p_max_pu=self.data_dict[self.region]['onwind'].values.flatten()
                    )

            elif tech == 'offwind':
                self.network.add("Generator", tech,
                    bus='electricity bus',
                    p_nom_extendable=True,
                    carrier='offwind',
                    capital_cost=self.costs.at[tech, "capital_cost"],
                    marginal_cost=self.costs.at[tech, "marginal_cost"],
                    p_max_pu=self.data_dict[self.region]['offwind'].values.flatten()
                    )

            elif tech == 'battery storage':
                self.network.add("Bus", 'battery bus')

                self.network.add("Link", 'battery charge',
                    bus0='electricity bus',
                    bus1='battery bus',
                    carrier='battery charge',
                    p_nom_extendable=True,
                    capital_cost=self.costs.at["battery inverter", "capital_cost"]/2,    # Divide by two as only one inverter will be baught in reality
                    efficiency=self.costs.at["battery inverter", "efficiency"]
                    )

                self.network.add("Link", 'battery discharge',
                    bus0='battery bus',
                    bus1='electricity bus',
                    carrier='battery discharge',
                    p_nom_extendable=True,
                    capital_cost=self.costs.at["battery inverter", "capital_cost"]/2,     # Divide by two as only one inverter will be baught in reality
                    efficiency=self.costs.at["battery inverter", "efficiency"]
                    )        

                self.network.add("Store", tech,
                    bus='battery bus',
                    e_nom_extendable=True,
                    e_cyclic=False,
                    capital_cost=self.costs.at[tech, "capital_cost"]
                    )

            elif tech == 'electrolysis':
                self.network.add("Link", tech,
                    bus0='electricity bus',
                    bus1='hydrogen bus',
                    carrier='electrolysis',
                    p_nom_extendable=True,
                    capital_cost=self.costs.at[tech, "capital_cost"],
                    efficiency=self.costs.at[tech, "efficiency"]
                    )

            elif tech == 'fuel cell':
                self.network.add("Link", tech,
                    bus0='hydrogen bus',
                    bus1='electricity bus',
                    carrier='fuel cell',
                    p_nom_extendable=True,
                    capital_cost=self.costs.at[tech, "capital_cost"],
                    efficiency=self.costs.at[tech, "efficiency"]
                    )

            elif tech == 'Hydrogen storage':
                self.network.add("Store", tech,
                    bus='hydrogen bus',
                    e_nom_extendable=True,
                    e_cyclic=False,
                    capital_cost=self.costs.at["H2 (l) storage tank", "capital_cost"],
                    carrier='hydrogen storage'
                    )
                
            elif tech == 'Reservoir hydro storage':
                self.network.add("StorageUnit", tech,
                    bus='electricity bus',
                    carrier='hydro',
                    p_nom_extendable=False,
                    p_nom = 12700,  # 12 GW
                    max_hours=1300,
                    efficiency_store=0,
                    efficiency_dispatch=self.costs.at["Pumped-Storage-Hydro-bicharger", "efficiency"],
                    cyclic_state_of_charge=False,
                    state_of_charge_initial= (12700 * 1300)*0.3 ,  # Initial storage capacity in MWh
                    inflow=self.data_dict[self.region]['hydro'].values.flatten(),
                    marginal_cost=self.costs.at["onwind", "marginal_cost"]*1.2,  # higher than wind to prioritize wind usage
                    capital_cost=0
                    )

    def extract_data(self, region: str, weather_year: int, hydro_year: int, demand_year: int):
        extracted = {}
        if demand_year not in self.all_data["demand"].index.year:
            print(f"Demand year {demand_year} not in data. Using 2017 instead.")
            demand_year = 2017
        if weather_year not in self.all_data["solar"].index.year:
            print(f"Weather year {weather_year} not in data. Using 2012 instead.")
            weather_year = 2012
        if hydro_year not in self.all_data["hydro_inflow"].index.year:
            print(f"Hydro year {hydro_year} not in data. Using 2011 instead.")
            hydro_year = 2011

        if region in self.all_data["demand"].columns:
            demand_series = self.all_data["demand"].loc[self.all_data["demand"].index.year == demand_year, region]
            extracted["demand"] = demand_series[demand_series.index.strftime('%m-%d') != '02-29'][:8760]

        for carrier in ["solar", "onwind", "offwind"]:
            if region in self.all_data[carrier].columns:
                weather_series = self.all_data[carrier].loc[self.all_data[carrier].index.year == weather_year, region]
                extracted[carrier] = weather_series[weather_series.index.strftime('%m-%d') != '02-29'][:8760]

        if region in self.all_data["hydro_inflow"].columns:
            hydro_series = self.all_data["hydro_inflow"].loc[self.all_data["hydro_inflow"].index.year == hydro_year, region]
            extracted["hydro"] = hydro_series[hydro_series.index.strftime('%m-%d') != '02-29'][:8760]

        return extracted  



class Build_network_capacity_exp_bat_S_unit:
    def __init__(
        self,
        weather_year: int = 2011,
        hydro_year: int = 2011,
        demand_year: int = 2018,
        data: dict = None,
        cost_data: tuple = None,
        setup: dict = None
    ):
        if setup is None:
            setup = {
                'NOR': {
                    'OCGT': True,
                    'CCGT': False,
                    'battery storage': True,
                    'onwind': True,
                    'offwind': False,
                    'solar': True,
                    'electrolysis': True,
                    'fuel cell': True,
                    'Hydrogen storage': True,
                    'Reservoir hydro storage': True,
                    'load shedding': True
                }
            }

        self.weather_year = weather_year
        self.hydro_year = hydro_year
        self.demand_year = demand_year
        self.setup = setup
        self.region = list(setup.keys())[0]  # Single region expected

        self.costs = cost_data.costs
        self.cost_units = cost_data.units

        self.all_data = data if data is not None else load_all_data()
        self.data_dict = {self.region: self.extract_data(self.region, self.weather_year, self.hydro_year, self.demand_year)}
        
        self.network = pypsa.Network()
        self.hours_in_year = pd.date_range(f'{weather_year}-01-01 00:00', f'{weather_year}-12-31 23:00', freq='h')
        if len(self.hours_in_year) > 8760:
            self.hours_in_year = self.hours_in_year[self.hours_in_year.strftime('%m-%d') != '02-29']
        self.network.set_snapshots(self.hours_in_year.values)

        self.carriers = [
            'gas', 'onwind', 'offwind', 'solar',
            'battery', 'hydro',
            'electrolysis', 'fuel cell', 'hydrogen', 'load shedding'
        ]

        self.colors = {
            'gas': 'gray', 'onwind': 'lightblue', 'offwind': 'dodgerblue', 'load shedding': 'red',
            'solar': 'orange', 'battery': 'gold',
            'electrolysis': 'green', 'fuel cell': 'limegreen', 'hydrogen': 'deepskyblue', 'hydro': 'slateblue'
        }

        self.network.add("Carrier",
            self.carriers,
            color=[self.colors[c] for c in self.carriers],
            co2_emissions=[self.costs.at[c, "CO2 intensity"] if c in self.costs.index else 0.0 for c in self.carriers]
        )


        self.network.add("Bus", 'electricity bus')
        self.network.add("Bus", 'hydrogen bus')
        self.network.add("Load", 'load',
                         bus='electricity bus',
                         p_set=self.data_dict[self.region]['demand'].values.flatten()
                         )


        technologies = self.setup[self.region].keys()
        for tech in technologies:
            if not self.setup[self.region][tech]:
                continue

            if tech in ['OCGT', 'CCGT']:
                self.network.add("Generator", tech,
                    bus='electricity bus',
                    p_nom_extendable=True,
                    carrier='gas',
                    capital_cost=self.costs.at[tech, "capital_cost"],
                    marginal_cost=self.costs.at[tech, "marginal_cost"]
                    )

            elif tech == 'load shedding':
                self.network.add("Generator", tech,
                    bus="electricity bus",
                    p_nom_extendable=True,
                    marginal_cost=2000,   # €/MWh, can adjust based on VoLL
                    capital_cost=0,
                    carrier="load shedding"
                    )
                
            elif tech == 'solar':
                self.network.add("Generator", tech,
                    bus='electricity bus',
                    p_nom_extendable=True,
                    carrier='solar',
                    capital_cost=self.costs.at[tech, "capital_cost"],
                    marginal_cost=self.costs.at[tech, "marginal_cost"],
                    p_max_pu=self.data_dict[self.region]['solar'].values.flatten()
                    )

            elif tech == 'onwind':
                self.network.add("Generator", tech,
                    bus='electricity bus',
                    p_nom_extendable=True,
                    carrier='onwind',
                    capital_cost=self.costs.at[tech, "capital_cost"],
                    marginal_cost=self.costs.at[tech, "marginal_cost"],
                    p_max_pu=self.data_dict[self.region]['onwind'].values.flatten()
                    )

            elif tech == 'offwind':
                self.network.add("Generator", tech,
                    bus='electricity bus',
                    p_nom_extendable=True,
                    carrier='offwind',
                    capital_cost=self.costs.at[tech, "capital_cost"],
                    marginal_cost=self.costs.at[tech, "marginal_cost"],
                    p_max_pu=self.data_dict[self.region]['offwind'].values.flatten()
                    )

            elif tech == 'battery storage':
                max_h = 4  # Max hours of storage
                self.network.add("StorageUnit", "battery",
                    bus='electricity bus', carrier='battery',
                    p_nom_extendable=True, 
                    max_hours=max_h,
                    efficiency_store=self.costs.at["battery inverter", "efficiency"],
                    efficiency_dispatch=self.costs.at["battery inverter", "efficiency"],
                    capital_cost=self.costs.at["battery inverter", "capital_cost"] + self.costs.at["battery storage", "capital_cost"] * max_h,
                    #capital_cost= max_h * self.costs.at["battery storage", "capital_cost"],
                    cyclic_state_of_charge=False
                    )   

            elif tech == 'electrolysis':
                self.network.add("Link", tech,
                    bus0='electricity bus',
                    bus1='hydrogen bus',
                    carrier='electrolysis',
                    p_nom_extendable=True,
                    capital_cost=self.costs.at[tech, "capital_cost"],
                    efficiency=self.costs.at[tech, "efficiency"]
                    )

            elif tech == 'fuel cell':
                self.network.add("Link", tech,
                    bus0='hydrogen bus',
                    bus1='electricity bus',
                    carrier='fuel cell',
                    p_nom_extendable=True,
                    capital_cost=self.costs.at[tech, "capital_cost"],
                    efficiency=self.costs.at[tech, "efficiency"]
                    )

            elif tech == 'Hydrogen storage':
                self.network.add("Store", tech,
                    bus='hydrogen bus',
                    e_nom_extendable=True,
                    e_cyclic=False,
                    capital_cost=self.costs.at["H2 (l) storage tank", "capital_cost"],
                    carrier='hydrogen storage'
                    )
                
            elif tech == 'Reservoir hydro storage':
                self.network.add("StorageUnit", tech,
                    bus='electricity bus',
                    carrier='hydro',
                    p_nom_extendable=False,
                    p_nom = 12700,  # 12 GW
                    max_hours=1300,
                    efficiency_store=0,
                    efficiency_dispatch=self.costs.at["Pumped-Storage-Hydro-bicharger", "efficiency"],
                    cyclic_state_of_charge=False,
                    state_of_charge_initial= (12700 * 1300)*0.3 ,  # Initial storage capacity in MWh
                    inflow=self.data_dict[self.region]['hydro'].values.flatten(),
                    marginal_cost=self.costs.at["onwind", "marginal_cost"]*1.2,  # higher than wind to prioritize wind usage
                    capital_cost=0
                    )

    def extract_data(self, region: str, weather_year: int, hydro_year: int, demand_year: int):
        extracted = {}
        if demand_year not in self.all_data["demand"].index.year:
            print(f"Demand year {demand_year} not in data. Using 2017 instead.")
            demand_year = 2017
        if weather_year not in self.all_data["solar"].index.year:
            print(f"Weather year {weather_year} not in data. Using 2012 instead.")
            weather_year = 2012
        if hydro_year not in self.all_data["hydro_inflow"].index.year:
            print(f"Hydro year {hydro_year} not in data. Using 2011 instead.")
            hydro_year = 2011

        if region in self.all_data["demand"].columns:
            demand_series = self.all_data["demand"].loc[self.all_data["demand"].index.year == demand_year, region]
            extracted["demand"] = demand_series[demand_series.index.strftime('%m-%d') != '02-29'][:8760]

        for carrier in ["solar", "onwind", "offwind"]:
            if region in self.all_data[carrier].columns:
                weather_series = self.all_data[carrier].loc[self.all_data[carrier].index.year == weather_year, region]
                extracted[carrier] = weather_series[weather_series.index.strftime('%m-%d') != '02-29'][:8760]

        if region in self.all_data["hydro_inflow"].columns:
            hydro_series = self.all_data["hydro_inflow"].loc[self.all_data["hydro_inflow"].index.year == hydro_year, region]
            extracted["hydro"] = hydro_series[hydro_series.index.strftime('%m-%d') != '02-29'][:8760]

        return extracted  



class Build_dispatch_network:
    def __init__(
        self,
        opt_capacities_df: pd.DataFrame,
        weather_year: int = 2011,
        hydro_year: int = 2011,
        demand_year: int = 2018,
        data: dict = None,
        cost_data: tuple = None,
        setup: dict = None
    ):
        if setup is None:
            setup = {
                'NOR': {
                    'OCGT': True,
                    'CCGT': False,
                    'battery storage': True,
                    'onwind': True,
                    'offwind': False,
                    'solar': True,
                    'electrolysis': True,
                    'fuel cell': True,
                    'Hydrogen storage': True,
                    'Reservoir hydro storage': True,
                    'Load shedding': True
                }
            }

        self.weather_year = weather_year
        self.hydro_year = hydro_year
        self.demand_year = demand_year
        self.setup = setup
        self.region = list(setup.keys())[0]

        self.costs = cost_data.costs
        self.cost_units = cost_data.units

        self.opt_caps = opt_capacities_df

        self.all_data = data if data is not None else load_all_data()
        self.data_dict = {self.region: self.extract_data(self.region, self.weather_year, self.hydro_year, self.demand_year)}

        self.network = pypsa.Network()
        self.hours_in_year = pd.date_range(f'{weather_year}-01-01 00:00', f'{weather_year}-12-31 23:00', freq='h')
        if len(self.hours_in_year) > 8760:
            self.hours_in_year = self.hours_in_year[self.hours_in_year.strftime('%m-%d') != '02-29']
        self.network.set_snapshots(self.hours_in_year.values)

        self.carriers = ['gas', 'onwind', 'offwind', 'solar', 'battery charge', 'battery discharge', 'electrolysis', 'fuel cell', 'hydrogen', 'hydro', 'load shedding']

        self.colors = {
            'gas': 'gray', 'onwind': 'lightblue', 'offwind': 'dodgerblue', 'solar': 'orange',
            'battery charge': 'gold', 'battery discharge': 'darkorange', 'electrolysis': 'green',
            'fuel cell': 'limegreen', 'hydrogen': 'deepskyblue', 'hydro': 'slateblue', 'load shedding': 'red'
        }

        self.network.add("Carrier", self.carriers, color=[self.colors[c] for c in self.carriers],
                         co2_emissions=[self.costs.at[c, "CO2 intensity"] if c in self.costs.index else 0.0 for c in self.carriers])

        self.network.add("Bus", 'electricity bus')
        self.network.add("Bus", 'hydrogen bus')
        self.network.add("Load", 'load', bus='electricity bus', p_set=self.data_dict[self.region]['demand'].values.flatten())


        for tech, active in self.setup[self.region].items():
            if not active:
                continue

            if tech in ['OCGT', 'CCGT', 'onwind', 'offwind', 'solar']:
                self.network.add("Generator", tech,
                    bus='electricity bus',
                    p_nom=self.opt_caps.at["generators", tech],
                    p_nom_extendable=False,
                    carrier=tech if tech in ['solar', 'onwind', 'offwind'] else 'gas',
                    capital_cost=0,
                    marginal_cost=self.costs.at[tech, "marginal_cost"],
                    p_max_pu=self.data_dict[self.region][tech].values.flatten() if tech != 'OCGT' else None)

            elif tech == 'Load shedding':
                # Add load shedding generator
                self.network.add("Generator", tech,
                    bus="electricity bus",
                    p_nom_extendable=True,
                    marginal_cost=2000,
                    capital_cost=0,
                    carrier="load shedding")

            elif tech == 'battery storage':
                self.network.add("Bus", 'battery bus')
                self.network.add("Link", 'battery charge',
                    bus0='electricity bus', bus1='battery bus',
                    carrier='battery charge',
                    p_nom=self.opt_caps.at['links', 'battery charge'],
                    p_nom_extendable=False,
                    capital_cost=0,
                    efficiency=self.costs.at['battery inverter', 'efficiency'])
                
                self.network.add("Link", 'battery discharge',
                    bus0='battery bus', bus1='electricity bus',
                    carrier='battery discharge',
                    p_nom=self.opt_caps.at['links', 'battery discharge'],
                    p_nom_extendable=False,
                    capital_cost=0,
                    efficiency=self.costs.at['battery inverter', 'efficiency'])
                
                self.network.add("Store", tech,
                    bus='battery bus',
                    e_nom=self.opt_caps.at['stores', 'battery storage'],
                    e_nom_extendable=False,
                    e_cyclic=False,
                    capital_cost=0)

            elif tech == 'electrolysis':
                self.network.add("Link", tech,
                    bus0='electricity bus', bus1='hydrogen bus',
                    carrier='electrolysis',
                    p_nom=self.opt_caps.at['links', tech],
                    p_nom_extendable=False,
                    capital_cost=0,
                    efficiency=self.costs.at[tech, 'efficiency'])

            elif tech == 'fuel cell':
                self.network.add("Link", tech,
                    bus0='hydrogen bus', bus1='electricity bus',
                    carrier='fuel cell',
                    p_nom=self.opt_caps.at['links', tech],
                    p_nom_extendable=False,
                    capital_cost=0,
                    efficiency=self.costs.at[tech, 'efficiency'])

            elif tech == 'Hydrogen storage':
                self.network.add("Store", tech,
                    bus='hydrogen bus',
                    e_nom=self.opt_caps.at['stores', tech],
                    e_nom_extendable=False,
                    e_cyclic=False,
                    capital_cost=0,
                    carrier='hydrogen storage')

            elif tech == 'Reservoir hydro storage':
                self.network.add("StorageUnit", tech,
                    bus='electricity bus',
                    carrier='hydro',
                    p_nom=12700,
                    p_nom_extendable=False,
                    max_hours=1300,
                    efficiency_store=0,
                    efficiency_dispatch=self.costs.at['Pumped-Storage-Hydro-bicharger', 'efficiency'],
                    cyclic_state_of_charge=False,
                    inflow=self.data_dict[self.region]['hydro'].values.flatten(),
                    state_of_charge_initial= (12700 * 1300)*0.3 ,  # Initial storage capacity in MWh
                    capital_cost=0)
    
    def extract_data(self, region: str, weather_year: int, hydro_year: int, demand_year: int):
        extracted = {}
        if demand_year not in self.all_data["demand"].index.year:
            print(f"Demand year {demand_year} not in data. Using 2017 instead.")
            demand_year = 2017
        if weather_year not in self.all_data["solar"].index.year:
            print(f"Weather year {weather_year} not in data. Using 2012 instead.")
            weather_year = 2012
        if hydro_year not in self.all_data["hydro_inflow"].index.year:
            print(f"Hydro year {hydro_year} not in data. Using 2011 instead.")
            hydro_year = 2011

        if region in self.all_data["demand"].columns:
            demand_series = self.all_data["demand"].loc[self.all_data["demand"].index.year == demand_year, region]
            extracted["demand"] = demand_series[demand_series.index.strftime('%m-%d') != '02-29'][:8760]

        for carrier in ["solar", "onwind", "offwind"]:
            if region in self.all_data[carrier].columns:
                weather_series = self.all_data[carrier].loc[self.all_data[carrier].index.year == weather_year, region]
                extracted[carrier] = weather_series[weather_series.index.strftime('%m-%d') != '02-29'][:8760]

        if region in self.all_data["hydro_inflow"].columns:
            hydro_series = self.all_data["hydro_inflow"].loc[self.all_data["hydro_inflow"].index.year == hydro_year, region]
            extracted["hydro"] = hydro_series[hydro_series.index.strftime('%m-%d') != '02-29'][:8760]

        return extracted    

class Build_dispatch_network_hMC:
    def __init__(
        self,
        opt_capacities_df: pd.DataFrame,
        weather_year: int = 2011,
        hydro_year: int = 2011,
        demand_year: int = 2018,
        data: dict = None,
        cost_data: tuple = None,
        setup: dict = None
    ):
        if setup is None:
            setup = {
                'NOR': {
                    'OCGT': True,
                    'CCGT': False,
                    'battery storage': True,
                    'onwind': True,
                    'offwind': False,
                    'solar': True,
                    'electrolysis': True,
                    'fuel cell': True,
                    'Hydrogen storage': True,
                    'Reservoir hydro storage': True,
                    'Load shedding': True
                }
            }

        self.weather_year = weather_year
        self.hydro_year = hydro_year
        self.demand_year = demand_year
        self.setup = setup
        self.region = list(setup.keys())[0]

        self.costs = cost_data.costs
        self.cost_units = cost_data.units

        self.opt_caps = opt_capacities_df

        self.all_data = data if data is not None else load_all_data()
        self.data_dict = {self.region: self.extract_data(self.region, self.weather_year, self.hydro_year, self.demand_year)}

        self.network = pypsa.Network()
        self.hours_in_year = pd.date_range(f'{weather_year}-01-01 00:00', f'{weather_year}-12-31 23:00', freq='h')
        if len(self.hours_in_year) > 8760:
            self.hours_in_year = self.hours_in_year[self.hours_in_year.strftime('%m-%d') != '02-29']
        self.network.set_snapshots(self.hours_in_year.values)

        self.carriers = ['gas', 'onwind', 'offwind', 'solar', 'battery charge', 'battery discharge', 'electrolysis', 'fuel cell', 'hydrogen', 'hydro', 'load shedding']

        self.colors = {
            'gas': 'gray', 'onwind': 'lightblue', 'offwind': 'dodgerblue', 'solar': 'orange',
            'battery charge': 'gold', 'battery discharge': 'darkorange', 'electrolysis': 'green',
            'fuel cell': 'limegreen', 'hydrogen': 'deepskyblue', 'hydro': 'slateblue', 'load shedding': 'red'
        }

        self.network.add("Carrier", self.carriers, color=[self.colors[c] for c in self.carriers],
                         co2_emissions=[self.costs.at[c, "CO2 intensity"] if c in self.costs.index else 0.0 for c in self.carriers])

        self.network.add("Bus", 'electricity bus')
        self.network.add("Bus", 'hydrogen bus')
        self.network.add("Load", 'load', bus='electricity bus', p_set=self.data_dict[self.region]['demand'].values.flatten())


        for tech, active in self.setup[self.region].items():
            if not active:
                continue

            if tech in ['OCGT', 'CCGT', 'onwind', 'offwind', 'solar']:
                self.network.add("Generator", tech,
                    bus='electricity bus',
                    p_nom=self.opt_caps.at["generators", tech],
                    p_nom_extendable=False,
                    carrier=tech if tech in ['solar', 'onwind', 'offwind'] else 'gas',
                    capital_cost=0,
                    marginal_cost=self.costs.at[tech, "marginal_cost"],
                    p_max_pu=self.data_dict[self.region][tech].values.flatten() if tech != 'OCGT' else None)

            elif tech == 'Load shedding':
                # Add load shedding generator
                self.network.add("Generator", tech,
                    bus="electricity bus",
                    p_nom_extendable=True,
                    marginal_cost=2000,
                    capital_cost=0,
                    carrier="load shedding")

            elif tech == 'battery storage':
                self.network.add("Bus", 'battery bus')
                self.network.add("Link", 'battery charge',
                    bus0='electricity bus', bus1='battery bus',
                    carrier='battery charge',
                    p_nom=self.opt_caps.at['links', 'battery charge'],
                    p_nom_extendable=False,
                    capital_cost=0,
                    efficiency=self.costs.at['battery inverter', 'efficiency'])
                
                self.network.add("Link", 'battery discharge',
                    bus0='battery bus', bus1='electricity bus',
                    carrier='battery discharge',
                    p_nom=self.opt_caps.at['links', 'battery discharge'],
                    p_nom_extendable=False,
                    capital_cost=0,
                    efficiency=self.costs.at['battery inverter', 'efficiency'])
                
                self.network.add("Store", tech,
                    bus='battery bus',
                    e_nom=self.opt_caps.at['stores', 'battery storage'],
                    e_nom_extendable=False,
                    e_cyclic=False,
                    capital_cost=0)

            elif tech == 'electrolysis':
                self.network.add("Link", tech,
                    bus0='electricity bus', bus1='hydrogen bus',
                    carrier='electrolysis',
                    p_nom=self.opt_caps.at['links', tech],
                    p_nom_extendable=False,
                    capital_cost=0,
                    efficiency=self.costs.at[tech, 'efficiency'])

            elif tech == 'fuel cell':
                self.network.add("Link", tech,
                    bus0='hydrogen bus', bus1='electricity bus',
                    carrier='fuel cell',
                    p_nom=self.opt_caps.at['links', tech],
                    p_nom_extendable=False,
                    capital_cost=0,
                    efficiency=self.costs.at[tech, 'efficiency'])

            elif tech == 'Hydrogen storage':
                self.network.add("Store", tech,
                    bus='hydrogen bus',
                    e_nom=self.opt_caps.at['stores', tech],
                    e_nom_extendable=False,
                    e_cyclic=False,
                    capital_cost=0,
                    carrier='hydrogen storage')

            elif tech == 'Reservoir hydro storage':
                self.network.add("StorageUnit", tech,
                    bus='electricity bus',
                    carrier='hydro',
                    p_nom=12700,
                    p_nom_extendable=False,
                    max_hours=1300,
                    efficiency_store=0,
                    efficiency_dispatch=self.costs.at['Pumped-Storage-Hydro-bicharger', 'efficiency'],
                    cyclic_state_of_charge=False,
                    inflow=self.data_dict[self.region]['hydro'].values.flatten(),
                    state_of_charge_initial= (12700 * 1300)*0.3 ,  # Initial storage capacity in MWh
                    marginal_cost=self.costs.at["onwind", "marginal_cost"]*1.2,  # higher than wind to prioritize wind usage
                    capital_cost=0)
    
    def extract_data(self, region: str, weather_year: int, hydro_year: int, demand_year: int):
        extracted = {}
        if demand_year not in self.all_data["demand"].index.year:
            print(f"Demand year {demand_year} not in data. Using 2017 instead.")
            demand_year = 2017
        if weather_year not in self.all_data["solar"].index.year:
            print(f"Weather year {weather_year} not in data. Using 2012 instead.")
            weather_year = 2012
        if hydro_year not in self.all_data["hydro_inflow"].index.year:
            print(f"Hydro year {hydro_year} not in data. Using 2011 instead.")
            hydro_year = 2011

        if region in self.all_data["demand"].columns:
            demand_series = self.all_data["demand"].loc[self.all_data["demand"].index.year == demand_year, region]
            extracted["demand"] = demand_series[demand_series.index.strftime('%m-%d') != '02-29'][:8760]

        for carrier in ["solar", "onwind", "offwind"]:
            if region in self.all_data[carrier].columns:
                weather_series = self.all_data[carrier].loc[self.all_data[carrier].index.year == weather_year, region]
                extracted[carrier] = weather_series[weather_series.index.strftime('%m-%d') != '02-29'][:8760]

        if region in self.all_data["hydro_inflow"].columns:
            hydro_series = self.all_data["hydro_inflow"].loc[self.all_data["hydro_inflow"].index.year == hydro_year, region]
            extracted["hydro"] = hydro_series[hydro_series.index.strftime('%m-%d') != '02-29'][:8760]

        return extracted    



from typing import NamedTuple
from typing import Optional


class extreme_period(NamedTuple):
    period: pd.Interval
    peak_hour: pd.Timestamp

def get_peak_hour_from_period(
    n: pypsa.Network,
    p: pd.Interval,
) -> list:
    """Find the hour with the highest system cost (load * nodal_price) for a given interval.

    Parameters
    ----------
    n : pypsa.Network
        The network for which to find difficult periods.
    p: pd.Interval
        Period of interest, represented as pd.Interval.

    Returns
    -------
    peak_hours: list[pd.Timestamp]
        A list of the most extreme timestamp for the list of periods of interest."""

    return (
        (
            n.buses_t["marginal_price"].loc[p.left : p.right]
            * n.loads_t.p.loc[p.left : p.right]
        )
        .sum(axis=1)
        .idxmax()
    )


def global_difficult_periods(
    n: pypsa.Network,
    min_length: int,
    max_length: int,
    T: float,
    month_bounds: Optional[tuple[int, int]] = None,
) -> pd.DataFrame:
    """Find intervals with high global system cost.

    The intervals will have a length between `min_length` and
    `max_length`, over which the total system costs adds up to a value
    greater than `T`.

    NOTE: for now, this function assumes that the network `n` has
    hourly resolution!

    Parameters
    ----------
    n : pypsa.Network
        The network for which to find difficult periods.
    min_length : int
        The minimum length of the intervals to consider, in hours.
    max_length : int
        The maximum length of the intervals to consider, in hours.
    T : float
        The threshold for the total system costs to be exceeded, in EUR.
    month_bounds : Optional[tuple[int, int]] = None
        Optionally, specify in which months to search for difficult
        periods. In this argument is not None, only periods in given
        month interval are returned. The interval is inclusive and
        cyclic. For example, if `month_bounds == (11, 2)`, only
        periods contained entirely within the November-February range
        (inclusive) are returned.

    Returns
    -------
    namedtuple consisting of
        list[pd.Interval]
            A list of the periods of interest, represented as pd.Interval.
        list[pd.Timestamp]
            A list of the most extreme timestamp for the list of periods of interest.
    """

    # TODO: the following only works for hourly resolution!

    nodal_costs = n.buses_t["marginal_price"]["electricity bus"] * n.loads_t["p_set"]["load"]
    total_costs = nodal_costs

    if month_bounds is not None:
        total_costs = total_costs.loc[
            (total_costs.index.month >= month_bounds[0])
            & (total_costs.index.month <= month_bounds[1])
        ]
    #print("total_costs after if", total_costs)

    # Create an empty series, but specifying the type of index it's
    # going to have: an datetime interval index.
    C = pd.Series(
        index=pd.IntervalIndex.from_tuples(
            [], closed="both", dtype="interval[datetime64[ns], both]"
        ),
        dtype="float64",
    )

    for w in range(min_length - 1, max_length - 1):
        # Create array of intervals of width w+1
        intervals = pd.IntervalIndex.from_arrays(
            left=total_costs.index[:-w], right=total_costs.index[w:], closed="both"
        )
        # Find total costs for all intervals of width w+1
        costs = total_costs.rolling(w + 1).sum().iloc[w:]
        costs.index = intervals
        # In case we are only looking at intervals within some given
        # months, the index of `costs` might actually consist of two
        # disjoint seasons (e.g. July-October and April-June), leading
        # to some intervals that span the "gap" between these seasons.
        # Filter those out by only keeping intervals with an actual
        # duration of w+1 hours
 
        
        costs = costs.loc[costs.index.length <= pd.Timedelta(hours=w + 1)]

        # Filter out the intervals costing less than T
        costs = costs.loc[costs > T]

        # Filter out the intervals that overlap with existing intervals
        if len(C) > 0:
            costs = costs.loc[
                ~np.array([costs.index.overlaps(I) for I in C.index]).any(axis=0)
            ]

        # Also filter out intervals in `non_overlapping_I` that
        # overlap with each other. Sort the intervals by cost (from
        # highest to lowest) and take each interval in turn as long as
        # it doesn't overlap any of the previously taken intervals.
        costs = costs.sort_values(ascending=False)
        # (Again, we need to specify the type of index explicitly when
        # initialising it empty.)
        non_overlapping_I = pd.IntervalIndex.from_tuples(
            [], closed="both", dtype="interval[datetime64[ns], both]"
        )
        for I in costs.index:
            if not any(non_overlapping_I.overlaps(I)):
                # Now that we have committed to selecting the interval
                # I, we can see if it's actually natural to "expand" I
                # in either direction. We only want to expand I by
                # times at which the cost is greater than the average
                # cost of I. First, find the average cost of I.
                avg_cost = total_costs.loc[I.left : I.right].mean()
                # Now, expand I in both directions as long as the cost
                # is greater than the average cost of I
                while (
                    I.left > total_costs.index[0]
                    and total_costs.iloc[total_costs.index.searchsorted(I.left) - 1]
                    > avg_cost
                ):
                    I = pd.Interval(
                        left=total_costs.index[
                            total_costs.index.searchsorted(I.left) - 1
                        ],
                        right=I.right,
                        closed="both",
                    )
                while (
                    I.right < total_costs.index[-1]
                    and total_costs.iloc[total_costs.index.searchsorted(I.right) + 1]
                    > avg_cost
                ):
                    I = pd.Interval(
                        left=I.left,
                        right=total_costs.index[
                            total_costs.index.searchsorted(I.right) + 1
                        ],
                        closed="both",
                    )

                # Insert in sorted order
                i = non_overlapping_I.searchsorted(I)
                non_overlapping_I = non_overlapping_I.insert(i, I)

        # Add the intervals we found one by one. However, since the
        # intervals may be been extended, they may now still overlap
        # with some of the intervals in the index of C.
        # NOTE: this code path is not taken for the periods of our paper!
        for I in non_overlapping_I:
            if len(C) == 0:
                C.loc[I] = total_costs.loc[I.left : I.right].sum()
            else:
                # Find intervals in C that overlap with I
                overlapping_I = C.index[C.index.overlaps(I)]
                # Remove them from C
                C = C.drop(overlapping_I)
                # Create the union of I and all the intervals it overlaps with.
                left = min([I.left, *overlapping_I.left])
                right = max([I.right, *overlapping_I.right])
                I = pd.Interval(left=left, right=right, closed="both")
                # Add the union to C
                C.loc[I] = total_costs.loc[I.left : I.right].sum()

        C = C.sort_index()

    return [extreme_period(p, get_peak_hour_from_period(n, p)) for p in C.index]
