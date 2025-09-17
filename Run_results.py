from functions import *
from Classes import *

#%%
# Define the use technologies and regions(s)
region = 'ESP'  
setup_exp = {
    f'{region}': {
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
        'Load shedding': False
    }
}

setup_dispatch = {
    f'{region}': {
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

# Default weather, hydro and demand years
w_year_exp = 2011
h_year_exp = 2007
d_year_exp = 2018

# Dispatch and rolling horizon settings
w_year_dispatch = 2017
h_year_dispatch = 2007
d_year_dispatch = 2019

base_dir = os.getcwd()  # Gets the current working directory (where notebook is running)
N_results_path = os.path.join(base_dir, "Network_results")
Results_path = os.path.join(base_dir, "Results")
os.makedirs(N_results_path, exist_ok=True)
weather_years = All_data['solar'].index.year.unique()

Test_name = "_gas_noise"  # Name to append to files for identification

#%%         RUN EXPANSION MODEL
" Running just one time for testing purposes "
# N_class = Build_network_capacity_exp(weather_year=w_year_exp, hydro_year=h_year_exp, demand_year=d_year_exp,
#     data=All_data, cost_data=Cost, setup=setup_exp)
# N = N_class.network

# silent_optimize(N)

# print_Results(N)

" Running for all weather years "

Saved_Networks_names = []

save_network = True
if save_network:

    for year in weather_years:
        N = Build_network_capacity_exp_gas(weather_year=year, hydro_year=h_year_exp, demand_year=d_year_exp,
            data=All_data, cost_data=Cost, setup=setup_exp).network

        silent_optimize(N)
        
        #print(f"\nResults for weather year {year}:\n")
        #print_Results(N)

        network_name = f"N_w-{year}_d-{d_year_exp}_h-{h_year_exp}_{region}{Test_name}.nc"
        N.export_to_netcdf(os.path.join(N_results_path, network_name))
        
        Saved_Networks_names.append(network_name)
        print(f"Saved network for year {year} as {network_name}")

##%        Read saved networks an create summary of capacities 
" Read saved networks and create summary of capacities "

New_results = True  # Set to False to load existing results file
if New_results:
    # Initialize
    optmized_network_names_exp = []
    networks_exp = {}

    # Load network files
    for year in weather_years:
        net_name = f'N_{year}_{region}'
        N_file_name = f"N_w-{year}_d-{d_year_exp}_h-{h_year_exp}_{region}{Test_name}.nc"
        file_path = os.path.join(N_results_path, N_file_name)
        if os.path.exists(file_path):
            networks_exp[net_name] = pypsa.Network(file_path)
            optmized_network_names_exp.append(net_name)

    # Prepare results dict
    results_exp_wo_LS = {
        "generators": pd.DataFrame(index=weather_years),
        "links": pd.DataFrame(index=weather_years),
        "stores": pd.DataFrame(index=weather_years),
    }

    # Extract data per component
    for year, name in zip(weather_years, optmized_network_names_exp):
        net = networks_exp[name]
        if not net.generators.empty:
            results_exp_wo_LS["generators"].loc[year, net.generators.index] = net.generators["p_nom_opt"]
        if not net.links.empty:
            results_exp_wo_LS["links"].loc[year, net.links.index] = net.links["p_nom_opt"]
        if not net.stores.empty:
            results_exp_wo_LS["stores"].loc[year, net.stores.index] = net.stores["e_nom_opt"]

    # Combine all into one DataFrame
    aggregated_df = pd.concat(results_exp_wo_LS, axis=1)

    # Compute extended statistics
    stats_dict = {
        component: df.describe(percentiles=[0.25, 0.5, 0.75]).loc[
            ["mean", "std", "min", "25%", "50%", "75%", "max"]
        ]
        for component, df in aggregated_df.items()
    }

    opt_capacities_df = pd.concat(stats_dict, axis=1)

    # Save final result
    file_name = f"optimized_capacities_exp_model_d{d_year_exp}_h{h_year_exp}{Test_name}.csv"
    opt_capacities_df.to_csv(os.path.join(Results_path, file_name))
else:
    Name = f"optimized_capacities_exp_model_d{d_year_exp}_h{h_year_exp}{Test_name}.csv"
    opt_capacities_df = pd.read_csv(os.path.join(Results_path, Name), header=[0,1], index_col=0)

#print(opt_capacities_df.head(7))

#%%        RUN DISPATCH MODEL
" Running just one time for testing purposes "
# N_dispatch_class = Build_dispatch_network(
#     opt_capacities_df=opt_capacities_df.loc["75%"],
#     weather_year=2012, hydro_year=h_year_dispatch, demand_year=d_year_dispatch,
#     data=All_data, cost_data=Cost, setup=setup_dispatch)

# N_dispatch = N_dispatch_class.network

# silent_optimize(N_dispatch)

# print_Results(N_dispatch)

#%%        RUN ROLLING HORIZON
" Running for all weather years "

d_horizon = 7 
h_overlap = 0  

save_networks_RH = False  

if save_networks_RH:

    for year in weather_years:

        N_dispatch_class = Build_dispatch_network(
            opt_capacities_df=opt_capacities_df.loc["75%"], # HUSK AT SØRG FOR OPT CAPACITIES ER LOADED FØRST 
            weather_year=year, hydro_year=h_year_dispatch, demand_year=d_year_dispatch,
            data=All_data, cost_data=Cost, setup=setup_dispatch)

        N_dispatch = N_dispatch_class.network
        silent_optimize(N_dispatch)
            
        N_rlh_class = Build_dispatch_network(
            opt_capacities_df=opt_capacities_df.loc["75%"],
            weather_year=year, 
            hydro_year=h_year_dispatch, 
            demand_year=d_year_dispatch,
            data=All_data, cost_data=Cost, setup=setup_dispatch)
        N_rlh = N_rlh_class.network

        N_rlh.optimize.optimize_with_rolling_horizon(
            snapshots=N_rlh.snapshots,
            horizon=(24*d_horizon),               
            overlap= h_overlap,                
            solver_name="gurobi", solver_options={"OutputFlag": 0})


        network_name_rlh = f"N_RHL_w-{year}_d-{d_year_dispatch}_h-{h_year_dispatch}_{region}{Test_name}.nc"
        network_name_pf = f"N_PF_w-{year}_d-{d_year_dispatch}_h-{h_year_dispatch}_{region}{Test_name}.nc"
        N_rlh.export_to_netcdf(os.path.join(N_results_path, network_name_rlh))
        N_dispatch.export_to_netcdf(os.path.join(N_results_path, network_name_pf))

" Running just for one year for testing purposes "
#%%
test_year = 1995

N_dispatch_class = Build_dispatch_network_hMC(
    opt_capacities_df=opt_capacities_df.loc["75%"], # HUSK AT SØRG FOR OPT CAPACITIES ER LOADED FØRST 
    weather_year=test_year, hydro_year=h_year_dispatch, demand_year=d_year_dispatch,
    data=All_data, cost_data=Cost, setup=setup_dispatch)

N_dispatch = N_dispatch_class.network
silent_optimize(N_dispatch)

N_rlh_class = Build_dispatch_network_hMC(
    opt_capacities_df=opt_capacities_df.loc["75%"],
    weather_year=test_year, 
    hydro_year=h_year_dispatch, 
    demand_year=d_year_dispatch,
    data=All_data, cost_data=Cost, setup=setup_dispatch)
N_rlh = N_rlh_class.network

N_rlh.optimize.optimize_with_rolling_horizon(
    snapshots=N_rlh.snapshots,
    horizon=(24*d_horizon),               
    overlap= h_overlap,                
    solver_name="gurobi", solver_options={"OutputFlag": 0})

#%%       COMPARE RESULTS
# Get marginal prices for N_dispatch and N_rlh (rolling horizon)
MC_N_pf = N_dispatch.buses_t['marginal_price'].iloc[:, 0]
MC_N_rlh = N_rlh.buses_t['marginal_price'].iloc[:, 0]
tot_cost_Np = (MC_N_pf * N_dispatch.loads_t.p_set['load']).sum() / 1e6
tot_cost_Nr = (MC_N_rlh * N_rlh.loads_t.p_set["load"]).sum() / 1e6
print(f"Hydro year {h_year_dispatch}, demand year {d_year_dispatch} and weather year {test_year}:")
print("Total cost (N_perfect) [MEUR]:", round(tot_cost_Np, 2))
print("Total cost (N_rolling) [MEUR]:", round(tot_cost_Nr, 2))

total_difference = tot_cost_Np - tot_cost_Nr
print("Total cost difference PF minus RLH [MEUR]:", round(total_difference, 2))

