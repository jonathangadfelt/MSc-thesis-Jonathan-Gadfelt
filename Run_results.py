from functions import *
from Classes import *

#%%
# Define the use technologies and regions(s)
region = 'ESP'  
setup_exp = {
    f'{region}': {
        'OCGT': True,
        'CCGT': False,
        'battery storage': False,
        'onwind': True,
        'offwind': False,
        'solar': True,
        'electrolysis': False,
        'fuel cell': False,
        'Hydrogen storage': False,
        'Reservoir hydro storage': True,
        'load shedding': False
    }
}

setup_dispatch = {
    f'{region}': {
        'OCGT': True,
        'CCGT': False,
        'battery storage': False,
        'onwind': True,
        'offwind': False,
        'solar': True,
        'electrolysis': False,
        'fuel cell': False,
        'Hydrogen storage': False,
        'Reservoir hydro storage': True,
        'load shedding': True
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

Test_name = "_0BS_100SPC_hMC_50"  # Name to append to files for identification

#%%        RUN EXPANSION MODEL
" Running just one time for testing purposes "
# N_class = Build_network_capacity_exp(weather_year=w_year_exp, hydro_year=h_year_exp, demand_year=d_year_exp,
#     data=All_data, cost_data=Cost, setup=setup_exp)
# N = N_class.network

# silent_optimize(N)

# print_Results(N)

" Running for all weather years "

save_network = True

if save_network:
    # make experiment subfolder
    exp_folder = f"N_EXP_d_{d_year_exp}_h_{h_year_exp}{Test_name}"
    exp_path = os.path.join(N_results_path, exp_folder)
    os.makedirs(exp_path, exist_ok=True)

    for year in weather_years:
        N = Build_network_capacity_exp(weather_year=year, hydro_year=h_year_exp, demand_year=d_year_exp,
            data=All_data, cost_data=Cost, setup=setup_exp).network

        silent_optimize(N)
        
        network_name = f"N_w-{year}_d-{d_year_exp}_h-{h_year_exp}_{region}{Test_name}.nc"
        N.export_to_netcdf(os.path.join(exp_path, network_name))
        
        print(f"Saved network for year {year} as {network_name}")

##%        Read saved networks an create summary of capacities 
" Read saved networks and create summary of capacities "

New_results = True  # Set to False to load existing results file
Test_key = "_hMC"   # for loading existing results file

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
    Name = f"optimized_capacities_exp_model_d{d_year_exp}_h{h_year_exp}{Test_key}.csv"
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
    # create subfolders for PF and RH
    pf_folder = f"N_PF_d_{d_year_dispatch}_h_{h_year_dispatch}{Test_name}"
    rh_folder = f"N_RH_d_{d_year_dispatch}_h_{h_year_dispatch}{Test_name}"

    pf_path = os.path.join(N_results_path, pf_folder)
    rh_path = os.path.join(N_results_path, rh_folder)
    os.makedirs(pf_path, exist_ok=True)
    os.makedirs(rh_path, exist_ok=True)


    df = pd.DataFrame(index=weather_years, columns=["total objective"])
    

    for year in weather_years:
        # PF
        N_dispatch_class = Build_dispatch_network_hMC(
            opt_capacities_df=opt_capacities_df.loc["75%"], 
            weather_year=year, 
            hydro_year=h_year_dispatch, 
            demand_year=d_year_dispatch,
            data=All_data, cost_data=Cost, setup=setup_dispatch
        )
        N_dispatch = N_dispatch_class.network
        silent_optimize(N_dispatch)

        # RH
        N_rlh_class = Build_dispatch_network_hMC(
            opt_capacities_df=opt_capacities_df.loc["75%"],
            weather_year=year, 
            hydro_year=h_year_dispatch, 
            demand_year=d_year_dispatch,
            data=All_data, cost_data=Cost, setup=setup_dispatch
        )
        N_rlh = N_rlh_class.network


        # --- monkey-patch onto your network instance ---
        # net_rh = ...  # your network
        N_rlh.optimize.optimize_with_rolling_horizon = MethodType(
            optimize_with_rolling_horizon_collect, N_rlh.optimize
        )

        N_rlh.optimize.optimize_with_rolling_horizon(
            snapshots=N_rlh.snapshots,
            horizon=(24 * d_horizon),
            overlap=h_overlap,
            solver_name="gurobi",
            solver_options={"OutputFlag": 0}
        )

        df.loc[year, "total objective"] = sum(N_rlh.generators.attrs["rolling_objectives"])

        # filenames
        network_name_pf = f"N_PF_w-{year}_d-{d_year_dispatch}_h-{h_year_dispatch}_{region}{Test_name}.nc"
        network_name_rlh = f"N_RH_w-{year}_d-{d_year_dispatch}_h-{h_year_dispatch}_{region}{Test_name}.nc"

        # save to their respective folders
        N_dispatch.export_to_netcdf(os.path.join(pf_path, network_name_pf))
        N_rlh.export_to_netcdf(os.path.join(rh_path, network_name_rlh))

        print(f"Saved PF network for {year} as {network_name_pf} in {pf_folder}")
        print(f"Saved RH network for {year} as {network_name_rlh} in {rh_folder}")
    df.to_csv(f"rolling_objectives_{network_name_rlh}.csv")

# --- imports you need ---


" Running just for one year for testing purposes "
#%%        TEST PERFECT AND ROLLING HORIZON FOR ONE YEAR
one_rh_pf_test = False
test_year = 1995
if one_rh_pf_test:
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


    # --- monkey-patch onto your network instance ---
    # net_rh = ...  # your network
    N_rlh.optimize.optimize_with_rolling_horizon = MethodType(
        optimize_with_rolling_horizon_collect, N_rlh.optimize
    )

    objectives = []
    run_counter = {"i": 0}   # mutable container to keep state across calls

    # def collect_objective(n, snapshots):
    #     run_counter["i"] += 1
    #     print(f"Rolling horizon run {run_counter['i']} with snapshots {snapshots[0]} → {snapshots[-1]}")
    #     objectives.append(n.objective)

    # N_rlh.optimize.optimize_with_rolling_horizon_selfmade(
    #     snapshots=N_rlh.snapshots,
    #     horizon=24*d_horizon,
    #     overlap=h_overlap,
    #     solver_name="gurobi",
    #     solver_options={"OutputFlag": 0}
    #     #extra_functionality=collect_objective
    # )


    # --- call exactly like before ---
    N_rlh.optimize.optimize_with_rolling_horizon(
        snapshots=N_rlh.snapshots,
        horizon=24*7,
        overlap=0,
        solver_name="gurobi",
        solver_options={"OutputFlag": 0},
        assign_all_duals=True
    )



#%%        COMPARE RESULTS
# Get marginal prices for N_dispatch and N_rlh (rolling horizon)

if one_rh_pf_test:
    MC_N_pf = N_dispatch.buses_t['marginal_price'].iloc[:, 0]
    MC_N_rlh = N_rlh.buses_t['marginal_price'].iloc[:, 0]
    tot_cost_Np = (MC_N_pf * N_dispatch.loads_t.p_set['load']).sum() / 1e6
    tot_cost_Nr = (MC_N_rlh * N_rlh.loads_t.p_set["load"]).sum() / 1e6
    print(f"Hydro year {h_year_dispatch}, demand year {d_year_dispatch} and weather year {test_year}:")
    print("Total cost (N_perfect) [MEUR]:", round(tot_cost_Np, 2))
    print("Total cost (N_rolling) [MEUR]:", round(tot_cost_Nr, 2))

    total_difference = tot_cost_Nr - tot_cost_Np
    print("Total cost difference RH minus PF [MEUR]:", round(total_difference, 2))
    print("spill RH:", N_rlh.storage_units_t.spill["Reservoir hydro storage"].sum()
        , "spill PF:", N_dispatch.storage_units_t.spill["Reservoir hydro storage"].sum())

    #total_objective_rh = sum(objectives_rh)
    #print("Rolling horizon total objective:", total_objective_rh)
    print("Perfect foresight objective:", N_dispatch.objective)
    print("Total objective n_rlh:", sum(getattr(N_rlh, "rolling_objectives", [])))
    print("Differences in obj N_rlh - N_pf:", sum(getattr(N_rlh, "rolling_objectives", [])) - N_dispatch.objective)
    print("accessing rh obj using rolling_objectives: ", sum(N_rlh.rolling_objectives))

