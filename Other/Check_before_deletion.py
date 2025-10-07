

# %% Rolling horizon patching for old pypsa
# OLD VERSION for OLDER PYPSA VERSIONS
def optimize_with_rolling_horizon_collect(self, snapshots=None, horizon=100, overlap=0, **kwargs):
    """
    Custom rolling horizon optimization that also collects objectives
    and stores them in n.attrs.
    """
    n = self.n   # <- works for your PyPSA version

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

    # Save to attrs so it survives export_to_netcdf
    n.generators.attrs["rolling_objectives"] = objs
    n.generators.attrs["rolling_runs"] = runs

    return n
#%% Rolling horizon patching for old pypsa
def rh_pf_test_yearly(test_year):    
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
        optimize_with_rolling_horizon_collect, N_rlh.optimize    )
    
    N_rlh.optimize_with_rolling_horizon(
        snapshots=N_rlh.snapshots,
        horizon=24*7,
        overlap=0,
        solver_name="gurobi",
        solver_options={"OutputFlag": 0},
        assign_all_duals=True
    )

    return N_dispatch, N_rlh, N_rlh.generators.attrs["rolling_objectives"]