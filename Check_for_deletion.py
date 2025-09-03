# Dispatch plots for all regions
# %%
def plot_dispatch(network_obj, save_plots=False):
    generators = network_obj.network.generators.index
    storage_units = network_obj.network.storage_units
    regions = list(network_obj.setup.keys()) if hasattr(network_obj, "setup") else [""]

    for region in regions:
        plt.figure(figsize=(10, 5))

        # Plot generator dispatch
        for generator in generators:
            name = generator  # no region suffix anymore
            if network_obj.network.generators.p_nom_opt.get(generator, 0) > 10:
                plt.plot(
                    network_obj.network.generators_t.p[generator][0:7 * 24],
                    label=name,
                    color=network_obj.colors.get(name, None)
                )

        # Plot hydro dispatch (StorageUnit)
        hydro_mask = (storage_units.carrier == "hydro") & (storage_units.bus == "electricity bus")
        if hydro_mask.any():
            for idx in storage_units[hydro_mask].index:
                plt.plot(
                    network_obj.network.storage_units_t.p_dispatch[idx][0:7 * 24],
                    label="hydro dispatch",
                    color=network_obj.colors.get("hydro", "blue"),
                    linestyle="--"
                )
                plt.plot(
                    network_obj.network.storage_units_t.state_of_charge[idx][0:7 * 24],
                    label="hydro soc",
                    color="green",
                    linestyle="-.",
                    linewidth=1.5
                )

        # Plot load
        load_name = 'load'
        if load_name in network_obj.network.loads.index:
            plt.plot(
                network_obj.network.loads_t.p_set[load_name][0:7 * 24],
                label="load",
                color="black",
                linestyle=":"
            )

        plt.title(f'Dispatch Winter - {region}', y=1.07)
        plt.ylabel('Generation in MWh')
        plt.grid(True, which='major', alpha=0.25)
        plt.legend()
        if save_plots:
            plt.savefig(f'./Plots/dispatch_{region}_winter.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Summer Plot
        plt.figure(figsize=(10, 5))
        for generator in generators:
            name = generator  # updated
            if network_obj.network.generators.p_nom_opt.get(generator, 0) > 10:
                plt.plot(
                    network_obj.network.generators_t.p[generator][4993: 4993 + 7 * 24],
                    label=name,
                    color=network_obj.colors.get(name, None)
                )

        if hydro_mask.any():
            for idx in storage_units[hydro_mask].index:
                plt.plot(
                    network_obj.network.storage_units_t.p_dispatch[idx][4993: 4993 + 7 * 24],
                    label="hydro dispatch",
                    color=network_obj.colors.get("hydro", "blue"),
                    linestyle="--"
                )
                plt.plot(
                    network_obj.network.storage_units_t.state_of_charge[idx][4993: 4993 + 7 * 24],
                    label="hydro soc",
                    color="green",
                    linestyle="-.",
                    linewidth=1.5
                )

        if load_name in network_obj.network.loads.index:
            plt.plot(
                network_obj.network.loads_t.p_set[load_name][4993: 4993 + 7 * 24],
                label="load",
                color="black",
                linestyle=":"
            )

        plt.title(f'Dispatch Summer - {region}', y=1.07)
        plt.ylabel('Generation in MWh')
        plt.legend()
        plt.grid(True, which='major', alpha=0.25)
        if save_plots:
            plt.savefig(f'./Plots/dispatch_{region}_summer.png', dpi=300, bbox_inches='tight')
        plt.show()


# another dispatch plot function
# %% another dispatch plot
def plot_dispatch(network_obj, save_plots=False, start_hour=0, duration_hours=7 * 24):
    generators = network_obj.network.generators.index
    storage_units = network_obj.network.storage_units
    regions = list(network_obj.setup.keys()) if hasattr(network_obj, "setup") else [""]

    end_hour = start_hour + duration_hours

    for region in regions:
        plt.figure(figsize=(10, 5))

        # Plot generator dispatch
        for generator in generators:
            name = generator  # no region suffix
            if network_obj.network.generators.p_nom_opt.get(generator, 0) > 10:
                plt.plot(
                    network_obj.network.generators_t.p[generator][start_hour:end_hour],
                    label=name,
                    color=network_obj.colors.get(name, None)
                )

        # Plot hydro dispatch
        hydro_mask = (storage_units.carrier == "hydro") & (storage_units.bus == "electricity bus")
        if hydro_mask.any():
            for idx in storage_units[hydro_mask].index:
                plt.plot(
                    network_obj.network.storage_units_t.p_dispatch[idx][start_hour:end_hour],
                    label="hydro dispatch",
                    color=network_obj.colors.get("hydro", "green"),
                    linestyle="--"
                )
                # plt.plot(
                #     network_obj.network.storage_units_t.state_of_charge[idx][start_hour:end_hour],
                #     label="hydro soc",
                #     color="green",
                #     linestyle="-.",
                #     linewidth=1.5
                # )

        # Plot load
        load_name = 'load'
        if load_name in network_obj.network.loads.index:
            plt.plot(
                network_obj.network.loads_t.p_set[load_name][start_hour:end_hour],
                label="load",
                color="black",
                linestyle=":"
            )

        # Plot formatting
        plt.title(f'Dispatch [{start_hour}:{end_hour}] - {region}', y=1.07)
        plt.ylabel('Generation in MWh')
        plt.grid(True, which='major', alpha=0.25)
        plt.legend()
        if save_plots:
            plt.savefig(f'./Plots/dispatch_{region}_{start_hour}.png', dpi=300, bbox_inches='tight')
        plt.show()


# %% 

#%%