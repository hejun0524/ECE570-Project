import json
from src.instance.constants import *
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"


class Grapher:

    def __init__(
        self,
        fnames: list[str],
        fshape: str,
        figsize: tuple = (6, 3),
    ) -> None:
        self.data = []
        for fname in fnames:
            with open(fname, "r") as f:
                self.data.append(json.load(f))
        with open(fshape, "r") as f:
            self.shape = json.load(f)
        self.n_sims = len(self.data)
        self.figsize = figsize
        self.n_steps_one_day = self.data[0][N_STEPS_ONE_DAY]
        self.hour_per_step = 24 // self.n_steps_one_day
        self.n_agents = self.data[0][N_AGENTS]

    def _line_plot(
        self,
        data_dict: dict,
        title: str,
        xlabel: str,
        ylabel: str,
        data_length: int,
        figsize: tuple | None = None,
        save_as: str | None = None,
    ):
        """
        `data_dict`:dict, must be { <label>: <data_list> }
        """
        xs = [t for t in range(data_length)]
        xnames = [t % self.n_steps_one_day * self.hour_per_step for t in xs]
        xfreq = self.n_steps_one_day // 4

        # Plotting
        fig = plt.figure(figsize=self.figsize if figsize is None else figsize)
        for label, data_list in data_dict.items():
            if isinstance(data_list[0], list):
                # compute mean and std
                ymean = np.mean(data_list, axis=0)
                ystd = np.std(data_list, axis=0)
                ymin = np.array(data_list).min(axis=0)
                ymax = np.array(data_list).max(axis=0)
                plt.plot(xs, ymean, marker="o", linestyle="-", label=label)
                plt.fill_between(xs, ymean - ystd, ymean + ystd, alpha=0.3)
                # plt.fill_between(xs, ymin, ymax, alpha=0.3)
            else:
                plt.plot(xs, data_list, marker="o", linestyle="-", label=label)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(xs[::xfreq], xnames[::xfreq])
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        if save_as is not None:
            fig.savefig(save_as, dpi=fig.dpi)

    def _get_day_range(self, n_days):
        if n_days >= 0:
            return f"First {n_days}"
        return f"Last {-n_days}"

    def _truncate(self, arrays, T):
        # trancate the arrays (arr of arr) based on T
        filtered_arrays = []
        for arr in arrays:
            filtered_arrays.append(arr[:T] if T > 0 else arr[T:])
        return filtered_arrays

    def _average(self, aggregate_arrays):
        # average the arrays over the number of agents (for aggregate)
        return self._scale(aggregate_arrays, self.n_agents)
    
    def _cumulative_sum(self, arrays):
        cumulative_arrays = []
        for arr in arrays:
            cumulative_arrays.append([sum(arr[:i]) for i in range(len(arr))])
        return cumulative_arrays
    
    def _scale(self, arrays, scale):
        scaled_arrays = []
        for arr in arrays:
            scaled_arrays.append([x / scale for x in arr])
        return scaled_arrays
    
    def _get_arrays(self, category_key, target_key):
        # get arrays from data for all simulations
        # eg. category_key: LMP, target_key: bname
        return [self.data[i][category_key][target_key] for i in range(self.n_sims)]

    def plot_shapes(self, figsize: tuple | None = None, save_as: str | None = None,):
        nd_prosumer = self.shape["Net demand (prosumer)"]["Data"]
        nd_consumer = self.shape["Net demand (consumer)"]["Data"]
        # Plot
        self._line_plot(
            data_dict={"Prosumer": nd_prosumer, "Consumer": nd_consumer},
            title="Prosumer & Consumer Net Demand Shape",
            xlabel="Hour of the Day",
            ylabel="Percentage w.r.t. Storage Capacity",
            data_length=len(nd_prosumer),
            figsize=figsize,
            save_as=save_as,
        )

    def plot_prices(self, n_days: int, bname=HUB, figsize: tuple | None = None, save_as: str | None = None,):
        """
        n_days: int, plot first `n_days` if >0, or last if <0.
        """
        T = n_days * self.n_steps_one_day
        lmps = self._truncate(self._get_arrays(LMP, bname), T)
        lmps0 = self._truncate(self._get_arrays(LMP0, bname), T)
        if bname == HUB:
            price_name = "Hub Price"
        else:
            price_name = f"Locational Marginal Price of Bus {bname}"
        # Plot
        self._line_plot(
            data_dict={"With battery": lmps, "Without battery": lmps0},
            title=f"{price_name} for the {self._get_day_range(n_days)} Days",
            xlabel="Hour of the Day",
            ylabel="Price ($/MW)",
            data_length=abs(T),
            figsize=figsize,
            save_as=save_as,
        )

    def plot_costs(self, n_days: int, bname=HUB, figsize: tuple | None = None, save_as: str | None = None,):
        """
        n_days: int, plot first `n_days` if >0, or last if <0.
        """
        T = n_days * self.n_steps_one_day
        costs = self._cumulative_sum(self._get_arrays(COST, bname))
        costs0 = self._cumulative_sum(self._get_arrays(COST0, bname))
        costs = self._truncate(self._scale(costs, 1000), T)
        costs0 = self._truncate(self._scale(costs0, 1000), T)
        if bname == HUB:
            costs = self._average(costs)
            costs0 = self._average(costs0)
            cost_name = "Network Average Payoff (Costs)"
        else:
            cost_name = f"Payoffs (Costs) at Bus {bname}"
        # Plot
        self._line_plot(
            data_dict={"With battery": costs, "Without battery": costs0},
            title=f"{cost_name} for the {self._get_day_range(n_days)} Days",
            xlabel="Hour of the Day",
            ylabel="Cost (k$)",
            data_length=abs(T),
            figsize=figsize,
            save_as=save_as,
        )

    def plot_actions(self, n_days: int, aid=AGGREGATE, figsize: tuple | None = None, save_as: str | None = None,):
        """
        n_days: int, plot first `n_days` if >0, or last if <0.
        """
        T = n_days * self.n_steps_one_day
        actions = self._truncate(self._get_arrays(ACTION, aid), T)
        if aid == AGGREGATE:
            actions = self._average(actions)
            action_name = "Network Average Action"
        else:
            action_name = f"Action of Agent {aid}"
        self._line_plot(
            data_dict={"Action": actions},
            title=f"{action_name} for the {self._get_day_range(n_days)} Days",
            xlabel="Hour of the Day",
            ylabel="Percentage of Charging/Discharging",
            data_length=abs(T),
            figsize=figsize,
            save_as=save_as,
        )

    def plot_battery_levels(
        self, n_days: int, aid=AGGREGATE, figsize: tuple | None = None, save_as: str | None = None,
    ):
        """
        n_days: int, plot first `n_days` if >0, or last if <0.
        """
        T = n_days * self.n_steps_one_day
        battery_levels = self._truncate(
            self._get_arrays(BATTERY_LEVEL, aid), T)
        if aid == AGGREGATE:
            battery_levels = self._average(battery_levels)
            battery_name = "Network Average Battery Level"
        else:
            battery_name = f"Battery Level of Agent {aid}"
        self._line_plot(
            data_dict={"Battery Level": battery_levels},
            title=f"{battery_name} for the {self._get_day_range(n_days)} Days",
            xlabel="Hour of the Day",
            ylabel="Battery Level",
            data_length=abs(T),
            figsize=figsize,
            save_as=save_as,
        )

    def plot_renewables(
        self, n_days: int, gname=AGGREGATE, figsize: tuple | None = None, wind_save_as: str | None = None, solar_save_as: str | None = None,
    ):
        """
        n_days: int, plot first `n_days` if >0, or last if <0.
        """
        T = n_days * self.n_steps_one_day
        winds = self._truncate(self._get_arrays(WIND, gname), T)
        solars = self._truncate(self._get_arrays(SOLAR, gname), T)
        if gname == AGGREGATE:
            wind_name = "Average Wind Generation Capacity Factor at Grid"
            solar_name = "Average Solar Generation Capacity Factor at Grid"
        else:
            wind_name = f"Wind Generator Capacity Factor of Generator {gname}"
            solar_name = f"Solar Generator Capacity Factor of Generator {gname}"
        self._line_plot(
            data_dict={"Wind": winds},
            title=f"{wind_name} for the {self._get_day_range(n_days)} Days",
            xlabel="Hour of the Day",
            ylabel="Capacity Factor",
            data_length=abs(T),
            figsize=figsize,
            save_as=wind_save_as,
        )
        self._line_plot(
            data_dict={"Solar": solars},
            title=f"{solar_name} for the {self._get_day_range(n_days)} Days",
            xlabel="Hour of the Day",
            ylabel="Capacity Factor",
            data_length=abs(T),
            figsize=figsize,
            save_as=solar_save_as,
        )
    
    def plot_imvs(self, n_days: int, bname=HUB, figsize: tuple | None = None, save_as: str | None = None,):
        """
        n_days: int, plot first `n_days` if >0, or last if <0.
        """
        T = n_days * self.n_steps_one_day
        imvs = self._truncate(self._get_arrays(IMV, bname), T)
        imvs0 = self._truncate(self._get_arrays(IMV0, bname), T)
        if bname == HUB:
            imv_name = "IMV of Hub Price"
        else:
            imv_name = f"IMV of Locational Marginal Price of Bus {bname}"
        # Plot
        self._line_plot(
            data_dict={"With battery": imvs, "Without battery": imvs0},
            title=f"{imv_name} for the {self._get_day_range(n_days)} Days",
            xlabel="Hour of the Day",
            ylabel="IMV",
            data_length=abs(T),
            figsize=figsize,
            save_as=save_as,
        )