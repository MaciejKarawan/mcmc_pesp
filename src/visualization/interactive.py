"""Interactive visualization widgets for Jupyter notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets

if TYPE_CHECKING:
    from src.core.network import Network
    from src.core.transfers import TransferSet
    from src.cost.transfer_cost import TransferCost
    from src.visualization.mcmc_history import MCMCHistory
    from src.visualization.network_plot import NetworkPlotter


@dataclass
class MCMCPlayer:
    """Interactive player for MCMC history visualization.

    Allows stepping through MCMC history with controls.
    """

    history: MCMCHistory
    network: Network
    transfers: TransferSet
    cost_fn: TransferCost
    plotter: NetworkPlotter | None = None

    def __post_init__(self) -> None:
        from src.visualization.network_plot import NetworkPlotter
        from src.core.schedule import Schedule

        if self.plotter is None:
            self.plotter = NetworkPlotter(
                network=self.network,
                transfers=self.transfers,
            )

        self._current_step = 0
        self._output = widgets.Output()
        self._playing = False
        self._play_speed = 100  # ms between frames

    def _get_schedule(self, step: int):
        """Get Schedule object at a given step."""
        from src.core.schedule import Schedule

        offsets = self.history.get_schedule_at_step(step)
        return Schedule(offsets=offsets, period=self.history.period)

    def _render_step(self, step: int) -> None:
        """Render visualization for a specific step."""
        with self._output:
            clear_output(wait=True)

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # 1. Network diagram
            schedule = self._get_schedule(step)
            summary = self.history.summary_at_step(step)
            cost = summary["cost"]
            best_cost = summary["best_cost"]
            accepted = summary["accepted"]

            title = f"Step {step}: Cost = {cost:.2f} (Best: {best_cost:.2f})"
            if step >= 0:
                if accepted:
                    title += " ✓ ACCEPTED"
                else:
                    title += " ✗ rejected"

            self.plotter.plot_schedule(schedule, ax=axes[0, 0], title=title)

            # 2. Cost trace
            costs = self.history.get_costs()
            ax_cost = axes[0, 1]
            ax_cost.plot(costs[:step + 1], "b-", linewidth=1, alpha=0.7)
            if step >= 0 and step < len(costs):
                ax_cost.scatter([step], [costs[step]], color="red", s=100, zorder=5)

            # Best cost line
            best_costs = []
            current_best = float("inf")
            for c in costs[:step + 1]:
                current_best = min(current_best, c)
                best_costs.append(current_best)
            ax_cost.plot(best_costs, "g-", linewidth=2, label="Best")

            # Optimal line if known
            if self.history.optimal_cost is not None:
                ax_cost.axhline(
                    y=self.history.optimal_cost,
                    color="green",
                    linestyle="--",
                    label=f"Optimal ({self.history.optimal_cost:.2f})",
                )

            ax_cost.set_xlabel("Step")
            ax_cost.set_ylabel("Cost")
            ax_cost.set_title("Cost Trace")
            ax_cost.legend()
            ax_cost.grid(True, alpha=0.3)
            if len(costs) > 0:
                ax_cost.set_xlim(-1, len(costs))

            # 3. Acceptance rate
            ax_ar = axes[1, 0]
            if step > 0:
                acceptance_rates = [
                    self.history.get_acceptance_rate_up_to(i)
                    for i in range(step + 1)
                ]
                ax_ar.plot(acceptance_rates, "b-", linewidth=2)
                ax_ar.axhline(y=0.234, color="red", linestyle="--",
                              label="Optimal (0.234)")
                ax_ar.axhline(y=0.44, color="orange", linestyle="--",
                              label="High-dim optimal (0.44)")
                ax_ar.scatter([step], [acceptance_rates[-1]], color="red", s=100, zorder=5)
            ax_ar.set_xlabel("Step")
            ax_ar.set_ylabel("Acceptance Rate")
            ax_ar.set_title("Cumulative Acceptance Rate")
            ax_ar.set_ylim(0, 1)
            ax_ar.legend()
            ax_ar.grid(True, alpha=0.3)

            # 4. Step info
            ax_info = axes[1, 1]
            ax_info.axis("off")

            info_text = f"""
Step Information
{'='*40}

Step:           {step} / {len(self.history) - 1}
Current Cost:   {cost:.4f}
Best Cost:      {best_cost:.4f}
Acceptance Rate: {summary['acceptance_rate']:.3f}

Proposal:       {summary.get('proposal_name', 'N/A')}
Changed Line:   {summary.get('proposed_line', 'N/A')}
Delta:          {summary.get('proposed_delta', 0):+d} minutes
Accept Prob:    {summary.get('acceptance_prob', 0):.3f}
Result:         {'ACCEPTED' if accepted else 'REJECTED'}

Schedule Offsets:
"""
            offsets = self.history.get_schedule_at_step(step)
            for line_id, offset in sorted(offsets.items()):
                info_text += f"  {line_id}: {offset} min\n"

            if self.history.optimal_schedule:
                info_text += f"\nOptimal Offsets:\n"
                for line_id, offset in sorted(self.history.optimal_schedule.items()):
                    info_text += f"  {line_id}: {offset} min\n"

            ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                        fontsize=11, verticalalignment="top", fontfamily="monospace",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

            plt.tight_layout()
            plt.show()

    def _on_step_change(self, change: dict) -> None:
        """Handle step slider change."""
        self._current_step = change["new"]
        self._render_step(self._current_step)

    def _on_prev(self, _) -> None:
        """Go to previous step."""
        if self._current_step > 0:
            self._step_slider.value = self._current_step - 1

    def _on_next(self, _) -> None:
        """Go to next step."""
        if self._current_step < len(self.history) - 1:
            self._step_slider.value = self._current_step + 1

    def _on_first(self, _) -> None:
        """Go to first step."""
        self._step_slider.value = 0

    def _on_last(self, _) -> None:
        """Go to last step."""
        self._step_slider.value = len(self.history) - 1

    def _on_play(self, _) -> None:
        """Start/stop auto-play."""
        import time
        import threading

        if self._playing:
            self._playing = False
            self._play_button.description = "▶ Play"
            return

        self._playing = True
        self._play_button.description = "⏸ Pause"

        def play_loop():
            while self._playing and self._current_step < len(self.history) - 1:
                time.sleep(self._play_speed / 1000)
                if self._playing:
                    self._step_slider.value = self._current_step + 1
            self._playing = False
            self._play_button.description = "▶ Play"

        thread = threading.Thread(target=play_loop)
        thread.start()

    def _on_speed_change(self, change: dict) -> None:
        """Handle speed slider change."""
        self._play_speed = change["new"]

    def display(self) -> None:
        """Display the interactive player."""
        # Controls
        self._step_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.history) - 1,
            step=1,
            description="Step:",
            continuous_update=False,
            layout=widgets.Layout(width="60%"),
        )
        self._step_slider.observe(self._on_step_change, names="value")

        btn_first = widgets.Button(description="⏮ First")
        btn_prev = widgets.Button(description="◀ Prev")
        btn_next = widgets.Button(description="Next ▶")
        btn_last = widgets.Button(description="Last ⏭")
        self._play_button = widgets.Button(description="▶ Play")

        btn_first.on_click(self._on_first)
        btn_prev.on_click(self._on_prev)
        btn_next.on_click(self._on_next)
        btn_last.on_click(self._on_last)
        self._play_button.on_click(self._on_play)

        speed_slider = widgets.IntSlider(
            value=100,
            min=10,
            max=500,
            step=10,
            description="Speed (ms):",
            layout=widgets.Layout(width="40%"),
        )
        speed_slider.observe(self._on_speed_change, names="value")

        controls = widgets.HBox([
            btn_first, btn_prev, self._play_button, btn_next, btn_last,
            speed_slider,
        ])

        # Layout
        display(widgets.VBox([
            self._step_slider,
            controls,
            self._output,
        ]))

        # Initial render
        self._render_step(0)


def create_mcmc_player(
    history: MCMCHistory,
    network: Network,
    transfers: TransferSet,
    cost_fn: TransferCost,
) -> MCMCPlayer:
    """Create an interactive MCMC player.

    Args:
        history: Recorded MCMC history.
        network: Transport network.
        transfers: Transfer connections.
        cost_fn: Cost function.

    Returns:
        MCMCPlayer widget.
    """
    return MCMCPlayer(
        history=history,
        network=network,
        transfers=transfers,
        cost_fn=cost_fn,
    )
