"""Project implementation status tracker."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from IPython.display import HTML, display


class Status(Enum):
    """Implementation status."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PLANNED = "planned"


@dataclass
class ModuleInfo:
    """Information about a module/function."""

    name: str
    description: str
    status: Status
    module_path: str
    phase: int
    check_fn: Callable[[], bool] | None = None

    def check_implemented(self) -> bool:
        """Check if module is actually implemented."""
        if self.check_fn:
            try:
                return self.check_fn()
            except Exception:
                return False
        return self.status == Status.COMPLETED


# Define all project modules
def _check_station() -> bool:
    from src.core.network import Station
    return Station is not None


def _check_line() -> bool:
    from src.core.network import Line
    return Line is not None


def _check_network() -> bool:
    from src.core.network import Network
    return Network is not None


def _check_schedule() -> bool:
    from src.core.schedule import Schedule
    return Schedule is not None


def _check_transfer() -> bool:
    from src.core.transfers import Transfer, TransferParams
    return Transfer is not None and TransferParams is not None


def _check_transfer_cost() -> bool:
    from src.cost.transfer_cost import TransferCost, soft_transfer_cost
    return TransferCost is not None


def _check_single_line_shift() -> bool:
    from src.mcmc.proposals.single_line import SingleLineShift
    return SingleLineShift is not None


def _check_sampler() -> bool:
    from src.mcmc.sampler import MCMCSampler
    return MCMCSampler is not None


def _check_triangle() -> bool:
    from src.io.synthetic import TriangleNetwork
    return TriangleNetwork is not None


def _check_component_move() -> bool:
    try:
        from src.mcmc.proposals.component import ComponentMove
        return ComponentMove is not None
    except ImportError:
        return False


def _check_smart_repair() -> bool:
    try:
        from src.mcmc.proposals.smart_repair import SmartRepair
        return SmartRepair is not None
    except ImportError:
        return False


def _check_parallel_tempering() -> bool:
    try:
        from src.mcmc.parallel_tempering import ParallelTempering
        return ParallelTempering is not None
    except ImportError:
        return False


def _check_pesplib_loader() -> bool:
    try:
        from src.io.pesplib_loader import load_pesplib
        return load_pesplib is not None
    except ImportError:
        return False


def _check_robustness() -> bool:
    try:
        from src.cost.robustness import CVaRCost
        return CVaRCost is not None
    except ImportError:
        return False


def _check_operational_cost() -> bool:
    try:
        from src.cost.operational import OperationalCost
        return OperationalCost is not None
    except ImportError:
        return False


PROJECT_MODULES = [
    # Phase 1: Core + PoC
    ModuleInfo("Station", "Physical location with modality", Status.COMPLETED,
               "src/core/network.py", 1, _check_station),
    ModuleInfo("Line", "Transport line with stations and times", Status.COMPLETED,
               "src/core/network.py", 1, _check_line),
    ModuleInfo("Network", "Collection of lines", Status.COMPLETED,
               "src/core/network.py", 1, _check_network),
    ModuleInfo("Schedule", "MCMC state (line offsets)", Status.COMPLETED,
               "src/core/schedule.py", 1, _check_schedule),
    ModuleInfo("Transfer", "Transfer connection with parameters", Status.COMPLETED,
               "src/core/transfers.py", 1, _check_transfer),
    ModuleInfo("TransferCost", "Soft transfer cost with barrier", Status.COMPLETED,
               "src/cost/transfer_cost.py", 1, _check_transfer_cost),
    ModuleInfo("SingleLineShift", "Basic Gaussian shift proposal", Status.COMPLETED,
               "src/mcmc/proposals/single_line.py", 1, _check_single_line_shift),
    ModuleInfo("MCMCSampler", "Metropolis-Hastings loop", Status.COMPLETED,
               "src/mcmc/sampler.py", 1, _check_sampler),
    ModuleInfo("TriangleNetwork", "Synthetic test network", Status.COMPLETED,
               "src/io/synthetic.py", 1, _check_triangle),

    # Phase 2: PESPlib + Advanced Proposals
    ModuleInfo("PESPlib Loader", "Load PESPlib instances", Status.PLANNED,
               "src/io/pesplib_loader.py", 2, _check_pesplib_loader),
    ModuleInfo("ComponentMove", "Shift correlated lines together", Status.PLANNED,
               "src/mcmc/proposals/component.py", 2, _check_component_move),
    ModuleInfo("SmartRepair", "Targeted constraint repair", Status.PLANNED,
               "src/mcmc/proposals/smart_repair.py", 2, _check_smart_repair),
    ModuleInfo("TotalCost", "Combined cost function", Status.PLANNED,
               "src/cost/total.py", 2, None),
    ModuleInfo("Convergence Metrics", "R-hat, ESS diagnostics", Status.PLANNED,
               "src/metrics/convergence.py", 2, None),

    # Phase 3: Parallel Tempering
    ModuleInfo("ParallelTempering", "PT with inertia", Status.PLANNED,
               "src/mcmc/parallel_tempering.py", 3, _check_parallel_tempering),
    ModuleInfo("CVaR Robustness", "Robustness cost term", Status.PLANNED,
               "src/cost/robustness.py", 3, _check_robustness),
    ModuleInfo("OperationalCost", "Turnaround and fleet proxy", Status.PLANNED,
               "src/cost/operational.py", 3, _check_operational_cost),

    # Phase 4: Scaling + GTFS
    ModuleInfo("GTFS Loader", "Load real-world GTFS data", Status.PLANNED,
               "src/io/gtfs_loader.py", 4, None),
    ModuleInfo("Incremental Cost", "Optimized delta computation", Status.PLANNED,
               "src/cost/incremental.py", 4, None),
]


class ProjectStatus:
    """Display and track project implementation status."""

    def __init__(self) -> None:
        self.modules = PROJECT_MODULES

    def get_phase_modules(self, phase: int) -> list[ModuleInfo]:
        """Get all modules for a specific phase."""
        return [m for m in self.modules if m.phase == phase]

    def get_implemented_count(self) -> tuple[int, int]:
        """Get (implemented, total) counts."""
        implemented = sum(1 for m in self.modules if m.check_implemented())
        return implemented, len(self.modules)

    def display_status(self) -> None:
        """Display interactive status table in Jupyter."""
        html = self._generate_html()
        display(HTML(html))

    def _generate_html(self) -> str:
        """Generate HTML status table."""
        impl, total = self.get_implemented_count()
        progress_pct = (impl / total) * 100 if total > 0 else 0

        html = f"""
        <style>
            .status-table {{
                border-collapse: collapse;
                width: 100%;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                font-size: 14px;
            }}
            .status-table th, .status-table td {{
                border: 1px solid #ddd;
                padding: 8px 12px;
                text-align: left;
            }}
            .status-table th {{
                background-color: #4a5568;
                color: white;
            }}
            .status-table tr:nth-child(even) {{
                background-color: #f7fafc;
            }}
            .status-completed {{
                color: #22c55e;
                font-weight: bold;
            }}
            .status-in-progress {{
                color: #f59e0b;
                font-weight: bold;
            }}
            .status-planned {{
                color: #94a3b8;
            }}
            .status-not-started {{
                color: #ef4444;
            }}
            .phase-header {{
                background-color: #1e40af !important;
                color: white;
                font-weight: bold;
            }}
            .progress-bar {{
                width: 100%;
                background-color: #e5e7eb;
                border-radius: 4px;
                overflow: hidden;
                margin-bottom: 20px;
            }}
            .progress-fill {{
                height: 24px;
                background-color: #22c55e;
                text-align: center;
                color: white;
                font-weight: bold;
                line-height: 24px;
            }}
            .check-mark {{ color: #22c55e; }}
            .cross-mark {{ color: #ef4444; }}
        </style>

        <h2>MCMC-PESP Project Status</h2>

        <div class="progress-bar">
            <div class="progress-fill" style="width: {progress_pct:.0f}%">
                {impl}/{total} modules ({progress_pct:.0f}%)
            </div>
        </div>

        <table class="status-table">
            <tr>
                <th>Module</th>
                <th>Description</th>
                <th>Path</th>
                <th>Status</th>
                <th>Implemented</th>
            </tr>
        """

        current_phase = 0
        for module in self.modules:
            if module.phase != current_phase:
                current_phase = module.phase
                phase_names = {
                    1: "Phase 1: Core + PoC",
                    2: "Phase 2: PESPlib + Advanced Proposals",
                    3: "Phase 3: Parallel Tempering",
                    4: "Phase 4: Scaling + GTFS",
                }
                html += f"""
                <tr class="phase-header">
                    <td colspan="5">{phase_names.get(current_phase, f"Phase {current_phase}")}</td>
                </tr>
                """

            status_class = f"status-{module.status.value.replace('_', '-')}"
            status_emoji = {
                Status.COMPLETED: "✅",
                Status.IN_PROGRESS: "🔄",
                Status.PLANNED: "📋",
                Status.NOT_STARTED: "❌",
            }

            implemented = module.check_implemented()
            impl_mark = '<span class="check-mark">✓</span>' if implemented else '<span class="cross-mark">✗</span>'

            html += f"""
            <tr>
                <td><strong>{module.name}</strong></td>
                <td>{module.description}</td>
                <td><code>{module.module_path}</code></td>
                <td class="{status_class}">{status_emoji[module.status]} {module.status.value.replace('_', ' ').title()}</td>
                <td style="text-align: center">{impl_mark}</td>
            </tr>
            """

        html += "</table>"
        return html

    def print_status(self) -> None:
        """Print status to console (non-Jupyter)."""
        impl, total = self.get_implemented_count()
        print(f"\n{'='*60}")
        print(f"MCMC-PESP Project Status: {impl}/{total} modules implemented")
        print(f"{'='*60}\n")

        current_phase = 0
        for module in self.modules:
            if module.phase != current_phase:
                current_phase = module.phase
                print(f"\n--- Phase {current_phase} ---")

            implemented = module.check_implemented()
            status_mark = "✓" if implemented else "✗"
            status_str = module.status.value.replace("_", " ")
            print(f"  [{status_mark}] {module.name:20} | {status_str:12} | {module.module_path}")
