"""
Run Manager - Persistent storage for simulation runs
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class RunManager:
    """Manages saving and loading of simulation runs"""

    def __init__(self, storage_path: str = "logs/run_history.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_storage_exists()

    def _ensure_storage_exists(self):
        """Create the storage file if it doesn't exist"""
        if not self.storage_path.exists():
            self.storage_path.write_text(json.dumps({"runs": []}, indent=2))

    def save_run(
        self,
        run_name: str,
        scenario: str,
        feedstock_scenario: str,
        steps: int,
        seed: int,
        config: Dict[str, Any],
        boolean_config: Dict[str, Any],
        results_summary: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a simulation run to persistent storage

        Returns:
            run_id: Unique identifier for the run
        """
        timestamp = datetime.now()
        run_id = timestamp.strftime("%Y%m%d_%H%M%S")

        run_data = {
            "run_id": run_id,
            "run_name": run_name,
            "timestamp": timestamp.isoformat(),
            "display_date": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": scenario,
            "feedstock_scenario": feedstock_scenario,
            "steps": steps,
            "seed": seed,
            "config": config,
            "boolean_config": boolean_config,
            "results_summary": results_summary or {},
        }

        # Load existing runs
        data = self._load_data()

        # Add new run
        data["runs"].append(run_data)

        # Save back
        self._save_data(data)

        return run_id

    def get_all_runs(self) -> List[Dict[str, Any]]:
        """Get all saved runs, sorted by timestamp (newest first)"""
        data = self._load_data()
        runs = data.get("runs", [])
        # Sort by timestamp, newest first
        runs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return runs

    def get_run_by_id(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific run by its ID"""
        runs = self.get_all_runs()
        for run in runs:
            if run.get("run_id") == run_id:
                return run
        return None

    def delete_run(self, run_id: str) -> bool:
        """Delete a run by its ID"""
        data = self._load_data()
        runs = data.get("runs", [])

        # Filter out the run to delete
        new_runs = [r for r in runs if r.get("run_id") != run_id]

        if len(new_runs) == len(runs):
            return False  # Run not found

        data["runs"] = new_runs
        self._save_data(data)
        return True

    def clear_all_runs(self):
        """Clear all saved runs"""
        self._save_data({"runs": []})

    def _load_data(self) -> Dict[str, Any]:
        """Load data from storage file"""
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"runs": []}

    def _save_data(self, data: Dict[str, Any]):
        """Save data to storage file"""
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# Global instance
_run_manager = None


def get_run_manager() -> RunManager:
    """Get the global RunManager instance"""
    global _run_manager
    if _run_manager is None:
        _run_manager = RunManager()
    return _run_manager
