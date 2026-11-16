"""Recalcula las celdas neuronales de Allen–Cahn en las tablas 7 y 8."""

import argparse
import json
import sys
from pathlib import Path, PureWindowsPath
from statistics import mean, stdev

import numpy as np
import torch


CAMPAIGN = "five_seed_campaign_7_11_23_37_53"
METHODS = {"bihyco": "Bi-HYCO-SYN", "xpinn_c": "XPINN-C"}
METRICS = (
    ("1000 V_val^syn", 1000, 3),
    ("100 e_u(Q)", 100, 2),
    ("100 e_u(T)", 100, 2),
    ("1000 V_te", 1000, 3),
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def observation_paths(reference_dir):
    paths = {}
    for path in (reference_dir / "observations").glob("*/metadata.json"):
        metadata = read_json(path)
        key = (metadata["seed"], metadata["data_id"])
        if key in paths:
            raise ValueError(f"Observaciones ambiguas para semilla e ID {key}")
        paths[key] = path.with_name("observations.npz")
    return paths


def evaluate(ac, pilot_dir):
    campaign_dir = pilot_dir / CAMPAIGN
    manifest = read_json(campaign_dir / "report_manifest.json")
    design = read_json(campaign_dir / "campaign_configuration.json")
    seeds = manifest["seeds"]
    if not manifest["complete"] or seeds != design["seeds"]:
        raise ValueError("La campaña de cinco semillas está incompleta o es incoherente")

    reference_key = PureWindowsPath(design["shared_reference"]).name
    reference_dir = pilot_dir / "reference_cache" / reference_key
    reference = ac.load_trajectory(reference_dir / "reference_heterogeneous.npz")
    observations = observation_paths(reference_dir)
    torch.set_num_threads(design["frozen_config"]["threads"])

    results = {}
    for source in manifest["sources"]:
        method = source["method"]
        if method not in METHODS:
            continue
        seed, data_id = source["seed"], source["data_id"]
        key = (method, seed)
        if key in results or data_id != manifest["per_seed_data_ids"][str(seed)]:
            raise ValueError(f"Ejecución duplicada o ID de datos incorrecto para {key}")

        run_suffix = PureWindowsPath(source["output_folder"]).parts[-2:]
        run_dir = pilot_dir.joinpath("runs", *run_suffix)
        run = read_json(run_dir / "configuration.json")
        config, signature = run["config"], run["signature"]
        if (signature["method"], signature["seed"], signature["data_id"]) != (method, seed, data_id):
            raise ValueError(f"La configuración no corresponde a la campaña: {run_dir}")

        checkpoint = ac.load_checkpoint(run_dir / "best_physical_checkpoint.pt", device="cpu")
        step = checkpoint["step"]
        if step != source["physical_checkpoint"]:
            raise ValueError(f"Checkpoint físico inconsistente con el manifiesto: {run_dir}")
        model = ac.StateNet(config["hyco_width"]) if method == "bihyco" else ac.XPINN(config["xp_width"])
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()

        times = np.linspace(0, ac.T_END, config["evaluation_frames"])
        truth = ac.interpolate_frames(reference, times, reference.x).astype(np.float32)
        prediction = ac.network_frames(model, times, reference.x)
        final_error = np.linalg.norm(prediction[-1] - truth[-1]) / np.linalg.norm(truth[-1])
        with np.load(observations[(seed, data_id)], allow_pickle=False) as obs:
            values = (
                ac.neural_validation(model, obs, "val"),
                ac.field_error(prediction, truth, times),
                float(final_error),
                ac.neural_validation(model, obs, "test"),
            )
        if not np.isfinite(values).all():
            raise ValueError(f"Métricas no finitas para {key} en el checkpoint {step}")
        results[key] = (step, values)

    if set(results) != {(method, seed) for method in METHODS for seed in seeds}:
        raise ValueError("Faltan ejecuciones de Bi-HYCO o XPINN-C en el manifiesto")
    return seeds, results


def print_results(seeds, results):
    columns = " | ".join(label for label, _, _ in METRICS)
    print(f"Salida | Semilla | Paso físico | {columns}")
    for method, label in METHODS.items():
        for seed in seeds:
            step, values = results[(method, seed)]
            cells = (f"{value * scale:.6f}" for value, (_, scale, _) in zip(values, METRICS))
            print(f"{label} | {seed} | {step} | {' | '.join(cells)}")

    print(f"\nMedia +/- desviación estándar muestral (n={len(seeds)})")
    print(f"Salida | {columns}")
    for method, label in METHODS.items():
        cells = []
        for index, (_, scale, digits) in enumerate(METRICS):
            values = [results[(method, seed)][1][index] * scale for seed in seeds]
            cells.append(f"{mean(values):.{digits}f} +/- {stdev(values):.{digits}f}")
        print(f"{label} | {' | '.join(cells)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Carpeta Allen-Cahn 08 (por defecto, la carpeta de este script).",
    )
    project_dir = parser.parse_args().project_dir.resolve()
    sys.path.insert(0, str(project_dir))
    import AllenCahn_TwoParameter as ac

    pilot_dir = project_dir / "AllenCahn_TwoParameter_Results" / "pilot"
    print_results(*evaluate(ac, pilot_dir))


if __name__ == "__main__":
    main()
