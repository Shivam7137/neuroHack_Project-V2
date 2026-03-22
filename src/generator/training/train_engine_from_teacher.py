"""Fit the procedural EEG engine against the trained baseline teacher."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from src.baseline.teacher_api import TeacherAPI
from src.generator.inference.sampler import GeneratorCondition, ProceduralEngineConfig, SyntheticSampler
from src.utils.io import save_json


@dataclass(slots=True)
class TeacherFitResult:
    """Result of fitting one task-specific procedural engine."""

    task_name: str
    config: ProceduralEngineConfig
    objective: float
    iterations: int
    channel_names: list[str]
    sample_rate: float


def _task_spec(teacher: TeacherAPI, task_name: str) -> tuple[list[str], Callable[[np.ndarray, float, list[str]], float]]:
    models = teacher.scorer.models
    if task_name not in models:
        raise ValueError(f"Unsupported task '{task_name}'.")
    if task_name == "concentration":
        return list(models[task_name].preprocessor.channel_names), teacher.predict_concentration
    if task_name == "stress":
        return list(models[task_name].preprocessor.channel_names), teacher.predict_stress
    raise ValueError(f"Unsupported task '{task_name}'.")


def _condition_grid(task_name: str) -> list[GeneratorCondition]:
    if task_name == "concentration":
        return [
            GeneratorCondition(concentration_level=0.1, stress_level=0.1),
            GeneratorCondition(concentration_level=0.3, stress_level=0.2),
            GeneratorCondition(concentration_level=0.5, stress_level=0.2),
            GeneratorCondition(concentration_level=0.7, stress_level=0.25),
            GeneratorCondition(concentration_level=0.9, stress_level=0.3),
        ]
    return [
        GeneratorCondition(concentration_level=0.2, stress_level=0.1),
        GeneratorCondition(concentration_level=0.3, stress_level=0.3),
        GeneratorCondition(concentration_level=0.35, stress_level=0.5),
        GeneratorCondition(concentration_level=0.4, stress_level=0.7),
        GeneratorCondition(concentration_level=0.45, stress_level=0.9),
    ]


def fit_procedural_engine(
    teacher: TeacherAPI,
    task_name: str,
    duration_sec: float = 2.0,
    sample_rate: float = 250.0,
    seeds: tuple[int, ...] = (7, 11),
    maxiter: int = 40,
) -> TeacherFitResult:
    """Fit the procedural engine to one baseline task using teacher outputs."""
    channel_names, predictor = _task_spec(teacher, task_name)
    preprocessor = teacher.scorer.models[task_name].preprocessor
    effective_duration = max(
        float(duration_sec),
        float(getattr(preprocessor, "trim_seconds_start", 0.0) + getattr(preprocessor, "trim_seconds_end", 0.0) + 4.0),
    )
    conditions = _condition_grid(task_name)
    targets = np.asarray(
        [condition.concentration_level if task_name == "concentration" else condition.stress_level for condition in conditions],
        dtype=float,
    )
    initial = ProceduralEngineConfig()
    penalty_weights = np.asarray([0.03] * len(initial.to_vector()), dtype=float)
    penalty_weights[-1] = 0.01

    def objective(vector: np.ndarray) -> float:
        config = ProceduralEngineConfig.from_vector(vector)
        condition_losses: list[float] = []
        for index, condition in enumerate(conditions):
            target = targets[index]
            seed_losses: list[float] = []
            for seed in seeds:
                sampler = SyntheticSampler(
                    channel_names=channel_names,
                    sample_rate=sample_rate,
                    random_seed=seed,
                    engine_config=config,
                )
                sample = sampler.sample(condition, duration_sec=effective_duration)
                predicted = predictor(sample.data, sample.sample_rate, sample.channel_names)
                seed_losses.append(abs(float(predicted) - float(target)))
            condition_losses.append(float(np.mean(seed_losses)))
        regularizer = float(np.mean(np.square(vector - initial.to_vector()) * penalty_weights))
        return float(np.mean(condition_losses) + regularizer)

    result = minimize(
        objective,
        x0=initial.to_vector(),
        method="Powell",
        options={"maxiter": maxiter, "disp": False},
    )
    fitted = ProceduralEngineConfig.from_vector(np.asarray(result.x, dtype=float))
    return TeacherFitResult(
        task_name=task_name,
        config=fitted,
        objective=float(result.fun),
        iterations=int(result.nit if result.nit is not None else 0),
        channel_names=channel_names,
        sample_rate=float(sample_rate),
    )


def save_fit_result(result: TeacherFitResult, output_dir: Path) -> Path:
    """Save a fitted teacher-guided engine config."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{result.task_name}_procedural_engine.json"
    save_json(
        {
            "task_name": result.task_name,
            "objective": result.objective,
            "iterations": result.iterations,
            "sample_rate": result.sample_rate,
            "channel_names": result.channel_names,
            "engine_config": result.config.to_dict(),
        },
        path,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the procedural EEG engine from the trained baseline teacher.")
    parser.add_argument("--task", choices=["concentration", "stress", "both"], default="both")
    parser.add_argument("--duration-sec", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=float, default=250.0)
    parser.add_argument("--maxiter", type=int, default=40)
    parser.add_argument("--seeds", default="7,11", help="Comma-separated random seeds used during teacher fitting.")
    parser.add_argument("--output-dir", default="artifacts/generator")
    args = parser.parse_args()

    teacher = TeacherAPI()
    task_names = ["concentration", "stress"] if args.task == "both" else [args.task]
    seeds = tuple(int(part.strip()) for part in args.seeds.split(",") if part.strip())
    for task_name in task_names:
        result = fit_procedural_engine(
            teacher,
            task_name=task_name,
            duration_sec=float(args.duration_sec),
            sample_rate=float(args.sample_rate),
            seeds=seeds,
            maxiter=int(args.maxiter),
        )
        path = save_fit_result(result, Path(args.output_dir))
        print(f"{task_name}: objective={result.objective:.4f} iterations={result.iterations} saved={path.resolve()}")


if __name__ == "__main__":
    main()
