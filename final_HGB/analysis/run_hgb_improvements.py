"""Sequential Off-market HGB v2 experiments with pre-December model selection."""
import argparse
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import check_submission
import data
from analysis.hgb_local_comps import ComparableSales, feature_names
from analysis.hgb_off_market_candidate import OffMarketCandidate
from analysis.run_hgb_baselines import (
    atomic_json, baseline_guard, identity, load_predictions, now, read_json,
)
from analysis.run_tree_baselines import (
    DEVELOPMENT_MONTHS, HOLDOUT_MONTHS, environment, grouped_metrics, point_metrics,
    sha256, temporal_fold,
)

VERSION = "hgb_v2"
TRACK = "off_market"
OUTPUT = ROOT / "outputs/hgb_baselines" / VERSION
V1 = OUTPUT.with_name("hgb_v1")
DEV_ORIGINS = ("2025-04-01", "2025-08-01")
EVAL_ORIGINS = ("2026-02-01",)
FINAL_CUTOFF = pd.Timestamp("2026-06-01")
CONTROL = dict(comp_mode="none", half_life_months=None, max_leaf_nodes=63, max_iter=800)
CHECKPOINTS = (800, 1200, 1600)
SOURCES = (
    "models.py", "features.py", "data.py", "check_submission.py", "run.py", "requirements.txt",
    "analysis/run_tree_baselines.py", "analysis/run_hgb_baselines.py",
    "analysis/hgb_local_comps.py", "analysis/hgb_off_market_candidate.py",
    "analysis/run_hgb_improvements.py",
)


def compare_identity(left, right):
    if left.ml_num.duplicated().any() or right.ml_num.duplicated().any():
        raise ValueError("Evaluation rows must contain distinct property IDs")
    pd.testing.assert_frame_equal(identity(left), identity(right), check_exact=True, check_dtype=False)


def compare_predictions(left, right):
    compare_identity(left, right)
    np.testing.assert_allclose(left.sort_values("ml_num").predicted_price_raw,
                               right.sort_values("ml_num").predicted_price_raw, rtol=1e-10, atol=1e-6)


def score(frame):
    return point_metrics(frame.sold_price, frame.predicted_price_raw)


def staged_prices(estimator, matrix, checkpoints):
    if checkpoints == (estimator.n_iter_,):
        return {estimator.n_iter_: np.exp(estimator.predict(matrix))}
    result = {}
    for iteration, prediction in enumerate(estimator.staged_predict(matrix), 1):
        if iteration in checkpoints:
            result[iteration] = np.exp(prediction)
        if iteration == checkpoints[-1]:
            break
    if tuple(result) != checkpoints:
        raise ValueError("Estimator did not produce every requested checkpoint")
    return result


def fold_layout(months, origins):
    """Share a fit when a rolling month is also a fixed-horizon origin."""
    layout = {month.start_time: [month] for month in months}
    for origin in origins:
        cutoff = pd.Timestamp(origin)
        layout[cutoff] = list(pd.period_range(cutoff, periods=4, freq="M"))
    return dict(sorted(layout.items()))


class ImprovementRun:
    def __init__(self):
        self.listings = data.load("listings")
        dates = self.listings.date_sold
        if (len(self.listings) != 150000 or self.listings.ml_num.duplicated().any()
                or dates.isna().any() or not (dates < FINAL_CUTOFF).all()):
            raise ValueError("Expected the 150,000 distinct published pre-June sales")
        self.env = environment()
        original = read_json(V1 / "execution_contract.json")
        for label, entry in self.env["data_identity"].items():
            old = original["data_identity"][label]
            if any(entry[key] != old[key] for key in ("sha256", "bytes")):
                raise ValueError(f"Current data do not match the v1 snapshot: {label}")
        if self.env["packages"] != read_json(
            V1 / read_json(V1 / "final_summary.json")["final_run"] / "environment.json"
        )["packages"]:
            raise ValueError("Use the original HGB v1 package versions")
        for name, digest in original["model_sources"].items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f"v1 model source differs: {name}")
        frozen_path = V1 / "frozen_config.json"
        frozen = read_json(frozen_path)
        frozen_hash = sha256(frozen_path)
        if (frozen["execution_contract_sha256"] != sha256(V1 / "execution_contract.json")
                or frozen_hash != read_json(V1 / "development_summary.json")["frozen_config_sha256"]):
            raise ValueError("v1 frozen configuration does not match its recorded contract and development")
        self.baseline_options = frozen["tracks"][TRACK]["model_config"]["options"]
        if any(self.baseline_options[k] != CONTROL[k] for k in ("max_leaf_nodes", "max_iter")):
            raise ValueError("Unexpected v1 Off-market capacity")
        self.guard = baseline_guard(V1)
        self.reference_evidence = {}
        self.comp_cache = {}
        protocol = {
            "version": VERSION, "track": TRACK,
            "development_rolling": list(map(str, DEVELOPMENT_MONTHS)),
            "development_fixed_origins": list(DEV_ORIGINS),
            "evaluation_rolling": list(map(str, HOLDOUT_MONTHS)),
            "evaluation_fixed_origins": list(EVAL_ORIGINS),
            "horizon_months_per_origin": 4,
            "steps": [
                {"stage": 1, "comp_modes": ["community", "community_and_site"], "history_months": 12,
                 "min_comparables": 3, "as_of": "min(query date_listed, fit cutoff); strictly earlier sales"},
                {"stage": 2, "half_life_months": 24},
                {"stage": 3, "max_leaf_nodes": [63, 127], "checkpoints": list(CHECKPOINTS)},
            ],
            "promotion": "Strictly lower pooled MdAPE in both development cohorts than parent. "
                         "Rank qualifying candidates by rolling MdAPE, rolling MAPE, fixed MdAPE, "
                         "fixed MAPE, smaller leaves and fewer iterations.",
            "holdout_previously_observed": True,
            "selection_used_post_november_scores": False,
        }
        self.contract = {
            "protocol": protocol,
            "code_sha256": {name: sha256(ROOT / name) for name in SOURCES},
            "data_identity": self.env["data_identity"],
            "packages": self.env["packages"],
            "thread_settings": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "v1_frozen_sha256": frozen_hash,
            "v1_baseline_guard": self.guard,
        }
        path = OUTPUT / "execution_contract.json"
        if path.exists():
            if read_json(path) != self.contract:
                raise ValueError("Existing HGB v2 run used a different execution contract")
        else:
            atomic_json(path, self.contract)
            atomic_json(OUTPUT / "environment.json", self.env)
            for name in SOURCES:
                dest = OUTPUT / "source_snapshot" / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, dest)
            shutil.copy2(V1 / "requirements-lock.txt", OUTPUT / "requirements-lock.txt")
        self.contract_hash = sha256(path)

    def model(self, spec, cutoff):
        options = {**self.baseline_options, **{k: spec[k] for k in ("max_leaf_nodes", "max_iter")}}
        return OffMarketCandidate(options, cutoff, spec["comp_mode"], spec["half_life_months"])

    def folds(self, cutoff, months):
        train, _, _ = temporal_fold(self.listings, months[0], cutoff)
        valid = pd.concat([temporal_fold(self.listings, month, cutoff)[1].assign(month=str(month))
                           for month in months])
        frame = valid[["ml_num", "house_type_name", "community", "sold_price", "month"]].copy()
        frame["training_cutoff"] = cutoff.isoformat()
        return train, data.hide_answers(valid, TRACK), frame

    def comparable_features(self, train, homes, cutoff, mode):
        if mode == "none":
            return None, None, None
        if cutoff not in self.comp_cache:
            index = ComparableSales(train)
            train_comp = index.transform(train, cutoff, mode="community_and_site")
            valid_comp = index.transform(homes, cutoff, mode="community_and_site")
            self.comp_cache[cutoff] = (index, train_comp, valid_comp)
        index, train_comp, valid_comp = self.comp_cache[cutoff]
        if mode == "community":
            train_comp = train_comp.loc[:, list(feature_names(mode))]
            valid_comp = valid_comp.loc[:, list(feature_names(mode))]
            train_comp.attrs["comp_mode"] = valid_comp.attrs["comp_mode"] = mode
        if not train_comp.index.equals(train.index) or not valid_comp.index.equals(homes.index):
            raise ValueError("Cached comparable features are from a different cohort")
        return index, train_comp, valid_comp

    def fit_fold(self, name, spec, cutoff, months, checkpoints):
        dest = OUTPUT / "physical_runs" / name / str(cutoff.date())
        train, homes, truth = self.folds(cutoff, months)
        if dest.is_dir():
            completed = read_json(dest / "fit.json")
            if (completed["execution_contract_sha256"] != self.contract_hash
                    or completed["spec"] != spec or completed["checkpoints"] != list(checkpoints)):
                raise ValueError(f"Cached fold belongs to a different experiment: {dest}")
            frames = {}
            for number in checkpoints:
                path = dest / f"predictions_{number}.csv"
                if sha256(path) != completed["prediction_sha256"][str(number)]:
                    raise ValueError(f"Cached predictions changed: {path}")
                frames[number] = load_predictions(path)
                compare_identity(frames[number], truth)
            return frames
        started = perf_counter()
        index, train_comp, valid_comp = self.comparable_features(train, homes, cutoff, spec["comp_mode"])
        model = self.model(spec, cutoff).fit(train, comp_index=index, train_comp_features=train_comp)
        if model.estimator.n_iter_ != spec["max_iter"]:
            raise ValueError("Unexpected number of boosting iterations")
        matrix = model.prepare_features(homes, comp_features=valid_comp)
        predictions = staged_prices(model.estimator, matrix, checkpoints)
        train_matrix = model.prepare_features(train, comp_features=train_comp)
        train_prices = staged_prices(model.estimator, train_matrix, checkpoints)
        recent = train.date_sold >= cutoff - pd.DateOffset(months=12)
        frames = {number: truth.assign(predicted_price_raw=prices) for number, prices in predictions.items()}
        timing = perf_counter() - started
        dest.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=dest.parent) as temp:
            attempt = Path(temp)
            digests = {}
            for number, frame in frames.items():
                path = attempt / f"predictions_{number}.csv"
                frame.to_csv(path, index=False)
                digests[str(number)] = sha256(path)
            atomic_json(attempt / "fit.json", {
                "execution_contract_sha256": self.contract_hash, "spec": spec,
                "checkpoints": list(checkpoints), "prediction_sha256": digests,
                "training_cutoff": cutoff.isoformat(), "max_training_sale": train.date_sold.max().isoformat(),
                "months": list(map(str, months)), "n_train": len(train), "n_validation": len(truth),
                "model_config": model.configuration(), "seconds": timing,
                "diagnostics": {str(n): {
                    "full_training": point_metrics(train.sold_price, train_prices[n]),
                    "recent_training": point_metrics(train.loc[recent, "sold_price"], train_prices[n][recent]),
                    "future": score(frames[n]),
                } for n in checkpoints},
            })
            attempt.rename(dest)
        metrics_text = ", ".join(f"i{n}={score(frame)['mdape']:.4f}%" for n, frame in frames.items())
        print(f"{name} {cutoff.date()} n_train={len(train):,} n_valid={len(truth):,} "
              f"{metrics_text} ({timing:.1f}s)", flush=True)
        return frames

    def reference(self, phase):
        frozen = read_json(V1 / "frozen_config.json")["tracks"][TRACK]
        if phase == "development":
            experiment = frozen["selected_experiment_id"]
            expected = frozen["development_metrics"]
            months, origin = DEVELOPMENT_MONTHS, None
        else:
            summary = read_json(V1 / f"{phase}_summary.json")
            if summary["frozen_config_sha256"] != self.contract["v1_frozen_sha256"]:
                raise ValueError(f"Unexpected v1 frozen hash for {phase}")
            selected = next(row for row in summary["results"] if row["track"] == TRACK)
            experiment, expected = selected["experiment_id"], selected["metrics"]
            if phase == "backtest":
                months, origin = HOLDOUT_MONTHS, None
            else:
                origin = pd.Timestamp(EVAL_ORIGINS[0])
                months = pd.period_range(origin, periods=4, freq="M")
        path = V1 / "experiments" / experiment / "predictions.csv"
        frame = load_predictions(path)
        truth = pd.concat([self.folds(origin or m.start_time, [m])[2] for m in months])
        compare_identity(frame, truth)
        observed = score(frame)
        for key, value in expected.items():
            np.testing.assert_allclose(observed[key], value, rtol=0, atol=1e-10)
        self.reference_evidence[phase] = {"file": str(path), "sha256": sha256(path), "metrics": observed}
        return frame

    def evaluate(self, name, spec, rolling_months, origins, checkpoints=None):
        checkpoints = checkpoints or (spec["max_iter"],)
        layout = fold_layout(rolling_months, origins)
        fits = {cutoff: self.fit_fold(name, spec, cutoff, months, checkpoints)
                for cutoff, months in layout.items()}
        results = {}
        for number in checkpoints:
            results[number] = {"fixed": pd.concat([fits[pd.Timestamp(origin)][number] for origin in origins])}
            if len(rolling_months):
                results[number]["rolling"] = pd.concat([
                    fits[m.start_time][number].loc[lambda df: df.month == str(m)] for m in rolling_months
                ])
        return results

    def result(self, name, spec, frames, reference):
        for cohort, frame in frames.items():
            compare_identity(frame, reference[cohort])
        per_origin = {cutoff: score(rows) for cutoff, rows in frames["fixed"].groupby("training_cutoff")}
        return {"name": name, "spec": spec, "rolling": score(frames["rolling"]),
                "fixed": score(frames["fixed"]), "fixed_by_origin": per_origin}

    def select(self, stage, parent, candidates):
        passing = [r for r in candidates
                   if all(r[c]["mdape"] < parent[c]["mdape"] for c in ("rolling", "fixed"))]
        chosen = min(passing, key=lambda r: (
            r["rolling"]["mdape"], r["rolling"]["mape"], r["fixed"]["mdape"],
            r["fixed"]["mape"], r["spec"]["max_leaf_nodes"], r["spec"]["max_iter"],
        )) if passing else parent
        step = {"stage": stage, "parent": parent["name"], "selected": chosen["name"],
                "qualifying": [r["name"] for r in passing], "results": candidates}
        print(f"STAGE {stage} selected {chosen['name']}: "
              f"rolling={chosen['rolling']['mdape']:.4f}%, fixed={chosen['fixed']['mdape']:.4f}%", flush=True)
        return chosen, step

    def development(self):
        path = OUTPUT / "frozen_config.json"
        if path.exists():
            frozen = read_json(path)
            if frozen["execution_contract_sha256"] != self.contract_hash:
                raise ValueError("Frozen configuration belongs to another contract")
            return frozen
        prior = self.reference("development")
        fixed = self.evaluate("v1_fixed_control", CONTROL, (), DEV_ORIGINS)[800]["fixed"]
        compare_predictions(fixed.loc[fixed.month == "2025-08"], prior.loc[prior.month == "2025-08"])
        reference = {"rolling": prior, "fixed": fixed}
        control = self.result("hgb_v1", CONTROL, reference, reference)
        candidate_frames = {"hgb_v1": reference}

        def run(name, spec):
            frames = self.evaluate(name, spec, DEVELOPMENT_MONTHS, DEV_ORIGINS)[spec["max_iter"]]
            candidate_frames[name] = frames
            return self.result(name, spec, frames, reference)

        candidates = [run(f"stage1_{mode}", {**CONTROL, "comp_mode": mode})
                      for mode in ("community", "community_and_site")]
        parent, stage1 = self.select(1, control, candidates)
        weighted = run("stage2_half_life_24", {**parent["spec"], "half_life_months": 24})
        parent, stage2 = self.select(2, parent, [weighted])
        candidates = []
        for leaves in (63, 127):
            name = f"stage3_leaves_{leaves}"
            physical_spec = {**parent["spec"], "max_leaf_nodes": leaves, "max_iter": CHECKPOINTS[-1]}
            by_iteration = self.evaluate(name, physical_spec, DEVELOPMENT_MONTHS, DEV_ORIGINS, CHECKPOINTS)
            for iteration, frames in by_iteration.items():
                spec = {**physical_spec, "max_iter": iteration}
                result = self.result(f"{name}_i{iteration}", spec, frames, reference)
                if spec == parent["spec"]:
                    for cohort in ("rolling", "fixed"):
                        compare_predictions(frames[cohort], candidate_frames[parent["name"]][cohort])
                    continue
                candidates.append(result)
        winner, stage3 = self.select(3, parent, candidates)
        frozen = {"version": VERSION, "created_utc": now(), "execution_contract_sha256": self.contract_hash,
                  "reference": control, "stages": [stage1, stage2, stage3], "selected": winner,
                  "selection_reference_evidence": self.reference_evidence}
        atomic_json(path, frozen)
        return frozen

    def backtest(self, frozen):
        winner = frozen["selected"]
        refs = {"rolling": self.reference("backtest"), "fixed": self.reference("horizon")}
        if winner["spec"] == CONTROL:
            frames = refs
        else:
            frames = self.evaluate("frozen_backtest", winner["spec"], HOLDOUT_MONTHS,
                                   EVAL_ORIGINS)[winner["spec"]["max_iter"]]
        results = [self.result("hgb_v1", CONTROL, refs, refs),
                   self.result(winner["name"], winner["spec"], frames, refs)]
        for cohort, frame in frames.items():
            frame.to_csv(OUTPUT / f"reused_{cohort}_predictions.csv", index=False)
        groups = [grouped_metrics(frame).assign(cohort=cohort, model=model)
                  for model, table in (("hgb_v1", refs), (VERSION, frames))
                  for cohort, frame in table.items()]
        pd.concat(groups, ignore_index=True).to_csv(OUTPUT / "reused_grouped_metrics.csv", index=False)
        summary = {
            "frozen_config_sha256": sha256(OUTPUT / "frozen_config.json"), "results": results,
            "reference_evidence": self.reference_evidence, "holdout_previously_observed": True,
            "improves_both_reused_mdapes": all(results[1][c]["mdape"] < results[0][c]["mdape"]
                                              for c in ("rolling", "fixed")),
        }
        atomic_json(OUTPUT / "backtest_summary.json", summary)
        rows = [dict(stage="development_control", **frozen["reference"])]
        rows += [dict(stage=str(step["stage"]), **result)
                 for step in frozen["stages"] for result in step["results"]]
        rows += [dict(stage="reused_evaluation", **r) for r in results]
        pd.json_normalize(rows).to_csv(OUTPUT / "comparison.csv", index=False)
        for result in results:
            print(f"REUSED {result['name']}: rolling={result['rolling']['mdape']:.4f}%, "
                  f"fixed={result['fixed']['mdape']:.4f}%", flush=True)
        return summary

    def predict(self, frozen, evaluation):
        if not evaluation["improves_both_reused_mdapes"]:
            return
        path = OUTPUT / "final_summary.json"
        if path.exists():
            existing = read_json(path)
            for artifact, expected in existing["sha256"].items():
                if sha256(ROOT / artifact) != expected:
                    raise ValueError(f"Final artifact changed: {artifact}")
            return
        spec = frozen["selected"]["spec"]
        dest = ROOT / "submissions" / VERSION
        model_path = OUTPUT / "final_models" / f"{TRACK}.joblib"
        csv_path = dest / f"submission_{TRACK}.csv"
        run = OUTPUT / "final_run"
        frozen_hash = sha256(OUTPUT / "frozen_config.json")
        if not run.exists():
            model = self.model(spec, FINAL_CUTOFF).fit(self.listings)
            homes = data.load(TRACK)
            prices = model.predict(homes)
            with TemporaryDirectory(dir=OUTPUT) as temp:
                stage = Path(temp)
                local_model, local_csv = stage / model_path.name, stage / csv_path.name
                joblib.dump(model, local_model, compress=3)
                np.testing.assert_array_equal(joblib.load(local_model).predict(homes), prices)
                pd.DataFrame({"ml_num": homes.ml_num, "predicted_price": np.round(prices)}).to_csv(
                    local_csv, index=False
                )
                problems = check_submission.check_file(local_csv, homes.ml_num)
                if problems:
                    raise ValueError(f"Invalid candidate submission: {problems}")
                atomic_json(stage / "fit.json", {
                    "spec": spec, "configuration": model.configuration(),
                    "frozen_config_sha256": frozen_hash, "n_train": len(self.listings), "n_test": len(homes),
                    "reload_exact_match": True, "submission_checker": "passed",
                    "sha256": {p.name: sha256(p) for p in (local_model, local_csv)},
                })
                stage.rename(run)
        completed = read_json(run / "fit.json")
        if completed["frozen_config_sha256"] != frozen_hash:
            raise ValueError("Final fit belongs to a different frozen configuration")
        exported = {}
        for target in (model_path, csv_path):
            source = run / target.name
            expected = completed["sha256"][source.name]
            if sha256(source) != expected or (target.exists() and sha256(target) != expected):
                raise ValueError(f"Final artifact changed: {target}")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with TemporaryDirectory(dir=target.parent) as temp:
                    staged = Path(temp) / target.name
                    shutil.copy2(source, staged)
                    staged.rename(target)
            exported[str(target.relative_to(ROOT))] = expected
        atomic_json(path, {**completed, "sha256": exported})
        print(f"FINAL candidate: {csv_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "backtest", "all"), default="all")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUTPUT / ".runner.lock"), timeout=0):
        runner = ImprovementRun()
        if args.phase == "backtest" and not (OUTPUT / "frozen_config.json").exists():
            parser.error("Complete the development phase first")
        frozen = runner.development()
        if args.phase != "development":
            evaluation = runner.backtest(frozen)
            if args.phase == "all":
                runner.predict(frozen, evaluation)
        baseline_guard(V1)
        print(f"Completed {args.phase}. Physical evaluation fits: "
              f"{len(list((OUTPUT / 'physical_runs').glob('*/*/fit.json')))}", flush=True)


if __name__ == "__main__":
    main()
