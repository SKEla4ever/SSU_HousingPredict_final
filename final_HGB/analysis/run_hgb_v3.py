"""Run sequential, versioned Off-market HGB v3 experiments."""
import argparse
from concurrent.futures import ThreadPoolExecutor
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
from analysis.hgb_v3_candidate import HGBV3Candidate
from analysis.run_hgb_baselines import atomic_json, baseline_guard, load_predictions, now, read_json
from analysis.run_hgb_improvements import (
    DEV_ORIGINS, EVAL_ORIGINS, FINAL_CUTOFF, compare_identity, fold_layout, score,
)
from analysis.run_tree_baselines import (
    DEVELOPMENT_MONTHS, HOLDOUT_MONTHS, environment, grouped_metrics, point_metrics, sha256, temporal_fold,
)

VERSION = "hgb_v3"
TRACK = "off_market"
OUTPUT = ROOT / "outputs/hgb_baselines" / VERSION
V2 = OUTPUT.with_name("hgb_v2")
CHECKPOINTS = (1200, 1600, 2000, 2400)
L2_VALUES = (1, 5)
WORKERS = 2
CONTROL = dict(max_iter=1600, l2_regularization=0.0, remarks=False,
               price_target="log_price", market_trend=False)
NEW_SOURCES = (
    "analysis/hgb_market_trend.py", "analysis/hgb_v3_candidate.py", "analysis/run_hgb_v3.py",
)


def price_checkpoints(model, homes, checkpoints):
    """Use the model's restored log price for both ordinary and staged prediction."""
    if checkpoints == (model.estimator.n_iter_,):
        return {checkpoints[0]: model.predict(homes)}
    prices = {}
    for i, log_price in enumerate(model.staged_predict_log(homes), 1):
        if i in checkpoints:
            prices[i] = np.exp(log_price)
        if i == checkpoints[-1]:
            break
    if tuple(prices) != checkpoints:
        raise ValueError("Missing boosting checkpoints")
    return prices


def cohorts(fits, rolling_months, origins):
    return {
        "rolling": pd.concat([fits[m.start_time].loc[lambda df: df.month == str(m)] for m in rolling_months]),
        "fixed": pd.concat([fits[pd.Timestamp(o)] for o in origins]),
    }


def matched_result(name, spec, frames, reference):
    for cohort, frame in frames.items():
        compare_identity(frame, reference[cohort])
    return {
        "name": name, "spec": spec, **{cohort: score(frame) for cohort, frame in frames.items()},
        "fixed_by_origin": {cutoff: score(rows) for cutoff, rows in frames["fixed"].groupby("training_cutoff")},
    }


def beats_both(candidate, parent):
    return all(candidate[c]["mdape"] < parent[c]["mdape"] for c in ("rolling", "fixed"))


def verify_scores(result, expected):
    for cohort in ("rolling", "fixed"):
        for key, value in expected[cohort].items():
            np.testing.assert_allclose(result[cohort][key], value, rtol=0, atol=1e-10)


class V3Run:
    def __init__(self):
        self.listings = data.load("listings")
        dates = self.listings.date_sold
        if (len(self.listings) != 150000 or self.listings.ml_num.duplicated().any()
                or self.listings.ml_num.isna().any() or dates.isna().any() or not dates.lt(FINAL_CUTOFF).all()):
            raise ValueError("Expected the 150,000 distinct pre-June sales")
        self.env = environment()
        old = read_json(V2 / "execution_contract.json")
        if self.env["packages"] != old["packages"]:
            raise ValueError("Use the HGB v2 package versions")
        for label, current in self.env["data_identity"].items():
            if any(current[key] != old["data_identity"][label][key] for key in ("bytes", "sha256")):
                raise ValueError(f"Data differ from HGB v2: {label}")
        for name, digest in old["code_sha256"].items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f"HGB v2 source differs: {name}")
        self.old_contract_hash = sha256(V2 / "execution_contract.json")
        frozen_path = V2 / "frozen_config.json"
        self.v2_frozen = read_json(frozen_path)
        self.v2_final = read_json(V2 / "final_summary.json")
        self.v2_frozen_hash = sha256(frozen_path)
        if (self.v2_frozen["execution_contract_sha256"] != self.old_contract_hash
                or self.v2_final["frozen_config_sha256"] != self.v2_frozen_hash
                or self.v2_final["spec"] != self.v2_frozen["selected"]["spec"]):
            raise ValueError("HGB v2 final fit and frozen configuration do not agree")
        self.options = self.v2_final["configuration"]["options"]
        spec = self.v2_final["spec"]
        if (self.options["max_iter"] != 1600 or self.options["max_leaf_nodes"] != 127
                or self.options["l2_regularization"] != 0 or spec["comp_mode"] != "none"
                or spec["half_life_months"] is not None):
            raise ValueError("Unexpected HGB v2 control configuration")
        self.protected = dict(self.v2_final["sha256"])
        self.protected.update({
            str(frozen_path.relative_to(ROOT)): self.v2_frozen_hash,
            str((V2 / "final_summary.json").relative_to(ROOT)): sha256(V2 / "final_summary.json"),
            str((V2 / "execution_contract.json").relative_to(ROOT)): self.old_contract_hash,
        })
        self.verify_protected()
        sources = {**old["code_sha256"], **{name: sha256(ROOT / name) for name in NEW_SOURCES}}
        protocol = {
            "track": TRACK, "version": VERSION, "leaves": self.options["max_leaf_nodes"],
            "development_rolling": list(map(str, DEVELOPMENT_MONTHS)),
            "development_fixed_origins": list(DEV_ORIGINS),
            "evaluation_rolling": list(map(str, HOLDOUT_MONTHS)), "evaluation_fixed_origins": list(EVAL_ORIGINS),
            "horizon_months": 4, "final_cutoff_exclusive": FINAL_CUTOFF.isoformat(),
            "stages": [
                {"l2_regularization": list(L2_VALUES), "trained_rounds": CHECKPOINTS[-1],
                 "checkpoints": list(CHECKPOINTS), "v2_pure_truncation_control": 1200},
                {"remarks": True, "flags": 10},
                {"price_target": "log_ppsf"},
                {"market_trend": True, "window_days": [90, 90], "minimum_comps_per_window": 5},
            ],
            "promotion": "Both development pooled MdAPEs must strictly improve over the previous winner. "
                         "Rank qualifiers by rolling MdAPE, rolling MAPE, fixed MdAPE, fixed MAPE, and fewer rounds.",
            "candidate_export": "Both reused backtest pooled MdAPEs must improve over v2.",
            "holdout_previously_observed": True, "post_november_scores_used_for_selection": False,
        }
        self.contract = {
            "protocol": protocol, "source_sha256": sources, "v2_protected_sha256": self.protected,
            "data_identity": self.env["data_identity"], "packages": self.env["packages"],
            "thread_settings": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "parallel_fold_workers": WORKERS,
        }
        contract_path = OUTPUT / "execution_contract.json"
        if contract_path.exists() and read_json(contract_path) != self.contract:
            raise ValueError("Existing v3 run uses a different experiment contract")
        if not contract_path.exists():
            atomic_json(OUTPUT / "environment.json", self.env)
            for name in sources:
                dest = OUTPUT / "source_snapshot" / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, dest)
            shutil.copy2(V2 / "requirements-lock.txt", OUTPUT / "requirements-lock.txt")
            atomic_json(contract_path, self.contract)
        self.contract_hash = sha256(contract_path)
        self.dev_evidence = {}

    def verify_protected(self):
        for name, expected in self.protected.items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f"Protected HGB v2 artifact changed: {name}")

    def model(self, spec, cutoff):
        options = {**self.options, "max_iter": spec["max_iter"], "l2_regularization": spec["l2_regularization"]}
        return HGBV3Candidate(options, cutoff, remarks=spec["remarks"], price_target=spec["price_target"],
                              market_trend=spec["market_trend"])

    def split(self, cutoff, months):
        train, _, _ = temporal_fold(self.listings, months[0], cutoff)
        valid = pd.concat([temporal_fold(self.listings, m, cutoff)[1].assign(month=str(m)) for m in months])
        truth = valid[["ml_num", "house_type_name", "community", "sold_price", "month"]].copy()
        truth["training_cutoff"] = cutoff.isoformat()
        return train, data.hide_answers(valid, TRACK), truth

    def v2_development(self, iteration):
        fits = {}
        for cutoff, months in fold_layout(DEVELOPMENT_MONTHS, DEV_ORIGINS).items():
            folder = V2 / "physical_runs" / "stage3_leaves_127" / str(cutoff.date())
            meta = read_json(folder / "fit.json")
            train, _, truth = self.split(cutoff, months)
            spec = meta["spec"]
            if (meta["execution_contract_sha256"] != self.old_contract_hash
                    or spec != self.v2_frozen["selected"]["spec"]
                    or meta["model_config"]["options"] != self.options
                    or meta["n_train"] != len(train) or meta["max_training_sale"] != train.date_sold.max().isoformat()):
                raise ValueError(f"HGB v2 development control differs at {cutoff}")
            path = folder / f"predictions_{iteration}.csv"
            digest = sha256(path)
            if digest != meta["prediction_sha256"][str(iteration)]:
                raise ValueError(f"HGB v2 reference predictions changed: {path}")
            frame = load_predictions(path)
            compare_identity(frame, truth)
            fits[cutoff] = frame
            self.dev_evidence[str(path.relative_to(ROOT))] = digest
        return cohorts(fits, DEVELOPMENT_MONTHS, DEV_ORIGINS)

    def fit_fold(self, name, spec, cutoff, months, checkpoints):
        train, homes, truth = self.split(cutoff, months)
        dest = OUTPUT / "physical_runs" / name / str(cutoff.date())
        if dest.exists():
            meta = read_json(dest / "fit.json")
            if (meta["execution_contract_sha256"] != self.contract_hash or meta["spec"] != spec
                    or meta["checkpoints"] != list(checkpoints)):
                raise ValueError(f"Incompatible saved fit at {dest}")
            frames = {}
            for i in checkpoints:
                path = dest / f"predictions_{i}.csv"
                if sha256(path) != meta["prediction_sha256"][str(i)]:
                    raise ValueError(f"Saved predictions changed: {path}")
                frames[i] = load_predictions(path)
                compare_identity(frames[i], truth)
            return frames
        started = perf_counter()
        model = self.model(spec, cutoff).fit(train)
        if model.estimator.n_iter_ != spec["max_iter"]:
            raise ValueError("Unexpected early stopping")
        predicted = price_checkpoints(model, homes, checkpoints)
        train_prices = price_checkpoints(model, data.hide_answers(train, TRACK), checkpoints)
        frames = {i: truth.assign(predicted_price_raw=prices) for i, prices in predicted.items()}
        training = {str(i): point_metrics(train.sold_price, prices) for i, prices in train_prices.items()}
        future = {str(i): score(frame) for i, frame in frames.items()}
        elapsed = perf_counter() - started
        dest.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=dest.parent) as tmp:
            stage = Path(tmp)
            digests = {}
            for i, frame in frames.items():
                path = stage / f"predictions_{i}.csv"
                frame.to_csv(path, index=False)
                digests[str(i)] = sha256(path)
            atomic_json(stage / "fit.json", {
                "execution_contract_sha256": self.contract_hash, "spec": spec, "checkpoints": list(checkpoints),
                "training_cutoff": cutoff.isoformat(), "max_training_sale": train.date_sold.max().isoformat(),
                "n_train": len(train), "n_validation": len(truth), "months": list(map(str, months)),
                "model_config": model.configuration(), "training": training, "validation": future,
                "prediction_sha256": digests, "seconds": elapsed,
            })
            stage.rename(dest)
        scores = ", ".join(f"i{i}={future[str(i)]['mdape']:.4f}%" for i in checkpoints)
        print(f"{name} {cutoff.date()} {scores} ({elapsed:.1f}s)", flush=True)
        return frames

    def evaluate(self, name, spec, rolling_months=DEVELOPMENT_MONTHS, origins=DEV_ORIGINS, checkpoints=None):
        checkpoints = checkpoints or (spec["max_iter"],)
        layout = fold_layout(rolling_months, origins)

        def fit(entry):
            cutoff, months = entry
            return cutoff, self.fit_fold(name, spec, cutoff, months, checkpoints)

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            fits = dict(pool.map(fit, layout.items()))
        return {i: cohorts({cutoff: frames[i] for cutoff, frames in fits.items()}, rolling_months, origins)
                for i in checkpoints}

    def development(self):
        path = OUTPUT / "frozen_config.json"
        if path.exists():
            frozen = read_json(path)
            if frozen["execution_contract_sha256"] != self.contract_hash:
                raise ValueError("Frozen selection differs from the current experiment contract")
            return frozen
        reference = self.v2_development(1600)
        control = matched_result("hgb_v2", CONTROL, reference, reference)
        verify_scores(control, self.v2_frozen["selected"])
        stages = []

        def choose(label, parent, candidates):
            passing = [row for row in candidates if beats_both(row, parent)]
            winner = min(passing, key=lambda r: (
                r["rolling"]["mdape"], r["rolling"]["mape"], r["fixed"]["mdape"],
                r["fixed"]["mape"], r["spec"]["max_iter"],
            )) if passing else parent
            stages.append({"stage": label, "parent": parent["name"], "selected": winner["name"],
                           "qualifying": [r["name"] for r in passing], "results": candidates})
            atomic_json(OUTPUT / "development_progress.json", {"stages": stages, "current_winner": winner})
            print(f"SELECT {label}: {winner['name']} rolling={winner['rolling']['mdape']:.4f}% "
                  f"fixed={winner['fixed']['mdape']:.4f}%", flush=True)
            return winner

        candidates = [matched_result("v2_truncated_1200", {**CONTROL, "max_iter": 1200},
                                     self.v2_development(1200), reference)]
        for value in L2_VALUES:
            name = f"l2_{value}"
            spec = {**CONTROL, "l2_regularization": value, "max_iter": CHECKPOINTS[-1]}
            for i, frames in self.evaluate(name, spec, checkpoints=CHECKPOINTS).items():
                candidates.append(matched_result(f"{name}_i{i}", {**spec, "max_iter": i}, frames, reference))
        parent = choose("regularization_and_truncation", control, candidates)

        for name, change in (
            ("remarks_10_flags", {"remarks": True}),
            ("log_price_per_sqft", {"price_target": "log_ppsf"}),
            ("market_trend_90d", {"market_trend": True}),
        ):
            spec = {**parent["spec"], **change}
            frames = self.evaluate(name, spec)[spec["max_iter"]]
            parent = choose(name, parent, [matched_result(name, spec, frames, reference)])
        frozen = {"version": VERSION, "created_utc": now(), "execution_contract_sha256": self.contract_hash,
                  "control": control, "stages": stages, "selected": parent,
                  "development_reference_sha256": self.dev_evidence}
        atomic_json(path, frozen)
        return frozen

    def backtest(self, frozen):
        old_summary = read_json(V2 / "backtest_summary.json")
        if old_summary["frozen_config_sha256"] != self.v2_frozen_hash:
            raise ValueError("HGB v2 backtest differs from its frozen configuration")
        refs = {}
        evidence = {}
        for cohort in ("rolling", "fixed"):
            path = V2 / f"reused_{cohort}_predictions.csv"
            refs[cohort] = load_predictions(path)
            evidence[str(path.relative_to(ROOT))] = sha256(path)
        truth = {cutoff: self.split(cutoff, months)[2]
                 for cutoff, months in fold_layout(HOLDOUT_MONTHS, EVAL_ORIGINS).items()}
        control = matched_result("hgb_v2", CONTROL, refs, cohorts(truth, HOLDOUT_MONTHS, EVAL_ORIGINS))
        verify_scores(control, old_summary["results"][1])
        selected = frozen["selected"]
        spec = selected["spec"]
        frames = (refs if spec == CONTROL else
                  self.evaluate("frozen_backtest", spec, HOLDOUT_MONTHS, EVAL_ORIGINS)[spec["max_iter"]])
        winner = matched_result(selected["name"], spec, frames, refs)
        for name, frame in frames.items():
            frame.to_csv(OUTPUT / f"reused_{name}_predictions.csv", index=False)
        groups = []
        for model_name, table in (("hgb_v2", refs), (VERSION, frames)):
            for cohort, frame in table.items():
                if cohort == "fixed":
                    start = pd.Timestamp(EVAL_ORIGINS[0]).to_period("M").ordinal
                    frame = frame.assign(horizon_months=[m.ordinal - start + 1 for m in pd.PeriodIndex(frame.month, freq="M")])
                groups.append(grouped_metrics(frame).assign(cohort=cohort, model=model_name))
        pd.concat(groups, ignore_index=True).to_csv(OUTPUT / "reused_grouped_metrics.csv", index=False)
        rows = [dict(stage="development_control", **frozen["control"])]
        rows += [dict(stage=step["stage"], **result) for step in frozen["stages"] for result in step["results"]]
        rows += [dict(stage="reused_backtest", **result) for result in (control, winner)]
        pd.json_normalize(rows).to_csv(OUTPUT / "comparison.csv", index=False)
        summary = {"frozen_config_sha256": sha256(OUTPUT / "frozen_config.json"),
                   "control": control, "selected": winner, "holdout_previously_observed": True,
                   "improves_both_reused_mdapes": beats_both(winner, control), "reference_sha256": evidence}
        atomic_json(OUTPUT / "backtest_summary.json", summary)
        for result in (control, winner):
            print(f"REUSED {result['name']}: rolling={result['rolling']['mdape']:.4f}% "
                  f"fixed={result['fixed']['mdape']:.4f}%", flush=True)
        return summary

    def predict(self, frozen, evaluation):
        if not evaluation["improves_both_reused_mdapes"]:
            return
        frozen_hash = sha256(OUTPUT / "frozen_config.json")
        if frozen_hash != evaluation["frozen_config_sha256"]:
            raise ValueError("Evaluation used a different frozen configuration")
        run = OUTPUT / "final_run"
        model_name, csv_name = "off_market.joblib", "submission_off_market.csv"
        if not run.exists():
            model = self.model(frozen["selected"]["spec"], FINAL_CUTOFF).fit(self.listings)
            homes = data.load(TRACK)
            prices = model.predict(homes)
            with TemporaryDirectory(dir=OUTPUT) as tmp:
                stage = Path(tmp)
                joblib.dump(model, stage / model_name, compress=3)
                np.testing.assert_array_equal(joblib.load(stage / model_name).predict(homes), prices)
                pd.DataFrame({"ml_num": homes.ml_num, "predicted_price": np.round(prices)}).to_csv(
                    stage / csv_name, index=False
                )
                problems = check_submission.check_file(stage / csv_name, homes.ml_num)
                if problems:
                    raise ValueError(f"Invalid v3 submission: {problems}")
                atomic_json(stage / "fit.json", {
                    "frozen_config_sha256": frozen_hash, "configuration": model.configuration(),
                    "n_train": len(self.listings), "n_test": len(homes),
                    "reload_exact_match": True, "submission_checker": "passed",
                    "sha256": {name: sha256(stage / name) for name in (model_name, csv_name)},
                })
                stage.rename(run)
        meta = read_json(run / "fit.json")
        if meta["frozen_config_sha256"] != frozen_hash:
            raise ValueError("Final fit belongs to a different frozen configuration")
        exported = {}
        for dest in (OUTPUT / "final_models" / model_name, ROOT / "submissions" / VERSION / csv_name):
            source = run / dest.name
            expected = meta["sha256"][dest.name]
            if sha256(source) != expected or (dest.exists() and sha256(dest) != expected):
                raise ValueError(f"Final artifact digest differs: {dest}")
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                with TemporaryDirectory(dir=dest.parent) as tmp:
                    staged = Path(tmp) / dest.name
                    shutil.copy2(source, staged)
                    staged.rename(dest)
            exported[str(dest.relative_to(ROOT))] = expected
        atomic_json(OUTPUT / "final_summary.json", {**meta, "sha256": exported})
        print(f"FINAL candidate: {ROOT / 'submissions' / VERSION / csv_name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "development", "backtest"), default="all")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUTPUT / ".runner.lock"), timeout=0):
        run = V3Run()
        if args.phase == "backtest" and not (OUTPUT / "frozen_config.json").exists():
            parser.error("Complete development before backtest")
        frozen = run.development()
        if args.phase != "development":
            evaluated = run.backtest(frozen)
            if args.phase == "all":
                run.predict(frozen, evaluated)
        run.verify_protected()
        baseline_guard(OUTPUT.with_name("hgb_v1"))
        n_fits = len(list((OUTPUT / "physical_runs").glob("*/*/fit.json")))
        print(f"Completed {args.phase}. Physical evaluation fits: {n_fits}", flush=True)


if __name__ == "__main__":
    main()
