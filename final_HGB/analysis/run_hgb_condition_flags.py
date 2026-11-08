"""Compare fixed condition cues against the saved Off-market V3 leaf20 control."""
import argparse
import os
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import data
from analysis.hgb_v3_candidate import HGBV3Candidate, remark_features
from analysis.run_hgb_baselines import atomic_json, load_predictions, read_json, unique_id
from analysis.run_hgb_improvements import DEV_ORIGINS, FINAL_CUTOFF, compare_identity, fold_layout, score
from analysis.run_hgb_v3 import beats_both, cohorts, matched_result, verify_scores
from analysis.run_tree_baselines import DEVELOPMENT_MONTHS, environment, sha256, temporal_fold


OUTPUT = ROOT / "outputs/hgb_baselines/hgb_v3_condition_flags"
V3 = OUTPUT.with_name("hgb_v3")
TRACK = "off_market"
CONDITION_PATTERNS = {
    "remark_aged_system": (
        r"\b(?:older|old|original|dated|aging|ageing|outdated) "
        r"(?:furnace|roof|windows|electrical|wiring|plumbing|hvac|mechanicals)\b"
    ),
    "remark_original_finishes": (
        r"\b(?:original (?:kitchen|bathrooms?|finishes|fixtures|light fixtures)|"
        r"(?:kitchen|bathrooms?|finishes|fixtures) (?:is|are|remain) original)\b"
    ),
    "remark_major_repair": (
        r"\b(?:(?:significant|substantial|major|extensive) (?:work|renovations?|updates?|updating|repairs?|tlc)|"
        r"bring your toolbox|down to (?:the )?studs|needs? (?:a )?(?:full|complete) (?:redo|renovation|overhaul))\b"
    ),
    "remark_unfinished_basement": r"\bunfinished (?:basement|bsmt)\b",
}
AGED_SYSTEM_PATTERNS = {"remark_aged_system": CONDITION_PATTERNS["remark_aged_system"]}


def condition_flags(homes, patterns=CONDITION_PATTERNS):
    """Record phrase mentions. A zero means the rule did not find a mention."""
    remarks = homes["remarks"].fillna("").str.lower()
    return pd.DataFrame({name: remarks.str.contains(pattern).astype("int8")
                         for name, pattern in patterns.items()}, index=homes.index)


class ConditionFlagsCandidate(HGBV3Candidate):
    def __init__(self, options, cutoff, *, condition_patterns, **kwargs):
        super().__init__(options, cutoff, **kwargs)
        self.condition_patterns = dict(condition_patterns)

    def configuration(self):
        config = super().configuration()
        config["schema"]["numeric_features"].extend(self.condition_patterns)
        config["schema"]["input_feature_order"].extend(self.condition_patterns)
        config["preprocessing"]["condition_remarks"] = {
            "extractor": "analysis.run_hgb_condition_flags.condition_flags",
            "normalization": "fill missing with empty string and lowercase",
            "flags": dict(self.condition_patterns),
            "dtype": "int8",
        }
        config["candidate"]["class"] = "analysis.run_hgb_condition_flags.ConditionFlagsCandidate"
        return config

    def prepare_features(self, homes, fit=False):
        base = super().prepare_features(homes, fit=fit)
        matrix = pd.concat([base, condition_flags(homes, self.condition_patterns)], axis=1)
        if fit:
            self.model_feature_order_ = matrix.columns.tolist()
        return matrix


def split_fold(listings, cutoff, months):
    train, _, _ = temporal_fold(listings, months[0], cutoff)
    valid = pd.concat([temporal_fold(listings, month, cutoff)[1].assign(month=str(month)) for month in months])
    truth = valid[["ml_num", "house_type_name", "community", "sold_price", "month"]].assign(
        training_cutoff=cutoff.isoformat())
    return train, valid, truth


def fit_predictions(model, train, valid, truth):
    model.fit(train)
    if model.estimator.n_iter_ != model.estimator.max_iter:
        raise ValueError("Unexpected number of boosting iterations")
    return truth.assign(predicted_price_raw=model.predict(data.hide_answers(valid, TRACK)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aged-system-only", action="store_true",
                        help="Add only the aged-system cue instead of the four-flag block.")
    args = parser.parse_args()
    candidate_name = "aged_system_only" if args.aged_system_only else "four_condition_flags"
    patterns = AGED_SYSTEM_PATTERNS if args.aged_system_only else CONDITION_PATTERNS

    frozen_path = V3 / "frozen_config.json"
    contract_path = V3 / "execution_contract.json"
    release_path = V3 / "final_summary.json"
    frozen, contract, release = map(read_json, (frozen_path, contract_path, release_path))
    if (release["frozen_config_sha256"] != sha256(frozen_path)
            or frozen["execution_contract_sha256"] != sha256(contract_path)):
        raise ValueError("Expected the frozen V3 leaf20 revision as control")
    control = frozen["selected"]
    options = release["configuration"]["options"]
    if (options["min_samples_leaf"] != 20 or control["name"] != "log_price_per_sqft"
            or control["spec"]["price_target"] != "log_ppsf" or not control["spec"]["remarks"]
            or control["spec"]["market_trend"]):
        raise ValueError("Unexpected V3 leaf20 model configuration")
    rounds = options["max_iter"]
    kwargs = dict(remarks=True, price_target="log_ppsf")
    if HGBV3Candidate(options, FINAL_CUTOFF, **kwargs).configuration() != release["configuration"]:
        raise ValueError("Current model code does not reproduce the frozen V3 configuration")
    env = environment()
    if env["packages"] != contract["packages"]:
        raise ValueError("Use the package versions from the saved V3 run")
    for name, entry in env["data_identity"].items():
        if entry["sha256"] != contract["data_identity"][name]["sha256"]:
            raise ValueError(f"V3 data snapshot differs for {name}")
    for name, digest in contract["source_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"V3 source snapshot differs for {name}")

    listings = data.load("listings")
    if not listings.ml_num.is_unique:
        raise ValueError("Listings must have distinct IDs")
    layout = fold_layout(DEVELOPMENT_MONTHS, DEV_ORIGINS)
    references = {}
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in (frozen_path, contract_path, release_path)}
    for cutoff in layout:
        folder = V3 / "physical_runs" / control["name"] / str(cutoff.date())
        fit_path, pred_path = folder / "fit.json", folder / f"predictions_{rounds}.csv"
        fit, digest = read_json(fit_path), sha256(pred_path)
        if (fit["execution_contract_sha256"] != sources[str(contract_path.relative_to(ROOT))]
                or fit["spec"] != control["spec"] or fit["training_cutoff"] != cutoff.isoformat()
                or fit["model_config"] != HGBV3Candidate(options, cutoff, **kwargs).configuration()
                or fit["prediction_sha256"][str(rounds)] != digest):
            raise ValueError(f"Unexpected V3 reference fit at {cutoff.date()}")
        sources[str(fit_path.relative_to(ROOT))] = sha256(fit_path)
        sources[str(pred_path.relative_to(ROOT))] = digest
        references[cutoff] = load_predictions(pred_path)
    reference = cohorts(references, DEVELOPMENT_MONTHS, DEV_ORIGINS)
    baseline = matched_result("v3_leaf20", control["spec"], reference, reference)
    verify_scores(baseline, control)

    output = OUTPUT / unique_id()
    script = Path(__file__).resolve()
    spec = {**control["spec"], "new_remarks_flags": patterns}
    atomic_json(output / "experiment.json", {
        "candidate_name": candidate_name, "candidate": spec, "control_configuration": release["configuration"],
        "reference_sha256": sources,
        "runner_sha256": sha256(script), "environment": env,
        "rolling_months": list(map(str, DEVELOPMENT_MONTHS)), "fixed_origins": list(DEV_ORIGINS),
        "fixed_horizon_months": 4, "evaluation_scope": "development only",
        "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    })
    print(f"OUTPUT {output}", flush=True)
    (output / script.name).write_bytes(script.read_bytes())
    fits = {}
    for cutoff, months in layout.items():
        started = perf_counter()
        train, valid, truth = split_fold(listings, cutoff, months)
        compare_identity(truth, references[cutoff])
        model = ConditionFlagsCandidate(options, cutoff, condition_patterns=patterns, **kwargs)
        fits[cutoff] = fit_predictions(model, train, valid, truth)
        print(f"{cutoff.date()} n_train={len(train)} n_eval={len(valid)} "
              f"MdAPE={score(fits[cutoff])['mdape']:.4f}% ({perf_counter() - started:.1f}s)", flush=True)

    candidate = cohorts(fits, DEVELOPMENT_MONTHS, DEV_ORIGINS)
    result = matched_result(candidate_name, spec, candidate, reference)
    flagged_homes = listings.loc[listings.ml_num.isin(candidate["fixed"].ml_num)].set_index("ml_num")
    flags = condition_flags(flagged_homes, patterns).assign(needs_work=remark_features(flagged_homes).remark_needs_work)
    subgroups = []
    for cohort, frame in candidate.items():
        paired = frame.merge(reference[cohort][["ml_num", "predicted_price_raw"]], on="ml_num",
                             suffixes=("", "_control"), validate="one_to_one")
        if cohort == "fixed":
            paired = paired.join(flags, on="ml_num", validate="one_to_one")
            for flag in patterns:
                for subset in ("all_flagged", "flagged_without_needs_work"):
                    rows = paired.loc[paired[flag].eq(1)]
                    if subset == "flagged_without_needs_work":
                        rows = rows.loc[rows.needs_work.eq(0)]
                    if not rows.empty:
                        control_rows = rows.assign(predicted_price_raw=rows.predicted_price_raw_control)
                        subgroups.append({"flag": flag, "subset": subset, "control": score(control_rows),
                                          "candidate": score(rows)})
        paired.to_csv(output / f"development_{cohort}_predictions.csv", index=False)

    summary = {"control": baseline, "candidate": result, "improves_both_mdapes": beats_both(result, baseline),
               "fixed_subgroups": subgroups, "physical_fits": len(fits)}
    atomic_json(output / "summary.json", summary)
    print(f"DEVELOPMENT rolling {baseline['rolling']['mdape']:.4f}% → {result['rolling']['mdape']:.4f}%, "
          f"fixed {baseline['fixed']['mdape']:.4f}% → {result['fixed']['mdape']:.4f}%", flush=True)


if __name__ == "__main__":
    main()
