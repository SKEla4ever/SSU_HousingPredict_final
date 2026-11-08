"""Run the whole pipeline. From this folder:

    python run.py validate   test the models on the last six months of sales
    python run.py report     score those predictions and draw a figure
    python run.py predict    predict the test homes and write the two submission files
    python run.py check      check the submission files before you hand them in

On Windows, type py instead of python if python is not found.
"""
from hashlib import file_digest
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import joblib

import data
import models
from analysis.run_hgb_condition_flags import ConditionFlagsCandidate
from check_submission import check_all

ROOT = Path(__file__).parent
TRACKS = ["off_market", "list_aware"]
CONFIG_PATH = ROOT / "config" / "frozen_models.json"
FINAL_CUTOFF = pd.Timestamp("2026-06-01")
HORIZON_CUTOFF = pd.Timestamp("2026-02-01")
VALIDATION_MONTHS = pd.period_range("2025-12", "2026-05", freq="M")


def load_release_data(name):
    expected = json.loads(CONFIG_PATH.read_text())["data_sha256"][name]
    path = data.find_file(name)
    with path.open("rb") as source:
        actual = file_digest(source, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Use the original competition parquet snapshot for {name}: {path}")
    return data.load(name)


def make_model(track, cutoff):
    if track == "list_aware":
        return models.make_hgb_model(track, CONFIG_PATH)
    selected = json.loads(CONFIG_PATH.read_text())["tracks"][track]["model_config"]
    candidate = selected["candidate"]
    model = ConditionFlagsCandidate(
        selected["options"],
        cutoff,
        remarks=candidate["remarks"],
        price_target=candidate["price_target"],
        market_trend=candidate["market_trend"],
        condition_patterns=selected["preprocessing"]["condition_remarks"]["flags"],
    )
    configuration = model.configuration()
    configuration["candidate"]["cutoff"] = candidate["cutoff"]
    if configuration != selected:
        raise ValueError("Off-market source differs from the frozen model configuration")
    return model


def evaluate_homes(model, homes, track, cutoff):
    return homes[["ml_num", "house_type_name", "community", "sold_price"]].assign(
        track=track,
        month=homes["date_sold"].dt.to_period("M").astype(str),
        training_cutoff=cutoff.date().isoformat(),
        predicted_price=model.predict(data.hide_answers(homes, track)),
    )


def validate():
    listings = load_release_data("listings")
    results = []
    horizon_results = []
    horizon_homes = listings[listings["date_sold"] >= HORIZON_CUTOFF]
    for month in VALIDATION_MONTHS:
        cutoff = month.start_time
        train = listings[listings["date_sold"] < cutoff]
        homes = listings[listings["date_sold"].dt.to_period("M") == month]
        for track in TRACKS:
            model = make_model(track, cutoff).fit(train)
            results.append(evaluate_homes(model, homes, track, cutoff))
            if cutoff == HORIZON_CUTOFF:
                horizon_results.append(evaluate_homes(model, horizon_homes, track, cutoff))
        print(f"{month}: trained on {len(train):,} earlier sales, predicted {len(homes):,} homes")
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    for name, frames in [("validation", results), ("horizon", horizon_results)]:
        path = output / f"{name}_predictions.csv"
        pd.concat(frames).to_csv(path, index=False)
        print(f"Wrote {path.relative_to(ROOT)}")


def bootstrap_ci(values, statistic, repeats=1000):
    """95% interval: recompute the statistic on 1,000 resamples of the homes
    (drawn with replacement) and keep the middle 95% of the results."""
    rng = np.random.default_rng(0)
    results = [statistic(rng.choice(values, size=len(values))) for _ in range(repeats)]
    return [round(float(x), 2) for x in np.percentile(results, [2.5, 97.5])]


def metrics(predictions):
    """Scores for one track. Errors are in percent of the sold price."""
    signed = 100 * (predictions["predicted_price"] - predictions["sold_price"]) / predictions["sold_price"]
    absolute = signed.abs().to_numpy()
    return {
        "n": len(predictions),
        "mape": round(float(np.mean(absolute)), 2),          # mean absolute % error
        "mape_95ci": bootstrap_ci(absolute, np.mean),
        "mdape": round(float(np.median(absolute)), 2),       # median absolute % error
        "mdape_95ci": bootstrap_ci(absolute, np.median),
        "median_signed_error": round(float(np.median(signed)), 2),  # above 0: predicts too high
    }


def report():
    path = ROOT / "outputs" / "validation_predictions.csv"
    if not path.exists():
        sys.exit(f"{path} does not exist yet: run `python run.py validate` first")
    predictions = pd.read_csv(path)
    scores = {track: metrics(predictions[predictions["track"] == track]) for track in TRACKS}

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "validation_report.json").write_text(json.dumps(scores, indent=2))
    horizon = pd.read_csv(ROOT / "outputs" / "horizon_predictions.csv")
    fixed_scores = {track: metrics(horizon[horizon["track"] == track]) for track in TRACKS}
    (ROOT / "reports" / "horizon_report.json").write_text(json.dumps(fixed_scores, indent=2))
    lines = [
        "# Validation report", "",
        f"Walk-forward validation, {VALIDATION_MONTHS[0]} to {VALIDATION_MONTHS[-1]}: each month was "
        "predicted by models trained only on the sales before it.",
        "Errors are (predicted - sold) / sold. Brackets: bootstrap 95% interval.", "",
        "| Track | Homes | MAPE | MdAPE | Median signed error |", "|---|---|---|---|---|"]
    for track, s in scores.items():
        lines.append(f"| {track} | {s['n']:,} | {s['mape']:.2f}% ({s['mape_95ci'][0]:.2f} to {s['mape_95ci'][1]:.2f}) "
                     f"| {s['mdape']:.2f}% ({s['mdape_95ci'][0]:.2f} to {s['mdape_95ci'][1]:.2f}) "
                     f"| {s['median_signed_error']:+.2f}% |")
    lines += [
        "", "MAPE: mean absolute percentage error. MdAPE: median absolute percentage error.",
        "A median signed error above 0 means the model tends to predict too high.",
        "The intervals treat homes as independent. Homes sold in the same month share the same",
        "market swings, so the real uncertainty is somewhat wider.", "",
        "![Error by price range](../figures/error_by_price_range.png)", ""]
    (ROOT / "reports" / "validation_report.md").write_text("\n".join(lines))
    write_latex_table(scores)
    error_by_price_range(predictions)
    print("\n".join(lines))
    print("Wrote reports/validation_report.md, reports/validation_report.json, reports/results_table.tex "
          "and figures/error_by_price_range.png")


def write_latex_table(scores):
    """The same scores as a LaTeX table, for report/report.tex to include."""
    rows = []
    for track, s in scores.items():
        name = track.replace("_", "-")
        rows.append(f"{name} & {s['n']:,} & {s['mdape']:.2f} ({s['mdape_95ci'][0]:.2f}--{s['mdape_95ci'][1]:.2f}) "
                    f"& {s['mape']:.2f} ({s['mape_95ci'][0]:.2f}--{s['mape_95ci'][1]:.2f}) "
                    f"& {s['median_signed_error']:+.2f} \\\\")
    table = "\n".join([
        "% Written by `python run.py report`. Do not edit by hand: rerun the command instead.",
        "\\begin{tabular}{lrccc}", "\\hline",
        "Track & Homes & MdAPE \\% (95\\% CI) & MAPE \\% (95\\% CI) & Median signed error \\% \\\\", "\\hline",
        *rows, "\\hline", "\\end{tabular}"])
    (ROOT / "reports" / "results_table.tex").write_text(table + "\n")


def error_by_price_range(predictions):
    """Figure: how big the errors are, from the cheapest tenth of homes to the dearest.

    Want more figures? Copy this function and group the errors by something else:
    house_type_name, month, or community (add it to the predictions file in validate()).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colours = {"off_market": "#2a78d6", "list_aware": "#eb6834"}
    for track in TRACKS:
        rows = predictions[predictions["track"] == track]
        error = 100 * (rows["predicted_price"] - rows["sold_price"]).abs() / rows["sold_price"]
        decile = pd.qcut(rows["sold_price"], 10, labels=False) + 1   # 1 = cheapest tenth
        by_decile = error.groupby(decile)
        middle = by_decile.median()
        ax.plot(middle.index, middle.values, marker="o", linewidth=2, color=colours[track], label=track)
        ax.fill_between(middle.index, by_decile.quantile(0.25), by_decile.quantile(0.75),
                        color=colours[track], alpha=0.15, linewidth=0)
    prices = rows["sold_price"].groupby(decile).median()
    ax.set_xticks(prices.index, [f"{d}\n${p / 1e6:.2f}M" for d, p in prices.items()])
    ax.set_xlabel("Sold-price decile (1 = cheapest tenth of homes), with its median price")
    ax.set_ylabel("Absolute % error\n(line: median, band: middle 50%)")
    ax.set_title(f"Validation error by price range, {VALIDATION_MONTHS[0]} to {VALIDATION_MONTHS[-1]}")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#e5e5e5")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "error_by_price_range.png", dpi=150)
    plt.close(fig)


def predict():
    listings = load_release_data("listings")
    if not (listings["date_sold"] < FINAL_CUTOFF).all():
        raise ValueError("Final training sales must all precede the hidden test cutoff, 2026-06-01")
    submissions = ROOT / "submissions"
    submissions.mkdir(exist_ok=True)
    model_folder = ROOT / "outputs" / "fitted_models"
    with TemporaryDirectory(dir=ROOT) as temp:
        staging = Path(temp)
        for track in TRACKS:
            homes = load_release_data(track)
            model = make_model(track, FINAL_CUTOFF).fit(listings)
            prices = np.round(model.predict(homes))
            path = staging / f"submission_{track}.csv"
            pd.DataFrame({"ml_num": homes["ml_num"], "predicted_price": prices}).to_csv(path, index=False)
            model_path = staging / f"{track}.joblib"
            joblib.dump(model, model_path, compress=3)
            np.testing.assert_array_equal(np.round(joblib.load(model_path).predict(homes)), prices)
        if not check_all(staging):
            raise ValueError("Generated submissions failed the official checker")
        model_folder.mkdir(parents=True, exist_ok=True)
        for track in TRACKS:
            name = f"submission_{track}.csv"
            (staging / f"{track}.joblib").replace(model_folder / f"{track}.joblib")
            (staging / name).replace(submissions / name)
            print(f"Wrote submissions/{name}")


def check():
    return check_all(ROOT / "submissions")


COMMANDS = {"validate": validate, "report": report, "predict": predict, "check": check}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    result = COMMANDS[sys.argv[1]]()
    sys.exit(1 if result is False else 0)
