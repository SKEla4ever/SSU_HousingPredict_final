# Organizer starter and submission workflow

The `starter/` directory contains the organizer's starter repository as archived from commit `6c5831e12f523d617eccd1418d46a5cf2a0bb8b4`. It includes the case handout, judging rubric, original Python baseline and checker, optional notebook and R baseline, and the LaTeX report template. [Organizer repository snapshot](https://github.com/erwinzhang7/SSU-Data-Dash/tree/6c5831e12f523d617eccd1418d46a5cf2a0bb8b4)

- [Case handout](starter/handout.md) and [judging rubric](starter/rubric.md)
- [Starter setup and commands](starter/README.md)
- [Original LaTeX report template](starter/report/report.tex)
- [Synthetic competition dataset](https://huggingface.co/datasets/Baliyang/SSU_data_dash)

To run the organizer's tests, enter `official/starter/` from the `final_HGB` root and run `MPLBACKEND=Agg python -m pytest -q` in the release Python environment. `MPLBACKEND=Agg` lets the starter render its figure in a terminal without a graphical session.

## Run the workflow with the final models

The runner in `starter/` reproduces the organizer's original baselines. For the final models, use the commands below from the `final_HGB` root after configuring Python and data. [Release setup](../README.md)

```sh
python -m pip install -r requirements.txt
python run.py validate
python run.py report
python run.py predict
python run.py check
```

`validate` runs the organizer's December 2025–May 2026 walk-forward and the team's supplementary fixed-origin backtest. `report` writes the results table and figure. `predict` fits the frozen release models and writes both CSVs. `check` verifies the two-column schema, one prediction per expected ID, and positive, finite prices.

| Submission CSV | Required rows |
|---|---:|
| [submission_off_market.csv](../submissions/submission_off_market.csv) | 3,974 |
| [submission_list_aware.csv](../submissions/submission_list_aware.csv) | 4,064 |

## Round 1 hand-in

The handout specifies one PDF of at most four pages, submitted through the official form by November 8, 2026 at 11:59 p.m. EST. The PDF must include every team member's full name and the walk-forward MdAPE for each track, and link to both a YouTube video of up to five minutes and a GitHub repository containing the code and both submission CSVs. An unlisted video is acceptable. Give the organizers access to a private repository. [Submission instructions](starter/handout.md#what-to-submit)

Recompile the PDF after editing the source and verify that the YouTube and GitHub links open before submitting. [Report source](../report/report.tex) · [Round 1 report PDF](../report/report.pdf)
