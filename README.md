# SAM 2 Failure Recovery

A training-free wrapper around SAM 2.1 for semi-supervised video object segmentation. SAM 2 is
prompted with a box on the first frame in which an object is visible. The wrapper then watches
SAM 2's own outputs (object score, change in mask area, jump of the mask centre, abrupt
disappearance) to flag frames where SAM 2 has lost the object, either with hand-set rules or with
a learned classifier. At a flagged frame it re-prompts SAM 2 with a box from a CSRT tracker that
is kept aligned with SAM 2's last reliable mask, and re-propagates the following frames. SAM 2
weights and the official scorers are used unmodified.

We compare three systems, each on DAVIS 2017 val (30 videos, 61 objects) and on a seeded
60-video subset of the SA-V test set (105 objects), with SAM 2.1 Hiera-Tiny and Hiera-Large:

- a CSRT + GrabCut baseline (no SAM 2),
- SAM 2 alone,
- SAM 2 + our failure detector and recovery step.

SA-V val is used only for tuning and fitting (rule thresholds, the learned detector, the
verification rule). DAVIS val and the SA-V test subset are run only with settings that were
frozen beforehand. Scores are J, F and J&F from the official evaluators, plus FPS, detector
precision / recall / PR-AUC, and per-video improved / worse / unchanged counts with bootstrap
intervals.

## Layout

```
src/common/      config loader, mask PNG I/O, boxes, per-frame signal schema
src/data/        DAVIS and SA-V loaders, first-frame boxes, seeded SA-V test subset
src/baseline/    CSRT + GrabCut baseline
src/sam2_runner/ SAM 2 wrapper, per-frame signals, caching, timing
src/detection/   failure labels, rule detector, learned detector, streaming adapter
src/oracle/      ground-truth flags and boxes (oracle runs only, named "oracle")
src/recovery/    aligned CSRT, box check, recovery module, online runner
src/eval/        official-scorer wrappers, completeness checks, bootstrap, main table
experiments/     one package per experiment (e1 ... e13) plus run / tuning helpers
configs/         YAML / JSON settings; configs/detector/ and configs/verify/ are the frozen ones
scripts/         checkpoint download, demo video
notebooks/       Colab notebooks
tests/           pytest suite with a tiny synthetic video (tests/fixtures/)
results/         per-video and summary CSVs behind every number we report
figures/         generated figures
```

## Setup

### Mac (CPU: tests, tables, figures)

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

`requirements.txt` pins the runtime and development packages (NumPy, opencv-contrib, pandas,
scikit-learn, matplotlib, PyYAML, scikit-image, scipy, pytest, ruff, black, mypy). The tests run
on a synthetic fixture and need no data and no GPU. Everything that reads only the CSVs in
`results/` (all tables and figures) runs on the Mac.

### GPU (SAM 2 runs, recovery runs, timing)

Every GPU run reported here was done on the UCF Newton cluster: one NVIDIA H100 PCIe per job,
Slurm batch jobs, Python 3.11, torch 2.5.1+cu124 (torchvision 0.20.1), CUDA 12.4 toolkit, gcc
12.2. Each full run was split over disjoint video lists with the runner's `--videos` option so
that several jobs could run at once. Any CUDA machine with the same stack should work.

```
python3.11 -m venv sam2fr-env            # or a conda environment with Python 3.11
source sam2fr-env/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
TORCH_CUDA_ARCH_LIST="7.0;9.0" SAM2_BUILD_ALLOW_ERRORS=0 pip install -r requirements-colab.txt
```

The last line installs SAM 2.1 from github.com/facebookresearch/sam2 at the pinned commit
(package name `SAM-2`; a different `sam2` exists on PyPI) and builds its CUDA extension. It needs
a CUDA toolkit and a compiler on the machine (on a cluster: load the matching modules first).

An alternative for the SAM 2-alone runs is the Colab notebook `notebooks/run_sam2.ipynb`
together with `requirements-colab.txt` and `notebooks/colab_setup.ipynb`. Our Colab sessions
disconnected and ran out of compute units, so no reported number comes from Colab.

#### Example Slurm job (SAM 2 alone, one shard of one split)

Generic placeholders (`GPU_TYPE`, `CUDA_12_4_MODULE`, `/path/to/...`); the other GPU commands below use the same job with the
`python -m ...` line replaced.

```
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:GPU_TYPE:1
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/logs/%x-%j.out
# submit: sbatch --export=ALL,MODEL=tiny,SPLIT=davis_val,SHARD=0,NSHARDS=1 run_sam2.sbatch
module load CUDA_12_4_MODULE GCC_MODULE
source /path/to/venv/bin/activate
export SAM2FR_DATA_ROOT=/path/to/data
set -euo pipefail
cd /path/to/this/repo
PROMPTS="$SAM2FR_DATA_ROOT/outputs/prompts/$SPLIT.json"
# every NSHARDS-th video, starting at SHARD: disjoint lists, together all videos
VIDEOS=$(python -c "import json,sys; v=sorted(json.load(open(sys.argv[1]))['videos']); print(' '.join(v[int(sys.argv[2])::int(sys.argv[3])]))" "$PROMPTS" "$SHARD" "$NSHARDS")
python -m src.sam2_runner.run --model "$MODEL" --prompts "$PROMPTS" --videos $VIDEOS
```

The runner skips videos whose output is already complete, so a job can be resubmitted.

### Data layout

Datasets, checkpoints and every generated mask stay outside the repository. Set
`SAM2FR_DATA_ROOT` to one folder; every path in `configs/data.yaml` and `configs/models.yaml`
is relative to it:

```
$SAM2FR_DATA_ROOT/
  DAVIS/ImageSets/2017/val.txt          official DAVIS 2017 (480p) folders:
  DAVIS/JPEGImages/480p/<video>/*.jpg     frames
  DAVIS/Annotations/480p/<video>/*.png    indexed masks
  SA-V/sav_val/{sav_val.txt, JPEGImages_24fps/<video>/*.jpg, Annotations_6fps/<video>/<obj>/*.png}
  SA-V/sav_test/{sav_test.txt, JPEGImages_24fps/..., Annotations_6fps/...}   same layout
  checkpoints/sam2.1_hiera_tiny.pt, sam2.1_hiera_large.pt
  outputs/                              written by the code: prompts/<split>.json and
                                        <method>/<split>/{masks,signals,events}/<video>/...
```

```
export SAM2FR_DATA_ROOT=/path/to/data
python -m src.data.check --dataset davis --split val
python -m src.data.check --dataset sav --split val
python -m src.data.check --dataset sav --split test
python -m src.data.make_subset --check
```

The `src.data.check` lines compare the video counts with the expected ones (DAVIS val 30, SA-V val
155, SA-V test 150) and exit with an error on a different count; add `--full` to read every mask.
`make_subset --check` re-derives the seeded 60-video SA-V test subset (seed 5415) and compares it
with the committed `configs/splits/sav_test_subset60.txt`.

Checkpoints and the pinned official evaluators (both downloads; run once):

```
bash scripts/get_checkpoints.sh
python -m src.eval.get_evaluators
```

`get_checkpoints.sh` writes the two SAM 2.1 checkpoints into `$SAM2FR_DATA_ROOT/checkpoints`
(`--print-dir` only prints that folder). `get_evaluators` fetches the DAVIS 2017 evaluator and the
SA-V evaluator at pinned commits into `third_party/` (not tracked) and refuses a checkout that is
at another commit or modified. Both are called as libraries, unmodified.

## Reproduce

Every command takes `--help`. Steps are in run order. Each is marked:

- **GPU**: needs a GPU and the datasets (a Slurm job like the example above; only the
  `python -m ...` line changes).
- **DATA**: CPU only; needs the datasets and the masks written by the GPU steps (we ran these next
  to the masks).
- **MAC**: reads only the CSVs committed in `results/`; running the command regenerates the file
  named, and the committed file is what it produces.

Shell variables used below (set them once; `LISTS` holds one file per shard, one video name per
line: `davis_val_all.txt` and `sav_test_0of4.txt` ... `sav_test_3of4.txt`, any disjoint split of
the videos in the prompts file works, for example every 4th name of the sorted list as in the
example job; `sav_val_all.txt` is used in step 13):

```
export SAM2FR_DATA_ROOT=/path/to/data
PROMPTS=$SAM2FR_DATA_ROOT/outputs/prompts
LISTS=/path/to/video_lists
CFG=configs/recovery
```

### 0. Prompts, SAM 2 alone, scoring, failure counts

Each object is prompted with the tight box of its first annotated frame in which it is visible
(frame 0 in DAVIS), taken from the ground-truth mask of that one frame.

```
python -m experiments.sam2_runs.make_prompts --split davis_val
python -m experiments.sam2_runs.make_prompts --split sav_val
python -m experiments.sam2_runs.make_prompts --split sav_test
```

GPU: SAM 2 alone for `--model tiny` and `--model large` on every split (`davis_val`, `sav_val`,
`sav_test`), sharded over jobs as in the example job:

```
python -m src.sam2_runner.run --model tiny --prompts $PROMPTS/davis_val.json --videos $(cat $LISTS/davis_val_all.txt)
```

DATA, after all runs finished (completeness check, official scorers, failure counts):

```
python -m src.eval.check_complete
python -m experiments.sam2_runs.score_all
python -m experiments.sam2_runs.failure_stats --out results/failure_stats.csv
```

Produces `results/sam2_{tiny,large}_{davis_val,sav_val,sav_test}.csv`, their
`results/sam2_*_timing_*.csv` and `results/failure_stats.csv`. SA-V val is used only for tuning.

### 1. E1: CSRT + GrabCut baseline

Run on a Mac CPU (the CPU model is recorded in the timing CSV). SA-V scoring uses the 60-video
subset list.

```
python -m src.baseline.run --split davis_val
python -m src.baseline.run --split sav_test
python -m src.baseline.timing --split davis_val --out results/baseline_timing_davis_val.csv
python -m src.baseline.timing --split sav_test --out results/baseline_timing_sav_test.csv
python -m src.eval.score --dataset davis --split davis_val --method baseline_csrt_grabcut --out results/baseline_davis_val.csv
python -m src.eval.score --dataset sav --split sav_test --method baseline_csrt_grabcut --videos $(grep -v '^#' configs/splits/sav_test_subset60.txt) --out results/baseline_sav_test.csv
```

MAC: table and figure `results/e1.csv`, `figures/e1_jf_fps.png`.

```
python -m experiments.e1.table
```

### 2. Rule detector: tuning and freeze (SA-V val only)

DATA, once per model: per-frame signals, boxes and failure labels of SA-V val, written to
`$SAM2FR_DATA_ROOT/tune_rules`.

```
python -m experiments.tune_rules.export_sav_val --models tiny
python -m experiments.tune_rules.export_sav_val --models large
```

MAC, with those exports in place: exhaustive threshold search on SA-V val. Writes
`configs/detector/rules.yaml` and `results/rules_sav_val.csv`. After this step
`configs/detector/` is not changed.

```
python -m experiments.tune_rules.tune --models tiny large
```

### 3. E2: SAM 2 + rule detector + aligned-CSRT recovery (main result)

GPU: Tiny and Large, DAVIS val and every SA-V test shard (the example for Tiny; repeat for
`sav_test_1of4` ... `sav_test_3of4` and with `rules_csrt_large.yaml`):

```
python -m experiments.e2.run_online --config $CFG/rules_csrt_tiny.yaml --prompts $PROMPTS/davis_val.json --videos $(cat $LISTS/davis_val_all.txt)
python -m experiments.e2.run_online --config $CFG/rules_csrt_tiny.yaml --prompts $PROMPTS/sav_test.json --videos $(cat $LISTS/sav_test_0of4.txt)
```

DATA: scoring (refuses an incomplete run) writes `results/fix_rules_csrt_{tiny,large}_*.csv`
(scores, per-frame side tables, timing) and `results/events_fix_rules_csrt_*.csv` (re-prompt
events):

```
python -m experiments.e2.score --config $CFG/rules_csrt_tiny.yaml --splits davis_val sav_test
python -m experiments.e2.score --config $CFG/rules_csrt_large.yaml --splits davis_val sav_test
```

GPU, correctness check on DAVIS val: frames before the first re-prompt must equal the cached
SAM 2-alone masks (`results/events_pilot.csv`, `results/events_pilot_check.csv`; `--videos` limits
it to some videos):

```
python -m experiments.e2.pilot_check --config $CFG/rules_csrt_tiny.yaml --prompts $PROMPTS/davis_val.json --out results/events_pilot.csv
```

MAC: `results/e2.csv`, `results/events_e2.csv`.

```
python -m experiments.e2.table
```

Rerun check: five seeded DAVIS val videos are run again under a separate method name. GPU, then
DATA, then MAC (`results/e2_rerun_check.csv`, `results/fix_rerun_*.csv`):

```
python -m experiments.e2.run_online --config $CFG/rules_csrt_tiny_rerun.yaml --prompts $PROMPTS/davis_val.json --videos bmx-trees cows dog mbike-trick paragliding-launch
python -m experiments.e2.score --config $CFG/rules_csrt_tiny_rerun.yaml --splits davis_val --videos bmx-trees cows dog mbike-trick paragliding-launch
python -m experiments.e2.rerun_check
```

### 4. E3: rule ablation (Tiny)

GPU: one run per removed rule on DAVIS val and every SA-V test shard, same command as E2 with
`$CFG/rules_no_low_score_csrt_tiny.yaml`, `rules_no_area_change_csrt_tiny.yaml`,
`rules_no_centre_jump_csrt_tiny.yaml` and `rules_no_disappearance_csrt_tiny.yaml`; then DATA:
`experiments.e2.score` with each of those configs (as in step 3). DATA: recall of the frozen
detector with each rule removed (`results/e3_detector_{davis_val,sav_test}.csv`). MAC: table and
figure (`results/e3.csv`, `figures/e3_ablation.png`).

```
python -m experiments.e3.detector_eval --model tiny --splits davis_val sav_test
python -m experiments.e3.table
python -m experiments.e3.figure
```

### 5. E4: detector threshold sweep (SA-V val)

MAC, with the step 2 exports: `results/e4.csv`, `figures/e4_pr_curve.png`.

```
python -m experiments.e4.sweep --models tiny large
python -m experiments.e4.figure
```

### 6. E5: learned detector (Tiny)

MAC, with the step 2 exports: fits the logistic regression on SA-V val; writes
`configs/learned/logreg_tiny.json` and its `.sha256`, `results/e5_detector.csv` and
`results/e5_weights.csv`. GPU: the real system with the learned detector on the test sets, then
DATA scoring, then MAC table (`results/e5.csv`).

```
python -m experiments.e5.train --models tiny
python -m experiments.e5.run_learned --config $CFG/learned_csrt_tiny.yaml --prompts $PROMPTS/davis_val.json --videos $(cat $LISTS/davis_val_all.txt)
python -m experiments.e5.score --config $CFG/learned_csrt_tiny.yaml --splits davis_val sav_test
python -m experiments.e5.table
```

(`run_learned` is repeated for the four SA-V test shards.)

### 7. E6: does the fix matter less for the stronger model?

MAC (`results/e6.csv`, from the E2 files and the failure counts):

```
python -m experiments.e6.table
```

### 8. E7: oracle study (all oracle runs are named `oracle`)

GPU, Tiny, three arms: oracle flags with CSRT boxes, rule detector with oracle boxes, oracle flags
with oracle boxes (`configs/oracle/oracle_flags_csrt_tiny.yaml`, `rules_oracle_box_tiny.yaml`,
`oracle_flags_oracle_box_tiny.yaml`), on DAVIS val and every SA-V test shard. Oracle arms read
ground truth by design and are never reported as the real system.

```
python -m experiments.e7.run_oracle --config configs/oracle/oracle_flags_csrt_tiny.yaml --prompts $PROMPTS/davis_val.json --videos $(cat $LISTS/davis_val_all.txt)
python -m experiments.e7.score --config configs/oracle/oracle_flags_csrt_tiny.yaml --splits davis_val sav_test
```

DATA: check that all arms used the same prompts and videos (`results/e7_arms_check.csv`). MAC:
`results/e7.csv`.

```
python -m experiments.e7.check_arms
python -m experiments.e7.table
```

### 9. E8: CSRT re-initialised at the last good frame versus continuous tracking

GPU: Tiny with `$CFG/rules_csrt_continuous_tiny.yaml`, run and scored like E2. DATA: which
re-prompts were wrong (needs ground truth; writes `results/*_reprompts.csv`). MAC: table and
figure (`results/e8.csv`, `figures/e8_jf.png`).

```
python -m experiments.e2.run_online --config $CFG/rules_csrt_continuous_tiny.yaml --prompts $PROMPTS/davis_val.json --videos $(cat $LISTS/davis_val_all.txt)
python -m experiments.e2.score --config $CFG/rules_csrt_continuous_tiny.yaml --splits davis_val sav_test
python -m experiments.e8.wrong_reprompts --config $CFG/rules_csrt_tiny.yaml --splits davis_val sav_test
python -m experiments.e8.wrong_reprompts --config $CFG/rules_csrt_continuous_tiny.yaml --splits davis_val sav_test
python -m experiments.e8.table
python -m experiments.e8.figure
```

### 10. E9: what the fix costs

GPU, on its own GPU (not shared): time per component on three seeded DAVIS val videos
(`results/e9.csv`).

```
python -m experiments.e9.time_components --prompts $PROMPTS/davis_val.json --models tiny large --repeats 3 --out results/e9.csv
```

### 11. Main table

MAC: `results/main_table.csv` and `results/main_table.tex`.

```
python -m src.eval.tables
```

### 12. E10: overlays, demo video, failure types

MAC: the clips are chosen by a written rule from the E2 per-video results (`results/e10_selection.csv`).
DATA: overlay figures `figures/e10_*.png` and the demo video (written outside the repository).

```
python -m experiments.e10_overlays.select
python -m experiments.e10_overlays.overlays --selection results/e10_selection.csv --figures-dir figures
python -m scripts.make_demo --selection results/e10_selection.csv --out /path/to/e10_demo.mp4
```

Failure types: the first command draws a seeded sample of 40 flagged failures
(`results/e10_types_sample.csv`) and writes clips and a template into a folder outside the repo
(DATA). `results/e10_types_labels.csv` holds the type given to each sample; the second command
validates it and counts the types (MAC, `results/e10_types.csv`).

```
python -m experiments.e10_types.sample --clips-dir /path/to/e10_types
python -m experiments.e10_types.count --labels results/e10_types_labels.csv
```

### 13. E11 (extension): verified recovery

A rule that accepts or rejects each candidate re-prompt, tuned on SA-V val only. GPU: accept-all
runs with candidate logging on SA-V val (Large on the 60-video subset drawn by
`experiments.verify.draw_subset`); DATA: label each candidate with its gain over SAM 2 alone;
MAC: choose the thresholds and write `configs/verify/verify.yaml`, `results/verify_sav_val.csv`.

```
python -m experiments.verify.draw_subset
python -m experiments.e2.run_online --config $CFG/rules_csrt_tiny.yaml --prompts $PROMPTS/sav_val.json --videos $(cat $LISTS/sav_val_all.txt) --log-candidates
python -m experiments.verify.label_candidates --config $CFG/rules_csrt_tiny.yaml
python -m experiments.verify.tune
```

Then, with the frozen setting, GPU on DAVIS val Tiny, DATA scoring and the check against SAM 2
alone, and MAC table (`results/e11.csv`, `results/events_e11.csv`):

```
python -m experiments.e2.run_online --config $CFG/verified_rules_csrt_tiny.yaml --prompts $PROMPTS/davis_val.json --videos $(cat $LISTS/davis_val_all.txt)
python -m experiments.e2.score --config $CFG/verified_rules_csrt_tiny.yaml --splits davis_val
python -m experiments.verify.pilot_check --config $CFG/verified_rules_csrt_tiny.yaml --prompts $PROMPTS/davis_val.json --out results/verify_pilot_check_verified_rules_csrt_tiny_davis_val.csv
python -m experiments.verify.table --models tiny --splits davis_val
```

### 14. E12 (extension): where recovery helps and where it hurts

MAC, analysis only (some of its bins use ground truth and say so in their output):
`results/e12_by_quality.csv`, `results/e12_false_alarms.csv`, `results/e12_upper_bound.csv`,
`results/e12_types_outcome.csv`, and the figures `figures/e12_by_quality.png`,
`figures/e12_false_alarms.png`, `figures/e7_decomposition.png`.

```
python -m experiments.e12.by_quality
python -m experiments.e12.false_alarms
python -m experiments.e12.upper_bound
python -m experiments.e12.types_outcome --labels results/e10_types_labels.csv
python -m experiments.e12.figures
```

### 15. E13 (extension): a stronger re-prompt box

Pilot on SA-V val only (CPU job on cached SAM 2-alone outputs, sharded over videos, then merged;
Tiny and Large), then the choice of box source and gate per model. DATA for the pilot, MAC for the
choice (`results/e13_pilot.csv`, `results/e13_gate_grid.csv`, `results/e13_choice.csv`,
`configs/recovery/box_gate_<model>.yaml` on a go). The ViT tracker model file goes under
`$SAM2FR_DATA_ROOT/models/vittrack/` (the config `configs/recovery/vit.yaml` pins its sha256).

```
python -m experiments.e13.pilot --model tiny --shard 0 --nshards 8   # shards 0..7, then:
python -m experiments.e13.pilot --model tiny --merge                 # same for --model large
python -m experiments.e13.choose
```

The run configs are written from the choice, never by hand (a model that is a no-go gets none):

```
python -m experiments.e13.make_configs
```

Test runs only after the choice is committed and tagged `box-frozen`. GPU: Tiny on DAVIS val and every
SA-V test shard, (a) with the chosen box and no gate and (b) with the frozen gate; Large (b) only.
`make_configs --list` names each run; when the chosen source is CSRT, variant (a) is the existing
E2 (`rules_csrt_tiny.yaml`) or E8 (`rules_csrt_continuous_tiny.yaml`) run and is not run again. The
example is for one gated Tiny run; repeat for the other shards and configs:

```
python -m experiments.e2.run_online --config $CFG/rules_gated_<arm>_tiny.yaml --prompts $PROMPTS/davis_val.json --videos $(cat $LISTS/davis_val_all.txt)
python -m experiments.e2.run_online --config $CFG/rules_gated_<arm>_tiny.yaml --prompts $PROMPTS/sav_test.json --videos $(cat $LISTS/sav_test_0of4.txt)
```

DATA: scoring (as step 3), the wrong re-prompt file (as step 9; also run it for
`rules_csrt_large.yaml`) and the identity check: every video where no re-prompt happened must equal
SAM 2 alone on every frame, and every video must equal it before its first re-prompt
(`results/e13_gated_check_<method>_<split>.csv`, exit status 1 on a difference):

```
python -m experiments.e2.score --config $CFG/rules_gated_<arm>_tiny.yaml --splits davis_val sav_test
python -m experiments.e8.wrong_reprompts --config $CFG/rules_gated_<arm>_tiny.yaml --splits davis_val sav_test
python -m experiments.e13.check_gated --config $CFG/rules_gated_<arm>_tiny.yaml --splits davis_val sav_test
```

MAC: `results/e13.csv` (and `results/e13.tex`) with SAM 2 alone, E2, variant (a), variant (b) and the
E7 rules + oracle box row (marked oracle), with paired 95% bootstrap intervals against SAM 2 alone and
against E2, improved / worse / unchanged videos, re-prompts, wrong re-prompts, gated events and FPS;
and `figures/e13_ladder.png`:

```
python -m experiments.e13.table
python -m experiments.e13.figure
```

## Results

Main table (`results/main_table.csv`, same rounding as `python -m src.eval.tables`): J&F in
percent with a 95% bootstrap interval over videos (10,000 resamples, seed 5415); delta is the
paired difference to SAM 2 alone; better / worse / same counts videos. FPS is frames per second
of the whole method on the hardware shown. DAVIS 2017 val: 30 videos, 61 objects. SA-V test
subset: 60 videos, 105 objects.

| Method | DAVIS J&F (%) | DAVIS delta | DAVIS better / worse / same | DAVIS FPS | SA-V J&F (%) | SA-V delta | SA-V better / worse / same | SA-V FPS | Hardware |
|---|---|---|---|---|---|---|---|---|---|
| CSRT + GrabCut | 34.8 [30.1, 39.7] | -- | -- | 0.67 | 25.5 [21.1, 29.7] | -- | -- | 0.95 | CPU: arm |
| SAM 2 Tiny | 83.9 [77.1, 89.7] | -- | -- | 8.51 | 70.6 [65.1, 75.8] | -- | -- | 11.71 | GPU: NVIDIA H100 PCIe |
| SAM 2 Tiny + fix | 82.9 [75.5, 89.4] | -1.1 [-3.1, +0.5] | 5 / 4 / 21 | 4.62 | 66.7 [61.1, 71.9] | -3.9 [-8.6, +0.6] | 17 / 35 / 8 | 1.15 | GPU: NVIDIA H100 PCIe |
| SAM 2 Large | 87.0 [80.3, 92.6] | -- | -- | 5.87 | 76.7 [71.5, 81.2] | -- | -- | 6.22 | GPU: NVIDIA H100 PCIe |
| SAM 2 Large + fix | 86.5 [80.1, 91.9] | -0.5 [-1.6, +0.5] | 2 / 5 / 23 | 2.97 | 69.5 [63.6, 75.1] | -7.2 [-11.5, -3.2] | 15 / 37 / 8 | 0.83 | GPU: NVIDIA H100 PCIe |

In this study the fix lowers J&F in all four model x test-set cells. The interval of the change
includes zero on DAVIS val for both models and on the SA-V test subset for Tiny; it lies entirely
below zero only for Large on the SA-V test subset. More videos get worse than get better in
every cell except Tiny on DAVIS val, where the counts are 5 better and 4 worse. The fix also
runs at roughly half the speed of SAM 2 alone or less, because each re-prompt re-runs the
frames before it (E9 times the components). The oracle study (E7) and the extension runs
(E11, E12) look at this further; their tables are in `results/`.

The first-frame prompt is a box built from the ground-truth mask of the object's first annotated
frame. The DAVIS and SA-V leaderboards prompt SAM 2 with that mask itself. Our numbers therefore
are not comparable with leaderboard numbers. They are comparable with each other, because all
three systems get the same boxes.

## Credits

- Shreshta Reddy Saireddy wrote the SAM 2.1 install and setup, the SAM 2 runner
  (`src/sam2_runner/`), the official-scorer wrappers and completeness check (`src/eval/score.py`,
  `davis_scores.py`, `sav_scores.py`, `evaluators.py`, `get_evaluators.py`, `check_complete.py`,
  `completeness.py`, `rows.py`), the failure labels (`src/detection/labels.py`), the SAM 2 run
  helpers (`experiments/sam2_runs/`: prompts, scoring, failure counts), the checkpoint download
  script and the notebooks `notebooks/run_sam2.ipynb` and `notebooks/sam2_smoke.ipynb`.
- Vedith Reddy Kommula wrote the rest: data loaders, CSRT + GrabCut baseline, oracles, aligned
  CSRT and box check, rule and learned detectors, recovery module and online runner, bootstrap
  and tables, experiments E1 to E12, figures and demo video.
- Third party, used unmodified: SAM 2.1 (Meta, github.com/facebookresearch/sam2, pinned commit in
  `requirements-colab.txt`), the DAVIS 2017 evaluator and the SA-V evaluator (pinned commits,
  fetched by `src.eval.get_evaluators`), the DAVIS 2017 and SA-V datasets, OpenCV (CSRT,
  GrabCut).
