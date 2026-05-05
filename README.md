
# Probabilistic Edge Detection Based on Local Intensity Differences

Official implementation of the method described in:

**A Probabilistic Framework for Edge Detection Based on Local Intensity Differences**

The method is a simple, interpretable, gradient-free probabilistic edge detector. It assigns an edge probability to each pixel using local intensity differences with its direct neighbors.

## Method overview

For a normalized grayscale image `I`, the edge probability at pixel `(i, j)` is defined as:

```text
P_C(i,j) = 1 - exp(-lambda * sum_{(k,l) in N(i,j)} |I(i,j) - I(k,l)|)
```

where:

- `N(i,j)` is the set of four direct neighbors: up, down, left, and right;
- `lambda > 0` controls the sensitivity to local intensity variations;
- high local contrast produces high edge probability;
- homogeneous regions produce low edge probability.

## Main features

- Gradient-free edge detection
- Probabilistic and interpretable formulation
- Lightweight computation
- No training stage
- Robustness analysis under Gaussian noise
- BSDS500-style evaluation utilities
- Lambda and threshold grid search
- Visualization tools for threshold analysis

## Repository structure

```text
Probabilistic-Edge-Detection/
|-- README.md
|-- requirements.txt
|-- LICENSE
|-- CITATION.cff
|-- src/
|   |-- edge_detection.py
|   |-- evaluation.py
|   |-- io_utils.py
|   `-- visualization.py
|-- experiments/
|   |-- run_detection.py
|   |-- bsds500_evaluation.py
|   |-- lambda_threshold_search.py
|   `-- threshold_visualization.py
|-- notebooks/
|-- data/
|   |-- BSDS500_processed/
|   `-- sample_images/
`-- results/
    |-- figures/
    `-- metrics/
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick usage in Python

```python
from src.io_utils import load_gray01
from src.edge_detection import detect_edges

image = load_gray01("data/sample_images/example.png")
score = detect_edges(image, lambda_param=5.0, gaussian_sigma=0.6)
```

## Run detection on a folder

```bash
python experiments/run_detection.py \
  --img_dir data/sample_images \
  --out_dir results/figures \
  --lambda_param 5.0 \
  --gaussian_sigma 0.6 \
  --threshold 0.575
```

This saves:

- edge probability maps;
- optional binary edge maps if `--threshold` is provided.

## BSDS500-style evaluation

Expected dataset structure:

```text
BSDS500_processed/
|-- images/test/
`-- ground_truth/test/
```

Run:

```bash
python experiments/bsds500_evaluation.py \
  --img_dir data/BSDS500_processed/images/test \
  --gt_dir data/BSDS500_processed/ground_truth/test \
  --out_dir results/metrics/bsds500 \
  --lambda_param 5.0 \
  --gaussian_sigma 0.6 \
  --n_thresholds 530
```

Outputs:

```text
results/metrics/bsds500/ods_pr_curve.csv
results/metrics/bsds500/summary_metrics.csv
```

## Lambda and threshold search

To test several sensitivity and binarization values, run:

```bash
python experiments/lambda_threshold_search.py \
  --img_dir data/BSDS500_processed/images/test \
  --gt_dir data/BSDS500_processed/ground_truth/test \
  --out_dir results/metrics/bsds500_grid_search \
  --lambda_values 0.05,0.1,0.5,1,5,10 \
  --n_thresholds 200 \
  --gaussian_sigma 1.6
```

Outputs:

```text
results/metrics/bsds500_grid_search/grid_results.csv
results/metrics/bsds500_grid_search/lambda_summary.csv
results/metrics/bsds500_grid_search/best_params.csv
```

The best pair is selected by the highest ODS/F1 score.

Use `--threshold_values 0.5,0.575,0.7` instead of `--n_thresholds` only when testing specific thresholds.

For the nested Multicue annotation layout, use `--label_mode multicue`:

```bash
python experiments/lambda_threshold_search.py \
  --img_dir data/Multicue/multicue_image \
  --gt_dir data/Multicue/multicue_edges \
  --label_mode multicue \
  --out_dir results/metrics/multicue_grid_search \
  --lambda_values 2 \
  --n_thresholds 200 \
  --gaussian_sigma 2.9
```

## Runtime and memory benchmark

To measure inference time and memory usage, run:

```bash
python experiments/runtime_memory_benchmark.py \
  --img_dir data/BSDS500_processed/images/test \
  --out_dir results/metrics/runtime_memory \
  --lambda_param 2 \
  --gaussian_sigma 0.6 \
  --repeats 5 \
  --warmup 1
```

Outputs:

```text
results/metrics/runtime_memory/per_image_runtime_memory.csv
results/metrics/runtime_memory/summary_runtime_memory.csv
```

The benchmark reports inference time after image loading, FPS, throughput in megapixels per second, Python allocation peak measured with `tracemalloc`, and process RSS delta when available.

## Visual threshold analysis

```bash
python experiments/threshold_visualization.py \
  --img_dir data/sample_images \
  --lambda_param 5.0 \
  --gaussian_sigma 0.6 \
  --max_images 10
```

## Google Colab

The original experimental notebook can be placed in:

```text
notebooks/demo_colab.ipynb
```

A public Colab link can also be added here after uploading the notebook.

## Citation

If you use this code, please cite:

```bibtex
@article{nefzi2026probabilistic,
  title={A Probabilistic Framework for Edge Detection Based on Local Intensity Differences},
  author={Nefzi, Wiem and Lajili, Mohamed},
  journal={The Visual Computer},
  year={2026}
}
```

## License

This project is released under the MIT License.
=======
# A-Probabilistic-Framework-for-Edge-Detection
Full source code and datasets for the A Probabilistic Framework for Edge Detection Based on Local Intensity Differences (The Visual Computer submission).
>>>>>>> 8a501d9a6f00bf62a5c83e66a71c1585230bd590
