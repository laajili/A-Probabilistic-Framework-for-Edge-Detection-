# Data

This folder is used for sample images and local benchmark datasets.

Large benchmark datasets are not included in this repository due to their size and licensing/distribution conditions. Please download them from their official sources:

- BSDS500: https://www.kaggle.com/datasets/balraj98/berkeley-segmentation-dataset-500-bsds500
- Multicue: https://serre.lab.brown.edu/#/resources/the-multi-cue-boundary-detection-dataset



## BSDS500 structure

For BSDS500 evaluation, organize the dataset as:

```text
data/BSDS500_processed/
├── images/
│   └── test/
└── ground_truth/
    └── test/
```

Ground-truth files may be named either:

```text
image_name_1.png
image_name_2.png
...
```

or:

```text
image_name.png
```

## Multicue structure

For Multicue evaluation, organize the dataset as:

```text
data/Multicue/
├── multicue_image/
└── multicue_edges/
```

When running Multicue experiments, use:

```bash
--label_mode multicue
```

## Note

This repository provides the source code, evaluation scripts, usage instructions, and expected dataset folder structures. It does not redistribute BSDS500 or Multicue.
