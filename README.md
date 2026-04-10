# Statistical Radiomics Saliency Maps

Standalone publication-ready export of the saliency-map experiment from the original `image_processing` workspace.

## Included Code

- `statistical_radiomics_saliency_maps.py` as the experiment entrypoint
- `data/` contains dataset loaders
- `nn/pytorch/radiomics/` for statistical radiomics feature layers and classifiers
- `radiomics_rs/` for the Rust-backed texture-matrix and saliency-map extension

## Setup

1. Create and activate a Python environment.
2. Install a Rust toolchain because `radiomics_rs` is built locally.
3. Install maturin
4. Build radiomics_rs and install into env: maturin develop --release
3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Example Run

```bash
python statistical_radiomics_saliency_maps.py \
  -b 0 \
  -e 1 \
  -d 0 \
  -dt /path/to/datasets \
  -dn medmnist_derma \
  -r ./results/run_ \
  -m mlp \
  -st True
```

## Parameters

The script exposes the following command-line arguments:

- `-b`, `--begin`: First experiment iteration index, inclusive. The script loops over `range(begin, end)`.
- `-e`, `--end`: Final experiment iteration bound, exclusive.
- `-d`, `--device`: CUDA device index used as `cuda:{device}`. Example: `0` for `cuda:0`.
- `-dt`, `--datapath`: Path to the dataset root directory. The expected files depend on `--datasetname`.
- `-dn`, `--datasetname`: Dataset selector. Supported values in the exported loader are `tuberculosis`, `raw_tuberculosis`, `liver_fatty`, `liver_fatty_masked`, `medmnist_pneumonia`, `medmnist_derma`, `brain_mri`, and `mnist`.
- `-f`, `--features`: Experiment name suffix used in the output path as `resultdir + features + /iteration`. In the current script this does not change the extracted feature set.
- `-a`, `--alpha`: Integer alpha parameter forwarded to the experiment configuration. Default: `0`.
- `-dl`, `--delta`: Integer delta parameter forwarded to the experiment configuration. Default: `1`.
- `-z`, `--omitzeros`: Boolean flag controlling zero handling in radiomics computations. Default: `True`.
- `-r`, `--resultdir`: Output directory prefix. Each run is saved under `resultdir/features/iteration`.
- `-st`, `--skiptraining`: If `True`, reuse an existing `weights.pth` from the run directory instead of training a new model. Default: `False`.
- `-m`, `--model`: Classifier type. Supported values are `mlp` and `transformer`.
- `--saliency-batch-size`: Batch size used during saliency-map generation. Default: `4`.
- `--saliency-n-steps`: Number of Integrated Gradients steps used for saliency generation. Default: `5`.
- `--profile-saliency`: Enable timing/profiling output for saliency and activation map generation. When enabled, profiling data is written to `saliency_profile.json`.

## Dataset-Specific Inputs

- `tuberculosis`: expects `images_only_roi.npy` and `labels.npy` under `--datapath`.
- `raw_tuberculosis`: expects `images.npy` and `labels.npy` under `--datapath`.
- `liver_fatty`: expects `liver.mat` under `--datapath`.
- `liver_fatty_masked`: expects `liver.mat` and `mask.png` under `--datapath`.
- `brain_mri`: expects `brain_mri_train.npy`, `brain_mri_test.npy`, `brain_mri_train_labels.npy`, and `brain_mri_test_labels.npy` under `--datapath`.
- `medmnist_pneumonia`, `medmnist_derma`, `mnist`: downloaded automatically by the dataset loader.

## Notes

- Dataset assets are not bundled in this export.
- The script expects the same dataset naming conventions as the original experiment code.
- The editable dependency in `requirements.txt` installs the bundled `radiomics_rs` package from this repository.
- See `DEPENDENCIES.md` for the dependency walk used to assemble this export.