# tfsage-core

`tfsage-core` is a toolkit for working with genomic region sets, embeddings, and transcription factor (TF) binding site prediction. It provides utilities for downloading genomic data, preparing assets, extracting features, and generating embeddings using various integration methods (such as Seurat). The package is designed for bioinformatics workflows involving high-dimensional genomics data.

---

## Features

- **Download Genomic Data**: Easily fetch data from ChIP-Atlas and ENCODE.
- **Prepare Genome/Region Assets**: Tools to load and process genome annotations and region sets.
- **Extract Features**: Compute feature matrices from genomic BED files.
- **Generate Embeddings**: Integrate and reduce dimensionality using multiple methods (Seurat, Harmony, FastMNN, etc.).
- **Predict TF Binding Sites**: Utilities to synthesize and evaluate TF binding predictions.

---

## Installation

> **Note:** This package requires Python 3.8+ and R with certain libraries for embedding generation.

1. **Clone the repository**
   ```bash
   git clone https://github.com/brandonlukas/tfsage-core.git
   cd tfsage-core
   ```

2. **Install Python dependencies**
   ```bash
   pip install -e .
   ```
   If a `requirements.txt` is present, use:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install required R packages** (for embedding generation in Seurat)
   ```R
   install.packages(c('optparse', 'arrow', 'dplyr', 'tibble'))
   if (!requireNamespace("BiocManager", quietly = TRUE))
       install.packages("BiocManager")
   BiocManager::install("Seurat")
   BiocManager::install("SeuratWrappers")
   ```

---

## Usage

### Downloading Data

You can download ChIP-Atlas or ENCODE BED files with:

```python
from tfsage.download import download_chip_atlas, download_encode

download_chip_atlas("SRX502813", "downloads/chip.bed")
download_encode("ENCFF729IPL", "downloads/encode.bed")
```

### Preparing Genome and Region Sets

```python
from tfsage.features import prepare_genome, prepare_region_set, load_region_set

genome = prepare_genome("path/to/genome.fa")
region_set = prepare_region_set("path/to/regions.bed", genome)
```

### Extracting Features

```python
from tfsage.features import extract_features

features = extract_features("path/to/bedfile.bed", genome)
```

### Generating Embeddings

Embeddings are generated via Seurat integration methods. You can use the Python API or directly run the R script.

#### **Python API Example**

```python
from tfsage.embedding.generate_embeddings import generate_embeddings

# rp_matrix_df: pandas DataFrame (genes x cells)
# metadata_df: pandas DataFrame (cells x metadata)
embedding_df = generate_embeddings(rp_matrix_df, metadata_df, align_key="Assay", method="FastMNNIntegration")
```

#### **R Script Example**

You can run the R script directly for more control:

```bash
Rscript tfsage/assets/seurat_integration.R \
    --rp-matrix /path/to/rp_matrix.h5 \
    --metadata /path/to/metadata.parquet \
    --output-dir /path/to/output \
    --align-key Assay \
    --method CCAIntegration,HarmonyIntegration
```

- **--rp-matrix**: Path to RP matrix file (input, h5 format)
- **--metadata**: Path to metadata file (input, Parquet format)
- **--output-dir**: Output directory for embeddings
- **--align-key**: (default: Assay) column to split the metadata
- **--method**: (default: FastMNNIntegration) one or more of: CCAIntegration, HarmonyIntegration, JointPCAIntegration, RPCAIntegration, FastMNNIntegration, none

---

### Example: Synthesizing Experiments and Predicting TF Binding

```python
from tfsage.download import download_encode
from tfsage.generate import synthesize_experiments

# Download data
download_encode("ENCFF654RND", "downloads/ENCFF654RND.bed")
download_encode("ENCFF794LOO", "downloads/ENCFF794LOO.bed")

# Synthesize predictions (example weights)
bed_files = ["downloads/ENCFF654RND.bed", "downloads/ENCFF794LOO.bed"]
weights = [0.5, 0.5]
predictions = synthesize_experiments(bed_files, weights)
```

---

## Requirements

- Python 3.8 or higher
- R (with Seurat, SeuratWrappers, optparse, arrow, dplyr, tibble)
- pandas, numpy, pyarrow, h5py, tqdm, and other standard scientific Python libraries

---

## License

[MIT](LICENSE)

---

## Acknowledgements

This toolkit builds on [Seurat](https://satijalab.org/seurat/), [ChIP-Atlas](https://chip-atlas.org/), and [ENCODE](https://www.encodeproject.org/).

---

## Contact

For questions or contributions, please open an issue or pull request on [GitHub](https://github.com/brandonlukas/tfsage-core).