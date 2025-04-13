import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


def write_10x_h5(
    df: pd.DataFrame,
    output_file: str,
    dtype: np.dtype = np.float32,
) -> None:
    """
    Save a DataFrame as a 10x Genomics-style HDF5 file (CellRanger-style .h5).

    Parameters:
        df (pd.DataFrame): Rows = genes/features, Columns = barcodes/cells
        output_file (str): Path to output .h5 file (e.g., filtered_feature_bc_matrix.h5)
        dtype (np.dtype): Data type for the sparse matrix. Default is np.float32.
    """
    # Convert to CSC sparse matrix (column-major for 10x format)
    sparse = sp.csc_matrix(df.values, dtype=dtype)

    with h5py.File(output_file, "w") as f:
        grp = f.create_group("default")

        # Matrix data
        grp.create_dataset("data", data=sparse.data, compression="gzip")
        grp.create_dataset("indices", data=sparse.indices, compression="gzip")
        grp.create_dataset("indptr", data=sparse.indptr, compression="gzip")
        grp.create_dataset("shape", data=np.array(sparse.shape, dtype=np.int64))

        # Barcodes (columns)
        barcodes = df.columns.astype(str).to_numpy()
        grp.create_dataset(
            "barcodes", data=np.array(barcodes, dtype="S"), compression="gzip"
        )

        # Gene names (rows)
        feature_ids = df.index.astype(str).to_numpy()
        grp.create_dataset(
            "gene_names", data=np.array(feature_ids, dtype="S"), compression="gzip"
        )


def read_10x_h5(
    input_file: str,
    dtype: np.dtype = np.float32,
) -> pd.DataFrame:
    """
    Load a 10x-style HDF5 file written by `write_10x_h5` and return a DataFrame.

    Parameters:
        input_file (str): Path to input .h5 file (e.g., filtered_feature_bc_matrix.h5)
        dtype (np.dtype): Data type for the sparse matrix. Default is np.float32.

    Returns:
        df (pd.DataFrame): Rows = genes/features, Columns = barcodes/cells
    """
    with h5py.File(input_file, "r") as f:
        grp = f["default"]
        data = grp["data"][()]
        indices = grp["indices"][()]
        indptr = grp["indptr"][()]
        shape = tuple(grp["shape"][()])

        # Metadata
        barcodes = [x.decode("utf-8") for x in grp["barcodes"][:]]
        gene_names = [x.decode("utf-8") for x in grp["gene_names"][:]]

    # Reconstruct sparse matrix
    matrix = sp.csc_matrix((data, indices, indptr), shape=shape, dtype=dtype)

    # Return as DataFrame
    df = pd.DataFrame.sparse.from_spmatrix(matrix, index=gene_names, columns=barcodes)
    return df
