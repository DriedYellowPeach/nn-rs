use thiserror::Error;

#[derive(Debug, Error)]
pub enum NNError {
    #[error("Bad layers: {0}")]
    BadLayers(#[from] BadLayers),

    #[error("Failed to save model: {0}")]
    SaveModelError(#[from] std::io::Error),

    #[error("Failed to de/serialize model: {0}")]
    SerializationError(#[from] postcard::Error),

    #[error("Failed to load training data: {0}")]
    LoadTrainDataError(String),
}

#[derive(Debug, Error)]
pub enum BadLayers {
    #[error("Invalid input size for {idx} th layer: input size should be {expected}, got {actual}")]
    InvalidLayerInputSize {
        idx: usize,
        actual: usize,
        expected: usize,
    },
    #[error("Unmatched layer: weights row {w_rows} != biases row {b_rows}")]
    UnmatchedLayer { w_rows: usize, b_rows: usize },

    #[error("No neurons/input in the layer")]
    EmptyLayer,
}
