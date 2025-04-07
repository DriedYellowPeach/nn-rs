use mnist::*;
use ndarray::prelude::*;

use super::Float;
use crate::error::*;
use std::path::Path;

pub struct DataSet {
    // tuples of (input, label)
    pub train: Vec<(Array1<Float>, Array1<Float>)>,
    pub test: Vec<(Array1<Float>, Array1<Float>)>,
}

impl DataSet {
    pub fn new(
        trn_data: Array2<Float>,
        trn_lbl: Array2<Float>,
        test_data: Array2<Float>,
        test_lbl: Array2<Float>,
    ) -> Self {
        let train = trn_data
            .outer_iter()
            .zip(trn_lbl.outer_iter())
            .map(|(x, y)| (x.to_owned(), y.to_owned()))
            .collect();
        let test = test_data
            .outer_iter()
            .zip(test_lbl.outer_iter())
            .map(|(x, y)| (x.to_owned(), y.to_owned()))
            .collect();

        Self { train, test }
    }

    pub fn from_mnist<P: AsRef<Path>>(
        base_path: P,
        train_size: u32,
        test_size: u32,
    ) -> Result<Self, NNError> {
        let base_path = base_path
            .as_ref()
            .to_str()
            .ok_or(NNError::LoadTrainDataError(
                "base path contains invalid unicode".to_owned(),
            ))?;
        let Mnist {
            trn_img,
            trn_lbl,
            tst_img,
            tst_lbl,
            ..
        } = MnistBuilder::new()
            .base_path(base_path) // Comment out this and `use_fashion_data()` to run on the original MNIST
            .label_format_digit()
            .training_set_length(train_size)
            .test_set_length(test_size)
            .finalize();

        let trn_data = Array2::from_shape_vec((train_size as usize, 28 * 28), trn_img)
            .map_err(|e| {
                NNError::LoadTrainDataError(format!(
                    "convert training images to Array2 struct error: {e:?}"
                ))
            })?
            .mapv(|x| x as Float / 256.);

        let tst_img = Array2::from_shape_vec((test_size as usize, 28 * 28), tst_img)
            .map_err(|e| {
                NNError::LoadTrainDataError(format!(
                    "convert tst images to Array2 struct error: {e:?}"
                ))
            })?
            .mapv(|x| x as Float / 256.);

        let mut trn_lbl_populated = Array2::zeros((train_size as usize, 10));
        let mut tst_lbl_populated = Array2::zeros((test_size as usize, 10));

        for (i, lbl) in trn_lbl.iter().enumerate() {
            trn_lbl_populated[[i, *lbl as usize]] = 1.0;
        }

        for (i, lbl) in tst_lbl.iter().enumerate() {
            tst_lbl_populated[[i, *lbl as usize]] = 1.0;
        }

        Ok(DataSet::new(
            trn_data,
            trn_lbl_populated,
            tst_img,
            tst_lbl_populated,
        ))
    }
}
