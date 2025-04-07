use ndarray::{Data, prelude::*};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

use crate::Float;
use crate::error::*;

#[derive(Serialize, Deserialize, Debug)]
pub struct LayerBuf {
    // buf
    pub nabla_weights: Array2<Float>,
    pub nabla_biases: Array1<Float>,
    pub z: Array1<Float>,
    pub activation: Array1<Float>,
}

impl LayerBuf {
    pub fn new(row: usize, col: usize) -> Self {
        Self {
            nabla_weights: Array2::zeros((row, col)),
            nabla_biases: Array1::zeros(row),
            z: Array1::zeros(row),
            activation: Array1::zeros(row),
        }
    }
}
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Layer {
    pub weights: Array2<Float>,
    pub biases: Array1<Float>,
    pub input_size: usize,
    pub layer_size: usize,
}

impl Layer {
    pub fn new(layer_sz: usize, input_sz: usize) -> Self {
        let row = layer_sz;
        let col = input_sz;
        let mut rng = rand::rng();

        Self {
            weights: Array2::from_shape_fn((row, col), |_idx| StandardNormal.sample(&mut rng)),
            biases: Array1::from_shape_fn(row, |_idx| StandardNormal.sample(&mut rng)),
            input_size: input_sz,
            layer_size: layer_sz,
        }
    }

    pub fn load_from_slices(weights: &[&[Float]], biases: &[Float]) -> Result<Self, NNError> {
        if weights.len() != biases.len() {
            return Err(BadLayers::UnmatchedLayer {
                w_rows: weights.len(),
                b_rows: biases.len(),
            })?;
        }

        if weights.is_empty() || weights[0].is_empty() {
            return Err(BadLayers::EmptyLayer)?;
        }

        let layer_size = weights.len();
        let input_size = weights[0].len();
        let weights = Array2::from_shape_fn((layer_size, input_size), |(i, j)| weights[i][j]);
        let biases = Array1::from_vec(biases.to_vec());

        Ok(Self {
            weights,
            biases,
            layer_size,
            input_size,
        })
    }

    pub fn forward<S>(&self, input: &ArrayBase<S, Ix1>, act: fn(Float) -> Float) -> Array1<Float>
    where
        S: Data<Elem = Float>,
    {
        let ret = self.weights.dot(input) + &self.biases;
        ret.mapv_into(act)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_ndarry_hadamard_product() {
        let arr1 = Array1::from(vec![1.0, 2.0, 3.0]);
        let arr2 = Array1::from(vec![4.0, 5.0, 6.0]);

        let expected = ArrayView1::from(&[4., 10., 18.]);

        assert_eq!(arr1 * arr2, expected);
    }

    #[test]
    fn test_layer_forward() {
        let layer = Layer::new(10, 5);
        let inputs = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let outputs = layer.forward(&inputs, |x| 2. * x);

        // Assert the layer shape
        assert_eq!(layer.weights.shape(), &[10, 5]);

        // Calculate the output manually
        for i in 0..10 {
            let r = layer.weights.row(i);
            let mut ret = 0.;
            for (w, x) in r.iter().zip(inputs.iter()) {
                ret += w * x;
            }

            ret += layer.biases[i];
            assert!(
                (outputs[i] - 2. * ret).abs() < 1e-5,
                "Output mismatch at index {i}: expected {}, got {}",
                2. * ret,
                outputs[i]
            );
        }
    }
}
