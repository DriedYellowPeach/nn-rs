use na::{DMatrix, DVector};
use nalgebra as na;
use rand_distr::{Distribution, StandardNormal};
use thiserror::Error;

type Float = f32;

#[derive(Debug, Error)]
pub enum NNError {
    #[error("Invalid input size for {idx} th layer: input size should be {expected}, got {actual}")]
    InvalidLayerInputSize {
        idx: usize,
        actual: usize,
        expected: usize,
    },

    #[error("Unmatched layer: weights row {w_rows} != biases row {b_rows}")]
    UnmatchedLayer { w_rows: usize, b_rows: usize },
}

struct LayerBuf {
    // buf
    nabla_weights: DMatrix<Float>,
    nabla_biases: DVector<Float>,
    z: DVector<Float>,
    activation: DVector<Float>,
}

impl LayerBuf {
    fn new(row: usize, col: usize) -> Self {
        Self {
            nabla_weights: DMatrix::zeros(row, col),
            nabla_biases: DVector::zeros(row),
            z: DVector::zeros(row),
            activation: DVector::zeros(row),
        }
    }
}

pub struct Layer {
    weights: DMatrix<Float>,
    biases: DVector<Float>,
    input_size: usize,
    layer_size: usize,
}

impl Layer {
    fn new(layer_sz: usize, input_sz: usize) -> Self {
        let row = layer_sz;
        let col = input_sz;

        let mut rng = rand::rng();

        Self {
            weights: DMatrix::from_fn(row, col, |_i, _j| StandardNormal.sample(&mut rng)),
            biases: DVector::from_fn(row, |_i, _j| StandardNormal.sample(&mut rng)),
            input_size: input_sz,
            layer_size: layer_sz,
        }
    }

    fn load_from_slices(weights: &[&[Float]], biases: &[Float]) -> Result<Self, NNError> {
        if weights.len() != biases.len() {
            return Err(NNError::UnmatchedLayer {
                w_rows: weights.len(),
                b_rows: biases.len(),
            });
        }

        let weights = DMatrix::from_fn(weights.len(), weights[0].len(), |i, j| weights[i][j]);
        let biases = DVector::from_column_slice(biases);
        let layer_size = weights.nrows();
        let input_size = weights.ncols();

        Ok(Self {
            weights,
            biases,
            layer_size,
            input_size,
        })
    }

    pub fn forward(&self, input: &DVector<Float>, act: fn(Float) -> Float) -> DVector<Float> {
        let ret = &self.weights * input + &self.biases;
        ret.apply_into(|x| *x = act(*x))
    }
}

fn sigmoid(z: Float) -> Float {
    1. / (1. + (-z).exp())
}

fn sigmoid_derivative(z: Float) -> Float {
    sigmoid(z) * (1. - sigmoid(z))
}

fn cost(output_activation: Float, y: Float) -> Float {
    0.5 * (output_activation - y).powi(2)
}

fn cost_derivative(output_activation: Float, y: Float) -> Float {
    output_activation - y
}

pub struct NN {
    layers: Vec<Layer>,
    layers_buf: Vec<LayerBuf>,
    act: fn(Float) -> Float,
    learn_rate: Float,
    epochs: usize,
    mini_batch_size: usize,
    // target accuracy rate
}

pub struct NNBuilder {
    layers: Vec<Layer>,
    layers_buf: Vec<LayerBuf>,
    act: fn(Float) -> Float,
    learn_rate: Float,
    epochs: usize,
    mini_batch_size: usize,
}

impl Default for NNBuilder {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            layers_buf: Vec::new(),
            act: sigmoid,
            learn_rate: 0.01,
            epochs: 1000,
            mini_batch_size: 10,
        }
    }
}

impl NNBuilder {
    pub fn new_from_layers<I: IntoIterator<Item = Layer>>(layers: I) -> Self {
        let layers = layers.into_iter().collect::<Vec<_>>();
        let mut layers_buf = Vec::new();

        for l in &layers {
            let row = l.weights.nrows();
            let col = l.weights.ncols();

            layers_buf.push(LayerBuf::new(row, col));
        }

        Self {
            layers,
            layers_buf,
            ..Default::default()
        }
    }

    pub fn new_from_arch(arch: &[usize]) -> Self {
        let mut layers = Vec::new();
        let mut layers_buf = Vec::new();

        for i in 1..arch.len() {
            layers.push(Layer::new(arch[i], arch[i - 1]));
            layers_buf.push(LayerBuf::new(arch[i], arch[i - 1]));
        }

        Self {
            layers,
            layers_buf,
            ..Default::default()
        }
    }

    pub fn with_learn_rate(self, rate: Float) -> Self {
        let mut ret = self;
        ret.learn_rate = rate;
        ret
    }

    pub fn with_epochs(self, epochs: usize) -> Self {
        let mut ret = self;
        ret.epochs = epochs;
        ret
    }

    pub fn with_activator(self, act: fn(Float) -> Float) -> Self {
        let mut ret = self;
        ret.act = act;
        ret
    }

    pub fn with_mini_batch_size(self, size: usize) -> Self {
        let mut ret = self;
        ret.mini_batch_size = size;
        ret
    }

    pub fn build(self) -> Result<NN, NNError> {
        let mut layers_buf = Vec::new();

        // Check if layers are valid
        for i in 1..self.layers.len() {
            let prev_layer = &self.layers[i - 1];
            let cur_layer = &self.layers[i];

            if cur_layer.input_size != prev_layer.layer_size {
                return Err(NNError::InvalidLayerInputSize {
                    idx: i,
                    actual: cur_layer.input_size,
                    expected: prev_layer.layer_size,
                });
            }
        }

        Ok(NN {
            layers: self.layers,
            layers_buf,
            act: self.act,
            learn_rate: self.learn_rate,
            epochs: self.epochs,
            mini_batch_size: self.mini_batch_size,
        })
    }
}

impl NN {
    pub fn input_size(&self) -> usize {
        self.layers[0].input_size
    }

    pub fn output_size(&self) -> usize {
        self.layers.last().unwrap().layer_size
    }

    // TODO: currently, use plain slice for training data and test data
    // need better encapsulation later
    // idea1: don't load all the data into memory, however, using a stream like iterator
    // Trains on training_data using stochastic gradient descent
    pub fn train(&mut self, training_data: &[(&[Float], &[Float])]) {
        todo!()
    }

    pub fn update_mini_batch(&mut self, mini_batch: usize, learning_rate: Float) {
        todo!()
    }

    pub fn feedforwad(&self, x: &[Float]) -> DVector<Float> {
        let mut activation = DVector::from_column_slice(x);

        for l in &self.layers {
            activation = l.forward(&activation, self.act);
        }

        activation
    }

    // TEST: how to test backprop?
    pub fn backprop(&mut self, x: &[Float], y: &[Float]) {
        // feedforwad
        let input = DVector::from_column_slice(x);

        for i in 0..self.layers.len() {
            let activation = if i == 0 {
                &input
            } else {
                &self.layers_buf[i - 1].activation
            };
            let z = &self.layers[i].weights * activation + &self.layers[i].biases;
            let activation = z.map(|x| (self.act)(x));
            self.layers_buf[i].z = z;
            self.layers_buf[i].activation = activation;
        }

        // backward
        // element-wise vector multiplication
        let a_l = &self.layers_buf.last().unwrap().activation;
        let z = &self.layers_buf.last().unwrap().z;
        let y = DVector::from_column_slice(y);
        let partial_der_a_l_over_z = z.map(sigmoid_derivative);
        let delta = a_l
            .zip_map(&y, cost_derivative)
            .component_mul(&partial_der_a_l_over_z);

        self.layers_buf.last_mut().unwrap().nabla_biases = delta.clone();

        let a_l_minus = &self.layers_buf[self.layers_buf.len() - 2].activation;
        self.layers_buf.last_mut().unwrap().nabla_weights = &delta * a_l_minus.transpose();

        let mut prev_delta = delta;

        for i in (0..self.layers_buf.len() - 1).rev() {
            let z = &self.layers_buf[i].z;
            let sp = z.map(sigmoid_derivative);
            let delta = (&self.layers[i + 1].weights.transpose() * &prev_delta).component_mul(&sp);
            self.layers_buf[i].nabla_biases = delta.clone();
            self.layers_buf[i].nabla_weights =
                &delta * &self.layers_buf[i - 1].activation.transpose();

            prev_delta = delta;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_forward() {
        let layer = Layer::new(10, 5);
        let inputs = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let outputs = layer.forward(&inputs, |x| 2. * x);

        // Assert the layer shape
        assert_eq!(layer.weights.shape(), (10, 5));

        // Calculate the output manually
        for i in 0..10 {
            let r = layer.weights.row(i);
            let mut ret = 0.;
            for (w, x) in r.iter().zip(inputs.iter()) {
                ret += w * x;
            }

            ret += layer.biases[i];
            assert_eq!(outputs[i], 2. * ret);
        }
    }

    #[test]
    fn test_nn_feedforward() {
        let l1 = Layer::load_from_slices(&[&[0.5, 0.25]], &[-0.5]).unwrap();
        let l2 = Layer::load_from_slices(&[&[0.5]], &[0.]).unwrap();
        let nn = NNBuilder::new_from_layers(vec![l1, l2])
            .with_activator(|x| x)
            .build()
            .unwrap();

        let y = nn.feedforwad(&[1.0, 2.0]);

        assert_eq!(y.nrows(), 1);
        assert_eq!(y.ncols(), 1);
        assert_eq!(y[0], 0.25);
    }
}
