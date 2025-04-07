use ndarray::{Zip, prelude::*};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::{fs::File, io::Write, path::Path};

use crate::Float;
use crate::activator::{ActPair, ActivatorType};
use crate::data_loader::DataSet;
use crate::error::{BadLayers, NNError};

pub mod layer;
pub use layer::Layer;

use layer::LayerBuf;

pub struct NN {
    layers: Vec<Layer>,
    layers_buf: Vec<LayerBuf>,
    activator: ActPair,
    learn_rate: Float,
    epochs: usize,
    mini_batch_size: usize,
    target_accuracy: Float,
}

#[derive(Serialize, Deserialize)]
pub struct NNBuilder {
    layers: Vec<Layer>,
    activator: ActivatorType,
    learn_rate: Float,
    epochs: usize,
    mini_batch_size: usize,
    target_accuracy: Float,
}

impl Default for NNBuilder {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            activator: ActivatorType::Sigmoid,
            learn_rate: 0.01,
            epochs: 1000,
            mini_batch_size: 10,
            target_accuracy: 0.95,
        }
    }
}

impl NNBuilder {
    pub fn new_from_layers<I: IntoIterator<Item = Layer>>(layers: I) -> Self {
        let layers = layers.into_iter().collect::<Vec<_>>();

        Self {
            layers,
            ..Default::default()
        }
    }

    pub fn new_from_arch(arch: &[usize]) -> Self {
        let mut layers = Vec::new();

        for i in 1..arch.len() {
            layers.push(Layer::new(arch[i], arch[i - 1]));
        }

        Self {
            layers,
            ..Default::default()
        }
    }

    pub fn new_from_model_file<P: AsRef<Path>>(model_path: P) -> Result<Self, NNError> {
        let bts =
            std::fs::read(model_path).map_err(|e| NNError::LoadTrainDataError(e.to_string()))?;
        let builder = postcard::from_bytes(&bts)?;

        Ok(builder)
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

    pub fn with_activator(self, act: ActivatorType) -> Self {
        let mut ret = self;
        ret.activator = act;
        ret
    }

    pub fn with_mini_batch_size(self, size: usize) -> Self {
        let mut ret = self;
        ret.mini_batch_size = size;
        ret
    }

    pub fn with_target_accuracy(self, target: Float) -> Self {
        let mut ret = self;
        ret.target_accuracy = target;
        ret
    }

    fn filled_layer_buf(&self) -> Vec<LayerBuf> {
        let mut layers_buf = Vec::new();

        for l in &self.layers {
            let row = l.weights.nrows();
            let col = l.weights.ncols();

            layers_buf.push(LayerBuf::new(row, col));
        }

        layers_buf
    }

    pub fn build(self) -> Result<NN, NNError> {
        // Check if layers are valid
        for i in 1..self.layers.len() {
            let prev_layer = &self.layers[i - 1];
            let cur_layer = &self.layers[i];

            if cur_layer.input_size != prev_layer.layer_size {
                return Err(BadLayers::InvalidLayerInputSize {
                    idx: i,
                    actual: cur_layer.input_size,
                    expected: prev_layer.layer_size,
                })?;
            }
        }

        let layers_buf = self.filled_layer_buf();

        Ok(NN {
            layers: self.layers,
            layers_buf,
            activator: self.activator.into(),
            learn_rate: self.learn_rate,
            epochs: self.epochs,
            mini_batch_size: self.mini_batch_size,
            target_accuracy: self.target_accuracy,
        })
    }
}

fn outer(x: &Array<Float, Ix1>, y: &Array<Float, Ix1>) -> Array<Float, Ix2> {
    let (size_x, size_y) = (x.shape()[0], y.shape()[0]);
    let x_view = x.view();
    let y_view = y.view();
    let x_reshaped = x_view.to_shape((size_x, 1)).unwrap();
    let y_reshaped = y_view.to_shape((1, size_y)).unwrap();
    x_reshaped.dot(&y_reshaped)
}

// NOTE: cost function
// fn cost(output_activation: Float, y: Float) -> Float {
//     0.5 * (output_activation - y).powi(2)
// }

fn cost_derivative(output_activation: Float, y: Float) -> Float {
    output_activation - y
}

impl NN {
    pub fn input_size(&self) -> usize {
        self.layers[0].input_size
    }

    pub fn output_size(&self) -> usize {
        self.layers.last().unwrap().layer_size
    }

    pub fn train(&mut self, data: DataSet) {
        let mut rng = rand::rng();
        let mut data = data;
        for i in 0..self.epochs {
            data.train.shuffle(&mut rng);
            log::info!("Epoch {i}: begin");

            for mini_batches in data.train.chunks(self.mini_batch_size) {
                self.update_mini_batch(mini_batches, self.learn_rate);
            }

            let correct = self.evaluate(&data);
            let total = data.test.len();
            let precision = correct as Float / total as Float;

            log::info!("Epoch {i}: correct: {correct}, total: {total},  precision: {precision:.4}",);

            if precision >= self.target_accuracy {
                log::info!("Target accuracy reached: {precision:.4}");
                break;
            }
        }

        log::info!("Training completed.");
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), NNError> {
        let mut file = File::create(path)?;
        let builder = NNBuilder {
            layers: self.layers.clone(),
            activator: self.activator.act_type,
            learn_rate: self.learn_rate,
            epochs: self.epochs,
            mini_batch_size: self.mini_batch_size,
            target_accuracy: self.target_accuracy,
        };
        let encoded = postcard::to_allocvec(&builder)?;

        file.write_all(&encoded)?;

        Ok(())
    }

    pub fn update_mini_batch(
        &mut self,
        mini_batch: &[(Array1<Float>, Array1<Float>)],
        learning_rate: Float,
    ) {
        let mut nabla_b = Vec::new();
        let mut nabla_w = Vec::new();

        for lb in &self.layers_buf {
            nabla_b.push(Array1::zeros(lb.nabla_biases.len()));
            nabla_w.push(Array2::zeros(lb.nabla_weights.dim()));
        }

        for (x, y) in mini_batch {
            self.backprop(x.view(), y.view());

            for (nb, dnb) in nabla_b.iter_mut().zip(self.layers_buf.iter()) {
                *nb += &dnb.nabla_biases;
            }

            for (nw, dnw) in nabla_w.iter_mut().zip(self.layers_buf.iter()) {
                *nw += &dnw.nabla_weights;
            }
        }

        for (l, (nw, nb)) in self
            .layers
            .iter_mut()
            .zip(nabla_w.iter().zip(nabla_b.iter()))
        {
            let step_w = nw.mapv(|x| x / mini_batch.len() as Float) * learning_rate;
            let step_b = nb.mapv(|x| x / mini_batch.len() as Float) * learning_rate;

            l.weights -= &step_w;
            l.biases -= &step_b;
        }
    }

    pub fn evaluate(&self, data: &DataSet) -> usize {
        data.test
            .iter()
            .filter(|(td, tlbl)| {
                let output = self.feedforward(td.view());
                let mut max_idx = 0;

                for i in 1..output.len() {
                    if output[i] > output[max_idx] {
                        max_idx = i;
                    }
                }

                tlbl[max_idx] == 1.0
            })
            .count()
    }

    pub fn feedforward<'a, A: Into<ArrayView1<'a, Float>>>(&self, input: A) -> Array1<Float> {
        let view: ArrayView1<Float> = input.into();
        let mut activation = view.to_owned();

        for l in &self.layers {
            activation = l.forward(&activation, self.activator.act);
        }

        activation
    }

    // TEST: how to test backprop?
    pub fn backprop(&mut self, x: ArrayView1<Float>, y: ArrayView1<Float>) {
        // feedforwad
        let input = x.to_owned();

        for i in 0..self.layers.len() {
            let activation = if i == 0 {
                &input
            } else {
                &self.layers_buf[i - 1].activation
            };
            // let z = &self.layers[i].weights * activation + &self.layers[i].biases;
            let z = self.layers[i].weights.dot(activation) + &self.layers[i].biases;
            let activation = z.mapv(self.activator.act);
            self.layers_buf[i].z = z;
            self.layers_buf[i].activation = activation;
        }

        // backward
        // element-wise vector multiplication
        let a_l = &self.layers_buf.last().unwrap().activation;
        let z = &self.layers_buf.last().unwrap().z;
        let y = y.to_owned();
        let partial_der_a_l_over_z = z.mapv(self.activator.act_der);
        let delta = Zip::from(a_l)
            .and(&y)
            .map_collect(|a, b| cost_derivative(*a, *b))
            * &partial_der_a_l_over_z;

        self.layers_buf.last_mut().unwrap().nabla_biases = delta.clone();

        let a_l_minus = &self.layers_buf[self.layers_buf.len() - 2].activation;
        self.layers_buf.last_mut().unwrap().nabla_weights = outer(&delta, a_l_minus);

        let mut prev_delta = delta;

        for i in (0..self.layers_buf.len() - 1).rev() {
            let z = &self.layers_buf[i].z;
            let sp = z.mapv(self.activator.act_der);
            let delta = (&self.layers[i + 1].weights.t().dot(&prev_delta)) * &sp;
            self.layers_buf[i].nabla_biases = delta.clone();

            let prev_activation = if i == 0 {
                &input
            } else {
                &self.layers_buf[i - 1].activation
            };
            self.layers_buf[i].nabla_weights = outer(&delta, prev_activation);

            prev_delta = delta;
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_nn_feedforward() {
        let l1 = Layer::load_from_slices(&[&[0.5, 0.25]], &[-0.5]).unwrap();
        let l2 = Layer::load_from_slices(&[&[0.5]], &[0.]).unwrap();
        let nn = NNBuilder::new_from_layers(vec![l1, l2])
            .with_activator(ActivatorType::LinearDummy)
            .build()
            .unwrap();

        let y = nn.feedforward(ArrayView1::from(&[1.0, 2.0]));

        assert_eq!(y.shape(), &[1]);
        assert_eq!(y[0], 0.25);
    }

    #[test]
    fn test_nn_end_training_on_target_accuracy() {}

    #[test]
    fn test_model_save_and_load() {
        let mut nn = NNBuilder::new_from_arch(&[28 * 28, 32, 32, 10])
            .with_activator(ActivatorType::Sigmoid)
            .with_learn_rate(0.01)
            .with_epochs(5)
            .with_mini_batch_size(5)
            .build()
            .unwrap();
        let data = DataSet::from_mnist("data/", 100, 10).unwrap();

        nn.train(data);
        nn.save("data/test_model.bin").unwrap();

        let builder = NNBuilder::new_from_model_file("data/test_model.bin").unwrap();

        assert_eq!(builder.layers, nn.layers);
        assert_eq!(builder.activator, nn.activator.act_type);
        assert_eq!(builder.learn_rate, nn.learn_rate);
        assert_eq!(builder.epochs, nn.epochs);
        assert_eq!(builder.mini_batch_size, nn.mini_batch_size);
        assert_eq!(builder.target_accuracy, nn.target_accuracy);
    }
}
