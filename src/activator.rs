use serde::{Deserialize, Serialize};

use crate::Float;

fn sigmoid(z: Float) -> Float {
    1. / (1. + (-z).exp())
}

fn sigmoid_derivative(z: Float) -> Float {
    sigmoid(z) * (1. - sigmoid(z))
}

pub struct ActPair {
    pub act_type: ActivatorType,
    pub act: fn(Float) -> Float,
    pub act_der: fn(Float) -> Float,
}

#[derive(Serialize, Deserialize, Copy, Clone, Debug, PartialEq)]
pub enum ActivatorType {
    Sigmoid,
    LeakyRelu,
    Tanh,
    LinearDummy,
}

impl From<ActivatorType> for ActPair {
    fn from(value: ActivatorType) -> Self {
        match value {
            ActivatorType::Sigmoid => ActPair {
                act_type: ActivatorType::Sigmoid,
                act: sigmoid,
                act_der: sigmoid_derivative,
            },
            ActivatorType::LeakyRelu => ActPair {
                act_type: ActivatorType::LeakyRelu,
                act: |x| if x > 0. { x } else { 0.1 * x },
                act_der: |x| if x > 0. { 1. } else { 0.1 },
            },
            ActivatorType::Tanh => ActPair {
                act_type: ActivatorType::Tanh,
                act: |x| x.tanh(),
                act_der: |x| 1. - x.tanh().powi(2),
            },
            ActivatorType::LinearDummy => ActPair {
                act_type: ActivatorType::LinearDummy,
                act: |x| x,
                act_der: |_x| 1.,
            },
        }
    }
}
