#[macro_use]
extern crate impl_ops;

mod value;
pub use crate::value::Value;

mod neuron;
pub use crate::neuron::Neuron;

mod layer;
pub use crate::layer::Layer;

mod mlp;
pub use crate::mlp::MLP;
