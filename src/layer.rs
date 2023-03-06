use crate::{Neuron, Value};

#[derive(Debug)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: i32, nout: i32, nonlin: bool) -> Layer {
        Layer {
            neurons: (0..nout).map(|_| Neuron::new(nin, nonlin)).collect(),
        }
    }

    pub fn from(nin: i32) -> Neuron {
        Neuron::new(nin, true)
    }

    pub fn forward(&self, x: &Vec<Value>) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(x)).collect()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}
