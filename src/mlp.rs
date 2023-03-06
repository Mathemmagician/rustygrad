use crate::{Layer, Value};

#[derive(Debug)]
pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: i32, mut nouts: Vec<i32>) -> MLP {
        nouts.insert(0, nin);
        let mut layers: Vec<Layer> = vec![];
        let n = nouts.len() - 1;

        for i in 0..n {
            layers.push(Layer::new(nouts[i], nouts[i + 1], i != n - 1));
        }
        MLP { layers }
    }

    pub fn forward(&self, mut x: Vec<Value>) -> Vec<Value> {
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut params: Vec<Value> = vec![];
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }

    pub fn zero_grad(&self) {
        for p in self.parameters() {
            p.borrow_mut().grad = 0.0;
        }
    }
}
