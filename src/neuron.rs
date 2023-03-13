use crate::Value;
use rand::{distributions::Uniform, Rng};
use std::fmt::{self, Debug};

pub struct Neuron {
    w: Vec<Value>,
    b: Value,
    nonlin: bool,
}

impl Debug for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = if self.nonlin { "ReLU" } else { "Linear" };
        write!(f, "{}({})", name, self.w.len())
    }
}

impl Neuron {
    pub fn new(nin: i32, nonlin: bool) -> Neuron {
        let mut rng = rand::thread_rng();
        let range = Uniform::<f64>::new(-1.0, 1.0);

        Neuron {
            w: (0..nin).map(|_| Value::from(rng.sample(range))).collect(),
            b: Value::from(0.0),
            nonlin,
        }
    }

    pub fn from(nin: i32) -> Neuron {
        Neuron::new(nin, true)
    }

    pub fn forward(&self, x: &Vec<Value>) -> Value {
        let wixi_sum: Value = self.w.iter().zip(x).map(|(wi, xi)| wi * xi).sum();
        let out = wixi_sum + &self.b;

        if self.nonlin {
            return out.relu();
        }
        out
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut out = self.w.clone();
        out.insert(0, self.b.clone());
        out
    }
}
