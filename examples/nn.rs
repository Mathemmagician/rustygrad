use micrograd::Neuron;
use micrograd::Value;

fn main() {
    let n = Neuron::from(2);
    let x = vec![Value::from(1.0), Value::from(-2.0)];

    println!("n = {:?}", n);
    let y = n.forward(&x);
    println!("{:?}", y);
}
