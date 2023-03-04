use micrograd::Neuron;
use micrograd::Value;

#[test]
fn neuron_example() {
    let x = vec![Value::from(1.0), Value::from(-2.0)];
    let n = Neuron::from(2);
    let ws: Vec<f64> = n.parameters().iter().map(|w| w.borrow().data).collect();

    let z = n.forward(&x);

    assert_eq!(z.borrow().data, ws[0] + ws[1] * 1.0 + ws[2] * (-2.0));
}
