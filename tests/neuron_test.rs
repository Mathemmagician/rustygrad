use rustygrad::Neuron;
use rustygrad::Value;

#[test]
fn neuron_example() {
    for _ in 1..20 {
        let x = vec![Value::from(1.0), Value::from(-2.0)];
        let n = Neuron::from(2);
        let ws: Vec<f64> = n.parameters().iter().map(|w| w.borrow().data).collect();

        let z = n.forward(&x);
        assert_eq!(
            z.borrow().data,
            (ws[0] + ws[1] * 1.0 + ws[2] * (-2.0)).max(0.0)
        );
    }
}
