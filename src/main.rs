use micrograd::load_moons_data;
use micrograd::Neuron;
use micrograd::Value;
use micrograd::MLP;

fn main() {
    value();
    println!("\n===============\n");
    nn();
    println!("\n===============\n");
    mlp();
}

fn value() {
    // a = Value(-4.0)
    // b = Value(2.0)
    let a = Value::from(-4.0);
    let b = Value::from(2.0);

    // c = a + b
    // d = a * b + b**3
    let mut c = &a + &b;
    let mut d = &a * &b + &b.pow(3.0);

    // c += c + 1
    // c += 1 + c + (-a)
    // d += d * 2 + (b + a).relu()
    // d += 3 * d + (b - a).relu()
    c += &c + 1.0;
    c += 1.0 + &c + (-&a);
    d += &d * 2.0 + (&b + &a).relu();
    d += 3.0 * &d + (&b - &a).relu();

    // e = c - d
    // f = e**2
    // g = f / 2.0
    // g += 10.0 / f
    let e = &c - &d;
    let f = e.pow(2.0);
    let mut g = &f / 2.0;
    g += 10.0 / &f;

    // print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
    println!("{:.4}", g.borrow().data); // 24.7041

    // g.backward()
    // print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
    // print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
    g.backward();
    println!("{:.4}", a.borrow().grad); // 138.8338
    println!("{:.4}", b.borrow().grad); // 645.5773

    println!("a is {:?}", a);
    println!("b is {:?}", b);
    println!("c is {:?}", c);
    println!("d is {:?}", d);
    println!("e is {:?}", e);
    println!("f is {:?}", f);
    println!("g is {:?}", g);
}

fn nn() {
    let n = Neuron::from(2);
    let x = vec![Value::from(1.0), Value::from(-2.0)];

    println!("n = {:?}", n);
    let y = n.forward(&x);
    println!("{:?}", y);
}

fn mlp() {
    let model = MLP::new(2, vec![16, 16, 1]);

    let (xs, ys) = load_moons_data();

    // optimization
    for k in 0..100 {
        // forward
        let (total_loss, acc) = loss(&model, &xs, &ys);

        // backward
        model.zero_grad();
        total_loss.backward();

        // update (sgd)
        let learning_rate = 1.0 - 0.9 * (k as f64) / 100.0;
        for p in &model.parameters() {
            let delta = learning_rate * p.borrow().grad;
            p.borrow_mut().data -= delta;
        }

        println!(
            "step {k} loss {:.3}, accuracy {:.2}%",
            total_loss.borrow().data,
            acc * 100.0
        );
    }

    // ASCII contour plot
    let mut grid: Vec<Vec<String>> = Vec::new();
    let bound = 20;
    for y in -bound..bound {
        let mut row: Vec<String> = Vec::new();
        for x in -bound..bound {
            let k = &model.forward(vec![Value::from(x as f64 / bound as f64 * 2.0), Value::from(-y as f64 / bound as f64 * 2.0)])[0];
            row.push(if k.borrow().data > 0.0 {String::from("*")} else {String::from(".")});
        }
        grid.push(row);
    }

    for row in grid {
        for val in row {
            print!("{} ", val);
        }
        println!();
    }

}

fn loss(model: &MLP, xs: &[Vec<f64>], ys: &[f64]) -> (Value, f64) {
    let inputs: Vec<Vec<Value>> = xs
        .iter()
        .map(|xrow| vec![Value::from(xrow[0]), Value::from(xrow[1])])
        .collect();

    // forward the model to get scores
    let scores: Vec<Value> = inputs
        .iter()
        .map(|xrow| model.forward(xrow.clone())[0].clone())
        .collect();

    // svm "max-margin" loss
    let losses: Vec<Value> = ys
        .iter()
        .zip(&scores)
        .map(|(yi, scorei)| (1.0 + -yi * scorei).relu())
        .collect();
    let n: f64 = (&losses).len() as f64;
    let data_loss: Value = losses.into_iter().sum::<Value>() / n;

    // L2 regularization
    let alpha: f64 = 0.0001;
    let reg_loss: Value = alpha
        * model
            .parameters()
            .iter()
            .map(|p| p * p)
            .into_iter()
            .sum::<Value>();
    let total_loss = data_loss + reg_loss;

    // also get accuracy
    let accuracies: Vec<bool> = ys
        .iter()
        .zip(scores.iter())
        .map(|(yi, scorei)| (*yi > 0.0) == (scorei.borrow().data > 0.0))
        .collect();
    let accuracy = accuracies.iter().filter(|&a| *a).count() as f64 / n;

    (total_loss, accuracy)
}
