use rustygrad::load_moons_data;
use rustygrad::Value;
use rustygrad::MLP;

fn main() {
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
            let k = &model.forward(vec![
                Value::from(x as f64 / bound as f64 * 2.0),
                Value::from(-y as f64 / bound as f64 * 2.0),
            ])[0];
            row.push(if k.borrow().data > 0.0 {
                String::from("*")
            } else {
                String::from(".")
            });
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
