
# micrograd
![Thanks, Dalle](crab.png)

Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). This repo tries to create a tiny Autograd engine in Rust:

1. with a friendly api
2. an easy to understand implementation
3. in minimal amount of code

The engine and the neural net are implemented in about 150 and 100 lines of code respectivelly (vs Andrej's 100 and 50)! About twice as long, but twice as fast!

#### Example usage
Below is an example showing supported operations `// and their Python version`

```Rust
use micrograd::Value;

fn main() {
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
}
```

#### Training a neural net

The file `demo.rs` trains a MLP binary classifier (with 2 16-node hidden layers) on a toy `make_moons.csv` dataset. Since plots in rust are hard, for now, here is an ascii representation of the learned solution space:


```
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . * * * * * . . . . . . . . . . . . . . . . * 
. . . . . . . . . . . . . . . . . * * * * * * * . . . . . . . . . . . . . . * * 
. . . . . . . . . . . . . . . . . * * * * * * * * . . . . . . . . . . . . * * * 
. . . . . . . . . . . . . . . . . * * * * * * * * * . . . . . . . . . . * * * * 
. . . . . . . . . . . . . . . . . * * * * * * * * * * . . . . . . . . . * * * * 
. . . . . . . . . . . . . . . . * * * * * * * * * * * . . . . . . . . * * * * * 
. . . . . . . . . . . . . . . . * * * * * * * * * * * * . . . . . . * * * * * * 
. . . . . . . . . . . . . . . . * * * * * * * * * * * * * . . . . * * * * * * * 
. . . . . . . . . . . . . . . * * * * * * * * * * * * * * * * * * * * * * * * * 
. . . . . . . . . . . . . . . * * * * * * * * * * * * * * * * * * * * * * * * * 
. . . . . . . . . . . . . . * * * * * * * * * * * * * * * * * * * * * * * * * * 
. . . . . . . . . . . . . . * * * * * * * * * * * * * * * * * * * * * * * * * * 
. . . . . . . . . . . . . . * * * * * * * * * * * * * * * * * * * * * * * * * * 
. . . . . . . . . . . . . * * * * * * * * * * * * * * * * * * * * * * * * * * * 
. . . . . . . . . . . . . * * * * * * * * * * * * * * * * * * * * * * * * * * * 
. . . . . . . . . . . . * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
. . . . . . . . . . . . * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
. . . . . . . . . . . * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
. . . . . . . . . . . * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
. . . . . . . . . . . * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
. . . . . . . . . . * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
. . . . . . . . . . * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
. . . . . . . . . * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
. . . . . . . . . * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
. . . . . . . . * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
. . . . . . . . * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

```

#### Running tests
```
cargo test
```