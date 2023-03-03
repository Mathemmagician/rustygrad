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

    println!("a is {:?}", a);
    println!("b is {:?}", b);
    println!("c is {:?}", c);
    println!("d is {:?}", d);
    println!("e is {:?}", e);
    println!("f is {:?}", f);
    println!("g is {:?}", g);
}
