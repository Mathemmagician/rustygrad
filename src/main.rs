use std::{
    cell::RefCell,
    fmt::{Debug, self},
    ops::{Deref, Add, Mul, Neg},
    rc::Rc,
};

#[macro_export]
macro_rules! value {
    ( $x:expr, $y:expr ) => {{
        RValue(Rc::new(RefCell::new(Value::new($x, $y))))
    }};
}

#[derive(Debug)]
struct RValue(Rc<RefCell<Value>>);

impl Deref for RValue { 
    type Target = Rc<RefCell<Value>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

type BackwardsFn = fn(value: &Value);

struct Value {
    data: f64,
    grad: f64,
    label: String,
    _backward: Option<BackwardsFn>,
    _prev: Vec<RValue>,
}


impl Value {
    fn new(data: f64, label: &str) -> Value {
        Value {
            data,
            grad: 0.0,
            label: label.to_string(),
            _backward: None,
            _prev: Vec::new(),
        }
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Value")
         .field("data", &self.data)
         .field("grad", &self.grad)
         .field("label", &self.label)
         .finish()
    }
}

impl RValue {
    fn new(value: Value) -> RValue {
        RValue(Rc::new(RefCell::new(value)))
    }

    fn set_label(&self, label: &str) -> &RValue {
        self.borrow_mut().label = label.to_string();
        self
    }

    fn relu(&self) -> RValue {
        let out = value!(self.borrow().data.max(0.0), "relu");
        out.borrow_mut()._prev = vec![RValue(Rc::clone(&self))];
        out.borrow_mut()._backward = Some(|value: &Value| {
            value._prev[0].borrow_mut().grad += value.grad;
        });
        out
    }

    fn pow(&self, other: &RValue) -> RValue {
        let out = value!(self.borrow().data.powf(other.borrow().data), "**{other}");
        out.borrow_mut()._prev = vec![RValue(Rc::clone(&self)), RValue(Rc::clone(&other))];
        out.borrow_mut()._backward = Some(|value: &Value| {
            let base = value._prev[0].borrow().data;
            let p = value._prev[1].borrow().data;
            value._prev[0].borrow_mut().grad += p * base.powf(p - 1.0) * value.grad;
        });
        out
    }
}


impl Add<&RValue> for &RValue {
    type Output = RValue;

    fn add(self, other: &RValue) -> RValue {
        let out = value!(self.borrow().data + other.borrow().data, "+");
        out.borrow_mut()._prev = vec![RValue(Rc::clone(&self)), RValue(Rc::clone(&other))];
        out.borrow_mut()._backward = Some(|value: &Value| {
            value._prev[0].borrow_mut().grad += value.grad;
            value._prev[1].borrow_mut().grad += value.grad;
        });
        out
    }
}

impl Mul<&RValue> for &RValue {
    type Output = RValue;

    fn mul(self, other: &RValue) -> RValue {
        let out = value!(self.borrow().data * other.borrow().data, "*");
        out.borrow_mut()._prev = vec![RValue(Rc::clone(&self)), RValue(Rc::clone(&other))];
        out.borrow_mut()._backward = Some(|value: &Value| {
            value._prev[0].borrow_mut().grad += value._prev[1].borrow_mut().data * value.grad;
            value._prev[1].borrow_mut().grad += value._prev[0].borrow_mut().data * value.grad;
        });
        out
    }
}

impl Neg for &RValue {
    type Output = RValue;

    fn neg(self) -> RValue {
        let neg = value!(-1.0, "");
        self * &neg
    }
}


fn main() {

    let a = value!(-4.0, "a");
    let b = value!(2.0, "b");
    let c = &a + &b;
    c.set_label("a + b");
    let d = &(&a * &b) + &b.pow(&value!(3.0, ""));
    d.set_label("a * c");
    let e = -&d;
    e.set_label(" - d");
    let f = a.pow(&value!(-1.0, ""));
    f.set_label("a ** -1");
    // let e = d.tanh();

    // e.set_grad(-1.0);

    println!("a is {:?}", a);
    println!("b is {:?}", b);
    println!("c is {:?}", c);
    println!("d is {:?}", d);
    println!("e is {:?}", e);
    println!("f is {:?}", f);

    // propagate_gradient(&e);

    // dbg!(&x);
    // println!("a is {:?}", a);
    // println!("b is {:?}", b);
    // println!("c is {:?}", c);
    // println!("d is {:?}", d);
    // println!("e is {:?}", e);
}
