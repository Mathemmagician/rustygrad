use std::{
    cell::RefCell,
    fmt::Debug,
    ops::{Add, Mul},
    rc::Rc,
};

#[macro_export]
macro_rules! value {
    ( $x:expr, $y:expr ) => {{
        RValue(Rc::new(RefCell::new(Value::new($x, $y))))
    }};
}

#[derive(Debug, PartialEq, Eq)]
enum Operation {
    Add,
    Mul,
    Tanh,
    Relu
}

#[derive(Eq, PartialEq, Debug)]
struct RValue(Rc<RefCell<Value>>);

struct Value {
    data: f64,
    label: String,
    grad: f64,
    operation: Option<Operation>,
    children: Vec<RValue>,
}

impl Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Value")
            .field("label", &self.label)
            .field("data", &self.data)
            .field("grad", &self.grad)
            .field("operation", &self.operation)
            .finish()
    }
}

impl RValue {
    fn new(value: Value) -> RValue {
        RValue(Rc::new(RefCell::new(value)))
    }

    fn set_label(&self, label: &str) -> &RValue {
        self.0.borrow_mut().label = label.to_string();
        self
    }

    fn set_grad(&self, grad: f64) -> &RValue {
        self.0.borrow_mut().grad = grad;
        self
    }

    fn add_grad(&self, grad: f64) -> &RValue {
        self.0.borrow_mut().grad += grad;
        self
    }

    fn tanh(&self) -> RValue {
        let c = (2.0 * self.0.borrow().data).exp();
        let data = (c - 1.0) / (c + 1.0);

        RValue::new(Value {
            data,
            label: String::from("tanh"),
            grad: 0.0,
            operation: Some(Operation::Tanh),
            children: vec![RValue(Rc::clone(&self.0))],
        })
    }

    fn relu(&self) -> RValue {
        let data = self.0.borrow().data.max(0.0);

        RValue::new(Value {
            data,
            label: String::from("tanh"),
            grad: 0.0,
            operation: Some(Operation::Tanh),
            children: vec![RValue(Rc::clone(&self.0))],
        })
    }
}

impl Value {
    fn new(data: f64, label: &str) -> Value {
        Value {
            data,
            label: label.to_string(),
            grad: 0.0,
            operation: None,
            children: Vec::new(),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
            && self.grad == other.grad
            && self.label == other.label
            && self.operation == other.operation
            && self.children == other.children
    }
}

impl Eq for Value {}

impl Add<&RValue> for &RValue {
    type Output = RValue;

    fn add(self, other: &RValue) -> RValue {
        let data = self.0.borrow().data + other.0.borrow().data;

        RValue::new(Value {
            data,
            label: String::from("+"),
            grad: 0.0,
            operation: Some(Operation::Add),
            children: vec![RValue(Rc::clone(&self.0)), RValue(Rc::clone(&other.0))],
        })
    }
}

impl Mul<&RValue> for &RValue {
    type Output = RValue;

    fn mul(self, other: &RValue) -> RValue {
        let data = self.0.borrow().data * other.0.borrow().data;

        RValue::new(Value {
            data,
            label: String::from("*"),
            grad: 0.0,
            operation: Some(Operation::Mul),
            children: vec![RValue(Rc::clone(&self.0)), RValue(Rc::clone(&other.0))],
        })
    }
}

fn propagate_gradient(value: &RValue) {
    let operation = &value.0.borrow().operation;
    let data = value.0.borrow().data;
    let grad = value.0.borrow().grad;

    if let Some(opp) = operation {
        // println!("{:?}", opp);
        let children = &value.0.borrow().children;

        if let Some(left) = &children.get(0) {
            if let Some(right) = &children.get(1) {
                match opp {
                    Operation::Add => {
                        left.add_grad(grad);
                        right.add_grad(grad);
                    }
                    Operation::Mul => {
                        left.add_grad(grad * right.0.borrow().data);
                        right.add_grad(grad * left.0.borrow().data);
                    }
                    _ => (),
                };
                propagate_gradient(right);
            } else {
                match opp {
                    Operation::Tanh => {
                        let c = (2.0 * data).exp();
                        let t = (c - 1.0) / (c + 1.0);
                        left.add_grad((1.0 - t.powi(2)) * grad);
                    }
                    _ => (),
                }
            }
            propagate_gradient(left);
        }
    }
}

fn main() {
    // let a = RValue::new(Value::new(2.0, "a"));
    let a = value!(2.0, "a");
    let b = value!(-3.0, "b");
    let c = &a + &b;
    c.set_label("c");
    let d = &a * &c;
    d.set_label("d");
    let e = d.tanh();

    e.set_grad(-1.0);

    println!("a is {:?}", a);
    println!("b is {:?}", b);
    println!("c is {:?}", c);
    println!("d is {:?}", d);
    println!("e is {:?}", e);

    propagate_gradient(&e);

    // dbg!(&x);
    println!("a is {:?}", a);
    println!("b is {:?}", b);
    println!("c is {:?}", c);
    println!("d is {:?}", d);
    println!("e is {:?}", e);
}
