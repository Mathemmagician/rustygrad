#[macro_use]
extern crate impl_ops;
use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::{self, Debug},
    hash::{Hash, Hasher},
    ops,
    rc::Rc,
};
use uuid::Uuid;

#[macro_export]
macro_rules! value {
    ( $x:expr ) => {{
        Value::from($x)
    }};
}

struct ValueData {
    uuid: Uuid,
    data: f64,
    grad: f64,
    _backward: Option<fn(value: &ValueData)>,
    _prev: Vec<Value>,
}

#[derive(Debug)]
struct Value(Rc<RefCell<ValueData>>);

impl ops::Deref for Value {
    type Target = Rc<RefCell<ValueData>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.borrow().uuid.hash(state);
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.borrow().uuid == other.borrow().uuid
    }
}

impl Eq for Value {}

impl_op_ex!(+ |a: &Value, b: &Value| -> Value {
    let out = value!(a.borrow().data + b.borrow().data);
    out.borrow_mut()._prev = vec![Value(Rc::clone(a)), Value(Rc::clone(b))];
    out.borrow_mut()._backward = Some(|value: &ValueData| {
        value._prev[0].borrow_mut().grad += value.grad;
        value._prev[1].borrow_mut().grad += value.grad;
    });
    out
});

impl_op_ex!(*|a: &Value, b: &Value| -> Value {
    let out = value!(a.borrow().data * b.borrow().data);
    out.borrow_mut()._prev = vec![Value(Rc::clone(a)), Value(Rc::clone(b))];
    out.borrow_mut()._backward = Some(|value: &ValueData| {
        value._prev[0].borrow_mut().grad += value._prev[1].borrow_mut().data * value.grad;
        value._prev[1].borrow_mut().grad += value._prev[0].borrow_mut().data * value.grad;
    });
    out
});

impl_op_ex!(+= |a: &mut Value, b: &Value| { *a = &*a + b });
impl_op!(-|a: &Value| -> Value { a * value!(-1.0) });
impl_op_ex!(-|a: &Value, b: &Value| -> Value { a + (-b) });
impl_op_ex!(/ |a: &Value, b: &Value| -> Value { a * b.pow(-1.0) });

impl_op_ex_commutative!(+|a: &Value, b: f64| -> Value { a + value!(b) });
impl_op_ex_commutative!(*|a: &Value, b: f64| -> Value { a * value!(b) });
impl_op_ex!(/ |a: &Value, b: f64| -> Value { a / value!(b) });
impl_op_ex!(/ |a: f64, b: &Value| -> Value { value!(a) / b });

impl ValueData {
    fn new(data: f64) -> ValueData {
        ValueData {
            uuid: Uuid::new_v4(),
            data,
            grad: 0.0,
            _backward: None,
            _prev: Vec::new(),
        }
    }
}

impl Debug for ValueData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Value")
            .field("data", &self.data)
            .field("grad", &self.grad)
            .finish()
    }
}

impl<T: Into<f64>> From<T> for Value {
    fn from(t: T) -> Value {
        Value::new(ValueData::new(t.into()))
    }
}

impl Value {
    fn new(value: ValueData) -> Value {
        Value(Rc::new(RefCell::new(value)))
    }

    fn relu(&self) -> Value {
        let out = value!(self.borrow().data.max(0.0));
        out.borrow_mut()._prev = vec![Value(Rc::clone(self))];
        out.borrow_mut()._backward = Some(|value: &ValueData| {
            if value.data > 0.0 {
                value._prev[0].borrow_mut().grad += value.grad;
            }
        });
        out
    }

    fn pow(&self, power: f64) -> Value {
        let out = value!(self.borrow().data.powf(power));
        out.borrow_mut()._prev = vec![Value(Rc::clone(self)), value!(power)];
        out.borrow_mut()._backward = Some(|value: &ValueData| {
            let base = value._prev[0].borrow().data;
            let p = value._prev[1].borrow().data;
            value._prev[0].borrow_mut().grad += p * base.powf(p - 1.0) * value.grad;
        });
        out
    }

    fn backward(&self) {
        let mut topo: Vec<Value> = vec![];
        let mut visited: HashSet<Value> = HashSet::new();

        self._build_topo(&mut topo, &mut visited);
        topo.reverse();
        self.borrow_mut().grad = 1.0;

        for v in topo {
            if let Some(backprop) = v.borrow()._backward {
                backprop(&v.borrow());
            }
        }
    }

    fn _build_topo(&self, topo: &mut Vec<Value>, visited: &mut HashSet<Value>) {
        if !visited.contains(self) {
            visited.insert(Value(Rc::clone(self)));

            for child in &self.borrow()._prev {
                child._build_topo(topo, visited);
            }
            topo.push(Value(Rc::clone(self)));
        }
    }
}

fn main() {
    
    // a = Value(-4.0)
    // b = Value(2.0)
    let a = value!(-4.0);
    let b = value!(2.0);
    
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
