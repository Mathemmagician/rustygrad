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

pub struct ValueData {
    pub data: f64,
    pub grad: f64,
    uuid: Uuid,
    _backward: Option<fn(value: &ValueData)>,
    _prev: Vec<Value>,
}

#[derive(Debug)]
pub struct Value(Rc<RefCell<ValueData>>);

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
    let out = Value::from(a.borrow().data + b.borrow().data);
    out.borrow_mut()._prev = vec![Value(Rc::clone(a)), Value(Rc::clone(b))];
    out.borrow_mut()._backward = Some(|value: &ValueData| {
        value._prev[0].borrow_mut().grad += value.grad;
        value._prev[1].borrow_mut().grad += value.grad;
    });
    out
});

impl_op_ex!(*|a: &Value, b: &Value| -> Value {
    let out = Value::from(a.borrow().data * b.borrow().data);
    out.borrow_mut()._prev = vec![Value(Rc::clone(a)), Value(Rc::clone(b))];
    out.borrow_mut()._backward = Some(|value: &ValueData| {
        value._prev[0].borrow_mut().grad += value._prev[1].borrow_mut().data * value.grad;
        value._prev[1].borrow_mut().grad += value._prev[0].borrow_mut().data * value.grad;
    });
    out
});

impl_op_ex!(+= |a: &mut Value, b: &Value| { *a = &*a + b });
impl_op!(-|a: &Value| -> Value { a * Value::from(-1.0) });
impl_op_ex!(-|a: &Value, b: &Value| -> Value { a + (-b) });
impl_op_ex!(/ |a: &Value, b: &Value| -> Value { a * b.pow(-1.0) });

impl_op_ex_commutative!(+|a: &Value, b: f64| -> Value { a + Value::from(b) });
impl_op_ex_commutative!(*|a: &Value, b: f64| -> Value { a * Value::from(b) });
impl_op_ex!(/ |a: &Value, b: f64| -> Value { a / Value::from(b) });
impl_op_ex!(/ |a: f64, b: &Value| -> Value { Value::from(a) / b });

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

    pub fn relu(&self) -> Value {
        let out = Value::from(self.borrow().data.max(0.0));
        out.borrow_mut()._prev = vec![Value(Rc::clone(self))];
        out.borrow_mut()._backward = Some(|value: &ValueData| {
            if value.data > 0.0 {
                value._prev[0].borrow_mut().grad += value.grad;
            }
        });
        out
    }

    pub fn pow(&self, power: f64) -> Value {
        let out = Value::from(self.borrow().data.powf(power));
        out.borrow_mut()._prev = vec![Value(Rc::clone(self)), Value::from(power)];
        out.borrow_mut()._backward = Some(|value: &ValueData| {
            let base = value._prev[0].borrow().data;
            let p = value._prev[1].borrow().data;
            value._prev[0].borrow_mut().grad += p * base.powf(p - 1.0) * value.grad;
        });
        out
    }

    pub fn backward(&self) {
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
