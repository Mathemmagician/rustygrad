// #[macro_use]
// extern crate impl_ops;
use std::ops::Add;

#[derive(Debug)]
enum Operation {
    Add,
    Multiply,
}

#[derive(Debug)]
struct Value<'a> {
    data: f64,
    label: String,
    grad: f64,
    operation: Option<Operation>,
    children: Vec<&'a Value<'a>>
}

impl<'a> Value<'a> {
    fn new(data: f64, label: &str,) -> Value {
        Value {
            data,
            label: label.to_string(),
            grad: 0.0,
            operation: None,
            children: Vec::new(),
        }
    }

    // fn binary_operation( data: f64, operation: Operation, left: &'a Value, right: &'a Value ) -> Value<'a> {
    //     Value {
    //         data,
    //         label: String::new(),
    //         grad: 0.0,
    //         operation: Some(operation),
    //         children: vec![left, right],
    //     }
    // }
}

impl<'a> Add for &'a Value<'a> {
    type Output = Value<'a>;
    
    fn add(self, other: Self) -> Value<'a> {
        Value {
            data: self.data + other.data,
            label: String::new(),
            grad: 0.0,
            operation: Some(Operation::Add),
            children: vec![self, other],
        }
    }
}

// impl_op!(+ |left: & Value<'a>, right: & Value<'a>| -> Value<'a> {

//     Value::binary_operation(
//         5.0,
//         Operation::Add,
//         left,
//         right
//     )

//     // let out = Value::binary_operation(
//     //     left.data + right.data,
//     //     Operation::Add,
//     //     left,
//     //     right
//     // );
//     // let out = Value::new(a.data + b.data, "");
//     // out._backward = |value: &Value| {
//     //     a.grad = 1.0 * value.grad;
//     //     b.grad = 1.0 * value.grad;
//     // };
//     // Value {
//     //     data: a.data * b.data,
//     //     label: String::new(),
//     // }
// });

fn main() {

    let a = Value::new(2.0, "a");
    let b = Value::new(-3.0, "b");
    let c = Value::new(2.0, "c");

    // let d = Value::binary_operation(
    //     5.0,
    //     Operation::Add,
    //     &a,
    //     &b
    // );

    let d = &a + &b;

    // dbg!(&x);
    println!("a is {:?}", a);
    println!("b is {:?}", b);
    println!("c is {:?}", c);
    println!("d is {:?}", d);
}
