pub mod tensor;

use crate::tensor::{Device, Tensor};

fn main() {
    // dtype is inferred from the Vec element type — no enum needed
    let a = Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU, true)
        .expect("failed to build tensor a");

    let b = Tensor::new(vec![5.0_f32, 6.0, 7.0, 8.0], vec![2, 2], Device::CPU, false)
        .expect("failed to build tensor b");

    let c = a.add(&b).expect("addition failed");
    let z = Tensor::<i32>::randn(vec![2, 2], Device::CPU);

    println!(
        "a => type: {}, shape: {:?}, device: {:?}, requires_grad: {}",
        a.type_name(),
        a.shape(),
        a.device(),
        a.requires_grad()
    );
    println!("c = a + b => type: {}, data: {:?}", c.type_name(), c.data());
    println!("z => type: {}, data: {:?}", z.type_name(), z.data());
}
