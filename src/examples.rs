use crate::{
    af_tensor::Node,
    engine::backward,
    layer::{HasParameters, Linear, MLP},
    loss::mse,
    optimizer::{Optimizer, SGD},
};
use arrayfire::{Dim4, af_print};

/// This is a pretty common example of a dataset that cannot be learned
/// by a linear model. It can be learned by a simple MLP where the hidden layer
/// learns the non-linearity
pub fn mlp_example() {
    let x_dims = Dim4::new(&[4, 2, 1, 1]);
    let y_dims = Dim4::new(&[4, 1, 1, 1]);

    // column-major: col0 = x1, col1 = x2
    let x = Node::leaf(
        arrayfire::Array::new(&[0.0f32, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0], x_dims),
        false,
    );

    let y_true = Node::leaf(
        arrayfire::Array::new(&[0.0f32, 1.0, 1.0, 0.0], y_dims),
        false,
    );

    let model = MLP::new(&[2, 4, 1]);

    // let model = Linear::new(2, 1);

    let mut optimizer = SGD { lr: 0.1 };

    println!("learning XOR\n");

    for epoch in 0..200 {
        let y_pred = model.forward(x.clone());

        let loss = mse(&y_pred, &y_true);

        backward(loss.clone());

        if epoch % 40 == 0 {
            let (loss_val, _) = arrayfire::sum_all(loss.borrow().tensor());
            println!("Epoch {:>4} | Loss: {:.6}", epoch, loss_val);
        }

        let params = model.parameters();
        optimizer.step(&params);
        optimizer.zero_grad(&params);
    }

    // Final prediction
    let y_pred = model.forward(x.clone());

    println!("\n=== XOR Results ===\n");
    af_print!("x", x.borrow().tensor());
    af_print!("y_true", y_true.borrow().tensor());
    af_print!("y_pred", y_pred.borrow().tensor());
}

pub fn linear_example() {
    let dims = Dim4::new(&[4, 1, 1, 1]);

    // y = 3x + 1
    let x = Node::leaf(arrayfire::Array::new(&[1.0f32, 2.0, 3.0, 4.0], dims), false);
    let y_true = Node::leaf(
        arrayfire::Array::new(&[4.0f32, 7.0, 10.0, 13.0], dims),
        false,
    );

    let model = Linear::new(1, 1);
    let mut optimizer = SGD { lr: 0.01 };

    println!("learning y = 3x + 1\n");

    for epoch in 0..200 {
        let y_pred = model.forward(x.clone());
        let loss = mse(&y_pred, &y_true);

        backward(loss.clone());

        if epoch % 20 == 0 {
            let (loss_val, _) = arrayfire::sum_all(loss.borrow().tensor());
            println!("Epoch {:>3} | Loss: {:.6}", epoch, loss_val);
        }

        optimizer.step(&model.parameters());
        optimizer.zero_grad(&model.parameters());
    }

    // Final prediction
    let y_pred = model.forward(x.clone());

    println!("\n=== Results ===\n");
    af_print!("x", x.borrow().tensor());
    af_print!("y_true", y_true.borrow().tensor());
    af_print!("y_pred", y_pred.borrow().tensor());
    af_print!("weight (expect ~3)", model.weight.borrow().tensor());
    af_print!("bias   (expect ~1)", model.bias.borrow().tensor());
}
