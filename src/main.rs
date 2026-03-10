use arrayfire::{Dim4, af_print};
use mashinko::{
    af_tensor::Node,
    engine::backward,
    layer::Linear,
    loss::mse,
    optimizer::{Optimizer, SGD},
};

fn main() {
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
