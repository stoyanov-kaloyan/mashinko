use crate::{
    af_tensor::Node,
    data::{DataLoader, Dataset},
    engine::backward,
    layer::{Conv2D, Flatten, HasParameters, Linear, MLP, MaxPool, Permute, ReLU, Sequential},
    loss::{cross_entropy, mse},
    optimizer::{Adam, Optimizer, SGD},
    sequential,
    utils::{argmax, one_hot_encode_host, parse_idx_file, parse_idx_file_host},
};
use arrayfire::{Dim4, af_print};

pub fn mnist_example() {
    let train_images = parse_idx_file("datasets/mnist/train-images.idx3-ubyte", true);
    let train_image_dims = train_images.dims();
    let (train_labels, train_label_dims) =
        parse_idx_file_host("datasets/mnist/train-labels.idx1-ubyte", false);
    eprintln!("train_images dims: {:?}", train_image_dims);
    eprintln!("train_labels dims: {:?}", train_label_dims);

    // Images: (60000, 28, 28, 1) from IDX parser
    // DataLoader batches along dim 0 → (batch, 28, 28, 1)
    // Permute layer in model converts to (28, 28, 1, batch) for conv ops
    let n_samples = train_image_dims[0];
    let labels_onehot = one_hot_encode_host(&train_labels, n_samples as usize, 10);
    let labels_onehot = arrayfire::Array::new(&labels_onehot, Dim4::new(&[n_samples, 10, 1, 1]));

    let dataset = Dataset::new(
        Node::leaf(train_images, false),
        Node::leaf(labels_onehot, false),
    );

    // Large batches keep GPU busier but shrink gradient magnitude (we average by batch),
    // which made updates too small in practice for this handcrafted backend.
    let data_loader = DataLoader::new(&dataset, 2048, true);

    let model = sequential![
        Permute::nhwc_to_hwcn(),
        Conv2D::new(1, 8, 5),
        ReLU::new(),
        MaxPool::new(2, 2),
        Conv2D::new(8, 16, 5),
        ReLU::new(),
        MaxPool::new(2, 2),
        Flatten::new(),
        Linear::lazy(10),
    ];

    let mut optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
    println!("Training MNIST with CNN\n");

    let num_epochs = 5;
    for epoch in 0..num_epochs {
        let mut total_loss = 0.0f32;
        let mut batch_count = 0;

        for (x_batch, y_batch) in &data_loader {
            // eprintln!("x_batch dims: {:?}", x_batch.borrow().tensor().dims());
            let y_pred = model.forward(x_batch);
            let loss = cross_entropy(&y_pred, &y_batch);

            backward(loss.clone());

            let (loss_val, _) = arrayfire::sum_all(loss.borrow().tensor());
            total_loss += loss_val;
            batch_count += 1;

            let params = model.parameters();
            optimizer.step(&params);
            optimizer.zero_grad(&params);

            // drop(loss);
            // drop(y_pred);
            // arrayfire::device_gc();
        }

        let avg_loss = total_loss / batch_count as f32;
        println!("Epoch {:>2} | Avg Loss: {:.6}", epoch + 1, avg_loss);
    }

    println!("\nTraining complete!");

    println!("\n Validation on training set (for demonstration) \n");
    let accuracy = {
        let eval_loader = DataLoader::new(&dataset, 1024, false);
        let mut num_correct = 0.0f32;

        for (x_batch, y_batch) in &eval_loader {
            let y_pred = model.forward(x_batch);
            let predicted_labels = argmax(y_pred.borrow().tensor());
            let true_labels = argmax(y_batch.borrow().tensor());
            let correct = arrayfire::eq(&predicted_labels, &true_labels, false).cast::<f32>();
            let (batch_correct, _) = arrayfire::sum_all(&correct);
            num_correct += batch_correct;
        }

        num_correct / n_samples as f32
    };
    println!("Validation Accuracy: {:.6}", accuracy);
}

/// XOR is a pretty common example of a dataset that cannot be learned
/// by a linear model. It can be learned by a simple MLP where the hidden layer
/// learns the non-linearity
pub fn mlp_example() {
    let x_dims = Dim4::new(&[4, 2, 1, 1]);
    let y_dims = Dim4::new(&[4, 1, 1, 1]);

    let x = Node::leaf(
        arrayfire::Array::new(&[0.0f32, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0], x_dims),
        false,
    );

    let y_true = Node::leaf(
        arrayfire::Array::new(&[0.0f32, 1.0, 1.0, 0.0], y_dims),
        false,
    );

    let model = MLP::new(&[2, 4, 1]);
    // let model = sequential![Linear::new(2, 4), ReLU::new(), Linear::new(4, 1)];

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
    af_print!(
        "weight (expect ~3)",
        model
            .weight()
            .expect("linear example weight missing")
            .borrow()
            .tensor()
    );
    af_print!(
        "bias   (expect ~1)",
        model
            .bias()
            .expect("linear example bias missing")
            .borrow()
            .tensor()
    );
}
