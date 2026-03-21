use crate::af_tensor::Node;
use crate::data::{DataLoader, Dataset};
use crate::engine::backward;
use crate::layer::{
    Conv2D, Flatten, HasParameters, Layer, Linear as DynLinear, MaxPool, Permute, ReLU as DynReLU,
    Sequential,
};
use crate::loss as dyn_loss;
use crate::utils::{
    argmax, one_hot_encode_host, parse_cifar10_batch_host, parse_cifar10_train_host,
    parse_idx_file, parse_idx_file_host,
};
use crate::{
    Frozen, Linear, ReLU, Tensor, Trainable, binary_cross_entropy_with_logits, cross_entropy, mse,
    optimizer::{Adam, Optimizer, SGD},
    sequential2,
};
use arrayfire::{Dim4, af_print};

pub fn mnist_example() {
    const TRAIN_SAMPLES: usize = 60_000;
    const TEST_SAMPLES: usize = 10_000;
    const FEATURES: usize = 28 * 28;
    const CLASSES: usize = 10;

    let train_images = parse_idx_file("datasets/mnist/train-images.idx3-ubyte", true);
    let train_image_dims = train_images.dims();
    assert_eq!(
        train_image_dims,
        Dim4::new(&[TRAIN_SAMPLES as u64, 28, 28, 1]),
        "Expected train image dims [60000, 28, 28, 1], got {:?}",
        train_image_dims
    );
    let train_x_flat = arrayfire::moddims(
        &train_images,
        Dim4::new(&[TRAIN_SAMPLES as u64, FEATURES as u64, 1, 1]),
    );
    let train_x = Tensor::<TRAIN_SAMPLES, FEATURES, Frozen>::constant_from_array(train_x_flat);

    let (train_labels, train_label_dims) =
        parse_idx_file_host("datasets/mnist/train-labels.idx1-ubyte", false);
    assert_eq!(
        train_label_dims,
        Dim4::new(&[TRAIN_SAMPLES as u64, 1, 1, 1]),
        "Expected train label dims [60000, 1, 1, 1], got {:?}",
        train_label_dims
    );
    let train_labels_onehot = one_hot_encode_host(&train_labels, TRAIN_SAMPLES, CLASSES);
    let train_y =
        Tensor::<TRAIN_SAMPLES, CLASSES, Frozen>::constant_from_array(arrayfire::Array::new(
            &train_labels_onehot,
            Dim4::new(&[TRAIN_SAMPLES as u64, CLASSES as u64, 1, 1]),
        ));

    let model = sequential2(
        sequential2(Linear::<FEATURES, 128>::new(), ReLU::new()),
        Linear::<128, CLASSES>::new(),
    );
    let mut optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
    let params = model.parameters();

    println!("training typed MNIST MLP\n");
    for epoch in 0..20 {
        let logits = model.forward(train_x.clone());
        let loss = cross_entropy(&logits, &train_y);
        loss.backward();

        let (loss_val, _) = arrayfire::sum_all(loss.node().borrow().tensor());
        println!("Epoch {:>2} | Loss: {:.6}", epoch + 1, loss_val);

        optimizer.step(&params);
        optimizer.zero_grad(&params);
    }

    let train_logits = model.forward(train_x.clone());
    let train_pred = argmax(train_logits.node().borrow().tensor());
    let train_true = argmax(train_y.node().borrow().tensor());
    let train_correct = arrayfire::eq(&train_pred, &train_true, false).cast::<f32>();
    let (train_hits, _) = arrayfire::sum_all(&train_correct);
    println!("Train Accuracy: {:.6}", train_hits / TRAIN_SAMPLES as f32);

    let test_images = parse_idx_file("datasets/mnist/t10k-images.idx3-ubyte", true);
    let test_image_dims = test_images.dims();
    assert_eq!(
        test_image_dims,
        Dim4::new(&[TEST_SAMPLES as u64, 28, 28, 1]),
        "Expected test image dims [10000, 28, 28, 1], got {:?}",
        test_image_dims
    );
    let test_x_flat = arrayfire::moddims(
        &test_images,
        Dim4::new(&[TEST_SAMPLES as u64, FEATURES as u64, 1, 1]),
    );
    let test_x = Tensor::<TEST_SAMPLES, FEATURES, Frozen>::constant_from_array(test_x_flat);

    let (test_labels, test_label_dims) =
        parse_idx_file_host("datasets/mnist/t10k-labels.idx1-ubyte", false);
    assert_eq!(
        test_label_dims,
        Dim4::new(&[TEST_SAMPLES as u64, 1, 1, 1]),
        "Expected test label dims [10000, 1, 1, 1], got {:?}",
        test_label_dims
    );
    let test_labels_onehot = one_hot_encode_host(&test_labels, TEST_SAMPLES, CLASSES);
    let test_y =
        Tensor::<TEST_SAMPLES, CLASSES, Frozen>::constant_from_array(arrayfire::Array::new(
            &test_labels_onehot,
            Dim4::new(&[TEST_SAMPLES as u64, CLASSES as u64, 1, 1]),
        ));

    let test_logits = model.forward(test_x);
    let test_pred = argmax(test_logits.node().borrow().tensor());
    let test_true = argmax(test_y.node().borrow().tensor());
    let test_correct = arrayfire::eq(&test_pred, &test_true, false).cast::<f32>();
    let (test_hits, _) = arrayfire::sum_all(&test_correct);
    println!("Test Accuracy: {:.6}", test_hits / TEST_SAMPLES as f32);
}

pub fn mlp_example() {
    let x = Tensor::<4, 2, Frozen>::constant_from_array(arrayfire::Array::new(
        &[0.0f32, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        Dim4::new(&[4, 2, 1, 1]),
    ));
    let y_true = Tensor::<4, 1, Frozen>::constant_from_array(arrayfire::Array::new(
        &[0.0f32, 1.0, 1.0, 0.0],
        Dim4::new(&[4, 1, 1, 1]),
    ));

    let model = sequential2(
        sequential2(Linear::<2, 8>::new(), ReLU::new()),
        Linear::<8, 1>::new(),
    );
    let mut optimizer = Adam::new(0.03, 0.9, 0.999, 1e-8);
    let params = model.parameters();

    println!("learning XOR with typed API\n");
    for epoch in 0..600 {
        let logits = model.forward(x.clone());
        let loss = binary_cross_entropy_with_logits(&logits, &y_true);
        loss.backward();

        if epoch % 200 == 0 {
            let (loss_val, _) = arrayfire::sum_all(loss.node().borrow().tensor());
            println!("Epoch {:>4} | Loss: {:.6}", epoch, loss_val);
        }

        optimizer.step(&params);
        optimizer.zero_grad(&params);
    }

    let logits = model.forward(x.clone());
    let y_pred = Tensor::<4, 1, Trainable>::from_node(Node::sigmoid(&logits.node()));
    let y_pred_bin = arrayfire::gt(
        y_pred.node().borrow().tensor(),
        &arrayfire::constant(0.5f32, Dim4::new(&[4, 1, 1, 1])),
        false,
    )
    .cast::<f32>();
    println!("\n=== XOR Results ===\n");
    af_print!("x", x.node().borrow().tensor());
    af_print!("y_true", y_true.node().borrow().tensor());
    af_print!("y_pred", y_pred.node().borrow().tensor());
    af_print!("y_pred_bin", y_pred_bin);
}

pub fn linear_example() {
    let x = Tensor::<4, 1, Frozen>::constant_from_array(arrayfire::Array::new(
        &[1.0f32, 2.0, 3.0, 4.0],
        Dim4::new(&[4, 1, 1, 1]),
    ));
    let y_true = Tensor::<4, 1, Frozen>::constant_from_array(arrayfire::Array::new(
        &[4.0f32, 7.0, 10.0, 13.0],
        Dim4::new(&[4, 1, 1, 1]),
    ));

    let model = Linear::<1, 1>::new();
    let mut optimizer = SGD::new(0.01);
    let params = model.parameters();

    println!("learning y = 3x + 1 with typed API\n");
    for epoch in 0..200 {
        let y_pred = model.forward(x.clone());
        let loss = mse(&y_pred, &y_true);
        loss.backward();

        if epoch % 20 == 0 {
            let (loss_val, _) = arrayfire::sum_all(loss.node().borrow().tensor());
            println!("Epoch {:>3} | Loss: {:.6}", epoch, loss_val);
        }

        optimizer.step(&params);
        optimizer.zero_grad(&params);
    }

    let y_pred = model.forward(x.clone());
    println!("\n=== Linear Results ===\n");
    af_print!("x", x.node().borrow().tensor());
    af_print!("y_true", y_true.node().borrow().tensor());
    af_print!("y_pred", y_pred.node().borrow().tensor());
}

pub fn cifar10_vgg_example() {
    const TRAIN_SAMPLES: usize = 50_000;
    const TEST_SAMPLES: usize = 10_000;
    const CLASSES: usize = 10;
    const BATCH_SIZE: usize = 2048;
    const EPOCHS: usize = 10;

    let (train_images_host, train_image_dims, train_labels_host, train_label_dims) =
        parse_cifar10_train_host("datasets/cifar", true);
    assert_eq!(
        train_image_dims,
        Dim4::new(&[TRAIN_SAMPLES as u64, 32, 32, 3]),
        "Expected CIFAR train image dims [50000, 32, 32, 3], got {:?}",
        train_image_dims
    );
    assert_eq!(
        train_label_dims,
        Dim4::new(&[TRAIN_SAMPLES as u64, 1, 1, 1]),
        "Expected CIFAR train label dims [50000, 1, 1, 1], got {:?}",
        train_label_dims
    );
    let train_images = arrayfire::Array::new(&train_images_host, train_image_dims);
    let train_labels_onehot = one_hot_encode_host(&train_labels_host, TRAIN_SAMPLES, CLASSES);
    let train_labels = arrayfire::Array::new(
        &train_labels_onehot,
        Dim4::new(&[TRAIN_SAMPLES as u64, CLASSES as u64, 1, 1]),
    );

    let (test_images_host, test_image_dims, test_labels_host, test_label_dims) =
        parse_cifar10_batch_host("datasets/cifar/test_batch.bin", true);
    assert_eq!(
        test_image_dims,
        Dim4::new(&[TEST_SAMPLES as u64, 32, 32, 3]),
        "Expected CIFAR test image dims [10000, 32, 32, 3], got {:?}",
        test_image_dims
    );
    assert_eq!(
        test_label_dims,
        Dim4::new(&[TEST_SAMPLES as u64, 1, 1, 1]),
        "Expected CIFAR test label dims [10000, 1, 1, 1], got {:?}",
        test_label_dims
    );
    let test_images = arrayfire::Array::new(&test_images_host, test_image_dims);
    let test_labels_onehot = one_hot_encode_host(&test_labels_host, TEST_SAMPLES, CLASSES);
    let test_labels = arrayfire::Array::new(
        &test_labels_onehot,
        Dim4::new(&[TEST_SAMPLES as u64, CLASSES as u64, 1, 1]),
    );

    let train_dataset = Dataset::new(
        Node::leaf(train_images.clone(), false),
        Node::leaf(train_labels.clone(), false),
    );
    let test_dataset = Dataset::new(
        Node::leaf(test_images.clone(), false),
        Node::leaf(test_labels.clone(), false),
    );
    let train_loader = DataLoader::new(&train_dataset, BATCH_SIZE, true);
    let test_loader = DataLoader::new(&test_dataset, BATCH_SIZE, false);

    let model = Sequential::new(vec![
        Layer::from(Permute::nhwc_to_hwcn()),
        Layer::from(Conv2D::with_params(3, 32, 3, [1, 1], [1, 1], [1, 1])),
        Layer::from(DynReLU::new()),
        Layer::from(MaxPool::new(2, 2)),
        Layer::from(Conv2D::with_params(32, 64, 3, [1, 1], [1, 1], [1, 1])),
        Layer::from(DynReLU::new()),
        Layer::from(MaxPool::new(2, 2)),
        Layer::from(Flatten::new()),
        Layer::from(DynLinear::new(8 * 8 * 64, 256)),
        Layer::from(DynReLU::new()),
        Layer::from(DynLinear::new(256, CLASSES)),
    ]);
    let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
    let params = model.parameters();

    println!("training CIFAR-10 CNN\n");
    for epoch in 0..EPOCHS {
        let mut epoch_loss_sum = 0.0f32;
        let mut batches = 0usize;
        let mut train_hits = 0.0f32;
        let mut train_seen = 0usize;

        for (x_batch, y_batch) in &train_loader {
            let logits = model.forward(x_batch.clone());
            let loss = dyn_loss::cross_entropy(&logits, &y_batch);
            backward(loss.clone());

            let (loss_val, _) = arrayfire::sum_all(loss.borrow().tensor());
            epoch_loss_sum += loss_val;
            batches += 1;

            let pred = argmax(logits.borrow().tensor());
            let truth = argmax(y_batch.borrow().tensor());
            let correct = arrayfire::eq(&pred, &truth, false).cast::<f32>();
            let (batch_hits, _) = arrayfire::sum_all(&correct);
            train_hits += batch_hits;
            train_seen += pred.dims()[0] as usize;

            optimizer.step(&params);
            optimizer.zero_grad(&params);
        }

        let avg_loss = epoch_loss_sum / batches as f32;
        let train_acc = train_hits / train_seen as f32;
        println!(
            "Epoch {:>2} | Loss: {:.6} | Train Accuracy: {:.6}",
            epoch + 1,
            avg_loss,
            train_acc
        );
    }

    let mut train_hits = 0.0f32;
    let mut train_seen = 0usize;
    for (x_batch, y_batch) in &train_loader {
        let logits = model.forward(x_batch);
        let pred = argmax(logits.borrow().tensor());
        let truth = argmax(y_batch.borrow().tensor());
        let correct = arrayfire::eq(&pred, &truth, false).cast::<f32>();
        let (hits, _) = arrayfire::sum_all(&correct);
        train_hits += hits;
        train_seen += pred.dims()[0] as usize;
    }
    println!("Train Accuracy: {:.6}", train_hits / train_seen as f32);

    let mut test_hits = 0.0f32;
    let mut test_seen = 0usize;
    for (x_batch, y_batch) in &test_loader {
        let logits = model.forward(x_batch);
        let pred = argmax(logits.borrow().tensor());
        let truth = argmax(y_batch.borrow().tensor());
        let correct = arrayfire::eq(&pred, &truth, false).cast::<f32>();
        let (hits, _) = arrayfire::sum_all(&correct);
        test_hits += hits;
        test_seen += pred.dims()[0] as usize;
    }
    println!("Test Accuracy: {:.6}", test_hits / test_seen as f32);
}

pub fn typed_mlp_example() {
    mlp_example();
}
