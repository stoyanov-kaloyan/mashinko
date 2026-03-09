use mashinko::tensor::{Device, Tensor};

#[test]
fn test_tensor_add() {
    let a = Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU, true).unwrap();
    let b = Tensor::new(vec![5.0_f32, 6.0, 7.0, 8.0], vec![2, 2], Device::CPU, false).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.data(), &[6.0_f32, 8.0, 10.0, 12.0]);
}

#[test]
fn test_tensor_matmul() {
    let a = Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU, true).unwrap();
    let b = Tensor::new(vec![5.0_f32, 6.0, 7.0, 8.0], vec![2, 2], Device::CPU, false).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.data(), &[19.0_f32, 22.0, 43.0, 50.0]);
}

#[test]
fn test_tensor_dot() {
    let a = Tensor::new(vec![1.0_f32, 2.0, 3.0], vec![3], Device::CPU, true).unwrap();
    let b = Tensor::new(vec![4.0_f32, 5.0, 6.0], vec![3], Device::CPU, false).unwrap();
    let c = a.dot(&b).unwrap();
    assert_eq!(c.data(), &[32.0_f32]);
}

#[test]
fn test_tensor_transpose() {
    let a = Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU, true).unwrap();
    let b = a.transpose().unwrap();
    assert_eq!(b.data(), &[1.0_f32, 3.0, 2.0, 4.0]);
}

#[test]
fn test_tensor_throws_on_shape_mismatch() {
    let a = Tensor::new(vec![1.0_f32, 2.0, 3.0], vec![3], Device::CPU, true).unwrap();
    let b = Tensor::new(vec![4.0_f32, 5.0], vec![2], Device::CPU, false).unwrap();
    assert!(a.add(&b).is_err());
    assert!(a.matmul(&b).is_err());
    assert!(a.dot(&b).is_err());
}
