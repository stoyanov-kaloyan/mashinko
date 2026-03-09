use mashinko::comp_graph::{Op, Value};
use mashinko::tensor::{Device, Tensor};

fn val(data: Vec<f64>, shape: Vec<usize>) -> Value<f64> {
    Value::new(Tensor::new(data, shape, Device::CPU, false).unwrap())
}

fn approx_eq(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < tol,
            "element {} differs: {} vs {} (tol {})",
            i,
            x,
            y,
            tol
        );
    }
}

#[test]
fn test_forward_add() {
    let a = val(vec![1.0, 2.0, 3.0], vec![3]);
    let b = val(vec![4.0, 5.0, 6.0], vec![3]);
    let c = a.add(&b).unwrap();
    assert_eq!(c.data().data(), &[5.0, 7.0, 9.0]);
    assert_eq!(c.op(), Op::Add);
}

#[test]
fn test_forward_mul() {
    let a = val(vec![2.0, 3.0], vec![2]);
    let b = val(vec![4.0, 5.0], vec![2]);
    let c = a.mul(&b).unwrap();
    assert_eq!(c.data().data(), &[8.0, 15.0]);
}

#[test]
fn test_forward_matmul() {
    // [1 2; 3 4] @ [5 6; 7 8] = [19 22; 43 50]
    let a = val(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = val(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.data().data(), &[19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_backward_add() {
    // z = a + b  =>  da = 1 , db = 1
    let a = val(vec![1.0, 2.0, 3.0], vec![3]);
    let b = val(vec![4.0, 5.0, 6.0], vec![3]);
    let z = a.add(&b).unwrap().sum();
    z.backward();

    assert_eq!(a.grad().data(), &[1.0, 1.0, 1.0]);
    assert_eq!(b.grad().data(), &[1.0, 1.0, 1.0]);
}

#[test]
fn test_backward_sub() {
    // z = sum(a - b)  =>  da = 1 , db = -1
    let a = val(vec![3.0, 2.0], vec![2]);
    let b = val(vec![1.0, 1.0], vec![2]);
    let z = a.sub(&b).unwrap().sum();
    z.backward();

    assert_eq!(a.grad().data(), &[1.0, 1.0]);
    assert_eq!(b.grad().data(), &[-1.0, -1.0]);
}

#[test]
fn test_backward_mul() {
    // z = sum(a * b)  =>  da = b, db = a
    let a = val(vec![2.0, 3.0], vec![2]);
    let b = val(vec![4.0, 5.0], vec![2]);
    let z = a.mul(&b).unwrap().sum();
    z.backward();

    assert_eq!(a.grad().data(), &[4.0, 5.0]); // da = b
    assert_eq!(b.grad().data(), &[2.0, 3.0]); // db = a
}

#[test]
fn test_backward_div() {
    // z = sum(a / b)   da = 1/b,  db = -a/b^2
    let a = val(vec![6.0, 8.0], vec![2]);
    let b = val(vec![3.0, 2.0], vec![2]);
    let z = a.div(&b).unwrap().sum();
    z.backward();

    // da = [1/3, 1/2]
    approx_eq(a.grad().data(), &[1.0 / 3.0, 1.0 / 2.0], 1e-10);
    // db = [-6/9, -8/4] = [-2/3, -2]
    approx_eq(b.grad().data(), &[-6.0 / 9.0, -8.0 / 4.0], 1e-10);
}

#[test]
fn test_backward_matmul() {
    // C = A @ B, L = sum(C)
    // dA = dC @ B^T,  dB = A^T @ dC   (dC is all-ones because L=sum)
    let a = val(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = val(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let c = a.matmul(&b).unwrap();
    let loss = c.sum();
    loss.backward();

    // dC = ones(2,2)
    // dA = ones @ B^T = [[5+6, 7+8],[5+6, 7+8]] = [[11,15],[11,15]]
    assert_eq!(a.grad().data(), &[11.0, 15.0, 11.0, 15.0]);
    // dB = A^T @ ones = [[1+3, 1+3],[2+4, 2+4]] = [[4,4],[6,6]]
    assert_eq!(b.grad().data(), &[4.0, 4.0, 6.0, 6.0]);
}

#[test]
fn test_backward_mean() {
    let a = val(vec![2.0, 4.0, 6.0, 8.0], vec![4]);
    let z = a.mean();
    z.backward();

    approx_eq(a.grad().data(), &[0.25, 0.25, 0.25, 0.25], 1e-10);
}

#[test]
fn test_backward_chain() {
    // L = sum((a + b) * b)  = sum(a*b + b*b)
    // dL/da = b,  dL/db = a + 2b
    let a = val(vec![1.0, 2.0], vec![2]);
    let b = val(vec![3.0, 4.0], vec![2]);
    let c = a.add(&b).unwrap(); // a + b
    let d = c.mul(&b).unwrap(); // (a+b) * b
    let loss = d.sum();
    loss.backward();

    assert_eq!(a.grad().data(), &[3.0, 4.0]); // b
    assert_eq!(b.grad().data(), &[7.0, 10.0]); // a + 2b
}

#[test]
fn test_backward_transpose() {
    // L = sum(a^T)  =>  da = ones
    let a = val(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let t = a.transpose().unwrap();
    let loss = t.sum();
    loss.backward();

    assert_eq!(a.grad().data(), &[1.0; 6]);
}

#[test]
fn test_backward_neg() {
    let a = val(vec![1.0, 2.0], vec![2]);
    let z = a.neg().sum();
    z.backward();

    assert_eq!(a.grad().data(), &[-1.0, -1.0]);
}
