use arrayfire::{Array, abs, max_all};

pub fn assert_all_close(a: &Array<f32>, b: &Array<f32>, tol: f32) {
    let diff = abs(&(a - b));
    let (max_err, _) = max_all(&diff);
    assert!(
        max_err < tol as f32,
        "arrays differ: max error {} > tolerance {}",
        max_err,
        tol
    );
}
