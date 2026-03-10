use arrayfire::constant;
use std::collections::HashSet;
use std::rc::Rc;

use crate::af_tensor::NodeRef;

/// Perform full backpropagation starting from `root`
pub fn backward(root: NodeRef) {
    // Build topological order
    let mut topo = Vec::new();
    let mut visited = HashSet::new();
    build_topo(root.clone(), &mut topo, &mut visited);

    // Initialize root gradient (dL/dL = 1)
    {
        let mut r = root.borrow_mut();
        let dims = r.tensor().dims();
        r.set_grad(constant(1.0_f32, dims));
    }

    // Traverse graph in reverse order
    for node in topo.into_iter().rev() {
        node.borrow_mut().backward();
    }
}

/// Depth-first search to build topological order
fn build_topo(node: NodeRef, topo: &mut Vec<NodeRef>, visited: &mut HashSet<usize>) {
    let ptr = Rc::as_ptr(&node) as usize;

    if visited.contains(&ptr) {
        return;
    }

    visited.insert(ptr);

    let parents = node.borrow().parents().to_vec();

    for parent in parents {
        build_topo(parent, topo, visited);
    }

    topo.push(node);
}

#[cfg(test)]
mod tests {
    use crate::engine::backward;
    use crate::{af_tensor::Node, utils::assert_all_close};
    use arrayfire::{Array, Dim4, constant};

    #[test]
    fn test_add_backward() {
        let x_tensor = constant(2.0f32, Dim4::new(&[1, 1, 1, 1]));
        let x = Node::leaf(x_tensor, true);

        let y_tensor = constant(3.0f32, Dim4::new(&[1, 1, 1, 1]));
        let y = Node::leaf(y_tensor, true);

        let z = Node::add(&x, &y);

        backward(z);

        let x_grad = x.borrow().grad().unwrap().clone();
        let y_grad = y.borrow().grad().unwrap().clone();

        let expected = constant(1.0f32, Dim4::new(&[1, 1, 1, 1]));

        assert_all_close(&x_grad, &expected, 1e-5);
        assert_all_close(&y_grad, &expected, 1e-5);
    }

    #[test]
    fn test_chain_rule_mul_add() {
        let dims = Dim4::new(&[1, 1, 1, 1]);

        let x = Node::leaf(constant(2.0f32, dims), true);
        let y = Node::leaf(constant(3.0f32, dims), true);

        let a = Node::mul(&x, &y);

        let z = Node::add(&a, &y);

        backward(z);

        let x_grad = x.borrow().grad().unwrap().clone();
        let y_grad = y.borrow().grad().unwrap().clone();

        let expected_dx = constant(3.0f32, dims); // dz/dx = y
        let expected_dy = constant(3.0f32, dims); // dz/dy = x + 1

        assert_all_close(&x_grad, &expected_dx, 1e-5);
        assert_all_close(&y_grad, &expected_dy, 1e-5);
    }

    #[test]
    fn test_mul_add() {
        let dims = Dim4::new(&[1, 1, 1, 1]);

        let x = Node::leaf(constant(2.0f32, dims), true);
        let y = Node::leaf(constant(3.0f32, dims), true);

        let a = Node::mul(&x, &y);

        let b = Node::mul(&x, &x);

        let c = Node::add(&a, &b);

        let z = Node::mul(&c, &y);

        backward(z);

        let x_grad = x.borrow().grad().unwrap().clone();
        let y_grad = y.borrow().grad().unwrap().clone();

        let expected_dx = constant(21.0f32, dims);
        let expected_dy = constant(16.0f32, dims);

        assert_all_close(&x_grad, &expected_dx, 1e-5);
        assert_all_close(&y_grad, &expected_dy, 1e-5);
    }

    #[test]
    fn test_tensor_chain_rule_vector() {
        let dims = Dim4::new(&[4, 1, 1, 1]);

        let x_data = Array::new(&[1f32, 2., 3., 4.], dims);
        let x = Node::leaf(x_data, true);

        let y_data = Array::new(&[5f32, 6., 7., 8.], dims);
        let y = Node::leaf(y_data, true);

        let a = Node::mul(&x, &y);

        let z = Node::add(&a, &x);

        backward(z);

        let x_grad = x.borrow().grad().unwrap().clone();
        let y_grad = y.borrow().grad().unwrap().clone();

        let expected_dx = Array::new(&[6f32, 7., 8., 9.], dims);
        let expected_dy = Array::new(&[1f32, 2., 3., 4.], dims);

        assert_all_close(&x_grad, &expected_dx, 1e-5);
        assert_all_close(&y_grad, &expected_dy, 1e-5);
    }

    #[test]
    fn test_matmul_backward() {
        let dims = Dim4::new(&[2, 2, 1, 1]);

        let x_data = Array::new(&[1f32, 3., 2., 4.], dims);

        let w_data = Array::new(&[5f32, 7., 6., 8.], dims);

        let x = Node::leaf(x_data, true);
        let w = Node::leaf(w_data, true);

        // Z = X @ W
        let z = Node::matmul(&x, &w);

        // L = sum(Z)
        let loss = Node::sum(&z);

        backward(loss);

        let dx = x.borrow().grad().unwrap().clone();
        let dw = w.borrow().grad().unwrap().clone();

        let expected_dx = Array::new(&[11f32, 11., 15., 15.], dims);

        let expected_dw = Array::new(&[4f32, 6., 4., 6.], dims);

        assert_all_close(&dx, &expected_dx, 1e-5);
        assert_all_close(&dw, &expected_dw, 1e-5);
    }

    #[test]
    fn test_sigmoid() {
        let x_tensor = arrayfire::constant(0.0f32, arrayfire::Dim4::new(&[1, 1, 1, 1]));
        let x = crate::af_tensor::Node::leaf(x_tensor, true);
        let z = crate::af_tensor::Node::sigmoid(&x);
        crate::engine::backward(z);

        let expected_grad = arrayfire::constant(0.25f32, arrayfire::Dim4::new(&[1, 1, 1, 1]));
        assert_all_close(x.borrow().grad().unwrap(), &expected_grad, 1e-5);
    }

    #[test]
    fn test_relu() {
        let x_tensor = arrayfire::constant(-1.0f32, arrayfire::Dim4::new(&[1, 1, 1, 1]));
        let x = crate::af_tensor::Node::leaf(x_tensor, true);
        let z = crate::af_tensor::Node::relu(&x);
        crate::engine::backward(z);

        let expected_grad = arrayfire::constant(0.0f32, arrayfire::Dim4::new(&[1, 1, 1, 1]));
        assert_all_close(x.borrow().grad().unwrap(), &expected_grad, 1e-5);
    }

    #[test]
    fn test_tanh() {
        let x_tensor = arrayfire::constant(0.0f32, arrayfire::Dim4::new(&[1, 1, 1, 1]));
        let x = crate::af_tensor::Node::leaf(x_tensor, true);
        let z = crate::af_tensor::Node::tanh(&x);
        crate::engine::backward(z);

        let expected_grad = arrayfire::constant(1.0f32, arrayfire::Dim4::new(&[1, 1, 1, 1]));
        assert_all_close(x.borrow().grad().unwrap(), &expected_grad, 1e-5);
    }
}
