use crate::af_tensor::{Node, NodeRef};

pub trait Optimizer {
    fn step(&mut self, parameters: &[NodeRef]);
    fn zero_grad(&mut self, parameters: &[NodeRef]);
}

pub struct SGD {
    pub lr: f32,
}

impl Optimizer for SGD {
    fn step(&mut self, parameters: &[NodeRef]) {
        for param in parameters {
            let (grad, tensor) = {
                let p = param.borrow();
                (p.grad().cloned(), p.tensor().clone())
            };
            if let Some(grad) = grad {
                let update = grad * self.lr;
                let new_value = tensor - update;
                param.borrow_mut().set_tensor(new_value);
            }
        }
    }

    fn zero_grad(&mut self, parameters: &[NodeRef]) {
        for param in parameters {
            let zero_tensor = Node::constant(0.0_f32, param.borrow().tensor().dims());
            let mut p = param.borrow_mut();
            p.set_grad(zero_tensor.borrow().tensor().clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Optimizer;
    use crate::utils::assert_all_close;

    #[test]
    fn test_sgd() {
        let mut optimizer = super::SGD { lr: 0.1 };
        let param = super::Node::leaf(
            arrayfire::constant(1.0_f32, arrayfire::Dim4::new(&[1, 1, 1, 1])),
            true,
        );
        param.borrow_mut().set_grad(
            super::Node::constant(0.5_f32, arrayfire::Dim4::new(&[1, 1, 1, 1]))
                .borrow()
                .tensor()
                .clone(),
        );
        optimizer.step(&[param.clone()]);
        let updated_value = param.borrow().tensor().clone();
        let expected_value = arrayfire::constant(0.95_f32, arrayfire::Dim4::new(&[1, 1, 1, 1]));
        assert_all_close(&updated_value, &expected_value, 1e-5);
    }

    #[test]
    fn test_zero_grad() {
        let mut optimizer = super::SGD { lr: 0.1 };
        let param = super::Node::leaf(
            arrayfire::constant(1.0_f32, arrayfire::Dim4::new(&[1, 1, 1, 1])),
            true,
        );
        param.borrow_mut().set_grad(
            super::Node::constant(0.5_f32, arrayfire::Dim4::new(&[1, 1, 1, 1]))
                .borrow()
                .tensor()
                .clone(),
        );
        optimizer.zero_grad(&[param.clone()]);
        let zero_grad = param.borrow().grad().unwrap().clone();
        let expected_zero_grad = arrayfire::constant(0.0_f32, arrayfire::Dim4::new(&[1, 1, 1, 1]));
        assert_all_close(&zero_grad, &expected_zero_grad, 1e-5);
    }
}
