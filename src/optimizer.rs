use crate::af_tensor::{Node, NodeRef};

pub trait Optimizer {
    fn step(&mut self, parameters: &[NodeRef]);
    fn zero_grad(&mut self, parameters: &[NodeRef]);
}

pub struct SGD {
    pub lr: f32,
}

pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub m: Vec<arrayfire::Array<f32>>,
    pub v: Vec<arrayfire::Array<f32>>,
    pub t: usize,
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

impl Optimizer for Adam {
    fn step(&mut self, parameters: &[NodeRef]) {
        if self.m.len() != parameters.len() || self.v.len() != parameters.len() {
            self.m = parameters
                .iter()
                .map(|p| arrayfire::constant(0.0f32, p.borrow().tensor().dims()))
                .collect();
            self.v = parameters
                .iter()
                .map(|p| arrayfire::constant(0.0f32, p.borrow().tensor().dims()))
                .collect();
            self.t = 0;
        }

        self.t += 1;
        let t = self.t as i32;
        let one_minus_beta1_t = 1.0f32 - self.beta1.powi(t);
        let one_minus_beta2_t = 1.0f32 - self.beta2.powi(t);

        for (i, param) in parameters.iter().enumerate() {
            let (grad, tensor) = {
                let p = param.borrow();
                (p.grad().cloned(), p.tensor().clone())
            };
            if let Some(grad) = grad {
                let one_minus_beta1 = 1.0f32 - self.beta1;
                let one_minus_beta2 = 1.0f32 - self.beta2;

                self.m[i] = &self.m[i] * self.beta1 + &grad * one_minus_beta1;
                self.v[i] = &self.v[i] * self.beta2 + (&grad * &grad) * one_minus_beta2;

                let m_hat = &self.m[i] / one_minus_beta1_t;
                let v_hat = &self.v[i] / one_minus_beta2_t;
                let denom = arrayfire::sqrt(&v_hat) + self.epsilon;
                let update = (m_hat / denom) * self.lr;
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

impl SGD {
    pub fn new(lr: f32) -> Self {
        SGD { lr }
    }
}

impl Adam {
    pub fn new(lr: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Adam {
            lr,
            beta1,
            beta2,
            epsilon,
            m: vec![],
            v: vec![],
            t: 0,
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

    #[test]
    fn test_adam_single_step() {
        let mut optimizer = super::Adam {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m: vec![],
            v: vec![],
            t: 0,
        };
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
        let expected_value = arrayfire::constant(0.999_f32, arrayfire::Dim4::new(&[1, 1, 1, 1]));
        assert_all_close(&updated_value, &expected_value, 1e-6);
        assert_eq!(optimizer.t, 1);
        assert_eq!(optimizer.m.len(), 1);
        assert_eq!(optimizer.v.len(), 1);
    }
}
