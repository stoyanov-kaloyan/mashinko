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
