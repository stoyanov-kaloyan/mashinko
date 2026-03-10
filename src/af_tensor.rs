use arrayfire::{Array, MatProp, matmul};
use std::cell::RefCell;
use std::rc::Rc;

pub type NodeRef = Rc<RefCell<Node>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Dot,
    Transpose,
    Sum,
    Mean,
    ReLU,
    Sigmoid,
    Tanh,
    Clamp,
    Log,
    Neg,
}

pub struct Node {
    tensor: Array<f32>,
    grad: Option<Array<f32>>,
    requires_grad: bool,
    op: Option<Operation>,
    parents: Vec<NodeRef>,
}

impl Node {
    pub fn leaf(tensor: Array<f32>, requires_grad: bool) -> NodeRef {
        Rc::new(RefCell::new(Self {
            tensor,
            grad: None,
            requires_grad,
            op: None,
            parents: vec![],
        }))
    }

    pub fn from_op(
        tensor: Array<f32>,
        op: Operation,
        parents: Vec<NodeRef>,
        requires_grad: bool,
    ) -> NodeRef {
        Rc::new(RefCell::new(Self {
            tensor,
            grad: None,
            requires_grad,
            op: Some(op),
            parents,
        }))
    }

    pub fn set_tensor(&mut self, tensor: Array<f32>) {
        self.tensor = tensor;
    }

    pub fn constant(value: f32, dims: arrayfire::Dim4) -> NodeRef {
        let tensor = arrayfire::constant(value, dims);
        Self::leaf(tensor, false)
    }

    // operation constructors

    pub fn add(a: &NodeRef, b: &NodeRef) -> NodeRef {
        let tensor = a.borrow().tensor() + b.borrow().tensor();
        Self::from_op(tensor, Operation::Add, vec![a.clone(), b.clone()], true)
    }

    pub fn sub(a: &NodeRef, b: &NodeRef) -> NodeRef {
        let tensor = a.borrow().tensor() - b.borrow().tensor();
        Self::from_op(tensor, Operation::Sub, vec![a.clone(), b.clone()], true)
    }

    pub fn mul(a: &NodeRef, b: &NodeRef) -> NodeRef {
        let tensor = a.borrow().tensor() * b.borrow().tensor();
        Self::from_op(tensor, Operation::Mul, vec![a.clone(), b.clone()], true)
    }

    pub fn div(a: &NodeRef, b: &NodeRef) -> NodeRef {
        let tensor = a.borrow().tensor() / b.borrow().tensor();
        Self::from_op(tensor, Operation::Div, vec![a.clone(), b.clone()], true)
    }

    pub fn neg(a: &NodeRef) -> NodeRef {
        let tensor = -a.borrow().tensor().clone();
        Self::from_op(tensor, Operation::Neg, vec![a.clone()], true)
    }

    pub fn matmul(a: &NodeRef, b: &NodeRef) -> NodeRef {
        let tensor = matmul(
            a.borrow().tensor(),
            b.borrow().tensor(),
            MatProp::NONE,
            MatProp::NONE,
        );
        Self::from_op(tensor, Operation::MatMul, vec![a.clone(), b.clone()], true)
    }

    /// reducing sum
    pub fn sum(a: &NodeRef) -> NodeRef {
        let tensor = arrayfire::sum_all(a.borrow().tensor()).0;
        let tensor_array = Array::new(&[tensor], arrayfire::Dim4::new(&[1, 1, 1, 1]));
        Self::from_op(tensor_array, Operation::Sum, vec![a.clone()], true)
    }

    /// Reducing mean
    pub fn mean(a: &NodeRef) -> NodeRef {
        let tensor = arrayfire::mean_all(a.borrow().tensor()).0;
        let tensor = tensor as f32;
        let tensor_array = Array::new(&[tensor], arrayfire::Dim4::new(&[1, 1, 1, 1]));
        Self::from_op(tensor_array, Operation::Mean, vec![a.clone()], true)
    }

    pub fn clamp(a: &NodeRef, min: f32, max: f32) -> NodeRef {
        let tensor = arrayfire::clamp(a.borrow().tensor(), &min, &max, true);
        Self::from_op(tensor, Operation::Clamp, vec![a.clone()], true)
    }

    pub fn log(a: &NodeRef) -> NodeRef {
        let tensor = arrayfire::log(&a.borrow().tensor());
        Self::from_op(tensor, Operation::Log, vec![a.clone()], true)
    }

    // activation functions

    pub fn sigmoid(a: &NodeRef) -> NodeRef {
        let neg_a = -a.borrow().tensor().clone();
        let tensor = 1.0f32 / (1.0f32 + arrayfire::exp(&neg_a));
        Self::from_op(tensor, Operation::Sigmoid, vec![a.clone()], true)
    }

    /// relu - max(0, x)
    pub fn relu(a: &NodeRef) -> NodeRef {
        let a_tensor = a.borrow().tensor().clone();
        let zero = arrayfire::constant(0.0f32, a_tensor.dims());
        let mask = arrayfire::gt(&a_tensor, &zero, false);
        let tensor = arrayfire::select(&a_tensor, &mask, &zero);
        Self::from_op(tensor, Operation::ReLU, vec![a.clone()], true)
    }

    pub fn tanh(a: &NodeRef) -> NodeRef {
        let tensor = arrayfire::tanh(&a.borrow().tensor());
        Self::from_op(tensor, Operation::Tanh, vec![a.clone()], true)
    }

    pub fn tensor(&self) -> &Array<f32> {
        &self.tensor
    }

    pub fn grad(&self) -> Option<&Array<f32>> {
        self.grad.as_ref()
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub fn op(&self) -> Option<Operation> {
        self.op
    }

    pub fn parents(&self) -> &[NodeRef] {
        &self.parents
    }

    pub fn set_grad(&mut self, grad: Array<f32>) {
        self.grad = Some(grad);
    }

    /// Run a single backward step for this node
    /// This is where we compute gradients
    pub fn backward(&mut self) {
        let dz = match &self.grad {
            Some(g) => g.clone(),
            None => return,
        };

        match self.op {
            Some(Operation::Add) => {
                // z = a + b  ->  da = dz, db = dz
                for parent in &self.parents {
                    acc_grad(parent, &dz);
                }
            }
            Some(Operation::Sub) => {
                // z = a - b  ->  da = dz, db = -dz
                acc_grad(&self.parents[0], &dz);
                acc_grad(&self.parents[1], &(-dz));
            }
            Some(Operation::Mul) => {
                // z = a * b  ->  da = dz * b, db = dz * a
                let a_data = self.parents[0].borrow().tensor.clone();
                let b_data = self.parents[1].borrow().tensor.clone();
                acc_grad(&self.parents[0], &(&dz * &b_data));
                acc_grad(&self.parents[1], &(&dz * &a_data));
            }
            Some(Operation::Div) => {
                // z = a / b  ->  da = dz / b, db = -dz * a / b^2
                let a_data = self.parents[0].borrow().tensor.clone();
                let b_data = self.parents[1].borrow().tensor.clone();
                let b_squared = &b_data * &b_data;
                acc_grad(&self.parents[0], &(&dz / &b_data));
                acc_grad(&self.parents[1], &(-dz * &a_data / &b_squared));
            }
            Some(Operation::MatMul) => {
                // z = a @ b  ->  da = dz @ b^T, db = a^T @ dz
                let a_data = self.parents[0].borrow().tensor.clone();
                let b_data = self.parents[1].borrow().tensor.clone();
                acc_grad(
                    &self.parents[0],
                    &matmul(&dz, &b_data, MatProp::NONE, MatProp::TRANS),
                );
                acc_grad(
                    &self.parents[1],
                    &matmul(&a_data, &dz, MatProp::TRANS, MatProp::NONE),
                );
            }
            Some(Operation::Dot) => {
                // z = dot(a, b)  ->  da = dz * b, db = dz * a
                let a_data = self.parents[0].borrow().tensor.clone();
                let b_data = self.parents[1].borrow().tensor.clone();
                acc_grad(&self.parents[0], &(&dz * &b_data));
                acc_grad(&self.parents[1], &(&dz * &a_data));
            }
            Some(Operation::Sum) => {
                // z = sum(a)  ->  da = dz broadcast to a shape
                let parent_dims = self.parents[0].borrow().tensor.dims();
                let ones = arrayfire::constant(1.0f32, parent_dims);
                // dz is a scalar (1x1); multiply broadcasts it to parent shape
                acc_grad(&self.parents[0], &(&ones * &dz));
            }
            Some(Operation::Mean) => {
                // z = mean(a)  ->  da = dz broadcast to a shape / num_elements
                let parent_dims = self.parents[0].borrow().tensor.dims();
                let num_elements =
                    (parent_dims[0] * parent_dims[1] * parent_dims[2] * parent_dims[3]) as f32;
                let ones = arrayfire::constant(1.0f32 / num_elements, parent_dims);
                acc_grad(&self.parents[0], &(&ones * &dz));
            }
            Some(Operation::ReLU) => {
                // relu(x) = max(0, x)  ->  da = dz * (x > 0)
                let a_data = self.parents[0].borrow().tensor.clone();
                let zero = arrayfire::constant(0.0f32, a_data.dims());
                let mask = arrayfire::gt(&a_data, &zero, false);
                let mask_f32 = mask.cast::<f32>();
                acc_grad(&self.parents[0], &(&dz * &mask_f32));
            }
            Some(Operation::Sigmoid) => {
                // s(x) ->  da = dz * s(x) * (1 - s(x))
                let sig = &self.tensor;
                let one = arrayfire::constant(1.0f32, sig.dims());
                let da = &dz * sig * &(one - sig);
                acc_grad(&self.parents[0], &da);
            }
            Some(Operation::Tanh) => {
                // tanh(x) ->  da = dz * (1 - tanh(x)^2)
                let tanh_x = &self.tensor;
                let one = arrayfire::constant(1.0f32, tanh_x.dims());
                let da = &dz * &(one - tanh_x * tanh_x);
                acc_grad(&self.parents[0], &da);
            }
            Some(Operation::Log) => {
                // log(x) ->  da = dz / x
                let x = &self.parents[0].borrow().tensor;
                acc_grad(&self.parents[0], &(&dz / x));
            }
            Some(Operation::Clamp) => {
                // clamp(x, min, max) -> da = dz where min < x < max else 0
                let x = &self.parents[0].borrow().tensor;
                let min_mask = arrayfire::gt(x, &arrayfire::constant(0.0f32, x.dims()), false);
                let max_mask = arrayfire::lt(x, &arrayfire::constant(1.0f32, x.dims()), false);
                let mask = arrayfire::mul(&min_mask, &max_mask, false);
                let mask_f32 = mask.cast::<f32>();
                acc_grad(&self.parents[0], &(&dz * &mask_f32));
            }
            Some(Operation::Neg) => {
                // z = -a  ->  da = -dz
                acc_grad(&self.parents[0], &(-dz));
            }
            Some(op) => {
                unimplemented!("backward not implemented for {:?}", op)
            }

            None => {}
        }
    }
}

/// Accumulate gradient into a node
fn acc_grad(node: &NodeRef, contribution: &Array<f32>) {
    let mut node = node.borrow_mut();

    if !node.requires_grad {
        return;
    }

    node.grad = Some(match node.grad.take() {
        Some(existing) => existing + contribution,
        None => contribution.clone(),
    });
}
