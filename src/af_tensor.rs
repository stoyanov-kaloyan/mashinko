use arrayfire::{Array, MatProp, matmul};
use std::cell::RefCell;
use std::rc::Rc;

pub type NodeRef = Rc<RefCell<Node>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    Add,
    Sub,
    Mul,
    MatMul,
    Dot,
    Transpose,
    Sum,
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
                let parent_dims = self.parents[0].borrow().tensor.dims();
                let ones = arrayfire::constant(1.0f32, parent_dims);
                // dz is a scalar (1x1); multiply broadcasts it to parent shape
                acc_grad(&self.parents[0], &(&ones * &dz));
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
