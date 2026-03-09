use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::rc::Rc;

use crate::tensor::{Device, Tensor, TensorElement};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Leaf,
    Add,
    Sub,
    Mul,
    Div,
    Matmul,
    Sum,
    Mean,
    Transpose,
    Neg,
}

struct ValueInner<T: TensorElement> {
    data: Tensor<T>,
    grad: Tensor<T>,
    op: Op,
    parents: Vec<Value<T>>,
}

pub struct Value<T: TensorElement> {
    inner: Rc<RefCell<ValueInner<T>>>,
}

impl<T: TensorElement> Clone for Value<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Rc::clone(&self.inner),
        }
    }
}

impl<T: TensorElement> fmt::Debug for Value<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.inner.borrow();
        f.debug_struct("Value")
            .field("shape", &inner.data.shape())
            .field("op", &inner.op)
            .finish()
    }
}

impl<T: TensorElement> Value<T> {
    pub fn new(tensor: Tensor<T>) -> Self {
        let grad = tensor.zeros_like();
        Self {
            inner: Rc::new(RefCell::new(ValueInner {
                data: tensor,
                grad,
                op: Op::Leaf,
                parents: vec![],
            })),
        }
    }

    fn from_op(tensor: Tensor<T>, op: Op, parents: Vec<Value<T>>) -> Self {
        let grad = tensor.zeros_like();
        Self {
            inner: Rc::new(RefCell::new(ValueInner {
                data: tensor,
                grad,
                op,
                parents,
            })),
        }
    }

    pub fn data(&self) -> Tensor<T> {
        self.inner.borrow().data.clone()
    }

    pub fn grad(&self) -> Tensor<T> {
        self.inner.borrow().grad.clone()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.inner.borrow().data.shape().to_vec()
    }

    pub fn op(&self) -> Op {
        self.inner.borrow().op
    }

    pub fn add(&self, other: &Self) -> Result<Self, String> {
        let result = self.inner.borrow().data.add(&other.inner.borrow().data)?;
        Ok(Self::from_op(
            result,
            Op::Add,
            vec![self.clone(), other.clone()],
        ))
    }

    pub fn sub(&self, other: &Self) -> Result<Self, String> {
        let result = self.inner.borrow().data.sub(&other.inner.borrow().data)?;
        Ok(Self::from_op(
            result,
            Op::Sub,
            vec![self.clone(), other.clone()],
        ))
    }

    pub fn mul(&self, other: &Self) -> Result<Self, String> {
        let result = self.inner.borrow().data.mul(&other.inner.borrow().data)?;
        Ok(Self::from_op(
            result,
            Op::Mul,
            vec![self.clone(), other.clone()],
        ))
    }

    pub fn div(&self, other: &Self) -> Result<Self, String> {
        let result = self.inner.borrow().data.div(&other.inner.borrow().data)?;
        Ok(Self::from_op(
            result,
            Op::Div,
            vec![self.clone(), other.clone()],
        ))
    }

    pub fn matmul(&self, other: &Self) -> Result<Self, String> {
        let result = self
            .inner
            .borrow()
            .data
            .matmul(&other.inner.borrow().data)?;
        Ok(Self::from_op(
            result,
            Op::Matmul,
            vec![self.clone(), other.clone()],
        ))
    }

    pub fn sum(&self) -> Self {
        let result = self.inner.borrow().data.sum();
        Self::from_op(result, Op::Sum, vec![self.clone()])
    }

    pub fn mean(&self) -> Self {
        let result = self.inner.borrow().data.mean();
        Self::from_op(result, Op::Mean, vec![self.clone()])
    }

    pub fn transpose(&self) -> Result<Self, String> {
        let result = self.inner.borrow().data.transpose()?;
        Ok(Self::from_op(result, Op::Transpose, vec![self.clone()]))
    }

    pub fn neg(&self) -> Self {
        let result = self.inner.borrow().data.neg();
        Self::from_op(result, Op::Neg, vec![self.clone()])
    }
}

impl<T: TensorElement> Value<T> {
    /// Run backpropagation starting from this node (typically a scalar loss).
    /// Seeds `self.grad` with ones, then walks the graph in reverse
    /// topological order.
    pub fn backward(&self) {
        // Seed the output gradient with ones (dL/dL = 1).
        {
            let mut inner = self.inner.borrow_mut();
            inner.grad = inner.data.ones_like();
        }

        // Build reverse topological order.
        let topo = self.topo_sort();

        for node in topo.iter() {
            // Read everything we need from the current node.
            let (op, parents, grad, _data) = {
                let inner = node.inner.borrow();
                (
                    inner.op,
                    inner.parents.clone(),
                    inner.grad.clone(),
                    inner.data.clone(),
                )
            };

            match op {
                Op::Leaf => {}

                // z = a + b  =>  da += dz,  db += dz
                Op::Add => {
                    Self::acc_grad(&parents[0], &grad);
                    Self::acc_grad(&parents[1], &grad);
                }

                // z = a - b  =>  da += dz,  db += -dz
                Op::Sub => {
                    Self::acc_grad(&parents[0], &grad);
                    let neg_grad = grad.neg();
                    Self::acc_grad(&parents[1], &neg_grad);
                }

                // z = a * b  (element-wise)  =>  da += dz * b,  db += dz * a
                Op::Mul => {
                    let a_data = parents[0].inner.borrow().data.clone();
                    let b_data = parents[1].inner.borrow().data.clone();
                    let da = grad.mul(&b_data).expect("mul backward shape");
                    let db = grad.mul(&a_data).expect("mul backward shape");
                    Self::acc_grad(&parents[0], &da);
                    Self::acc_grad(&parents[1], &db);
                }

                // z = a / b  =>  da += dz / b,  db += -dz * a / (b * b)
                Op::Div => {
                    let a_data = parents[0].inner.borrow().data.clone();
                    let b_data = parents[1].inner.borrow().data.clone();
                    let da = grad.div(&b_data).expect("div backward shape");
                    let b_sq = b_data.mul(&b_data).expect("div backward shape");
                    let db = grad
                        .mul(&a_data)
                        .expect("div backward shape")
                        .div(&b_sq)
                        .expect("div backward shape")
                        .neg();
                    Self::acc_grad(&parents[0], &da);
                    Self::acc_grad(&parents[1], &db);
                }

                // z = a @ b  =>  da += dz @ b^T,  db += a^T @ dz
                Op::Matmul => {
                    let a_data = parents[0].inner.borrow().data.clone();
                    let b_data = parents[1].inner.borrow().data.clone();
                    let b_t = b_data.transpose().expect("matmul backward transpose");
                    let a_t = a_data.transpose().expect("matmul backward transpose");
                    let da = grad.matmul(&b_t).expect("matmul backward shape");
                    let db = a_t.matmul(&grad).expect("matmul backward shape");
                    Self::acc_grad(&parents[0], &da);
                    Self::acc_grad(&parents[1], &db);
                }

                // z = sum(a)  =>  da += broadcast(dz)
                Op::Sum => {
                    let parent_shape = parents[0].inner.borrow().data.shape().to_vec();
                    let size: usize = parent_shape.iter().product();
                    // dz is a scalar; broadcast to parent shape
                    let dz_val = grad.data()[0].clone();
                    let expanded =
                        Tensor::new(vec![dz_val; size], parent_shape, Device::CPU, false)
                            .expect("sum backward");
                    Self::acc_grad(&parents[0], &expanded);
                }

                // z = mean(a)  =>  da += broadcast(dz / n)
                Op::Mean => {
                    let parent_shape = parents[0].inner.borrow().data.shape().to_vec();
                    let size: usize = parent_shape.iter().product();
                    let scale = T::from_f64(1.0 / size as f64);
                    let dz_val = grad.data()[0].clone();
                    let scaled = T::default() + dz_val * scale;
                    let expanded =
                        Tensor::new(vec![scaled; size], parent_shape, Device::CPU, false)
                            .expect("mean backward");
                    Self::acc_grad(&parents[0], &expanded);
                }

                // z = a^T  =>  da += dz^T
                Op::Transpose => {
                    let da = grad.transpose().expect("transpose backward");
                    Self::acc_grad(&parents[0], &da);
                }

                // z = -a  =>  da += -dz
                Op::Neg => {
                    Self::acc_grad(&parents[0], &grad.neg());
                }
            }
        }
    }

    /// Accumulate `grad_contribution` into `target.grad`.
    fn acc_grad(target: &Value<T>, grad_contribution: &Tensor<T>) {
        let mut inner = target.inner.borrow_mut();
        inner.grad = inner
            .grad
            .add(grad_contribution)
            .expect("grad accumulation shape mismatch");
    }

    /// Build a reverse topological ordering starting from `self`.
    fn topo_sort(&self) -> Vec<Value<T>> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        self.topo_visit(&mut visited, &mut order);
        order.reverse();
        order
    }

    fn topo_visit(&self, visited: &mut HashSet<usize>, order: &mut Vec<Value<T>>) {
        let id = Rc::as_ptr(&self.inner) as usize;
        if visited.contains(&id) {
            return;
        }
        visited.insert(id);
        let parents = self.inner.borrow().parents.clone();
        for p in &parents {
            p.topo_visit(visited, order);
        }
        order.push(self.clone());
    }
}
