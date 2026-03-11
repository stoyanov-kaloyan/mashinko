use arrayfire::{Array, MatProp, matmul};
use std::cell::RefCell;
use std::rc::Rc;

// TODO: figure out how to do batching

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
    Conv2D {
        stride: [u64; 2],
        padding: [u64; 2],
        dilation: [u64; 2],
    },
    MaxPool {
        pool_size: u64,
        stride: u64,
    },
    Reshape {
        original_dims: [u64; 4],
    },
    Reorder {
        perm: [u64; 4],
    },
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

    pub fn transpose(a: &NodeRef) -> NodeRef {
        let tensor = arrayfire::transpose(&a.borrow().tensor(), false);
        Self::from_op(tensor, Operation::Transpose, vec![a.clone()], true)
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

    pub fn conv2d(
        input: &NodeRef,
        weight: &NodeRef,
        stride: [u64; 2],
        padding: [u64; 2],
        dilation: [u64; 2],
    ) -> NodeRef {
        let input_t = input.borrow().tensor().clone();
        let weight_t = weight.borrow().tensor().clone();

        let in_dims = input_t.dims();
        let w_dims = weight_t.dims();
        let h = in_dims[0];
        let w = in_dims[1];
        let n = in_dims[3];
        let kh = w_dims[0];
        let kw = w_dims[1];
        let c_in = w_dims[2];
        let c_out = w_dims[3];

        // eprintln!(
        //     "conv2d: input=({},{},{},{}), weight=({},{},{},{}), stride={:?}, padding={:?}",
        //     h, w, in_dims[2], n, kh, kw, c_in, c_out, stride, padding
        // );
        let oh = (h + 2 * padding[0] - kh) / stride[0] + 1;
        let ow = (w + 2 * padding[1] - kw) / stride[1] + 1;

        // im2col: (H,W,C_in,N) → (kH*kW, oH*oW, C_in, N)
        let cols = arrayfire::unwrap(
            &input_t,
            kh as i64,
            kw as i64,
            stride[0] as i64,
            stride[1] as i64,
            padding[0] as i64,
            padding[1] as i64,
            true,
        );

        // Rearrange: (kH*kW, oH*oW, C_in, N) → (kH*kW, C_in, oH*oW, N)
        let cols_perm = arrayfire::reorder_v2(&cols, 0, 2, Some(vec![1, 3]));

        // Reshape: (kH*kW*C_in, oH*oW, 1, N) for batched matmul
        let cols_2d = arrayfire::moddims(
            &cols_perm,
            arrayfire::Dim4::new(&[kh * kw * c_in, oh * ow, 1, n]),
        );

        // Reshape filter: (kH*kW*C_in, C_out)
        let filter_2d = arrayfire::moddims(
            &weight_t,
            arrayfire::Dim4::new(&[kh * kw * c_in, c_out, 1, 1]),
        );

        // Batched matmul: filter^T @ cols → (C_out, oH*oW, 1, N)
        let out_2d = matmul(&filter_2d, &cols_2d, MatProp::TRANS, MatProp::NONE);

        // Reshape: (C_out, oH, oW, N)
        let out_3d = arrayfire::moddims(&out_2d, arrayfire::Dim4::new(&[c_out, oh, ow, n]));

        // Reorder to standard layout: (oH, oW, C_out, N)
        let output = arrayfire::reorder_v2(&out_3d, 1, 2, Some(vec![0, 3]));

        Self::from_op(
            output,
            Operation::Conv2D {
                stride,
                padding,
                dilation,
            },
            vec![input.clone(), weight.clone()],
            true,
        )
    }

    pub fn max_pool(input: &NodeRef, pool_size: u64, stride: u64) -> NodeRef {
        let input_tensor = input.borrow().tensor().clone();
        let dims = input_tensor.dims();
        let h = dims[0];
        let w = dims[1];
        let c = dims[2];
        let n = dims[3];

        let oh = (h - pool_size) / stride + 1;
        let ow = (w - pool_size) / stride + 1;

        // Unwrap into patches: (pool_size*pool_size, oh*ow, c, n)
        let unwrapped = arrayfire::unwrap(
            &input_tensor,
            pool_size as i64,
            pool_size as i64,
            stride as i64,
            stride as i64,
            0,
            0,
            true,
        );

        // Max along dim 0
        let (max_vals, _) = arrayfire::imax(&unwrapped, 0);

        // Reshape to (oh, ow, c, n)
        let output = arrayfire::moddims(&max_vals, arrayfire::Dim4::new(&[oh, ow, c, n]));

        Self::from_op(
            output,
            Operation::MaxPool { pool_size, stride },
            vec![input.clone()],
            true,
        )
    }

    pub fn reshape(input: &NodeRef, new_dims: arrayfire::Dim4) -> NodeRef {
        let original_dims = input.borrow().tensor().dims();
        let tensor = arrayfire::moddims(input.borrow().tensor(), new_dims);
        Self::from_op(
            tensor,
            Operation::Reshape {
                original_dims: [
                    original_dims[0],
                    original_dims[1],
                    original_dims[2],
                    original_dims[3],
                ],
            },
            vec![input.clone()],
            true,
        )
    }

    pub fn reorder(input: &NodeRef, perm: [u64; 4]) -> NodeRef {
        let tensor = arrayfire::reorder_v2(
            input.borrow().tensor(),
            perm[0],
            perm[1],
            Some(vec![perm[2], perm[3]]),
        );
        Self::from_op(
            tensor,
            Operation::Reorder { perm },
            vec![input.clone()],
            true,
        )
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
                let x = self.parents[0].borrow().tensor.clone();
                acc_grad(&self.parents[0], &(&dz / &x));
            }
            Some(Operation::Clamp) => {
                // clamp(x, min, max) -> da = dz where min < x < max else 0
                let x = self.parents[0].borrow().tensor.clone();
                let min_mask = arrayfire::gt(&x, &arrayfire::constant(0.0f32, x.dims()), false);
                let max_mask = arrayfire::lt(&x, &arrayfire::constant(1.0f32, x.dims()), false);
                let mask = arrayfire::mul(&min_mask, &max_mask, false);
                let mask_f32 = mask.cast::<f32>();
                acc_grad(&self.parents[0], &(&dz * &mask_f32));
            }
            Some(Operation::Neg) => {
                // z = -a  ->  da = -dz
                acc_grad(&self.parents[0], &(-dz));
            }
            Some(Operation::Transpose) => {
                // z = a^T  ->  da = dz^T
                let dz_data = self.tensor.clone();
                acc_grad(&self.parents[0], &arrayfire::transpose(&dz_data, false));
            }
            // written by ai so i shoul look again here
            Some(Operation::Conv2D {
                stride, padding, ..
            }) => {
                let input_data = self.parents[0].borrow().tensor.clone();
                let weight_data = self.parents[1].borrow().tensor.clone();

                let in_dims = input_data.dims();
                let w_dims = weight_data.dims();
                let h = in_dims[0];
                let w_val = in_dims[1];
                let n = in_dims[3];
                let kh = w_dims[0];
                let kw = w_dims[1];
                let c_in = w_dims[2];
                let c_out = w_dims[3];

                let out_dims = self.tensor.dims();
                let oh = out_dims[0];
                let ow = out_dims[1];

                // Recompute im2col columns
                let cols = arrayfire::unwrap(
                    &input_data,
                    kh as i64,
                    kw as i64,
                    stride[0] as i64,
                    stride[1] as i64,
                    padding[0] as i64,
                    padding[1] as i64,
                    true,
                );
                let cols_perm = arrayfire::reorder_v2(&cols, 0, 2, Some(vec![1, 3]));
                let cols_2d = arrayfire::moddims(
                    &cols_perm,
                    arrayfire::Dim4::new(&[kh * kw * c_in, oh * ow, 1, n]),
                );

                let filter_2d = arrayfire::moddims(
                    &weight_data,
                    arrayfire::Dim4::new(&[kh * kw * c_in, c_out, 1, 1]),
                );

                // dz: (oH, oW, C_out, N) → (C_out, oH, oW, N) → (C_out, oH*oW, 1, N)
                let dz_perm = arrayfire::reorder_v2(&dz, 2, 0, Some(vec![1, 3]));
                let dz_2d =
                    arrayfire::moddims(&dz_perm, arrayfire::Dim4::new(&[c_out, oh * ow, 1, n]));

                // Gradient w.r.t. filter: sum_n dz_n @ cols_n^T → (C_out, kH*kW*C_in)
                let d_filter_batch = matmul(&dz_2d, &cols_2d, MatProp::NONE, MatProp::TRANS);
                let d_filter_sum = arrayfire::sum(&d_filter_batch, 3);
                let d_filter_t = arrayfire::transpose(&d_filter_sum, false);
                let d_filter =
                    arrayfire::moddims(&d_filter_t, arrayfire::Dim4::new(&[kh, kw, c_in, c_out]));
                acc_grad(&self.parents[1], &d_filter);

                // Gradient w.r.t. input: filter @ dz → col2im
                let d_cols_2d = matmul(&filter_2d, &dz_2d, MatProp::NONE, MatProp::NONE);
                let d_cols_3d = arrayfire::moddims(
                    &d_cols_2d,
                    arrayfire::Dim4::new(&[kh * kw, c_in, oh * ow, n]),
                );
                let d_cols = arrayfire::reorder_v2(&d_cols_3d, 0, 2, Some(vec![1, 3]));
                let d_input = arrayfire::wrap(
                    &d_cols,
                    h as i64,
                    w_val as i64,
                    kh as i64,
                    kw as i64,
                    stride[0] as i64,
                    stride[1] as i64,
                    padding[0] as i64,
                    padding[1] as i64,
                    true,
                );
                acc_grad(&self.parents[0], &d_input);
            }
            // ai wrote this gradient computation so i should look into it
            Some(Operation::MaxPool { pool_size, stride }) => {
                let input_data = self.parents[0].borrow().tensor.clone();
                let input_dims = input_data.dims();
                let h = input_dims[0];
                let w = input_dims[1];
                let c = input_dims[2];
                let n = input_dims[3];

                let oh = (h - pool_size) / stride + 1;
                let ow = (w - pool_size) / stride + 1;
                let ps = pool_size * pool_size;

                // Unwrap input into patches
                let unwrapped = arrayfire::unwrap(
                    &input_data,
                    pool_size as i64,
                    pool_size as i64,
                    stride as i64,
                    stride as i64,
                    0,
                    0,
                    true,
                );

                // Get max values for each patch
                let (max_vals, _) = arrayfire::imax(&unwrapped, 0);

                // Create mask: broadcast max_vals and compare
                let max_broadcast =
                    arrayfire::tile(&max_vals, arrayfire::Dim4::new(&[ps, 1, 1, 1]));
                let mask = arrayfire::eq(&unwrapped, &max_broadcast, false).cast::<f32>();

                // Normalize mask (handle ties)
                let mask_sum = arrayfire::sum(&mask, 0);
                let mask_sum_broadcast =
                    arrayfire::tile(&mask_sum, arrayfire::Dim4::new(&[ps, 1, 1, 1]));
                let mask_normalized = &mask / &mask_sum_broadcast;

                // Reshape and tile dz
                let dz_reshaped =
                    arrayfire::moddims(&dz, arrayfire::Dim4::new(&[1, oh * ow, c, n]));
                let dz_broadcast =
                    arrayfire::tile(&dz_reshaped, arrayfire::Dim4::new(&[ps, 1, 1, 1]));

                // Multiply gradient by mask
                let grad_unwrapped = &dz_broadcast * &mask_normalized;

                // Wrap back to input shape
                let grad_input = arrayfire::wrap(
                    &grad_unwrapped,
                    h as i64,
                    w as i64,
                    pool_size as i64,
                    pool_size as i64,
                    stride as i64,
                    stride as i64,
                    0,
                    0,
                    true,
                );

                acc_grad(&self.parents[0], &grad_input);
            }
            Some(Operation::Reshape { original_dims }) => {
                let orig = arrayfire::Dim4::new(&original_dims);
                acc_grad(&self.parents[0], &arrayfire::moddims(&dz, orig));
            }
            Some(Operation::Reorder { perm }) => {
                // Inverse permutation
                let mut inv_perm = [0u64; 4];
                for i in 0..4 {
                    inv_perm[perm[i] as usize] = i as u64;
                }
                let dz_reordered = arrayfire::reorder_v2(
                    &dz,
                    inv_perm[0],
                    inv_perm[1],
                    Some(vec![inv_perm[2], inv_perm[3]]),
                );
                acc_grad(&self.parents[0], &dz_reordered);
            }
            None => {}
        }
    }
}

/// Accumulate gradient into a node, reducing along broadcast dimensions
fn acc_grad(node: &NodeRef, contribution: &Array<f32>) {
    let mut node = node.borrow_mut();

    if !node.requires_grad {
        return;
    }

    // there was a bug here where dimensions got broadcast wrong
    let param_dims = node.tensor.dims();
    let grad_dims = contribution.dims();
    let mut reduced = contribution.clone();

    for dim in 0..4 {
        if param_dims[dim] == 1 && grad_dims[dim] > 1 {
            reduced = arrayfire::sum(&reduced, dim as i32);
        }
    }

    node.grad = Some(match node.grad.take() {
        Some(existing) => existing + &reduced,
        None => reduced,
    });
}
