use arrayfire::{Array, MatProp, matmul};
use std::cell::RefCell;
use std::ops::Mul;
use std::rc::Rc;

// TODO: figure out how to do batching

pub type NodeRef = Rc<RefCell<Node>>;

#[derive(Clone)]
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
    Softmax,
    CrossEntropy {
        probs: Array<f32>,
        target: Array<f32>,
    },
    BinaryCrossEntropyWithLogits {
        probs: Array<f32>,
        target: Array<f32>,
        scale: f32,
    },
    Conv2D,
    MaxPool {
        pool_size: u64,
        stride: u64,
        mask: Array<f32>,
    },
    Reshape,
    Reorder([u64; 4]),
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

    /// Numerically-stable softmax along dim1 (the classes axis).
    /// Expects input shape [batch, classes, 1, 1].
    pub fn softmax(a: &NodeRef) -> NodeRef {
        let x = a.borrow().tensor().clone();
        let n_classes = x.dims()[1];
        // Subtract per-sample max for numerical stability
        let max_x = arrayfire::max(&x, 1); // [batch, 1, 1, 1]
        let max_tiled = arrayfire::tile(&max_x, arrayfire::Dim4::new(&[1, n_classes, 1, 1]));
        let shifted = x - max_tiled;
        let exp_x = arrayfire::exp(&shifted);
        let sum_exp = arrayfire::sum(&exp_x, 1); // [batch, 1, 1, 1]
        let sum_tiled = arrayfire::tile(&sum_exp, arrayfire::Dim4::new(&[1, n_classes, 1, 1]));
        let s = exp_x / sum_tiled;
        Self::from_op(s, Operation::Softmax, vec![a.clone()], true)
    }

    /// Internal categorical cross-entropy from raw logits.
    /// `pred` and `target` are expected to have shape [batch, classes, 1, 1].
    pub(crate) fn cross_entropy_with_logits(pred: &NodeRef, target: &NodeRef) -> NodeRef {
        let logits = pred.borrow().tensor().clone();
        let target_tensor = target.borrow().tensor().clone();
        let batch = logits.dims()[0] as f32;
        let n_classes = logits.dims()[1];

        let max_logits = arrayfire::max(&logits, 1);
        let max_tiled = arrayfire::tile(&max_logits, arrayfire::Dim4::new(&[1, n_classes, 1, 1]));
        let shifted = &logits - &max_tiled;
        let exp_logits = arrayfire::exp(&shifted);
        let sum_exp = arrayfire::sum(&exp_logits, 1);
        let sum_exp_tiled = arrayfire::tile(&sum_exp, arrayfire::Dim4::new(&[1, n_classes, 1, 1]));
        let probs = &exp_logits / &sum_exp_tiled;

        let log_sum_exp = arrayfire::log(&sum_exp);
        let log_sum_exp_tiled =
            arrayfire::tile(&log_sum_exp, arrayfire::Dim4::new(&[1, n_classes, 1, 1]));
        let log_probs = &shifted - &log_sum_exp_tiled;
        let nll = -arrayfire::sum_all(&(&target_tensor * &log_probs)).0 as f32 / batch;
        let loss = Array::new(&[nll], arrayfire::Dim4::new(&[1, 1, 1, 1]));

        Self::from_op(
            loss,
            Operation::CrossEntropy {
                probs,
                target: target_tensor,
            },
            vec![pred.clone()],
            true,
        )
    }

    /// Internal binary cross-entropy from raw logits.
    /// `pred` and `target` are expected to have the same shape.
    pub(crate) fn binary_cross_entropy_with_logits(pred: &NodeRef, target: &NodeRef) -> NodeRef {
        let logits = pred.borrow().tensor().clone();
        let target_tensor = target.borrow().tensor().clone();
        let abs_logits = arrayfire::abs(&logits);
        let zeros = arrayfire::constant(0.0f32, logits.dims());
        let max_logits = arrayfire::maxof(&logits, &zeros, false);
        let stable_term = arrayfire::log1p(&arrayfire::exp(&(&abs_logits.mul(-1))));
        let per_elem = &max_logits - &(&logits * &target_tensor) + &stable_term;
        let num_elements = logits.elements() as f32;
        let loss_val = arrayfire::sum_all(&per_elem).0 as f32 / num_elements;
        let loss = Array::new(&[loss_val], arrayfire::Dim4::new(&[1, 1, 1, 1]));

        let probs = 1.0f32 / (1.0f32 + arrayfire::exp(&(&logits.mul(-1))));

        Self::from_op(
            loss,
            Operation::BinaryCrossEntropyWithLogits {
                probs,
                target: target_tensor,
                scale: 1.0f32 / num_elements,
            },
            vec![pred.clone()],
            true,
        )
    }

    pub fn conv2d(input: &NodeRef, weight: &NodeRef) -> NodeRef {
        let signal = input.borrow().tensor().clone(); // [H, W, C_in, N]
        let filter = weight.borrow().tensor().clone(); // [FH, FW, C_in, C_out]
        let c_out = filter.dims()[3];
        let mode = arrayfire::ConvMode::DEFAULT;
        let domain = arrayfire::ConvDomain::AUTO;

        // Loop over output channels: for each k, convolve signal[H,W,C_in,N] with
        // filter_k[FH,FW,C_in,1]. Batching: signal dim3=N vs filter dim3=1 → BATCH_LHS ✓
        // Then sum over C_in (dim2) → [H,W,1,N].
        let channels: Vec<Array<f32>> = (0..c_out)
            .map(|k| {
                let filter_k = arrayfire::index(
                    &filter,
                    &[
                        arrayfire::seq!(),
                        arrayfire::seq!(),
                        arrayfire::seq!(),
                        arrayfire::seq!(k as i32, k as i32, 1),
                    ],
                );
                let conv_k = arrayfire::convolve2(&signal, &filter_k, mode, domain);
                arrayfire::sum(&conv_k, 2)
            })
            .collect();

        let tensor = join_along(2, channels);
        Self::from_op(
            tensor,
            Operation::Conv2D,
            vec![input.clone(), weight.clone()],
            true,
        )
    }

    pub fn max_pool(input: &NodeRef, pool_size: u64, stride: u64) -> NodeRef {
        let tensor = input.borrow().tensor().clone();
        let dim0 = tensor.dims()[0] as i32;
        let dim1 = tensor.dims()[1] as i32;
        let ps = pool_size as i32;
        let st = stride as i32;

        let mut result: Option<Array<f32>> = None;

        for i in 0..ps {
            for j in 0..ps {
                let indices = &[
                    arrayfire::seq!(i, dim0 - 1, st),
                    arrayfire::seq!(j, dim1 - 1, st),
                    arrayfire::seq!(),
                    arrayfire::seq!(),
                ];
                let patch = arrayfire::index(&tensor, indices);
                result = Some(match result {
                    None => patch,
                    Some(curr) => arrayfire::maxof(&curr, &patch, false),
                });
            }
        }

        let output = result.unwrap();

        // Build input-sized mask: 1 where input[r,c] == max of its pool window,
        // i.e. input[r,c] == output[floor(r/stride), floor(c/stride)].
        // We nearest-neighbor upsample `output` to input size via a gather:
        // row_indices[r] = floor(r / stride), col_indices[c] = floor(c / stride)
        let input_dims = tensor.dims();
        let out_h = output.dims()[0];
        let out_w = output.dims()[1];
        let out_c = output.dims()[2];
        let out_n = output.dims()[3];

        // row gather: for each input row r, collect output row floor(r/stride).
        // In column-major, repeat-each-element along dim0 requires:
        //   moddims([1, H, W*C*N]) → tile([stride, 1, 1]) → moddims([H*stride, W, C, N])
        // (H must be in dim1 so tiling dim0 interleaves copies between elements)
        let row_expanded = {
            let flat = arrayfire::moddims(
                &output,
                arrayfire::Dim4::new(&[1, out_h, out_w * out_c * out_n, 1]),
            );
            let tiled = arrayfire::tile(&flat, arrayfire::Dim4::new(&[stride, 1, 1, 1]));
            arrayfire::moddims(
                &tiled,
                arrayfire::Dim4::new(&[out_h * stride, out_w, out_c, out_n]),
            )
        };
        // Step 2: expand each col by stride → [out_h*stride, out_w*stride, C, N]
        // Bring W to dim0 via transpose, repeat-interleave, transpose back.
        let col_expanded = {
            let transposed = arrayfire::reorder_v2(&row_expanded, 1, 0, Some(vec![2, 3]));
            let transposed_w = transposed.dims()[0];
            let transposed_rest =
                transposed.dims()[1] * transposed.dims()[2] * transposed.dims()[3];
            let flat = arrayfire::moddims(
                &transposed,
                arrayfire::Dim4::new(&[1, transposed_w, transposed_rest, 1]),
            );
            let tiled = arrayfire::tile(&flat, arrayfire::Dim4::new(&[stride, 1, 1, 1]));
            let expanded = arrayfire::moddims(
                &tiled,
                arrayfire::Dim4::new(&[
                    transposed_w * stride,
                    transposed.dims()[1],
                    transposed.dims()[2],
                    transposed.dims()[3],
                ]),
            );
            arrayfire::reorder_v2(&expanded, 1, 0, Some(vec![2, 3]))
        };
        // Crop to exact input size (in case H or W is not divisible by stride)
        let upsampled = arrayfire::index(
            &col_expanded,
            &[
                arrayfire::seq!(0, (input_dims[0] - 1) as i32, 1),
                arrayfire::seq!(0, (input_dims[1] - 1) as i32, 1),
                arrayfire::seq!(),
                arrayfire::seq!(),
            ],
        );
        let mask = arrayfire::eq(&tensor, &upsampled, false).cast::<f32>();

        Self::from_op(
            output,
            Operation::MaxPool {
                pool_size,
                stride,
                mask,
            },
            vec![input.clone()],
            true,
        )
    }

    pub fn reshape(input: &NodeRef, new_dims: arrayfire::Dim4) -> NodeRef {
        let output = arrayfire::moddims(input.borrow().tensor(), new_dims);
        Self::from_op(output, Operation::Reshape, vec![input.clone()], true)
    }

    pub fn reorder(input: &NodeRef, perm: [u64; 4]) -> NodeRef {
        let output = arrayfire::reorder_v2(
            input.borrow().tensor(),
            perm[0],
            perm[1],
            Some(vec![perm[2], perm[3]]),
        );
        Self::from_op(output, Operation::Reorder(perm), vec![input.clone()], true)
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
        self.op.clone()
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

        match &self.op {
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
            Some(Operation::Softmax) => {
                // s = softmax(x), shape [batch, classes, 1, 1]
                // dL/dx_i = s_i * (dz_i - sum_j(s_j * dz_j))
                let s = &self.tensor;
                let sdz = s * &dz;
                let sum_sdz = arrayfire::sum(&sdz, 1); // [batch, 1, 1, 1]
                let tiled =
                    arrayfire::tile(&sum_sdz, arrayfire::Dim4::new(&[1, s.dims()[1], 1, 1]));
                let da = s * &(&dz - &tiled);
                acc_grad(&self.parents[0], &da);
            }
            Some(Operation::CrossEntropy { probs, target }) => {
                let batch = probs.dims()[0] as f32;
                let scale = arrayfire::constant(1.0f32 / batch, probs.dims());
                let da = (probs - target) * &scale * &dz;
                acc_grad(&self.parents[0], &da);
            }
            Some(Operation::BinaryCrossEntropyWithLogits {
                probs,
                target,
                scale,
            }) => {
                let scale_tensor = arrayfire::constant(*scale, probs.dims());
                let da = (probs - target) * &scale_tensor * &dz;
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
                acc_grad(&self.parents[0], &arrayfire::transpose(&dz, false));
            }
            // vibecoded gradient ; should look into it :^)
            Some(Operation::Conv2D) => {
                let signal = self.parents[0].borrow().tensor.clone(); // [H, W, C_in, N]
                let filter = self.parents[1].borrow().tensor.clone(); // [FH, FW, C_in, C_out]
                let signal_dims = signal.dims();
                let filter_dims = filter.dims();
                let h = signal_dims[0] as i32;
                let w = signal_dims[1] as i32;
                let fh = filter_dims[0] as i32;
                let fw = filter_dims[1] as i32;
                let c_in = filter_dims[2] as i32;
                let c_out = filter_dims[3] as i32;
                let n_batch = signal_dims[3];
                let mode = arrayfire::ConvMode::DEFAULT;
                let domain = arrayfire::ConvDomain::AUTO;

                // === SIGNAL GRADIENT ===
                // d_signal[:,:,c,:] = sum_k convolve2(dz_k, flip(f_ck))
                // convolve2 does math convolution (flips 2nd arg), so we must flip the filter
                // to get the correct adjoint: dx = convolve2(dy, flip(w))
                let d_signal = {
                    let chans: Vec<Array<f32>> = (0..c_in)
                        .map(|c| {
                            let mut d_c = arrayfire::constant(
                                0.0f32,
                                arrayfire::Dim4::new(&[h as u64, w as u64, 1, n_batch]),
                            );
                            for k in 0..c_out {
                                let dz_k = arrayfire::index(
                                    &dz,
                                    &[
                                        arrayfire::seq!(),
                                        arrayfire::seq!(),
                                        arrayfire::seq!(k, k, 1),
                                        arrayfire::seq!(),
                                    ],
                                );
                                let f_ck = arrayfire::index(
                                    &filter,
                                    &[
                                        arrayfire::seq!(),
                                        arrayfire::seq!(),
                                        arrayfire::seq!(c, c, 1),
                                        arrayfire::seq!(k, k, 1),
                                    ],
                                );
                                let f_ck_flipped = arrayfire::flip(&arrayfire::flip(&f_ck, 0), 1);
                                d_c =
                                    d_c + arrayfire::convolve2(&dz_k, &f_ck_flipped, mode, domain);
                            }
                            d_c
                        })
                        .collect();
                    join_along(2, chans)
                };
                acc_grad(&self.parents[0], &d_signal);

                // === FILTER GRADIENT ===
                // dw[m,n,c,k] = sum_batch crop(convolve2_expand(flip(x_c), dz_k))
                // Verified: dw[m,n] = convolve2_expand(flip(x), dz)[H-1-ch+m, W-1-cw+n]
                let expand_mode = arrayfire::ConvMode::EXPAND;
                let d_filter = {
                    let k_chans: Vec<Array<f32>> = (0..c_out)
                        .map(|k| {
                            let dz_k = arrayfire::index(
                                &dz,
                                &[
                                    arrayfire::seq!(),
                                    arrayfire::seq!(),
                                    arrayfire::seq!(k, k, 1),
                                    arrayfire::seq!(),
                                ],
                            ); // [H,W,1,N]

                            let c_chans: Vec<Array<f32>> = (0..c_in)
                                .map(|c| {
                                    let signal_c = arrayfire::index(
                                        &signal,
                                        &[
                                            arrayfire::seq!(),
                                            arrayfire::seq!(),
                                            arrayfire::seq!(c, c, 1),
                                            arrayfire::seq!(),
                                        ],
                                    ); // [H,W,1,N]

                                    let signal_c_flipped =
                                        arrayfire::flip(&arrayfire::flip(&signal_c, 0), 1);

                                    // convolve2_expand(flip(x_c), dz_k): BATCH_SAME on dim3
                                    let full = arrayfire::convolve2(
                                        &signal_c_flipped,
                                        &dz_k,
                                        expand_mode,
                                        domain,
                                    ); // [2H-1, 2W-1, 1, N]

                                    // Crop at (H-1-ch, W-1-cw) with size (fh, fw)
                                    let start_h = h - 1 - (fh - 1) / 2;
                                    let start_w = w - 1 - (fw - 1) / 2;
                                    let cropped = arrayfire::index(
                                        &full,
                                        &[
                                            arrayfire::seq!(start_h, start_h + fh - 1, 1),
                                            arrayfire::seq!(start_w, start_w + fw - 1, 1),
                                            arrayfire::seq!(),
                                            arrayfire::seq!(),
                                        ],
                                    );
                                    arrayfire::sum(&cropped, 3) // [fh,fw,1,1]
                                })
                                .collect();
                            join_along(2, c_chans) // [fh,fw,c_in,1]
                        })
                        .collect();
                    join_along(3, k_chans) // [fh,fw,c_in,c_out]
                };
                acc_grad(&self.parents[1], &d_filter);
            }
            Some(Operation::MaxPool {
                pool_size: _,
                stride,
                mask,
            }) => {
                // Upsample dz back to input size (nearest-neighbor), multiply by mask.
                let parent_dims = self.parents[0].borrow().tensor.dims();
                let dz_h = dz.dims()[0];
                let dz_w = dz.dims()[1];
                let dz_c = dz.dims()[2];
                let dz_n = dz.dims()[3];
                let row_expanded = {
                    let flat = arrayfire::moddims(
                        &dz,
                        arrayfire::Dim4::new(&[1, dz_h, dz_w * dz_c * dz_n, 1]),
                    );
                    let tiled = arrayfire::tile(&flat, arrayfire::Dim4::new(&[*stride, 1, 1, 1]));
                    arrayfire::moddims(
                        &tiled,
                        arrayfire::Dim4::new(&[dz_h * stride, dz_w, dz_c, dz_n]),
                    )
                };
                let col_expanded = {
                    let transposed = arrayfire::reorder_v2(&row_expanded, 1, 0, Some(vec![2, 3]));
                    let transposed_w = transposed.dims()[0];
                    let transposed_rest =
                        transposed.dims()[1] * transposed.dims()[2] * transposed.dims()[3];
                    let flat = arrayfire::moddims(
                        &transposed,
                        arrayfire::Dim4::new(&[1, transposed_w, transposed_rest, 1]),
                    );
                    let tiled = arrayfire::tile(&flat, arrayfire::Dim4::new(&[*stride, 1, 1, 1]));
                    let expanded = arrayfire::moddims(
                        &tiled,
                        arrayfire::Dim4::new(&[
                            transposed_w * stride,
                            transposed.dims()[1],
                            transposed.dims()[2],
                            transposed.dims()[3],
                        ]),
                    );
                    arrayfire::reorder_v2(&expanded, 1, 0, Some(vec![2, 3]))
                };
                let upsampled_dz = arrayfire::index(
                    &col_expanded,
                    &[
                        arrayfire::seq!(0, (parent_dims[0] - 1) as i32, 1),
                        arrayfire::seq!(0, (parent_dims[1] - 1) as i32, 1),
                        arrayfire::seq!(),
                        arrayfire::seq!(),
                    ],
                );
                let da = upsampled_dz * mask;
                acc_grad(&self.parents[0], &da);
            }
            Some(Operation::Reshape) => {
                // z = reshape(a)  ->  da = dz reshaped to a's shape
                let parent_shape = self.parents[0].borrow().tensor.dims();
                let da = arrayfire::moddims(&dz, parent_shape);
                acc_grad(&self.parents[0], &da);
            }
            Some(Operation::Reorder(perm)) => {
                // z = reorder(a, perm)  ->  da = reorder dz with inverse permutation
                // inv_perm[perm[i]] = i
                let mut inv_perm = [0u64; 4];
                for i in 0..4 {
                    inv_perm[perm[i] as usize] = i as u64;
                }
                let da = arrayfire::reorder_v2(
                    &dz,
                    inv_perm[0],
                    inv_perm[1],
                    Some(vec![inv_perm[2], inv_perm[3]]),
                );
                acc_grad(&self.parents[0], &da);
            }
            None => {}
        }
    }
}

fn join_along(dim: i32, arrays: Vec<Array<f32>>) -> Array<f32> {
    arrays
        .into_iter()
        .reduce(|acc, a| arrayfire::join(dim, &acc, &a))
        .expect("join_along: empty array list")
}

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

#[cfg(test)]
mod tests {
    use arrayfire::{Array, af_print, dim4};

    use crate::{af_tensor::Node, utils::assert_all_close};

    #[test]
    fn test_max_pool() {
        let fourxfour =
            Array::new(&(0..16).into_iter().collect::<Vec<_>>(), dim4!(4, 4, 1, 1)).cast::<f32>();
        let input = Node::leaf(fourxfour, true);
        let output = Node::max_pool(&input, 2, 2);
        // expected output is [2 2 1 1]
        //  5         13
        //  7         15
        let expected = arrayfire::Array::new(
            &[5.0f32, 7.0, 13.0, 15.0],
            arrayfire::Dim4::new(&[2, 2, 1, 1]),
        );
        af_print!("output", output.borrow().tensor());
        af_print!("expected", expected);

        assert_all_close(output.borrow().tensor(), &expected, 1e-5);
    }

    #[test]
    fn test_max_pool_backward() {
        let fourxfour =
            Array::new(&(0..16).into_iter().collect::<Vec<_>>(), dim4!(4, 4, 1, 1)).cast::<f32>();
        let input = Node::leaf(fourxfour, true);
        let output = Node::max_pool(&input, 2, 2);

        //set output grad
        let out_tensor = { output.borrow().tensor.clone() };
        output
            .borrow_mut()
            .set_grad(arrayfire::constant(1.0f32, out_tensor.dims()));

        output.borrow_mut().backward();

        let expected_input_grad = arrayfire::Array::new(
            &[
                0.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
            ],
            dim4!(4, 4, 1, 1),
        );

        assert_all_close(input.borrow().grad().unwrap(), &expected_input_grad, 1e-5);
        assert_all_close(
            output.borrow().grad().unwrap(),
            &arrayfire::constant(1.0f32, out_tensor.dims()),
            1e-5,
        );
    }

    #[test]
    fn test_conv2d() {
        // in is [[123], [456], [789]]
        let input = Node::leaf(
            Array::new(
                &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                dim4!(3, 3, 1, 1),
            ),
            true,
        );

        //expected output of conv2d with 2x2 filter
        // [[-1, -2, -1], [0, 0, 0], [1, 2, 1]] is [[-13, -20, -17], [-18, -24, -18], [13, 20, 17]]
        let expected_output = Array::new(
            &[
                -13.0f32, -20.0, -17.0, -18.0, -24.0, -18.0, 13.0, 20.0, 17.0,
            ],
            dim4!(3, 3, 1, 1),
        );
        let filter = Node::leaf(
            Array::new(
                &[-1.0f32, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0],
                dim4!(3, 3, 1, 1),
            ),
            true,
        );

        let output = Node::conv2d(&input, &filter);

        assert_all_close(output.borrow().tensor(), &expected_output, 1e-5);
    }

    /// Numerical gradient check for conv2d (both signal and filter gradients)
    #[test]
    fn test_conv2d_backward_numerical() {
        use crate::engine::backward;

        let eps = 1e-3f32;

        // 3x3 input, 3x3 Sobel filter, 1 channel, 1 output channel
        let x_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let w_data = vec![-1.0f32, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];
        let x_arr = Array::new(&x_data, dim4!(3, 3, 1, 1));
        let w_arr = Array::new(&w_data, dim4!(3, 3, 1, 1));

        // Analytical gradients via backward
        let x = Node::leaf(x_arr.clone(), true);
        let w = Node::leaf(w_arr.clone(), true);
        let out = Node::conv2d(&x, &w);
        let loss = Node::sum(&out);
        backward(loss);

        let anal_dx = x.borrow().grad().unwrap().clone();
        let anal_dw = w.borrow().grad().unwrap().clone();

        // Numerical gradient for x (finite differences)
        let mut num_dx = vec![0.0f32; 9];
        for idx in 0..9 {
            let mut x_plus = x_data.clone();
            x_plus[idx] += eps;
            let xp = Node::leaf(Array::new(&x_plus, dim4!(3, 3, 1, 1)), false);
            let wp = Node::leaf(w_arr.clone(), false);
            let out_p = Node::conv2d(&xp, &wp);
            let (lp, _) = arrayfire::sum_all(out_p.borrow().tensor());

            let mut x_minus = x_data.clone();
            x_minus[idx] -= eps;
            let xm = Node::leaf(Array::new(&x_minus, dim4!(3, 3, 1, 1)), false);
            let wm = Node::leaf(w_arr.clone(), false);
            let out_m = Node::conv2d(&xm, &wm);
            let (lm, _) = arrayfire::sum_all(out_m.borrow().tensor());

            num_dx[idx] = (lp - lm) / (2.0 * eps);
        }
        let num_dx_arr = Array::new(&num_dx, dim4!(3, 3, 1, 1));
        assert_all_close(&anal_dx, &num_dx_arr, 0.1);

        // Numerical gradient for w
        let mut num_dw = vec![0.0f32; 9];
        for idx in 0..9 {
            let mut w_plus = w_data.clone();
            w_plus[idx] += eps;
            let xp = Node::leaf(x_arr.clone(), false);
            let wp = Node::leaf(Array::new(&w_plus, dim4!(3, 3, 1, 1)), false);
            let out_p = Node::conv2d(&xp, &wp);
            let (lp, _) = arrayfire::sum_all(out_p.borrow().tensor());

            let mut w_minus = w_data.clone();
            w_minus[idx] -= eps;
            let xm = Node::leaf(x_arr.clone(), false);
            let wm = Node::leaf(Array::new(&w_minus, dim4!(3, 3, 1, 1)), false);
            let out_m = Node::conv2d(&xm, &wm);
            let (lm, _) = arrayfire::sum_all(out_m.borrow().tensor());

            num_dw[idx] = (lp - lm) / (2.0 * eps);
        }
        let num_dw_arr = Array::new(&num_dw, dim4!(3, 3, 1, 1));
        assert_all_close(&anal_dw, &num_dw_arr, 0.1);
    }
}
