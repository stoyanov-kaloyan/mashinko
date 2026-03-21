use std::marker::PhantomData;

use arrayfire::{Array, Dim4, randu};

use crate::af_tensor::{Node, NodeRef};

#[derive(Clone, Copy, Debug)]
pub enum Trainable {}

#[derive(Clone, Copy, Debug)]
pub enum Frozen {}

#[derive(Clone)]
pub struct Tensor<const B: usize, const F: usize, G> {
    node: NodeRef,
    _grad_state: PhantomData<G>,
}

impl<const B: usize, const F: usize, G> Tensor<B, F, G> {
    pub fn from_node(node: NodeRef) -> Self {
        let dims = node.borrow().tensor().dims();
        assert_eq!(
            dims,
            Dim4::new(&[B as u64, F as u64, 1, 1]),
            "Tensor shape mismatch: expected [{B}, {F}, 1, 1], got {:?}",
            dims
        );
        Self {
            node,
            _grad_state: PhantomData,
        }
    }

    pub fn node(&self) -> NodeRef {
        self.node.clone()
    }

    pub fn data(&self) -> Array<f32> {
        self.node.borrow().tensor().clone()
    }

    pub fn backward(&self) {
        crate::engine::backward(self.node());
    }
}

impl<const B: usize, const F: usize> Tensor<B, F, Trainable> {
    pub fn variable_from_array(data: Array<f32>) -> Self {
        let dims = data.dims();
        assert_eq!(
            dims,
            Dim4::new(&[B as u64, F as u64, 1, 1]),
            "Tensor::variable_from_array shape mismatch: expected [{B}, {F}, 1, 1], got {:?}",
            dims
        );
        Self::from_node(Node::leaf(data, true))
    }

    pub fn freeze(self) -> Tensor<B, F, Frozen> {
        Tensor {
            node: self.node,
            _grad_state: PhantomData,
        }
    }
}

impl<const B: usize, const F: usize> Tensor<B, F, Frozen> {
    pub fn constant_from_array(data: Array<f32>) -> Self {
        let dims = data.dims();
        assert_eq!(
            dims,
            Dim4::new(&[B as u64, F as u64, 1, 1]),
            "Tensor::constant_from_array shape mismatch: expected [{B}, {F}, 1, 1], got {:?}",
            dims
        );
        Self::from_node(Node::leaf(data, false))
    }

    pub fn unfreeze(self) -> Tensor<B, F, Trainable> {
        Tensor {
            node: self.node,
            _grad_state: PhantomData,
        }
    }
}

pub struct Linear<const IN: usize, const OUT: usize> {
    weight: NodeRef,
    bias: NodeRef,
}

impl<const IN: usize, const OUT: usize> Linear<IN, OUT> {
    pub fn new() -> Self {
        let w_dims = Dim4::new(&[IN as u64, OUT as u64, 1, 1]);
        let b_dims = Dim4::new(&[1, OUT as u64, 1, 1]);
        let scale = (2.0f32 / IN as f32).sqrt();
        let weight = Node::leaf((randu::<f32>(w_dims) - 0.5f32) * (2.0f32 * scale), true);
        let bias = Node::leaf(arrayfire::constant(0.0f32, b_dims), true);
        Self { weight, bias }
    }

    pub fn forward<const B: usize, G>(&self, input: Tensor<B, IN, G>) -> Tensor<B, OUT, Trainable> {
        let matmul_out = Node::matmul(&input.node(), &self.weight);
        let output = Node::add(&matmul_out, &self.bias);
        Tensor::from_node(output)
    }

    pub fn weight(&self) -> NodeRef {
        self.weight.clone()
    }

    pub fn bias(&self) -> NodeRef {
        self.bias.clone()
    }

    pub fn parameters(&self) -> Vec<NodeRef> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

impl<const IN: usize, const OUT: usize> Default for Linear<IN, OUT> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        Self
    }

    pub fn forward<const B: usize, const F: usize, G>(&self, input: Tensor<B, F, G>) -> Tensor<B, F, Trainable> {
        Tensor::from_node(Node::relu(&input.node()))
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Sequential2<L1, L2> {
    l1: L1,
    l2: L2,
}

impl<L1, L2> Sequential2<L1, L2> {
    pub fn new(l1: L1, l2: L2) -> Self {
        Self { l1, l2 }
    }
}

impl<const IN: usize, const H: usize, const OUT: usize> Sequential2<Linear<IN, H>, Linear<H, OUT>> {
    pub fn forward<const B: usize, G>(&self, input: Tensor<B, IN, G>) -> Tensor<B, OUT, Trainable> {
        let x = self.l1.forward(input);
        self.l2.forward(x)
    }

    pub fn parameters(&self) -> Vec<NodeRef> {
        let mut p = self.l1.parameters();
        p.extend(self.l2.parameters());
        p
    }
}

impl<const IN: usize, const H: usize> Sequential2<Linear<IN, H>, ReLU> {
    pub fn forward<const B: usize, G>(&self, input: Tensor<B, IN, G>) -> Tensor<B, H, Trainable> {
        let x = self.l1.forward(input);
        self.l2.forward(x)
    }

    pub fn parameters(&self) -> Vec<NodeRef> {
        self.l1.parameters()
    }
}

impl<const IN: usize, const H: usize, const OUT: usize>
    Sequential2<Sequential2<Linear<IN, H>, ReLU>, Linear<H, OUT>>
{
    pub fn forward<const B: usize, G>(&self, input: Tensor<B, IN, G>) -> Tensor<B, OUT, Trainable> {
        let x = self.l1.forward(input);
        self.l2.forward(x)
    }

    pub fn parameters(&self) -> Vec<NodeRef> {
        let mut p = self.l1.parameters();
        p.extend(self.l2.parameters());
        p
    }
}

pub fn sequential2<L1, L2>(l1: L1, l2: L2) -> Sequential2<L1, L2> {
    Sequential2::new(l1, l2)
}

pub fn mse<const B: usize, const F: usize, GP, GT>(
    pred: &Tensor<B, F, GP>,
    target: &Tensor<B, F, GT>,
) -> Tensor<1, 1, Trainable> {
    Tensor::from_node(crate::loss::mse(&pred.node(), &target.node()))
}

pub fn binary_cross_entropy_with_logits<const B: usize, const F: usize, GP, GT>(
    pred: &Tensor<B, F, GP>,
    target: &Tensor<B, F, GT>,
) -> Tensor<1, 1, Trainable> {
    Tensor::from_node(crate::loss::binary_cross_entropy_with_logits(
        &pred.node(),
        &target.node(),
    ))
}

pub fn cross_entropy<const B: usize, const C: usize, GP, GT>(
    pred: &Tensor<B, C, GP>,
    target: &Tensor<B, C, GT>,
) -> Tensor<1, 1, Trainable> {
    Tensor::from_node(crate::loss::cross_entropy(&pred.node(), &target.node()))
}

#[cfg(test)]
mod tests {
    use arrayfire::{Array, Dim4, constant};

    use crate::engine::backward;
    use crate::optimizer::{Optimizer, SGD};
    use crate::typed::{
        Frozen, Linear, ReLU, Tensor, Trainable, binary_cross_entropy_with_logits, cross_entropy,
        mse, sequential2,
    };
    use crate::utils::assert_all_close;

    #[test]
    fn tensor_shape_check() {
        let ok = Array::new(&[0.0f32, 1.0, 2.0, 3.0], Dim4::new(&[2, 2, 1, 1]));
        let _ = Tensor::<2, 2, Frozen>::constant_from_array(ok);
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn tensor_shape_mismatch_panics() {
        let wrong = Array::new(&[0.0f32, 1.0, 2.0, 3.0], Dim4::new(&[1, 4, 1, 1]));
        let _ = Tensor::<2, 2, Frozen>::constant_from_array(wrong);
    }

    #[test]
    fn typed_linear_forward_has_expected_shape() {
        let x = Tensor::<4, 3, Frozen>::constant_from_array(constant(
            1.0f32,
            Dim4::new(&[4, 3, 1, 1]),
        ));
        let linear = Linear::<3, 5>::new();
        let y = linear.forward(x);
        assert_eq!(y.data().dims(), Dim4::new(&[4, 5, 1, 1]));
    }

    #[test]
    fn typed_sequential_parameters_and_optimizer_interop() {
        let model = sequential2(
            sequential2(Linear::<2, 4>::new(), ReLU::new()),
            Linear::<4, 1>::new(),
        );
        let x = Tensor::<4, 2, Frozen>::constant_from_array(Array::new(
            &[0.0f32, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            Dim4::new(&[4, 2, 1, 1]),
        ));
        let y_true = Tensor::<4, 1, Frozen>::constant_from_array(Array::new(
            &[0.0f32, 1.0, 1.0, 0.0],
            Dim4::new(&[4, 1, 1, 1]),
        ));

        let mut optimizer = SGD::new(0.1);
        let params = model.parameters();

        let y_pred = model.forward(x);
        let loss = mse(&y_pred, &y_true);
        backward(loss.node());
        optimizer.step(&params);
        optimizer.zero_grad(&params);

        for p in params {
            assert!(p.borrow().grad().is_none());
        }
    }

    #[test]
    fn typed_freeze_unfreeze_roundtrip() {
        let x_train = Tensor::<2, 3, Trainable>::variable_from_array(constant(
            0.5f32,
            Dim4::new(&[2, 3, 1, 1]),
        ));
        let x_frozen: Tensor<2, 3, Frozen> = x_train.freeze();
        let x_unfrozen: Tensor<2, 3, Trainable> = x_frozen.unfreeze();
        assert_all_close(
            x_unfrozen.node().borrow().tensor(),
            &constant(0.5f32, Dim4::new(&[2, 3, 1, 1])),
            1e-6,
        );
    }

    #[test]
    fn typed_loss_wrappers_return_scalar() {
        let logits = Tensor::<4, 3, Trainable>::variable_from_array(Array::new(
            &[1.0f32, 0.0, -1.0, 0.5, 1.0, -0.5, 0.2, -0.1, 0.0, 0.8, 0.6, -0.2],
            Dim4::new(&[4, 3, 1, 1]),
        ));
        let targets = Tensor::<4, 3, Frozen>::constant_from_array(Array::new(
            &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            Dim4::new(&[4, 3, 1, 1]),
        ));
        let ce = cross_entropy(&logits, &targets);
        assert_eq!(ce.data().dims(), Dim4::new(&[1, 1, 1, 1]));

        let bin_logits = Tensor::<4, 1, Trainable>::variable_from_array(Array::new(
            &[0.2f32, -0.3, 1.4, -1.0],
            Dim4::new(&[4, 1, 1, 1]),
        ));
        let bin_targets = Tensor::<4, 1, Frozen>::constant_from_array(Array::new(
            &[1.0f32, 0.0, 1.0, 0.0],
            Dim4::new(&[4, 1, 1, 1]),
        ));
        let bce = binary_cross_entropy_with_logits(&bin_logits, &bin_targets);
        assert_eq!(bce.data().dims(), Dim4::new(&[1, 1, 1, 1]));

        let mse_loss = mse(&bin_logits, &bin_targets);
        assert_eq!(mse_loss.data().dims(), Dim4::new(&[1, 1, 1, 1]));
    }
}
