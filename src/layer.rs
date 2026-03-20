use crate::af_tensor::{Node, NodeRef};
use arrayfire::{Dim4, randu};
use std::cell::RefCell;

pub struct Linear {
    in_features: Option<usize>,
    out_features: usize,
    weight: RefCell<Option<NodeRef>>,
    bias: RefCell<Option<NodeRef>>,
    pub name: Option<String>,
}

pub struct ReLU {
    pub name: Option<String>,
}

pub struct Conv2D {
    pub weight: NodeRef,
    pub bias: NodeRef,
    pub stride: [u64; 2],
    pub padding: [u64; 2],
    pub dilation: [u64; 2],
    pub name: Option<String>,
}

pub struct MaxPool {
    pub pool_size: u64,
    pub stride: u64,
    pub name: Option<String>,
}

pub struct Permute {
    pub perm: [u64; 4],
    pub name: Option<String>,
}

pub struct Flatten {
    pub name: Option<String>,
}

/// A simple Multi-Layer Perceptron with configurable layer sizes.
/// TODO: add support for different activation funcs and more
/// complex architectures
pub struct MLP {
    pub layers: Vec<Box<dyn HasParameters>>,
}

pub enum Layer {
    Linear(Linear),
    ReLU(ReLU),
    Conv2D(Conv2D),
    MaxPool(MaxPool),
    Flatten(Flatten),
    Permute(Permute),
}

pub struct Sequential {
    pub layers: Vec<Layer>,
}

pub trait HasParameters {
    fn forward(&self, input: NodeRef) -> NodeRef;
    fn parameters(&self) -> Vec<NodeRef>;
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            in_features: Some(in_features),
            out_features,
            weight: RefCell::new(Some(Self::init_weight(in_features, out_features))),
            bias: RefCell::new(Some(Self::init_bias(out_features))),
            name: None,
        }
    }

    pub fn lazy(out_features: usize) -> Self {
        Self {
            in_features: None,
            out_features,
            weight: RefCell::new(None),
            bias: RefCell::new(None),
            name: None,
        }
    }

    pub fn weight(&self) -> Option<NodeRef> {
        self.weight.borrow().clone()
    }

    pub fn bias(&self) -> Option<NodeRef> {
        self.bias.borrow().clone()
    }

    fn init_weight(in_features: usize, out_features: usize) -> NodeRef {
        let w_dims = Dim4::new(&[in_features as u64, out_features as u64, 1, 1]);
        let scale = (2.0f32 / in_features as f32).sqrt();
        Node::leaf((randu::<f32>(w_dims) - 0.5f32) * (2.0f32 * scale), true)
    }

    fn init_bias(out_features: usize) -> NodeRef {
        let b_dims = Dim4::new(&[1, out_features as u64, 1, 1]);
        Node::leaf(arrayfire::constant(0.0f32, b_dims), true)
    }

    fn ensure_initialized(&self, input: &NodeRef) {
        let inferred_in_features = input.borrow().tensor().dims()[1] as usize;

        if self.weight.borrow().is_some() {
            if let Some(expected) = self.in_features {
                assert_eq!(
                    inferred_in_features, expected,
                    "Linear expected {} input features, got {}",
                    expected, inferred_in_features
                );
            }
            return;
        }

        let weight = Self::init_weight(inferred_in_features, self.out_features);
        let bias = Self::init_bias(self.out_features);

        *self.weight.borrow_mut() = Some(weight);
        *self.bias.borrow_mut() = Some(bias);
    }
}

impl ReLU {
    pub fn new() -> Self {
        Self { name: None }
    }
}

/// simple mlp - first layer is linear, then relu, then linear again
impl MLP {
    pub fn new(layer_sizes: &[usize]) -> Self {
        let mut layers: Vec<Box<dyn HasParameters>> = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            layers.push(Box::new(Linear::new(layer_sizes[i], layer_sizes[i + 1])));
            if i < layer_sizes.len() - 2 {
                layers.push(Box::new(ReLU::new()));
            }
        }
        Self { layers }
    }
}

impl Conv2D {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self::with_params(
            in_channels,
            out_channels,
            kernel_size,
            [1, 1],
            [0, 0],
            [1, 1],
        )
    }

    pub fn with_params(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: [u64; 2],
        padding: [u64; 2],
        dilation: [u64; 2],
    ) -> Self {
        let w_dims = Dim4::new(&[
            kernel_size as u64,
            kernel_size as u64,
            in_channels as u64,
            out_channels as u64,
        ]);
        // Bias: (1, 1, out_channels, 1) to broadcast with (oH, oW, out_channels, batch)
        let b_dims = Dim4::new(&[1, 1, out_channels as u64, 1]);

        let fan_in = (in_channels * kernel_size * kernel_size) as f32;
        let scale = (2.0 / fan_in).sqrt();
        let w = Node::leaf((randu::<f32>(w_dims) - 0.5f32) * (2.0f32 * scale), true);
        let b = Node::leaf(arrayfire::constant(0.0f32, b_dims), true);

        Self {
            weight: w,
            bias: b,
            stride,
            padding,
            dilation,
            name: None,
        }
    }
}

impl MaxPool {
    pub fn new(pool_size: u64, stride: u64) -> Self {
        Self {
            pool_size,
            stride,
            name: None,
        }
    }
}

impl Flatten {
    pub fn new() -> Self {
        Self { name: None }
    }
}

impl Permute {
    pub fn new(perm: [u64; 4]) -> Self {
        Self { perm, name: None }
    }

    pub fn nhwc_to_hwcn() -> Self {
        Self::new([1, 2, 3, 0])
    }
}

impl From<Linear> for Layer {
    fn from(l: Linear) -> Self {
        Layer::Linear(l)
    }
}

impl From<ReLU> for Layer {
    fn from(r: ReLU) -> Self {
        Layer::ReLU(r)
    }
}

impl From<Conv2D> for Layer {
    fn from(c: Conv2D) -> Self {
        Layer::Conv2D(c)
    }
}

impl From<MaxPool> for Layer {
    fn from(m: MaxPool) -> Self {
        Layer::MaxPool(m)
    }
}

impl From<Flatten> for Layer {
    fn from(f: Flatten) -> Self {
        Layer::Flatten(f)
    }
}

impl From<Permute> for Layer {
    fn from(p: Permute) -> Self {
        Layer::Permute(p)
    }
}

impl HasParameters for Layer {
    fn forward(&self, input: NodeRef) -> NodeRef {
        match self {
            Layer::Linear(l) => l.forward(input),
            Layer::ReLU(r) => r.forward(input),
            Layer::Conv2D(c) => c.forward(input),
            Layer::MaxPool(m) => m.forward(input),
            Layer::Flatten(f) => f.forward(input),
            Layer::Permute(p) => p.forward(input),
        }
    }

    fn parameters(&self) -> Vec<NodeRef> {
        match self {
            Layer::Linear(l) => l.parameters(),
            Layer::ReLU(r) => r.parameters(),
            Layer::Conv2D(c) => c.parameters(),
            Layer::MaxPool(m) => m.parameters(),
            Layer::Flatten(f) => f.parameters(),
            Layer::Permute(p) => p.parameters(),
        }
    }
}

impl Layer {
    pub fn name(&self) -> Option<&str> {
        match self {
            Layer::Linear(l) => l.name.as_deref(),
            Layer::ReLU(r) => r.name.as_deref(),
            Layer::Conv2D(c) => c.name.as_deref(),
            Layer::MaxPool(m) => m.name.as_deref(),
            Layer::Flatten(f) => f.name.as_deref(),
            Layer::Permute(p) => p.name.as_deref(),
        }
    }

    pub fn set_name<S: Into<String>>(&mut self, name: S) {
        let name = Some(name.into());
        match self {
            Layer::Linear(l) => l.name = name,
            Layer::ReLU(r) => r.name = name,
            Layer::Conv2D(c) => c.name = name,
            Layer::MaxPool(m) => m.name = name,
            Layer::Flatten(f) => f.name = name,
            Layer::Permute(p) => p.name = name,
        }
    }

    pub fn named_parameters(&self) -> Vec<(String, NodeRef)> {
        let Some(prefix) = self.name() else {
            return vec![];
        };

        match self {
            Layer::Linear(l) => {
                let mut params = Vec::new();
                if let Some(weight) = l.weight() {
                    params.push((format!("{prefix}.weight"), weight));
                }
                if let Some(bias) = l.bias() {
                    params.push((format!("{prefix}.bias"), bias));
                }
                params
            }
            Layer::Conv2D(c) => vec![
                (format!("{prefix}.weight"), c.weight.clone()),
                (format!("{prefix}.bias"), c.bias.clone()),
            ],
            Layer::ReLU(_)
            | Layer::MaxPool(_)
            | Layer::Flatten(_)
            | Layer::Permute(_) => vec![],
        }
    }
}

/// Sequential allows composing multiple layers together
impl Sequential {
    pub fn new(mut layers: Vec<Layer>) -> Self {
        for (index, layer) in layers.iter_mut().enumerate() {
            if layer.name().is_none() {
                layer.set_name(format!("layers.{index}"));
            }
        }
        Self { layers }
    }

    pub fn named_parameters(&self) -> Vec<(String, NodeRef)> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.named_parameters());
        }
        params
    }
}

impl HasParameters for Linear {
    fn forward(&self, input: NodeRef) -> NodeRef {
        self.ensure_initialized(&input);
        let weight = self
            .weight()
            .expect("Linear::forward: weight should be initialized");
        let bias = self
            .bias()
            .expect("Linear::forward: bias should be initialized");
        let matmul_out = Node::matmul(&input, &weight);
        // might be more appropriate here but needs diagnosing
        // Node::matmul(&input, &Node::transpose(&self.weight))
        Node::add(&matmul_out, &bias)
    }

    fn parameters(&self) -> Vec<NodeRef> {
        let mut params = Vec::new();
        if let Some(weight) = self.weight() {
            params.push(weight);
        }
        if let Some(bias) = self.bias() {
            params.push(bias);
        }
        params
    }
}

impl HasParameters for ReLU {
    fn forward(&self, input: NodeRef) -> NodeRef {
        Node::relu(&input)
    }

    fn parameters(&self) -> Vec<NodeRef> {
        vec![]
    }
}

impl HasParameters for MLP {
    fn forward(&self, input: NodeRef) -> NodeRef {
        let mut x = input;
        for layer in &self.layers {
            x = layer.forward(x);
        }
        x
    }

    fn parameters(&self) -> Vec<NodeRef> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }
}

impl HasParameters for Conv2D {
    fn forward(&self, input: NodeRef) -> NodeRef {
        let conv_out = Node::conv2d(
            &input,
            &self.weight,
            self.stride,
            self.padding,
            self.dilation,
        );
        let conv_dims = conv_out.borrow().tensor().dims();
        let bias_dims = self.bias.borrow().tensor().dims();
        let conv_out = if conv_dims[2] == bias_dims[2] {
            conv_out
        } else if conv_dims[3] == bias_dims[2] {
            Node::reorder(&conv_out, [0, 1, 3, 2])
        } else {
            panic!(
                "Conv2D output channels do not match bias dims: conv={:?}, bias={:?}",
                conv_dims, bias_dims
            );
        };

        Node::add(&conv_out, &self.bias)
    }

    fn parameters(&self) -> Vec<NodeRef> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

impl HasParameters for MaxPool {
    fn forward(&self, input: NodeRef) -> NodeRef {
        Node::max_pool(&input, self.pool_size, self.stride)
    }

    fn parameters(&self) -> Vec<NodeRef> {
        vec![]
    }
}

impl HasParameters for Flatten {
    fn forward(&self, input: NodeRef) -> NodeRef {
        // (H, W, C, N) → (H*W*C, N) → transpose → (N, H*W*C)
        let dims = input.borrow().tensor().dims();
        let batch = dims[3];
        let features = dims[0] * dims[1] * dims[2];
        let reshaped = Node::reshape(&input, arrayfire::Dim4::new(&[features, batch, 1, 1]));
        Node::transpose(&reshaped)
    }

    fn parameters(&self) -> Vec<NodeRef> {
        vec![]
    }
}

impl HasParameters for Permute {
    fn forward(&self, input: NodeRef) -> NodeRef {
        Node::reorder(&input, self.perm)
    }

    fn parameters(&self) -> Vec<NodeRef> {
        vec![]
    }
}

impl HasParameters for Sequential {
    fn forward(&self, input: NodeRef) -> NodeRef {
        let mut x = input;
        for layer in &self.layers {
            x = layer.forward(x);
        }
        x
    }

    fn parameters(&self) -> Vec<NodeRef> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }
}

/// Macro to build a Sequential model from a list of layers.
/// Each layer is automatically converted into the Layer enum.
#[macro_export]
macro_rules! sequential {
    ($($layer:expr),+ $(,)?) => {
        Sequential::new(vec![$($crate::layer::Layer::from($layer)),+])
    };
}

#[cfg(test)]
mod tests {
    use super::{Conv2D, Layer, Linear, MaxPool, Node, Permute, ReLU, Sequential};
    use crate::layer::HasParameters;
    use arrayfire::Dim4;
    use std::rc::Rc;

    #[test]
    fn sequential_assigns_default_hierarchical_layer_names() {
        let model = Sequential::new(vec![
            Layer::from(Linear::new(4, 8)),
            Layer::from(ReLU::new()),
            Layer::from(MaxPool::new(2, 2)),
        ]);

        assert_eq!(model.layers[0].name(), Some("layers.0"));
        assert_eq!(model.layers[1].name(), Some("layers.1"));
        assert_eq!(model.layers[2].name(), Some("layers.2"));
    }

    #[test]
    fn sequential_preserves_existing_layer_names() {
        let mut linear = Linear::new(4, 8);
        linear.name = Some("encoder.proj".to_string());

        let model = Sequential::new(vec![Layer::from(linear), Layer::from(ReLU::new())]);

        assert_eq!(model.layers[0].name(), Some("encoder.proj"));
        assert_eq!(model.layers[1].name(), Some("layers.1"));
    }

    #[test]
    fn sequential_named_parameters_use_hierarchical_names() {
        let linear = Linear::new(4, 8);
        let linear_weight = linear.weight().expect("linear weight should be initialized");
        let linear_bias = linear.bias().expect("linear bias should be initialized");

        let conv = Conv2D::new(1, 2, 3);
        let conv_weight = conv.weight.clone();
        let conv_bias = conv.bias.clone();

        let model = Sequential::new(vec![
            Layer::from(linear),
            Layer::from(ReLU::new()),
            Layer::from(conv),
        ]);

        let named_params = model.named_parameters();

        assert_eq!(named_params.len(), 4);
        assert_eq!(named_params[0].0, "layers.0.weight");
        assert!(Rc::ptr_eq(&named_params[0].1, &linear_weight));
        assert_eq!(named_params[1].0, "layers.0.bias");
        assert!(Rc::ptr_eq(&named_params[1].1, &linear_bias));
        assert_eq!(named_params[2].0, "layers.2.weight");
        assert!(Rc::ptr_eq(&named_params[2].1, &conv_weight));
        assert_eq!(named_params[3].0, "layers.2.bias");
        assert!(Rc::ptr_eq(&named_params[3].1, &conv_bias));
    }

    #[test]
    fn lazy_linear_initializes_from_first_input_shape() {
        let linear = Linear::lazy(3);
        assert!(linear.weight().is_none());
        assert!(linear.bias().is_none());

        let input = Node::leaf(arrayfire::constant(0.0f32, Dim4::new(&[5, 7, 1, 1])), false);
        let _ = linear.forward(input);

        let weight = linear.weight().expect("lazy linear should initialize weight");
        let bias = linear.bias().expect("lazy linear should initialize bias");
        assert_eq!(weight.borrow().tensor().dims(), Dim4::new(&[7, 3, 1, 1]));
        assert_eq!(bias.borrow().tensor().dims(), Dim4::new(&[1, 3, 1, 1]));
    }

    #[test]
    fn permute_semantic_helper_uses_expected_axis_order() {
        let permute = Permute::nhwc_to_hwcn();
        assert_eq!(permute.perm, [1, 2, 3, 0]);
    }
}
