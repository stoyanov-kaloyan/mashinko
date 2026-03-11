use crate::af_tensor::{Node, NodeRef};
use arrayfire::{Dim4, randu};

pub struct Linear {
    pub weight: NodeRef,
    pub bias: NodeRef,
    pub name: Option<String>,
}

pub struct ReLU;

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

pub struct Flatten;

pub struct Permute {
    pub perm: [u64; 4],
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
        let w_dims = Dim4::new(&[in_features as u64, out_features as u64, 1, 1]);
        let b_dims = Dim4::new(&[1, out_features as u64, 1, 1]);

        let scale = (2.0f32 / in_features as f32).sqrt();
        let w = Node::leaf((randu::<f32>(w_dims) - 0.5f32) * (2.0f32 * scale), true);
        let b = Node::leaf(arrayfire::constant(0.0f32, b_dims), true);

        Self {
            weight: w,
            bias: b,
            name: None,
        }
    }
}

impl ReLU {
    pub fn new() -> Self {
        Self
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
        Self
    }
}

impl Permute {
    pub fn new(perm: [u64; 4]) -> Self {
        Self { perm }
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

/// Sequential allows composing multiple layers together
impl Sequential {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }
}

impl HasParameters for Linear {
    fn forward(&self, input: NodeRef) -> NodeRef {
        let matmul_out = Node::matmul(&input, &self.weight);
        // might be more appropriate here but needs diagnosing
        // Node::matmul(&input, &Node::transpose(&self.weight))
        Node::add(&matmul_out, &self.bias)
    }

    fn parameters(&self) -> Vec<NodeRef> {
        vec![self.weight.clone(), self.bias.clone()]
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
        let conv_out = Node::conv2d(&input, &self.weight);
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
