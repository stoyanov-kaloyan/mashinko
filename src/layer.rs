// we want to expose the following api here
// let layer = Linear::new(in_features, out_features);
// let y = layer.forward(x);
use crate::af_tensor::{Node, NodeRef};
use arrayfire::{Dim4, randu};

pub struct Linear {
    pub weight: NodeRef,
    pub bias: NodeRef,
    pub name: Option<String>,
}

pub struct ReLU;

/// A simple Multi-Layer Perceptron with configurable layer sizes.
pub struct MLP {
    pub layers: Vec<Box<dyn HasParameters>>,
}

pub trait HasParameters {
    fn forward(&self, input: NodeRef) -> NodeRef;
    fn parameters(&self) -> Vec<NodeRef>;
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let w_dims = Dim4::new(&[in_features as u64, out_features as u64, 1, 1]);
        let b_dims = Dim4::new(&[out_features as u64, 1, 1, 1]);

        let w = Node::leaf(randu::<f32>(w_dims) * 0.01f32, true);
        let b = Node::leaf(randu::<f32>(b_dims) * 0.01f32, true);

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
