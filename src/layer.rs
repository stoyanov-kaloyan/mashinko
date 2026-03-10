// we want to expose the following api here
// let layer = Linear::new(in_features, out_features);
// let y = layer.forward(x);
use crate::af_tensor::{Node, NodeRef};
use arrayfire::{Dim4, randu};

pub struct Linear {
    pub weight: NodeRef,
    pub bias: NodeRef,
}

pub trait HasParameters {
    fn forward(&self, input: NodeRef) -> NodeRef;
    fn parameters(&self) -> Vec<NodeRef>;
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let w_dims = Dim4::new(&[out_features as u64, in_features as u64, 1, 1]);
        let b_dims = Dim4::new(&[out_features as u64, 1, 1, 1]);

        let w = Node::leaf(randu::<f32>(w_dims) * 0.01f32, true);
        let b = Node::leaf(randu::<f32>(b_dims) * 0.01f32, true);

        Self { weight: w, bias: b }
    }

    pub fn forward(&self, input: NodeRef) -> NodeRef {
        let matmul_out = Node::matmul(&input, &self.weight);
        Node::add(&matmul_out, &self.bias)
    }

    pub fn parameters(&self) -> Vec<NodeRef> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}
