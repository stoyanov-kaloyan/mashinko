use crate::af_tensor::{Node, NodeRef};

/// Loss API design:
/// - Keep simple, numerically-safe losses as compositions of primitive Node ops.
/// - Use specialized internal autograd ops for losses that benefit from fused,
///   logits-based, or more stable forward/backward implementations.
///   There is guideline that coupling is usually bad but in this case
///   it is very practical since it's the easiest way to manage the internal
///   state of the computation graph

pub fn mse(pred: &NodeRef, target: &NodeRef) -> NodeRef {
    let diff = Node::sub(pred, target);
    let sq_diff = Node::mul(&diff, &diff);
    Node::mean(&sq_diff)
}

/// Binary cross-entropy from raw logits.
/// pred: logits
/// target: binary labels in [0, 1]
pub fn binary_cross_entropy_with_logits(pred: &NodeRef, target: &NodeRef) -> NodeRef {
    Node::binary_cross_entropy_with_logits(pred, target)
}

/// Binary cross-entropy from raw logits.
/// Kept as the short default BCE API for training.
pub fn bce(pred: &NodeRef, target: &NodeRef) -> NodeRef {
    binary_cross_entropy_with_logits(pred, target)
}

/// Categorical cross-entropy with fused softmax.
/// pred: raw logits [batch, classes, 1, 1]
/// target: one-hot [batch, classes, 1, 1]
/// loss = -mean_batch(sum_classes(target * log(softmax(pred))))
pub fn cross_entropy(pred: &NodeRef, target: &NodeRef) -> NodeRef {
    Node::cross_entropy_with_logits(pred, target)
}
