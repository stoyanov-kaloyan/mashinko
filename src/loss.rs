use crate::af_tensor::{Node, NodeRef};

pub fn mse(pred: &NodeRef, target: &NodeRef) -> NodeRef {
    let diff = Node::sub(pred, target);
    let sq_diff = Node::mul(&diff, &diff);
    Node::mean(&sq_diff)
}

pub fn bce(pred: &NodeRef, target: &NodeRef) -> NodeRef {
    let eps = 1e-7f32;
    let pred_clamped = Node::clamp(pred, eps, 1.0 - eps);
    let term1 = Node::mul(target, &Node::log(&pred_clamped));
    let term2 = Node::mul(
        &Node::sub(
            &Node::constant(1.0, pred_clamped.borrow().tensor().dims()),
            target,
        ),
        &Node::log(&Node::sub(
            &Node::constant(1.0, pred_clamped.borrow().tensor().dims()),
            &pred_clamped,
        )),
    );
    Node::neg(&Node::mean(&Node::add(&term1, &term2)))
}

/// Categorical cross-entropy with fused softmax.
/// pred: raw logits [batch, classes, 1, 1]
/// target: one-hot [batch, classes, 1, 1]
/// loss = -mean_batch(sum_classes(target * log(softmax(pred))))
pub fn cross_entropy(pred: &NodeRef, target: &NodeRef) -> NodeRef {
    Node::cross_entropy(pred, target)
}
