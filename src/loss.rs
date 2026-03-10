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

pub fn cross_entropy(pred: &NodeRef, target: &NodeRef) -> NodeRef {
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
