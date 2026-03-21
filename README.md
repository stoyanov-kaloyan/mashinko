# mashinko

Small neural net/autodiff experiments in Rust with ArrayFire.

## Usage

Pick the example you want in `src/main.rs`, then run:

```bash
cargo run
```

MNIST expects the IDX files under `datasets/mnist/`. They can be downloaded from Kaggle: <http://yann.lecun.com/exdb/mnist/>.

TODO: implement better dataset loading and preprocessing, e.g. normalization, data augmentation, etc.

## API

The public API is typed-first for MLP-style workflows with compile-time feature dimensions:

```rust
use mashinko::{Frozen, Linear, ReLU, Tensor, sequential2};
use arrayfire::{constant, Dim4};

let x = Tensor::<4, 2, Frozen>::constant_from_array(constant(1.0f32, Dim4::new(&[4, 2, 1, 1])));
let model = sequential2(sequential2(Linear::<2, 8>::new(), ReLU::new()), Linear::<8, 1>::new());
let y = model.forward(x); // Tensor<4, 1, _>
```

Loss wrappers are also provided at crate root: `mse`, `cross_entropy`, and
`binary_cross_entropy_with_logits`, each returning scalar `Tensor<1, 1, _>`.

Examples in `src/examples.rs` are typed-only.
