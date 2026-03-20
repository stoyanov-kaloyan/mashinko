# mashinko

Small neural net/autodiff experiments in Rust with ArrayFire.

## Usage

Pick the example you want in `src/main.rs`, then run:

```bash
cargo run
```

MNIST expects the IDX files under `datasets/mnist/`. They can be downloaded from Kaggle: <http://yann.lecun.com/exdb/mnist/>.

TODO: implement better dataset loading and preprocessing, e.g. normalization, data augmentation, etc.
