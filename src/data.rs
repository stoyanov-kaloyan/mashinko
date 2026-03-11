use crate::af_tensor::{Node, NodeRef};
use arrayfire::{self as af, Array};
use rand::seq::SliceRandom;

pub struct Dataset {
    pub x: NodeRef,
    pub y: NodeRef,
}

impl Dataset {
    pub fn new(x: NodeRef, y: NodeRef) -> Self {
        Self { x, y }
    }
}

pub struct DataLoader {
    pub dataset: Dataset,
    pub batch_size: usize,
    pub shuffle: bool,
}

impl DataLoader {
    pub fn new(dataset: Dataset, batch_size: usize, shuffle: bool) -> Self {
        Self {
            dataset,
            batch_size,
            shuffle,
        }
    }

    pub fn iter(&self) -> DataLoaderIter {
        let x_tensor = self.dataset.x.borrow().tensor().clone();
        let y_tensor = self.dataset.y.borrow().tensor().clone();
        let n_samples = x_tensor.dims()[0] as usize;

        let (x_data, y_data) = if self.shuffle {
            let x_dims = x_tensor.dims();
            let y_dims = y_tensor.dims();

            let x_elements = x_tensor.elements();
            let y_elements = y_tensor.elements();
            let x_stride = x_elements / n_samples;
            let y_stride = y_elements / n_samples;

            let mut x_host = vec![0.0f32; x_elements];
            let mut y_host = vec![0.0f32; y_elements];
            x_tensor.host(&mut x_host);
            y_tensor.host(&mut y_host);

            let mut perm: Vec<usize> = (0..n_samples).collect();
            perm.shuffle(&mut rand::rng());

            // Reorder rows according to permutation
            let mut x_shuffled = vec![0.0f32; x_elements];
            let mut y_shuffled = vec![0.0f32; y_elements];
            for (new_idx, &old_idx) in perm.iter().enumerate() {
                for col in 0..x_stride {
                    x_shuffled[new_idx + col * n_samples] = x_host[old_idx + col * n_samples];
                }
                for col in 0..y_stride {
                    y_shuffled[new_idx + col * n_samples] = y_host[old_idx + col * n_samples];
                }
            }

            (
                Array::new(&x_shuffled, x_dims),
                Array::new(&y_shuffled, y_dims),
            )
        } else {
            (x_tensor, y_tensor)
        };

        DataLoaderIter {
            x_data,
            y_data,
            batch_size: self.batch_size,
            n_samples,
            index: 0,
        }
    }

    pub fn len(&self) -> usize {
        let n_samples = self.dataset.x.borrow().tensor().dims()[0] as usize;
        (n_samples + self.batch_size - 1) / self.batch_size
    }

    pub fn is_empty(&self) -> bool {
        self.dataset.x.borrow().tensor().dims()[0] == 0
    }
}

impl<'a> IntoIterator for &'a DataLoader {
    type Item = (NodeRef, NodeRef);
    type IntoIter = DataLoaderIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub struct DataLoaderIter {
    x_data: Array<f32>,
    y_data: Array<f32>,
    batch_size: usize,
    n_samples: usize,
    index: usize,
}

impl Iterator for DataLoaderIter {
    type Item = (NodeRef, NodeRef);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.n_samples {
            return None;
        }

        let start = self.index;
        let end = (start + self.batch_size).min(self.n_samples);
        self.index = end;

        let batch_seq = af::Seq::new(start as f64, (end - 1) as f64, 1.0);

        let x_dims = self.x_data.dims();
        let x_batch = af::index(
            &self.x_data,
            &[
                batch_seq,
                af::Seq::new(0.0, (x_dims[1] as f64) - 1.0, 1.0),
                af::Seq::new(0.0, (x_dims[2] as f64) - 1.0, 1.0),
                af::Seq::new(0.0, (x_dims[3] as f64) - 1.0, 1.0),
            ],
        );

        let y_dims = self.y_data.dims();
        let y_batch = af::index(
            &self.y_data,
            &[
                batch_seq,
                af::Seq::new(0.0, (y_dims[1] as f64) - 1.0, 1.0),
                af::Seq::new(0.0, (y_dims[2] as f64) - 1.0, 1.0),
                af::Seq::new(0.0, (y_dims[3] as f64) - 1.0, 1.0),
            ],
        );

        Some((Node::leaf(x_batch, false), Node::leaf(y_batch, false)))
    }
}
