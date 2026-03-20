use crate::af_tensor::{Node, NodeRef};
use arrayfire::{self as af, Array, Dim4};
use rand::seq::SliceRandom;

pub struct Dataset {
    pub x: NodeRef,
    pub y: NodeRef,
}

pub struct HostDataset {
    pub x: Vec<f32>,
    pub x_dims: Dim4,
    pub y: Vec<f32>,
    pub y_dims: Dim4,
}

impl Dataset {
    pub fn new(x: NodeRef, y: NodeRef) -> Self {
        Self { x, y }
    }
}

impl HostDataset {
    pub fn new(x: Vec<f32>, x_dims: Dim4, y: Vec<f32>, y_dims: Dim4) -> Self {
        Self {
            x,
            x_dims,
            y,
            y_dims,
        }
    }
}

pub struct DataLoader {
    pub dataset: Dataset,
    pub batch_size: usize,
    pub shuffle: bool,
}

impl DataLoader {
    pub fn new(dataset: &Dataset, batch_size: usize, shuffle: bool) -> Self {
        Self {
            dataset: Dataset::new(dataset.x.clone(), dataset.y.clone()),
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

pub struct HostDataLoader {
    pub dataset: HostDataset,
    pub batch_size: usize,
    pub shuffle: bool,
}

impl HostDataLoader {
    pub fn new(dataset: &HostDataset, batch_size: usize, shuffle: bool) -> Self {
        Self {
            dataset: HostDataset::new(
                dataset.x.clone(),
                dataset.x_dims,
                dataset.y.clone(),
                dataset.y_dims,
            ),
            batch_size,
            shuffle,
        }
    }

    pub fn iter(&self) -> HostDataLoaderIter<'_> {
        let n_samples = self.dataset.x_dims[0] as usize;
        let mut indices: Vec<usize> = (0..n_samples).collect();
        if self.shuffle {
            indices.shuffle(&mut rand::rng());
        }

        HostDataLoaderIter {
            x_data: &self.dataset.x,
            x_dims: self.dataset.x_dims,
            y_data: &self.dataset.y,
            y_dims: self.dataset.y_dims,
            indices,
            batch_size: self.batch_size,
            index: 0,
        }
    }
}

impl<'a> IntoIterator for &'a HostDataLoader {
    type Item = (NodeRef, NodeRef);
    type IntoIter = HostDataLoaderIter<'a>;

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

pub struct HostDataLoaderIter<'a> {
    x_data: &'a [f32],
    x_dims: Dim4,
    y_data: &'a [f32],
    y_dims: Dim4,
    indices: Vec<usize>,
    batch_size: usize,
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

impl Iterator for HostDataLoaderIter<'_> {
    type Item = (NodeRef, NodeRef);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.indices.len() {
            return None;
        }

        let start = self.index;
        let end = (start + self.batch_size).min(self.indices.len());
        self.index = end;

        let batch_indices = &self.indices[start..end];
        let x_batch_host = gather_rows(&self.x_data, self.x_dims, batch_indices);
        let y_batch_host = gather_rows(&self.y_data, self.y_dims, batch_indices);

        let x_batch_dims = Dim4::new(&[
            batch_indices.len() as u64,
            self.x_dims[1],
            self.x_dims[2],
            self.x_dims[3],
        ]);
        let y_batch_dims = Dim4::new(&[
            batch_indices.len() as u64,
            self.y_dims[1],
            self.y_dims[2],
            self.y_dims[3],
        ]);

        let x_batch = Array::new(&x_batch_host, x_batch_dims);
        let y_batch = Array::new(&y_batch_host, y_batch_dims);
        Some((Node::leaf(x_batch, false), Node::leaf(y_batch, false)))
    }
}

fn gather_rows(data: &[f32], dims: Dim4, indices: &[usize]) -> Vec<f32> {
    let n_samples = dims[0] as usize;
    let row_stride = (dims[1] * dims[2] * dims[3]) as usize;
    let mut out = vec![0.0f32; indices.len() * row_stride];

    for (new_row, &old_row) in indices.iter().enumerate() {
        for col in 0..row_stride {
            out[new_row + col * indices.len()] = data[old_row + col * n_samples];
        }
    }

    out
}
