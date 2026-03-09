use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

// this is my future optimism that I'll eventually be implementing gpu support
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    CPU,
    GPU,
}

pub trait TensorElement:
    Clone
    + Default
    + fmt::Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Send
    + Sync
    + 'static
{
    fn type_name() -> &'static str;
    fn from_f64(val: f64) -> Self;
}

impl TensorElement for f64 {
    fn type_name() -> &'static str {
        "f64"
    }
    fn from_f64(val: f64) -> Self {
        val
    }
}

impl TensorElement for f32 {
    fn type_name() -> &'static str {
        "f32"
    }
    fn from_f64(val: f64) -> Self {
        val as f32
    }
}

impl TensorElement for i64 {
    fn type_name() -> &'static str {
        "i64"
    }
    fn from_f64(val: f64) -> Self {
        val as i64
    }
}

impl TensorElement for i32 {
    fn type_name() -> &'static str {
        "i32"
    }
    fn from_f64(val: f64) -> Self {
        val as i32
    }
}

pub struct Tensor<T: TensorElement> {
    data: Vec<T>,
    shape: Vec<usize>,
    device: Device,
    requires_grad: bool,
}

impl<T: TensorElement> Tensor<T> {
    pub fn new(
        data: Vec<T>,
        shape: Vec<usize>,
        device: Device,
        requires_grad: bool,
    ) -> Result<Self, String> {
        let expected_size: usize = shape.iter().product();

        if data.len() != expected_size {
            return Err(format!(
                "data length ({}) does not match shape {:?} (expected {})",
                data.len(),
                shape,
                expected_size
            ));
        }

        Ok(Self {
            data,
            shape,
            device,
            requires_grad,
        })
    }

    pub fn zeros(shape: Vec<usize>, device: Device) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![T::default(); size],
            shape,
            device,
            requires_grad: false,
        }
    }

    pub fn ones(shape: Vec<usize>, device: Device) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![T::from_f64(1.0); size],
            shape,
            device,
            requires_grad: false,
        }
    }

    pub fn randn(shape: Vec<usize>, device: Device) -> Self {
        let size: usize = shape.iter().product();
        let data = (0..size)
            .map(|_| T::from_f64(rand::random::<f64>().mul(100f64)))
            .collect();

        Self {
            data,
            shape,
            device,
            requires_grad: false,
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn type_name(&self) -> &'static str {
        T::type_name()
    }

    /// Element-wise addition.  Shape mismatch is the only possible runtime
    /// error — dtype mismatch is a **compile-time** error.
    pub fn add(&self, other: &Self) -> Result<Self, String> {
        if self.shape != other.shape {
            return Err(format!(
                "shape mismatch for addition: {:?} vs {:?}",
                self.shape, other.shape
            ));
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a.clone().add(b.clone()))
            .collect();

        Ok(Self {
            data,
            shape: self.shape.clone(),
            device: self.device,
            requires_grad: false,
        })
    }

    pub fn sub(&self, other: &Self) -> Result<Self, String> {
        if self.shape != other.shape {
            return Err(format!(
                "shape mismatch for subtraction: {:?} vs {:?}",
                self.shape, other.shape
            ));
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a.clone().sub(b.clone()))
            .collect();

        Ok(Self {
            data,
            shape: self.shape.clone(),
            device: self.device,
            requires_grad: false,
        })
    }

    pub fn mul(&self, other: &Self) -> Result<Self, String> {
        if self.shape != other.shape {
            return Err(format!(
                "shape mismatch for multiplication: {:?} vs {:?}",
                self.shape, other.shape
            ));
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a.clone().mul(b.clone()))
            .collect();

        Ok(Self {
            data,
            shape: self.shape.clone(),
            device: self.device,
            requires_grad: false,
        })
    }

    pub fn div(&self, other: &Self) -> Result<Self, String> {
        if self.shape != other.shape {
            return Err(format!(
                "shape mismatch for division: {:?} vs {:?}",
                self.shape, other.shape
            ));
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a.clone().div(b.clone()))
            .collect();

        Ok(Self {
            data,
            shape: self.shape.clone(),
            device: self.device,
            requires_grad: false,
        })
    }

    pub fn matmul(&self, other: &Self) -> Result<Self, String> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(format!(
                "matmul requires 2D tensors, got {:?} and {:?}",
                self.shape, other.shape
            ));
        }

        let (m, k1) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);

        if k1 != k2 {
            return Err(format!(
                "inner dimensions must match for matmul: {:?} vs {:?}",
                self.shape, other.shape
            ));
        }

        let mut data = vec![T::default(); m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = T::default();
                for k in 0..k1 {
                    sum = sum.add(
                        self.data[i * k1 + k]
                            .clone()
                            .mul(other.data[k * n + j].clone()),
                    );
                }
                data[i * n + j] = sum;
            }
        }

        Ok(Self {
            data,
            shape: vec![m, n],
            device: self.device,
            requires_grad: false,
        })
    }

    pub fn dot(&self, other: &Self) -> Result<Self, String> {
        if self.shape.len() != 1 || other.shape.len() != 1 {
            return Err(format!(
                "dot product requires 1D tensors, got {:?} and {:?}",
                self.shape, other.shape
            ));
        }

        let n1 = self.shape[0];
        let n2 = other.shape[0];

        if n1 != n2 {
            return Err(format!(
                "vectors must have the same length for dot product: {:?} vs {:?}",
                self.shape, other.shape
            ));
        }

        let mut sum = T::default();
        for i in 0..n1 {
            sum = sum.add(self.data[i].clone().mul(other.data[i].clone()));
        }

        Ok(Self {
            data: vec![sum],
            shape: vec![],
            device: self.device,
            requires_grad: false,
        })
    }

    pub fn transpose(&self) -> Result<Self, String> {
        if self.shape.len() != 2 {
            return Err(format!(
                "transpose requires a 2D tensor, got shape {:?}",
                self.shape
            ));
        }

        let (m, n) = (self.shape[0], self.shape[1]);
        let mut data = vec![T::default(); m * n];

        for i in 0..m {
            for j in 0..n {
                data[j * m + i] = self.data[i * n + j].clone();
            }
        }

        Ok(Self {
            data,
            shape: vec![n, m],
            device: self.device,
            requires_grad: false,
        })
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, String> {
        let expected_size: usize = new_shape.iter().product();

        if self.data.len() != expected_size {
            return Err(format!(
                "data length ({}) does not match new shape {:?} (expected {})",
                self.data.len(),
                new_shape,
                expected_size
            ));
        }

        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
            device: self.device,
            requires_grad: self.requires_grad,
        })
    }

    pub fn mean(&self) -> Self {
        let sum = self
            .data
            .iter()
            .fold(T::default(), |acc, x| acc.add(x.clone()));
        let count = T::from_f64(self.data.len() as f64);
        Self {
            data: vec![sum.div(count)],
            shape: vec![],
            device: self.device,
            requires_grad: false,
        }
    }

    pub fn sum(&self) -> Self {
        let sum = self
            .data
            .iter()
            .fold(T::default(), |acc, x| acc.add(x.clone()));
        Self {
            data: vec![sum],
            shape: vec![],
            device: self.device,
            requires_grad: false,
        }
    }
}
