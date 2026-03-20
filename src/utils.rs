use arrayfire::{Array, Dim4, abs, imax, max_all};

/// Asserts that all elements of two arrays are close within a given tolerance
pub fn assert_all_close(a: &Array<f32>, b: &Array<f32>, tol: f32) {
    let diff = abs(&(a - b));
    let (max_err, _) = max_all(&diff);
    assert!(
        max_err < tol as f32,
        "arrays differ: max error {} > tolerance {}",
        max_err,
        tol
    );
}

/// Parses idx file into an ArrayFire array
///
/// IDX format:
/// - Bytes 0-1: Always 0x00
/// - Byte 2: Data type (0x08 = unsigned byte, 0x09 = signed byte, 0x0B = short, 0x0C = int, 0x0D = float, 0x0E = double)
/// - Byte 3: Number of dimensions
/// - Followed by 4-byte big-endian dimension sizes
/// - Followed by raw data
///
/// # Arguments
/// * `path` - Path to the IDX file
/// * `normalize` - If true, divides by 255.0 (for image data). If false, returns raw values (for label data).
pub fn parse_idx_file(path: &str, normalize: bool) -> Array<f32> {
    let (host_data, dims) = parse_idx_file_host(path, normalize);
    Array::new(&host_data, dims)
}

/// Parses IDX into host memory laid out in ArrayFire's logical column-major order.
pub fn parse_idx_file_host(path: &str, normalize: bool) -> (Vec<f32>, Dim4) {
    let mut file = std::fs::File::open(path).expect("Failed to open file");
    let mut buffer = Vec::new();
    std::io::Read::read_to_end(&mut file, &mut buffer).expect("Failed to read file");

    // Parse magic number
    if buffer[0] != 0 || buffer[1] != 0 {
        panic!("Invalid IDX magic number: first two bytes must be 0");
    }
    let data_type = buffer[2];
    let num_dims = buffer[3] as usize;

    // Only unsigned byte (0x08) is currently supported
    if data_type != 0x08 {
        panic!(
            "Unsupported IDX data type: 0x{:02x} (only 0x08 unsigned byte is supported)",
            data_type
        );
    }

    // Parse dimension sizes (each is 4 bytes, big-endian)
    let mut dims = [1u64; 4]; // Default to 1 for unused dimensions
    for i in 0..num_dims {
        let offset = 4 + i * 4;
        let size = u32::from_be_bytes(buffer[offset..offset + 4].try_into().unwrap()) as u64;
        dims[i] = size;
    }
    // fill with 1s for unused dimensions
    for i in num_dims..4 {
        dims[i] = 1;
    }

    // Calculate where data starts (header is 4 bytes + num_dims*4 bytes)
    let data_offset = 4 + num_dims * 4;
    let data = &buffer[data_offset..];
    let dims_af = Dim4::new(&dims);
    let total_elements = data.len();

    let host_data = if num_dims == 1 {
        data.iter()
            .map(|&value| {
                let value = value as f32;
                if normalize { value / 255.0f32 } else { value }
            })
            .collect()
    } else {
        let logical_dims = &dims[..num_dims];
        let mut out = vec![0.0f32; total_elements];

        for (row_major_offset, &value_u8) in data.iter().enumerate() {
            let mut remainder = row_major_offset;
            let mut indices = [0usize; 4];

            for axis in (0..num_dims).rev() {
                let axis_size = logical_dims[axis] as usize;
                indices[axis] = remainder % axis_size;
                remainder /= axis_size;
            }

            let col_major_offset = indices[0]
                + dims[0] as usize
                    * (indices[1]
                        + dims[1] as usize * (indices[2] + dims[2] as usize * indices[3]));

            let value = value_u8 as f32;
            out[col_major_offset] = if normalize { value / 255.0f32 } else { value };
        }

        out
    };

    (host_data, dims_af)
}

/// One-hot encode labels stored as `[num_samples, 1, 1, 1]` into host memory
/// for an ArrayFire tensor with shape `[num_samples, num_classes, 1, 1]`.
pub fn one_hot_encode_host(labels: &[f32], num_samples: usize, num_classes: usize) -> Vec<f32> {
    let mut one_hot = vec![0.0f32; num_samples * num_classes];
    for (sample_idx, &label) in labels.iter().take(num_samples).enumerate() {
        let class_idx = label as usize;
        one_hot[sample_idx + num_samples * class_idx] = 1.0f32;
    }
    one_hot
}

/// Argmax along dim 1 (class dimension), returning per-sample class indices as u32
pub fn argmax(array: &Array<f32>) -> Array<u32> {
    let (_, indices) = imax(array, 1);
    indices
}
