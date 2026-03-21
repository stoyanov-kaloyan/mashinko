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

/// Parse a CIFAR-10 binary batch file (10000 records, each 1 label + 3072 image bytes).
/// Returns images as host data in shape [N, 32, 32, 3] (NHWC logical layout),
/// and labels as class indices in shape [N, 1, 1, 1].
pub fn parse_cifar10_batch_host(path: &str, normalize: bool) -> (Vec<f32>, Dim4, Vec<f32>, Dim4) {
    let mut file = std::fs::File::open(path).expect("Failed to open CIFAR-10 batch file");
    let mut buffer = Vec::new();
    std::io::Read::read_to_end(&mut file, &mut buffer).expect("Failed to read CIFAR-10 file");

    const IMAGE_SIZE: usize = 32 * 32 * 3;
    const RECORD_SIZE: usize = 1 + IMAGE_SIZE;
    assert!(
        buffer.len() % RECORD_SIZE == 0,
        "Invalid CIFAR-10 file length: {}",
        buffer.len()
    );

    let num_samples = buffer.len() / RECORD_SIZE;
    let mut images = vec![0.0f32; num_samples * IMAGE_SIZE];
    let mut labels = vec![0.0f32; num_samples];

    // CIFAR layout per record: [label][1024 R][1024 G][1024 B]
    // Output layout: [N, H, W, C] in ArrayFire logical indexing.
    for sample in 0..num_samples {
        let base = sample * RECORD_SIZE;
        labels[sample] = buffer[base] as f32;

        for channel in 0..3usize {
            for pixel in 0..1024usize {
                let row = pixel / 32;
                let col = pixel % 32;
                let src = base + 1 + channel * 1024 + pixel;
                let mut v = buffer[src] as f32;
                if normalize {
                    v /= 255.0f32;
                }
                let dst = sample
                    + num_samples * (row + 32usize * (col + 32usize * channel));
                images[dst] = v;
            }
        }
    }

    (
        images,
        Dim4::new(&[num_samples as u64, 32, 32, 3]),
        labels,
        Dim4::new(&[num_samples as u64, 1, 1, 1]),
    )
}

/// Parse all CIFAR-10 training batches from a directory containing data_batch_1..5.bin.
pub fn parse_cifar10_train_host(
    dir: &str,
    normalize: bool,
) -> (Vec<f32>, Dim4, Vec<f32>, Dim4) {
    let mut all_images: Vec<f32> = Vec::new();
    let mut all_labels: Vec<f32> = Vec::new();
    let mut total_samples = 0u64;

    for batch in 1..=5 {
        let path = format!("{dir}/data_batch_{batch}.bin");
        let (images, image_dims, labels, _) = parse_cifar10_batch_host(&path, normalize);
        if total_samples == 0 {
            all_images = vec![0.0f32; (50000 * 32 * 32 * 3) as usize];
            all_labels = vec![0.0f32; 50000];
        }

        let n = image_dims[0] as usize;
        let feature_stride = (32 * 32 * 3) as usize;
        for feature in 0..feature_stride {
            let dst_col_offset = feature * 50000usize;
            let src_col_offset = feature * n;
            for i in 0..n {
                all_images[(total_samples as usize) + i + dst_col_offset] = images[i + src_col_offset];
            }
        }
        for i in 0..n {
            all_labels[(total_samples as usize) + i] = labels[i];
        }
        total_samples += n as u64;
    }

    (
        all_images,
        Dim4::new(&[total_samples, 32, 32, 3]),
        all_labels,
        Dim4::new(&[total_samples, 1, 1, 1]),
    )
}
