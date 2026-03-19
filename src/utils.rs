use arrayfire::{Array, abs, imax, max_all};

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

    // IDX payloads are stored in row-major order, while ArrayFire arrays are
    // column-major. For tensors with more than one axis, load the raw bytes
    // using reversed storage dims so the linear buffer is interpreted
    // correctly, then reorder back to the logical IDX axis order expected by
    // the rest of the codebase.
    let mut storage_dims = [1u64; 4];
    for i in 0..num_dims {
        storage_dims[i] = dims[num_dims - 1 - i];
    }
    for i in num_dims..4 {
        storage_dims[i] = 1;
    }

    let array = Array::new(data, arrayfire::Dim4::new(&storage_dims)).cast::<f32>();
    let result = if num_dims > 1 {
        let mut perm = [0u64, 1, 2, 3];
        for (logical_axis, slot) in perm.iter_mut().enumerate().take(num_dims) {
            *slot = (num_dims - 1 - logical_axis) as u64;
        }
        arrayfire::reorder_v2(&array, perm[0], perm[1], Some(vec![perm[2], perm[3]]))
    } else {
        array
    };

    if normalize {
        result / 255.0f32 // Normalize pixel values to [0, 1]
    } else {
        result // Keep raw values for labels, etc.
    }
}

/// Argmax along dim 1 (class dimension), returning per-sample class indices as u32
pub fn argmax(array: &Array<f32>) -> Array<u32> {
    let (_, indices) = imax(array, 1);
    indices
}
