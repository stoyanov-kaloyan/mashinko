use std::path::PathBuf;

use arrayfire::{Array, abs, max_all};

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

    // Create ArrayFire array and cast to f32
    let array = Array::new(data, arrayfire::Dim4::new(&dims));
    let result = array.cast::<f32>();

    if normalize {
        result / 255.0f32 // Normalize pixel values to [0, 1]
    } else {
        result // Keep raw values for labels, etc.
    }
}
