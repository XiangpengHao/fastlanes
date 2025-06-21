use crate::BitPacking;
use alloc::vec::Vec;

/// Generic select function for bit-packed values that returns packed output
/// 
/// # Arguments
/// * `width` - The bit width used for packing
/// * `packed` - The bit-packed input array 
/// * `bitmask` - 1024-bit mask as 16 u64 words, where 1 means select this element
/// * `output` - Vector to store the selected values in packed format
/// 
/// # Returns
/// Number of elements selected
/// 
/// # Safety
/// The packed array must be properly formatted for the given width.
/// The caller must ensure the packed array has the correct length for the given type and width.
pub unsafe fn select<T: BitPacking>(
    width: usize,
    packed: &[T],
    bitmask: &[u64; 16],
    output: &mut Vec<T>,
) -> usize {
    // First, collect all selected unpacked values
    let mut selected_values = Vec::new();
    
    for (word_idx, &mask_word) in bitmask.iter().enumerate() {
        if mask_word == 0 {
            continue; // Skip words with no bits set
        }
        
        for bit_idx in 0..64 {
            if (mask_word >> bit_idx) & 1 == 1 {
                let global_idx = word_idx * 64 + bit_idx;
                if global_idx < 1024 {
                    let value = unsafe { T::unchecked_unpack_single(width, packed, global_idx) };
                    selected_values.push(value);
                }
            }
        }
    }
    
    let count = selected_values.len();
    
    if count == 0 {
        return 0;
    }
    
    // Pad to multiple of 1024 for efficient packing
    let padded_count = ((count + 1023) / 1024) * 1024;
    selected_values.resize(padded_count, unsafe { core::mem::zeroed() });
    
    // Pack the selected values in chunks of 1024
    for chunk in selected_values.chunks_exact(1024) {
        let chunk_array: &[T; 1024] = chunk.try_into().unwrap();
        let packed_len = 128 * width / core::mem::size_of::<T>();
        let mut packed_chunk = alloc::vec![unsafe { core::mem::zeroed() }; packed_len];
        
        unsafe { T::unchecked_pack(width, chunk_array, &mut packed_chunk) };
        output.extend_from_slice(&packed_chunk);
    }
    
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::array;

    #[test]
    fn test_select_u32() {
        const WIDTH: usize = 10;
        
        // Create test data
        let values: [u32; 1024] = array::from_fn(|i| (i % (1 << WIDTH)) as u32);
        
        // Pack the values
        let mut packed = [0u32; 128 * WIDTH / 4];
        unsafe { u32::unchecked_pack(WIDTH, &values, &mut packed); }
        
        // Create a bitmask that selects every 4th element
        let mut bitmask = [0u64; 16];
        for i in (0..1024).step_by(4) {
            let word_idx = i / 64;
            let bit_idx = i % 64;
            bitmask[word_idx] |= 1u64 << bit_idx;
        }
        
        // Select elements
        let mut output = Vec::new();
        let count = unsafe { select(WIDTH, &packed, &bitmask, &mut output) };
        
        // Verify results
        assert_eq!(count, 256); // 1024 / 4 = 256
        
        // Unpack the result to verify correctness
        let mut unpacked = [0u32; 1024];
        unsafe { u32::unchecked_unpack(WIDTH, &output, &mut unpacked) };
        
        for (idx, &value) in unpacked[..count].iter().enumerate() {
            let original_idx = idx * 4;
            assert_eq!(value, values[original_idx]);
        }
    }

    #[test]
    fn test_select_u16_all_ones() {
        const WIDTH: usize = 8;
        
        // Create test data  
        let values: [u16; 1024] = array::from_fn(|i| (i % (1 << WIDTH)) as u16);
        
        // Pack the values
        let mut packed = [0u16; 128 * WIDTH / 2];
        unsafe { u16::unchecked_pack(WIDTH, &values, &mut packed); }
        
        // Create a bitmask with all 1s
        let bitmask = [u64::MAX; 16];
        
        // Select elements
        let mut output = Vec::new();
        let count = unsafe { select(WIDTH, &packed, &bitmask, &mut output) };
        
        // Verify results - should match all original values
        assert_eq!(count, 1024);
        
        // Unpack and verify
        let mut unpacked = [0u16; 1024];
        unsafe { u16::unchecked_unpack(WIDTH, &output, &mut unpacked) };
        
        for (idx, &value) in unpacked.iter().enumerate() {
            assert_eq!(value, values[idx]);
        }
    }

    #[test]
    fn test_select_u64_empty() {
        const WIDTH: usize = 12;
        
        // Create test data
        let values: [u64; 1024] = array::from_fn(|i| (i % (1 << WIDTH)) as u64);
        
        // Pack the values
        let mut packed = [0u64; 128 * WIDTH / 8];
        unsafe { u64::unchecked_pack(WIDTH, &values, &mut packed); }
        
        // Create an empty bitmask (all zeros)
        let bitmask = [0u64; 16];
        
        // Select elements
        let mut output = Vec::new();
        let count = unsafe { select(WIDTH, &packed, &bitmask, &mut output) };
        
        // Verify results - should be empty
        assert_eq!(count, 0);
        assert_eq!(output.len(), 0);
    }

    #[test]
    fn test_select_u8_sparse() {
        const WIDTH: usize = 5;
        
        // Create test data
        let values: [u8; 1024] = array::from_fn(|i| (i % (1 << WIDTH)) as u8);
        
        // Pack the values
        let mut packed = [0u8; 128 * WIDTH];
        unsafe { u8::unchecked_pack(WIDTH, &values, &mut packed); }
        
        // Create a sparse bitmask - select elements at indices 0, 10, 100, 1000
        let mut bitmask = [0u64; 16];
        for &idx in &[0, 10, 100, 1000] {
            let word_idx = idx / 64;
            let bit_idx = idx % 64;
            bitmask[word_idx] |= 1u64 << bit_idx;
        }
        
        // Select elements
        let mut output = Vec::new();
        let count = unsafe { select(WIDTH, &packed, &bitmask, &mut output) };
        
        // Verify results
        assert_eq!(count, 4);
        
        // Unpack to verify
        let mut unpacked = [0u8; 1024];
        unsafe { u8::unchecked_unpack(WIDTH, &output, &mut unpacked) };
        
        assert_eq!(unpacked[0], values[0]);
        assert_eq!(unpacked[1], values[10]);
        assert_eq!(unpacked[2], values[100]);
        assert_eq!(unpacked[3], values[1000]);
    }
} 