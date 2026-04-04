// Merkle SHA-256 Oracle — Source of Truth
//
// Spec:
//   Input: 1 MiB chunk (1,048,576 bytes)
//   Leaves: 256 × 4 KiB sub-chunks
//   Leaf hash: SHA-256(sub_chunk)
//   Internal: SHA-256(left_hash || right_hash)
//   Tree: 8 rounds → single 32-byte root
//
// This produces a DIFFERENT hash than flat SHA-256(chunk).
// IC would adopt this under a new StateSyncV5 version.

use sha2::{Sha256, Digest};

pub const SUB_CHUNK_SIZE: usize = 4096;
pub const CHUNK_SIZE: usize = 1_048_576;
pub const NUM_LEAVES: usize = CHUNK_SIZE / SUB_CHUNK_SIZE; // 256

/// Compute Merkle root hash of a 1 MiB chunk.
/// Returns 32-byte root hash.
pub fn merkle_chunk_hash(data: &[u8]) -> [u8; 32] {
    assert_eq!(data.len(), CHUNK_SIZE, "Chunk must be exactly 1 MiB");

    // Phase 1: Hash 256 leaves
    let mut hashes: Vec<[u8; 32]> = Vec::with_capacity(NUM_LEAVES);
    for i in 0..NUM_LEAVES {
        let start = i * SUB_CHUNK_SIZE;
        let end = start + SUB_CHUNK_SIZE;
        let hash: [u8; 32] = Sha256::digest(&data[start..end]).into();
        hashes.push(hash);
    }

    // Phase 2: Build Merkle tree (8 rounds)
    while hashes.len() > 1 {
        let mut next_level: Vec<[u8; 32]> = Vec::with_capacity(hashes.len() / 2);
        for pair in hashes.chunks(2) {
            let mut hasher = Sha256::new();
            hasher.update(&pair[0]);
            hasher.update(&pair[1]);
            let hash: [u8; 32] = hasher.finalize().into();
            next_level.push(hash);
        }
        hashes = next_level;
    }

    hashes[0]
}

/// Compute Merkle root hashes for multiple chunks.
pub fn merkle_batch_hash(data: &[u8], num_chunks: usize) -> Vec<[u8; 32]> {
    assert_eq!(data.len(), num_chunks * CHUNK_SIZE);
    (0..num_chunks)
        .map(|i| {
            let start = i * CHUNK_SIZE;
            merkle_chunk_hash(&data[start..start + CHUNK_SIZE])
        })
        .collect()
}

/// Also output all intermediate hashes for a single chunk (for GPU debugging).
/// Returns (leaf_hashes, level_hashes_per_round, root).
pub fn merkle_chunk_hash_with_intermediates(data: &[u8]) -> (Vec<[u8; 32]>, Vec<Vec<[u8; 32]>>, [u8; 32]) {
    assert_eq!(data.len(), CHUNK_SIZE);

    // Phase 1: Leaves
    let mut hashes: Vec<[u8; 32]> = Vec::with_capacity(NUM_LEAVES);
    for i in 0..NUM_LEAVES {
        let start = i * SUB_CHUNK_SIZE;
        let end = start + SUB_CHUNK_SIZE;
        let hash: [u8; 32] = Sha256::digest(&data[start..end]).into();
        hashes.push(hash);
    }
    let leaves = hashes.clone();

    // Phase 2: Tree
    let mut levels: Vec<Vec<[u8; 32]>> = Vec::new();
    while hashes.len() > 1 {
        let mut next: Vec<[u8; 32]> = Vec::with_capacity(hashes.len() / 2);
        for pair in hashes.chunks(2) {
            let mut hasher = Sha256::new();
            hasher.update(&pair[0]);
            hasher.update(&pair[1]);
            next.push(hasher.finalize().into());
        }
        levels.push(next.clone());
        hashes = next;
    }

    let root = hashes[0];
    (leaves, levels, root)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> Vec<u8> {
        let mut data = vec![0u8; CHUNK_SIZE];
        for i in (0..CHUNK_SIZE).step_by(8) {
            let val = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
            let remaining = CHUNK_SIZE - i;
            let to_copy = remaining.min(8);
            data[i..i + to_copy].copy_from_slice(&val.to_le_bytes()[..to_copy]);
        }
        data
    }

    #[test]
    fn test_deterministic() {
        let data = make_test_data();
        let h1 = merkle_chunk_hash(&data);
        let h2 = merkle_chunk_hash(&data);
        assert_eq!(h1, h2, "Merkle hash must be deterministic");
    }

    #[test]
    fn test_different_from_flat() {
        let data = make_test_data();
        let merkle = merkle_chunk_hash(&data);
        let flat: [u8; 32] = Sha256::digest(&data).into();
        assert_ne!(merkle, flat, "Merkle hash must differ from flat SHA-256");
    }

    #[test]
    fn test_leaf_count() {
        let data = make_test_data();
        let (leaves, levels, root) = merkle_chunk_hash_with_intermediates(&data);
        assert_eq!(leaves.len(), 256);
        assert_eq!(levels.len(), 8); // 128, 64, 32, 16, 8, 4, 2, 1
        assert_eq!(levels[0].len(), 128);
        assert_eq!(levels[7].len(), 1);
        assert_eq!(levels[7][0], root);
    }

    #[test]
    fn test_batch_matches_single() {
        let data = make_test_data();
        let single = merkle_chunk_hash(&data);

        // Make 3 identical chunks
        let mut batch_data = vec![0u8; 3 * CHUNK_SIZE];
        for i in 0..3 {
            batch_data[i * CHUNK_SIZE..(i + 1) * CHUNK_SIZE].copy_from_slice(&data);
        }
        let batch = merkle_batch_hash(&batch_data, 3);
        assert_eq!(batch.len(), 3);
        for h in &batch {
            assert_eq!(h, &single);
        }
    }
}
