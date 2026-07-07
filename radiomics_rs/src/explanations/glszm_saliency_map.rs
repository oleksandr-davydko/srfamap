use ndarray::{Array2, ArrayViewD};
use crate::texture_matrices::glszm_routines::calculate_zone_stats;

pub fn glszm_saliency_map(image: ArrayViewD<u8>, attributions: ArrayViewD<f32>, omit_zeros: bool, mask: Option<ArrayViewD<u8>>) -> Array2<f32> {
    let width = image.shape()[1];
    let height = image.shape()[0];
    // Zone sizes larger than the attribution matrix has columns for were excluded when
    // the GLSZM matrix was built (glszm_routines drops freq > max_size - 1), so they
    // carry no attribution. Bound against the actual attribution width instead of a
    // hard-coded size to keep the two consistent and avoid an out-of-bounds panic on
    // images with large uniform zones (e.g. BloodMNIST).
    let max_frequency = attributions.shape()[1];
    let mut map = Array2::zeros((height, width));
    let (zone_map, _, unique_frequency) = calculate_zone_stats(&image, omit_zeros);
    let mut invalid_zone_ids = std::collections::HashSet::new();
    if let Some(mask_view) = mask {
        for i in 0..height {
            for j in 0..width {
                if mask_view[[i, j]] == 1 {
                    let zone_id = zone_map[[i, j]];
                    if zone_id != 0 {
                        invalid_zone_ids.insert(zone_id);
                    }
                }
            }
        }
    }
    for i in 0..height {
        for j in 0..width {
            let intensity = image[[i, j]];
            if omit_zeros && intensity == 0 {
                continue;
            }
            let zone_id = zone_map[[i, j]];
            if invalid_zone_ids.contains(&zone_id) {
                continue;
            }
            if !unique_frequency.contains_key(&zone_id) {
                continue;
            }
            let frequency = unique_frequency.get(&zone_id).unwrap().to_owned();
            if frequency as usize >= max_frequency {
                continue;
            }
            let attribution = attributions[[intensity as usize, frequency as usize]];
            map[[i, j]] = attribution;
        }
    }
    map
}