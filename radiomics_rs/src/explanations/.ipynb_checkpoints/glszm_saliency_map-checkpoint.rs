use ndarray::{Array2, ArrayViewD};
use crate::texture_matrices::glszm_routines::calculate_zone_stats;

pub fn glszm_saliency_map(image: ArrayViewD<u8>, attributions: ArrayViewD<f32>, omit_zeros: bool) -> Array2<f32> {
    let width = image.shape()[1];
    let height = image.shape()[0];
    let mut map = Array2::zeros((height, width));
    let (zone_map, _, unique_frequency) = calculate_zone_stats(&image, omit_zeros);
    for i in 0..height {
        for j in 0..width {
            let intensity = image[[i, j]];
            if omit_zeros && intensity == 0 {
                continue;
            }
            let zone_id = zone_map[[i, j]];
            if !unique_frequency.contains_key(&zone_id) {
                continue;
            }
            let frequency = unique_frequency.get(&zone_id).unwrap().to_owned();
            if frequency > 223 {
                continue;
            }
            let attribution = attributions[[intensity as usize, frequency as usize]];
            map[[i, j]] = attribution;
        }
    }
    map
}