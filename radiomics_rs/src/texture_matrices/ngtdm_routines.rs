use ndarray::{Array2, ArrayViewD, Axis, concatenate, s};

pub fn ngtdm(image: ArrayViewD<u8>, delta: usize, omit_zeros: bool, mask: Option<ArrayViewD<u8>>) -> Array2<f32> {
    let width = image.shape()[0];
    let height = image.shape()[1];
    let mut ngtdm = Array2::zeros((256, 3));
    let t = image.into_shape((width, height)).unwrap().to_owned();
    let mut padded = concatenate![
        Axis(0), Array2::zeros((delta, height)), t, Array2::zeros((delta, height))
    ];
    padded = concatenate![
        Axis(1), Array2::zeros((width + 2 * delta, delta)), padded, Array2::zeros((width + 2 * delta, delta))
    ];
    let mut padded_mask = Array2::zeros((width + 2 * delta, height + 2 * delta));
    if let Some(mask_view) = mask {
        for y in 0..width {
            for x in 0..height {
                padded_mask[[y + delta, x + delta]] = mask_view[[y, x]];
            }
        }
    }
    let number_of_voxels = width * height;
    for y in delta..delta+width {
        for x in delta..delta+height {
            let central_value = padded[[y, x]];
            if central_value == 0 && omit_zeros {
                continue;
            }
            if padded_mask.slice(s![y-delta..y + delta + 1, x-delta..x + delta + 1]).iter().any(|&v| v == 1) {
                continue;
            }
            let slice = padded.slice(s![y-delta..y + delta + 1, x-delta..x + delta + 1]);
            let non_zero_count = (slice.mapv(|e| (e > 0) as u8).sum() - 1) as f32;
            if non_zero_count <= 0.0 {
                continue
            }
            let sum = (slice.sum() - central_value) as f32;
            let value: f32 = (central_value as f32 - (sum / non_zero_count)).abs();
            ngtdm[[central_value as usize, 0]] += 1.0;
            ngtdm[[central_value as usize, 1]] = ngtdm[[central_value as usize, 0]] / number_of_voxels as f32;
            ngtdm[[central_value as usize, 2]] += value;
        }
    }
    ngtdm
}