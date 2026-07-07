use ndarray::{concatenate, s, Array2, ArrayViewD, Axis};

pub fn gldm_saliency_map(image: ArrayViewD<u8>, attributions: ArrayViewD<f32>,
                         alpha: u8, delta: usize, omit_zeros: bool, mask: Option<ArrayViewD<u8>>) -> Array2<f32> {
    let width = image.shape()[0];
    let height = image.shape()[1];
    let t = image.into_shape((width, height)).unwrap().to_owned();
    let mut padded = concatenate![
        Axis(0), Array2::zeros((delta, height)), t, Array2::zeros((delta, height))
    ];
    padded = concatenate![
        Axis(1), Array2::zeros((width + 2 * delta, delta)), padded, Array2::zeros((width + 2 * delta, delta))
    ];
    let mut padded_map = concatenate![
        Axis(0), Array2::zeros((delta, height)), Array2::zeros((width, height)), Array2::zeros((delta, height))
    ];
    padded_map = concatenate![
        Axis(1), Array2::zeros((width + 2 * delta, delta)), padded_map, Array2::zeros((width + 2 * delta, delta))
    ];
    let mut padded_mask = Array2::zeros((width + 2 * delta, height + 2 * delta));
    if let Some(mask_view) = mask {
        for y in 0..width {
            for x in 0..height {
                padded_mask[[y + delta, x + delta]] = mask_view[[y, x]];
            }
        }
    }
    for y in delta..delta+width {
        for x in delta..delta+height {
            let central_value = padded[[y, x]];
            if central_value == 0 && omit_zeros {
                continue;
            }
            if padded_mask.slice(s![y-delta..y + delta + 1, x-delta..x + delta + 1]).iter().any(|&v| v == 1) {
                continue;
            }
            let slice_parameters = s![y-delta..y + delta + 1, x-delta..x + delta + 1];
            let slice = padded.slice(slice_parameters);
            let diff = slice.mapv(|e| ((central_value as i16 - e as i16).abs() <= alpha as i16) as u8);
            let sum = diff.sum();
            let attribution = attributions[[central_value as usize, sum as usize]];
            // Credit only the pixels that actually formed this feature — the dependent
            // neighbours counted by `diff` (|central - e| <= alpha) — instead of smearing
            // the attribution over the whole (2*delta+1)^2 window, which would also credit
            // non-dependent and zero-padding pixels. Matches the forward feature and the
            // NGTDM back-projection.
            let contribution = diff.mapv(|d| d as f32 * attribution);
            let mut map_slice = padded_map.slice_mut(slice_parameters);
            map_slice += &contribution;
        }
    }
    // Here width = rows (shape[0]) and height = cols (shape[1]); crop the padding back
    // off using each axis's own extent (previously both used `height`, correct only when
    // the image is square).
    padded_map.slice(s![delta..delta + width, delta..delta + height]).to_owned()
}