use ndarray::{concatenate, s, Array2, ArrayViewD, Axis};

pub fn ngtdm_saliency_map(image: ArrayViewD<u8>, attributions: ArrayViewD<f32>, delta: usize, omit_zeros: bool, mask: Option<ArrayViewD<u8>>) -> Array2<f32> {
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
            let slice = padded.slice(s![y-delta..y + delta + 1, x-delta..x + delta + 1]);
            padded_map[[y, x]] += attributions[[central_value as usize, 0]] + attributions[[central_value as usize, 1]];
            let mut map_slice = padded_map.slice_mut(s![y-delta..y + delta + 1, x-delta..x + delta + 1]);
            let map_attributions = slice.mapv(|e| match e > 0 {
                false => 0.0,
                true => attributions[[central_value as usize, 2]]
            });
            map_slice += &map_attributions;
        }
    }
    // width = rows (shape[0]), height = cols (shape[1]); crop each axis by its own extent
    // (previously transposed, correct only for square images).
    padded_map.slice(s![delta..delta+width, delta..delta+height]).to_owned()
}