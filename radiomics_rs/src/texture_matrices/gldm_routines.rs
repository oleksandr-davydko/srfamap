use ndarray::{Array2, ArrayViewD, Axis, concatenate, s};

pub fn gldm(image: ArrayViewD<u8>, alpha: u8, delta: usize, omit_zeros: bool, mask: Option<ArrayViewD<u8>>) -> Array2<u16> {
    let width = image.shape()[0];
    let height = image.shape()[1];
    let mut gldm = Array2::zeros((256, 256));
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
            let diff = slice.mapv(|e| ((central_value as i16 - e as i16).abs() <= alpha as i16) as u8);
            let sum = diff.sum();
            gldm[[central_value as usize, sum as usize]] += 1;
        }
    }
    gldm
}