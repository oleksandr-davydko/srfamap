use ndarray::{Array2, ArrayViewD, Axis, concatenate, s};

pub fn gldm(image: ArrayViewD<u8>, alpha: u8, delta: usize, omit_zeros: bool) -> Array2<u16> {
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
    for y in delta..delta+width {
        for x in delta..delta+height {
            let central_value = padded[[y, x]];
            if central_value == 0 && omit_zeros {
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