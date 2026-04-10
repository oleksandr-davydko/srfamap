use numpy::Ix2;
use numpy::ndarray::{Array2, ArrayBase, ArrayViewD, OwnedRepr};
use numpy::npyffi::npy_uint8;

fn finish_run_length(
    omit_zeros: bool,
    matrix: &mut ArrayBase<OwnedRepr<u16>, Ix2>,
    now_val: npy_uint8,
    counter: i32) {
    if counter != 0 {
        if omit_zeros && now_val != 0 {
            matrix[[now_val as usize, counter as usize]] += 1;
        } else if !omit_zeros {
            matrix[[now_val as usize, counter as usize]] += 1;
        }
    }
}

pub fn glrlm(image: ArrayViewD<u8>, omit_zeros: bool) -> Array2<u16> {
    let width = image.shape()[1];
    let height = image.shape()[0];
    let mut matrix = Array2::zeros((256, width));
    for i in 0..height {
        let mut now_val: npy_uint8 = 0;
        let mut counter = 0;
        for j in 0..width {
            now_val = image[[i, j]];
            let mut next_val = match j + 1 >= width {
                false => image[[i, j + 1]],
                true => 0
            };
            if next_val != now_val && counter == 0 {
                matrix[[now_val as usize,counter as usize]] += 1
            } else if next_val == now_val {
                counter += 1
            } else {
                finish_run_length(omit_zeros, &mut matrix, now_val, counter);
                counter = 0
            }
        }
    }
    return matrix
}