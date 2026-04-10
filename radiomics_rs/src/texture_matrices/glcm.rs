use ndarray::{s, Array, Array2, ArrayViewD, Axis};

pub fn glcm(image: ArrayViewD<u8>, angle: u16, distance: usize, omit_zeros: bool, mask: Option<ArrayViewD<u8>>) -> Array2<u32> {
    if angle != 0 && angle != 45 && angle != 90 && angle != 135 {
        panic!("specified angle is not supported")
    }
    if distance < 1 {
        panic!("distance cannot be smaller than 1");
    }
    let width = image.shape()[1];
    let height = image.shape()[0];
    let mut glcm = Array2::zeros((256, 256));
    for i in 0..height-distance {
        for j in 0..width-distance {
            let first_pixel = image[[i, j]];
            let i_index = match angle {
                0 => i,
                45 => i + distance,
                90 => i + distance,
                135 => i + distance,
                _ => panic!("invalid angle")
            };
            let j_index = match angle {
                0 => j + distance,
                45 => j + distance,
                90 => j,
                135 => j - distance,
                _ => panic!("invalid angle")
            };
            if i_index < 0 || i_index >= width || j_index < 0 || j_index >= height {
                continue;
            }
            let second_pixel = image[[i_index, j_index]];
            if omit_zeros && first_pixel == 0 && second_pixel == 0 {
                continue;
            }
            if let Some(mask_view) = &mask {
                if mask_view[[i, j]] == 1 || mask_view[[i_index, j_index]] == 1 {
                    continue;
                }
            }
            glcm[[first_pixel as usize, second_pixel as usize]] += 1;
        }
    }
    glcm
}