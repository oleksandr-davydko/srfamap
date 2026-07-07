use ndarray::{Array2, ArrayViewD};

fn get_next_pair_coordinates(i: usize, j: usize, distance: usize, angle: u16) -> (usize, usize) {
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
    (i_index, j_index)
}

pub fn glcm_saliency_map(image: ArrayViewD<u8>, attributions: ArrayViewD<f32>, angle: u16, distance: usize, omit_zeros: bool, mask: Option<ArrayViewD<u8>>) -> Array2<f32> {
    if angle != 0 && angle != 45 && angle != 90 && angle != 135 {
        panic!("specified angle is not supported")
    }
    if distance < 1 {
        panic!("distance cannot be smaller than 1");
    }
    let width = image.shape()[1];
    let height = image.shape()[0];
    // map is indexed as map[[row, col]] with row in 0..height, col in 0..width, so it
    // must be allocated (height, width). i_index is a row index (bound by height) and
    // j_index a column index (bound by width); the previous code transposed both, which
    // was harmless only because the images are square.
    let mut map = Array2::zeros((height, width));
    for i in 0..height-distance {
        for j in 0..width-distance {
            let first_pixel = image[[i, j]];
            let (i_index, j_index) = get_next_pair_coordinates(i, j, distance, angle);
            if i_index >= height || j_index >= width {
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
            map[[i, j]] += attributions[[first_pixel as usize, second_pixel as usize]];
            map[[i_index, j_index]] += attributions[[first_pixel as usize, second_pixel as usize]];
        }
    }
    map
}