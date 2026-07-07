use ndarray::{Array2, ArrayViewD, s};
use numpy::npyffi::npy_uint8;

pub fn glrlm_saliency_map(image: ArrayViewD<u8>, attributions: ArrayViewD<f32>, omit_zeros: bool, mask: Option<ArrayViewD<u8>>) -> Array2<f32> {
    let width = image.shape()[1];
    let height = image.shape()[0];
    let mut map = Array2::zeros((height, width));
    for i in 0..height {
        let mut now_val: npy_uint8 = 0;
        let mut counter = 0;
        let mut run_masked = false;
        for j in 0..width {
            now_val = image[[i, j]];
            if let Some(mask_view) = &mask {
                run_masked = run_masked || mask_view[[i, j]] == 1;
            }
            let mut next_val = match j + 1 >= width {
                false => image[[i, j + 1]],
                true => 0
            };
            if next_val != now_val && counter == 0 {
                // Single-pixel run. Mirror the forward's omit_zeros handling so isolated
                // background pixels are neither counted (row 0) nor credited here.
                if ((!omit_zeros && now_val == 0) || now_val != 0) && !run_masked {
                    map[[i, j]] = attributions[[now_val as usize, counter]]
                }
                run_masked = false;
            } else if next_val == now_val {
                counter += 1
            } else {
                if ((!omit_zeros && now_val == 0) || now_val != 0) && !run_masked {
                    // The run spans pixels [j-counter, j] inclusive (counter+1 pixels);
                    // the upper bound must be j+1 so the final pixel of the run is credited.
                    let mut slice = map.slice_mut(s![i, (j-counter)..(j+1)]);
                    slice += attributions[[now_val as usize, counter]];
                }
                counter = 0;
                run_masked = false;
            }
        }
    }
    return map
}