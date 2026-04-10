use std::collections::{HashMap, VecDeque};

use numpy::Ix2;
use numpy::ndarray::{Array2, ArrayBase, ArrayViewD, Axis, OwnedRepr};

fn update_glszm_frequency(visited_pixels: &mut ArrayBase<OwnedRepr<u32>, Ix2>, mut current_zone_id: u32, unique_frequency: &mut HashMap<u32, u32>, y_current: usize, x_current: usize) {
    visited_pixels[[y_current, x_current]] = current_zone_id;
    let count = unique_frequency.entry(current_zone_id).or_insert(0);
    let incremented = *count + 1;
    unique_frequency.insert(current_zone_id, incremented);
}

fn process_element(queue: &mut VecDeque<(usize, usize)>,
                   omit_zeros: bool,
                   image: &ArrayViewD<u8>,
                   i: usize,
                   j: usize,
                   zone_id: &mut u32,
                   unique_frequency: &mut HashMap::<u32, u32>,
                   visited_pixels: &mut Array2<u32>) {
    if visited_pixels[[i, j]] != 0 {
        return;
    }
    if image[[i, j]] == 0 && omit_zeros {
        return;
    }
    let columns = image.shape()[1];
    let rows = image.shape()[0];
    let degree0equal = j + 1 < columns && image[[i, j]] == image[[i, j + 1]];
    let degree45equal = j + 1 < columns && i + 1 < rows && image[[i + 1, j + 1]] == image[[i, j]];
    let degree90equal = i + 1 < rows && image[[i + 1, j]] == image[[i, j]];
    let degree135equal = i + 1 < rows && j >= 1 && image[[i + 1, j - 1]] == image[[i, j]];
    let degree180equal = j >= 1 && image[[i, j - 1]] == image[[i, j]];
    let degree225equal = j >= 1 && i >= 1 && image[[i - 1, j - 1]] == image[[i, j]];
    let degree270equal = i >= 1 && image[[i - 1, j]] == image[[i, j]];
    let degree315equal = i >= 1 && j + 1 < columns && image[[i - 1, j + 1]] == image[[i, j]];
    if degree0equal {
        queue.push_back((i, j + 1));
    }
    if degree45equal {
        queue.push_back((i + 1, j + 1));
    }
    if degree90equal {
        queue.push_back((i + 1, j));
    }
    if degree135equal {
        queue.push_back((i + 1, j - 1));
    }
    if degree180equal {
        queue.push_back((i, j - 1));
    }
    if degree225equal {
        queue.push_back((i - 1, j - 1));
    }
    if degree270equal {
        queue.push_back((i - 1, j));
    }
    if degree315equal {
        queue.push_back((i - 1, j + 1));
    }
    update_glszm_frequency(visited_pixels, *zone_id, unique_frequency, i, j);

}

pub fn calculate_zone_stats(image: &ArrayViewD<u8>, omit_zeros: bool) -> (ArrayBase<OwnedRepr<u32>, Ix2>, HashMap<u32, u8>, HashMap<u32, u32>) {
    let columns = image.shape()[1];
    let rows = image.shape()[0];
    let mut zone_map = Array2::zeros((rows, columns));
    let mut current_zone_id = 1;
    let mut intensity_to_zone_map = HashMap::<u32, u8>::new();
    let mut unique_frequencies = HashMap::<u32, u32>::new();
    for i in 0..rows {
        for j in 0..columns {
            // skip visited pixel
            let pixel_val = zone_map[[i, j]];
            if pixel_val != 0 {
                continue
            }
            let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
            queue.push_back((i, j));
            while queue.len() > 0 {
                let (x, y) = queue.pop_front().unwrap();
                process_element(&mut queue,
                                omit_zeros,
                                &image,
                                x,
                                y,
                                &mut current_zone_id,
                                &mut unique_frequencies,
                                &mut zone_map)
            }
            intensity_to_zone_map.insert(current_zone_id, image[[i, j]]);
            current_zone_id = current_zone_id + 1;
        }
    }
    intensity_to_zone_map.insert(current_zone_id, image[[image.len_of(Axis(0)) - 1, image.len_of(Axis(1)) - 1]]);
    (zone_map, intensity_to_zone_map, unique_frequencies)
}

pub fn glszm(image: ArrayViewD<u8>, omit_zeros: bool, fixed_max_size: usize) -> (Array2<u16>, Array2<u32>) {
    let (zone_map, intensity_map, unique_frequency) = calculate_zone_stats(&image, omit_zeros);
    if unique_frequency.len() == 0 {
        return (Array2::zeros((256, fixed_max_size)), zone_map);
    }
    let max_size = match fixed_max_size {
        0 => *(unique_frequency.values().max().unwrap()) as usize,
        _ => fixed_max_size
    };
    let mut glszm = Array2::zeros((256, max_size));
    for (zone, freq) in unique_frequency.into_iter() {
        if !intensity_map.contains_key(&(zone)) {
            println!("Missing key for {zone} zone");
        } else {
            if freq > (max_size - 1).try_into().unwrap()  {
                continue;
            }
            let intensity = intensity_map[&(zone)];
            glszm[[intensity as usize, freq as usize]] += 1;
        }
    }
    (glszm, zone_map)
}
