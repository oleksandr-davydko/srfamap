extern crate numpy;
extern crate serde;
use numpy::ndarray::{Array2, Dimension};
use ndarray_npy::{read_npy, ReadNpyError, ReadNpyExt, WriteNpyError, WriteNpyExt};
use std::fs::File;
use ndarray::s;
use crate::explanations::ngtdm_saliency_map::ngtdm_saliency_map;
use crate::explanations::glrlm_saliency_map_routines::glrlm_saliency_map;
use crate::explanations::gldm_saliency_map::gldm_saliency_map;
use crate::texture_matrices::glcm::glcm;
use crate::texture_matrices::gldm_routines::gldm;
use crate::texture_matrices::glszm_routines::glszm;
use crate::texture_matrices::ngtdm_routines::ngtdm;
use crate::texture_matrices::glrlm_routines::glrlm;


#[test]
fn test_glcm_1() {
    let vec: Vec<[u8; 5]>= vec![
        [1, 2, 5, 2, 3],
        [3, 2, 1, 3, 1],
        [1, 3, 5, 5, 2],
        [1, 1, 1, 1, 2],
        [1, 2, 4, 3, 5]
    ];
    let test_input = Array2::from(vec).into_dyn();
    let input_view = test_input.view();
    let result = glcm(input_view, 0, 1, true, None);
    let actual_result = result.slice(s![1..6, 1..6]);

    let expected_vec: Vec<[u32; 5]> = vec![
        [6, 4, 3, 0, 0],
        [4, 0, 2, 1, 3],
        [3, 2, 0, 1, 2],
        [0, 1, 1, 0, 0],
        [0, 3, 2, 0, 2]
    ];
    let expected_result = Array2::from(expected_vec);
    println!("{}", actual_result);
    println!("{}", expected_result);
    assert_eq!(actual_result == expected_result, true);
}


#[test]
fn test_glrlm_1() {
    let vec: Vec<[u8; 4]>= vec![
        [1, 2, 3, 4],
        [1, 3, 4, 4],
        [3, 2, 2, 2],
        [4, 1, 4, 1]
    ];
    let test_input = Array2::from(vec).into_dyn();
    let input_view = test_input.view();
    let result = glrlm(input_view, true, None);
    let actual_result = result.slice(s![1..5, 0..4]);

    let expected_vec: Vec<[u16; 4]> = vec![
        [4, 0, 0, 0],
        [1, 0, 1, 0],
        [3, 0, 0, 0],
        [3, 1, 0, 0],
    ];
    let expected_result = Array2::from(expected_vec);
    println!("{}", actual_result);
    println!("{}", expected_result);
    assert_eq!(actual_result == expected_result, true);
}

#[test]
fn test_glrlm_2() {
    let vec: Vec<[u8; 5]>= vec![
        [5, 2, 5, 4, 4],
        [3, 3, 3, 1, 3],
        [2, 1, 1, 1, 3],
        [4, 2, 2, 2, 3],
        [3, 5, 3, 3, 2]
    ];
    let test_input = Array2::from(vec).into_dyn();
    let result = glrlm(test_input.view(), true, None);

    let actual_result = result.slice(s![1..6, 0..5]);

    let expected_vec: Vec<[u16; 5]> = vec![
        [1, 0, 1, 0, 0],
        [3, 0, 1, 0, 0],
        [4, 1, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [3, 0, 0, 0, 0]
    ];
    let expected_result = Array2::from(expected_vec);
    println!("{}", actual_result);
    println!("{}", expected_result);
    assert_eq!(actual_result == expected_result, true);
}


#[test]
fn test_glszm_1() {
    let vec: Vec<[u8; 5]>= vec![
        [5, 2, 5, 4, 4],
        [3, 3, 3, 1, 3],
        [2, 1, 1, 1, 3],
        [4, 2, 2, 2, 3],
        [3, 5, 3, 3, 2]
    ];
    let test_input = Array2::from(vec).into_dyn();
    let (result, mp) = glszm(test_input.view(), true, 5, None);

    let actual_result = result.slice(s![1..6, 1..6]);

    let expected_vec: Vec<[u16; 5]> = vec![
        [0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 1, 0, 0, 0],
        [3, 0, 0, 0, 0]
    ];
    let expected_result = Array2::from(expected_vec);
    println!("{}", mp);
    println!("{}", actual_result);
    println!("{}", expected_result);
    assert_eq!(actual_result == expected_result, true);
}

#[test]
fn test_glszm_3() {
    let test_input: Array2<u8> = read_npy("C:/temp/test.npy").unwrap();
    let (result, mp) = glszm(test_input.into_dyn().view(), true, 5, None);

    let actual_result = result.slice(s![1..5, 1..5]);

    let expected_vec: Vec<[u16; 5]> = vec![
        [0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 1, 0, 0, 0],
        [3, 0, 0, 0, 0]
    ];
    let expected_result = Array2::from(expected_vec);
    println!("{}", mp);
    println!("{}", actual_result);
    println!("{}", expected_result);
    assert_eq!(actual_result == expected_result, true);
}

#[test]
fn test_glszm_2() {
    let vec: Vec<[u8; 4]>= vec![
        [1, 2, 3, 4],
        [1, 3, 4, 4],
        [3, 2, 2, 2],
        [4, 1, 4, 1]
    ];
    let test_input = Array2::from(vec).into_dyn();
    let (result, mp) = glszm(test_input.view(), true, 5, None);

    let actual_result = result.slice(s![1..5, 1..5]);

    let expected_vec: Vec<[u16; 4]> = vec![
        [2, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 0, 1, 0],
        [2, 0, 1, 0],
    ];
    let expected_result = Array2::from(expected_vec);
    println!("{}", mp);
    println!("{}", actual_result);
    println!("{}", expected_result);
    assert_eq!(actual_result == expected_result, true);
}

#[test]
fn test_gldm_1() {
    let vec: Vec<[u8; 5]>= vec![
        [5, 2, 5, 4, 4],
        [3, 3, 3, 1, 3],
        [2, 1, 1, 1, 3],
        [4, 2, 2, 2, 3],
        [3, 5, 3, 3, 2]
    ];
    let test_input = Array2::from(vec).into_dyn();
    let result = gldm(test_input.view(), 0, 1, true, None);

    let actual_result = result.slice(s![1..6, 1..5]);

    let expected_vec: Vec<[u16; 4]> = vec![
        [0, 1, 2, 1],
        [1, 2, 3, 0],
        [1, 4, 4, 0],
        [1, 2, 0, 0],
        [3, 0, 0, 0]
    ];
    let expected_result = Array2::from(expected_vec);
    println!("{}", actual_result);
    println!("{}", expected_result);
    assert_eq!(actual_result == expected_result, true);
}


#[test]
fn test_gldm_2() {
    let test_input: Array2<u8> = read_npy("C:/temp/test.npy").unwrap();
    let result = gldm(test_input.into_dyn().view(), 0, 1, true, None);

    let actual_result = result.slice(s![1..6, 1..5]);

    let expected_vec: Vec<[u16; 4]> = vec![
        [0, 1, 2, 1],
        [1, 2, 3, 0],
        [1, 4, 4, 0],
        [1, 2, 0, 0],
        [3, 0, 0, 0]
    ];
    let expected_result = Array2::from(expected_vec);
    println!("{}", actual_result);
    println!("{}", expected_result);
    assert_eq!(actual_result == expected_result, true);
}

#[test]
fn test_ngtdm_1() {
    let vec: Vec<[u8; 4]>= vec![
        [1, 2, 5, 2],
        [3, 5, 1, 3],
        [1, 3, 5, 5],
        [3, 1, 1, 1]
    ];
    let test_input = Array2::from(vec).into_dyn();
    let result = ngtdm(test_input.view(), 1, true, None);

    let actual_result = result.slice(s![1..6, 0..3]);

    let expected_vec: Vec<[f32; 3]> = vec![
        [6.0, 0.375, 13.35],
        [2.0, 0.125, 2.0],
        [4.0, 0.25, 3.03],
        [0.0, 0.0, 0.0],
        [4.0, 0.25, 10.075]
    ];
    let expected_result = Array2::from(expected_vec);

    let saliency: Vec<[f32; 3]> = vec![
        [6.0, 0.375, 13.35],
        [2.0, 0.125, 2.0],
        [4.0, 0.25, 3.03],
        [0.0, 0.0, 0.0],
        [4.0, 0.25, 10.075],
        [4.0, 0.25, 10.075]
    ];

    let saliency = ngtdm_saliency_map(test_input.view(), Array2::from(saliency).into_dyn().view(), 1, true, None);
    println!("{}", actual_result);
    println!("{}", saliency);
    //println!("{}", expected_result);
    //assert_eq!(actual_result == expected_result, true);
}

#[test]
fn test_glcm_masked_excludes_pairs() {
    let vec: Vec<[u8; 2]>= vec![
        [1, 2],
        [3, 4],
    ];
    let mask_vec: Vec<[u8; 2]>= vec![
        [0, 1],
        [0, 0],
    ];
    let image = Array2::from(vec).into_dyn();
    let mask = Array2::from(mask_vec).into_dyn();

    let unmasked = glcm(image.view(), 0, 1, false, None);
    let masked = glcm(image.view(), 0, 1, false, Some(mask.view()));

    assert_eq!(unmasked[[1, 2]], 1);
    assert_eq!(masked[[1, 2]], 0);
}

#[test]
fn test_glrlm_masked_excludes_run() {
    let vec: Vec<[u8; 4]>= vec![
        [2, 2, 2, 1],
    ];
    let mask_vec: Vec<[u8; 4]>= vec![
        [0, 1, 0, 0],
    ];
    let image = Array2::from(vec).into_dyn();
    let mask = Array2::from(mask_vec).into_dyn();

    let unmasked = glrlm(image.view(), false, None);
    let masked = glrlm(image.view(), false, Some(mask.view()));

    assert!(unmasked.sum() > masked.sum());
}

#[test]
fn test_glszm_masked_excludes_zone() {
    let vec: Vec<[u8; 3]>= vec![
        [5, 5, 1],
        [5, 5, 1],
        [2, 2, 2],
    ];
    let mask_vec: Vec<[u8; 3]>= vec![
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ];
    let image = Array2::from(vec).into_dyn();
    let mask = Array2::from(mask_vec).into_dyn();

    let (unmasked, _) = glszm(image.view(), false, 10, None);
    let (masked, _) = glszm(image.view(), false, 10, Some(mask.view()));

    assert!(unmasked.sum() > masked.sum());
}

#[test]
fn test_glrlm_saliency_map_credits_full_run() {
    // Regression test for the run back-projection off-by-one (B1): every pixel of a
    // run must receive the run's attribution, including the final pixel. On [2, 2, 1]
    // the length-2 run of 2's must credit BOTH pixels 0 and 1 (not just pixel 0).
    let image = Array2::from(vec![[2u8, 2, 1]]).into_dyn();
    let mut attributions = Array2::<f32>::zeros((256, 4));
    attributions[[2, 1]] = 10.0; // run of two 2's: length 2 -> column 1
    attributions[[1, 0]] = 5.0;  // single 1: length 1 -> column 0
    let attributions = attributions.into_dyn();

    let map = glrlm_saliency_map(image.view(), attributions.view(), true, None);

    let expected = Array2::from(vec![[10.0f32, 10.0, 5.0]]);
    assert_eq!(map, expected);
}

#[test]
fn test_glrlm_omit_zeros_excludes_background_singletons() {
    // B4: with omit_zeros, an isolated background (0) pixel must be excluded from the
    // GLRLM matrix (row 0) in the forward AND receive no attribution in the map, keeping
    // the two consistent. On [3, 0, 3] the middle background pixel must stay uncredited.
    let image = Array2::from(vec![[3u8, 0, 3]]).into_dyn();

    let matrix = glrlm(image.view(), true, None);
    assert_eq!(matrix[[0, 0]], 0); // background singleton not counted
    assert_eq!(matrix[[3, 0]], 2); // the two isolated 3's

    let mut attributions = Array2::<f32>::zeros((256, 4));
    attributions[[3, 0]] = 7.0;
    attributions[[0, 0]] = 99.0; // must never be applied
    let attributions = attributions.into_dyn();
    let map = glrlm_saliency_map(image.view(), attributions.view(), true, None);

    let expected = Array2::from(vec![[7.0f32, 0.0, 7.0]]);
    assert_eq!(map, expected);
}

#[test]
fn test_gldm_saliency_map_credits_only_dependent_pixels() {
    // B3: attribution must land only on the dependent pixels (|central - e| <= alpha),
    // not the whole (2*delta+1)^2 window. With alpha=0 and all-distinct values every
    // pixel is dependent only on itself, so a bin set for the centre pixel (value 5,
    // dependency count 1) must credit ONLY the centre pixel, not its 8 neighbours.
    let image = Array2::from(vec![
        [1u8, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]).into_dyn();
    let mut attributions = Array2::<f32>::zeros((256, 16));
    attributions[[5, 1]] = 100.0;
    let attributions = attributions.into_dyn();

    let map = gldm_saliency_map(image.view(), attributions.view(), 0, 1, false, None);

    let mut expected = Array2::<f32>::zeros((3, 3));
    expected[[1, 1]] = 100.0;
    assert_eq!(map, expected);
}