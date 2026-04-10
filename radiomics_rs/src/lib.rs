extern crate itertools;
extern crate numpy;
extern crate pyo3;

mod explanations;
pub mod texture_matrices;
mod tests;

use itertools::Itertools;
use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn};
use numpy::ndarray::ArrayD;
use numpy::ndarray::Dimension;
use numpy::ndarray::prelude::*;
use pyo3::{pyfunction, pymodule, types::{PyAnyMethods, PyDictMethods, PyModule}, wrap_pyfunction, Bound, FromPyObject, PyResult, Python};
use pyo3::prelude::*;
use pyo3::prelude::PyModuleMethods;
use pyo3::exceptions::PyValueError;
use texture_matrices::glszm_routines::glszm as glszm_internal;
use texture_matrices::glcm::glcm as glcm_internal;
use texture_matrices::glrlm_routines::glrlm as glrlm_internal;
use texture_matrices::gldm_routines::gldm as gldm_internal;
use texture_matrices::ngtdm_routines::ngtdm as ngtdm_internal;
use explanations::glcm_saliency_map_routines::glcm_saliency_map as glcm_saliency_map_internal;
use explanations::glrlm_saliency_map_routines::glrlm_saliency_map as glrlm_saliency_map_internal;
use explanations::glszm_saliency_map::glszm_saliency_map as glszm_saliency_map_internal;
use explanations::gldm_saliency_map::gldm_saliency_map as gldm_saliency_map_internal;
use explanations::ngtdm_saliency_map::ngtdm_saliency_map as ngtdm_saliency_map_internal;

fn parse_optional_mask<'py>(
    mask: Option<&Bound<'py, PyAny>>,
    image_shape: &[usize]
) -> PyResult<Option<ArrayD<u8>>> {
    let Some(mask_any) = mask else {
        return Ok(None);
    };

    if let Ok(mask_u8) = mask_any.extract::<PyReadonlyArrayDyn<'py, u8>>() {
        let mask_view = mask_u8.as_array();
        if mask_view.shape() != image_shape {
            return Err(PyValueError::new_err(format!(
                "mask shape {:?} must match image shape {:?}",
                mask_view.shape(),
                image_shape
            )));
        }
        return Ok(Some(mask_view.to_owned().into_dyn()));
    }

    if let Ok(mask_bool) = mask_any.extract::<PyReadonlyArrayDyn<'py, bool>>() {
        let mask_view = mask_bool.as_array();
        if mask_view.shape() != image_shape {
            return Err(PyValueError::new_err(format!(
                "mask shape {:?} must match image shape {:?}",
                mask_view.shape(),
                image_shape
            )));
        }
        let converted = mask_view.mapv(|v| if v { 1u8 } else { 0u8 }).into_dyn();
        return Ok(Some(converted));
    }

    Err(PyValueError::new_err(
        "mask must be a numpy ndarray with dtype uint8 or bool"
    ))
}

#[pyfunction]
#[pyo3(signature = (image, angle, distance, omit_zeros, mask=None))]
fn glcm<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    angle: u16,
    distance: usize,
    omit_zeros: bool,
    mask: Option<&Bound<'py, PyAny>>
) -> PyResult<Bound<'py, PyArrayDyn<u32>>> {
    let x = image.as_array();
    let mask_array = parse_optional_mask(mask, x.shape())?;
    let z = glcm_internal(x, angle, distance, omit_zeros, mask_array.as_ref().map(|m| m.view())).into_dyn();
    Ok(z.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (image, omit_zeros, mask=None))]
fn glrlm<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    omit_zeros: bool,
    mask: Option<&Bound<'py, PyAny>>
) -> PyResult<Bound<'py, PyArrayDyn<u16>>> {
    let x = image.as_array();
    let mask_array = parse_optional_mask(mask, x.shape())?;
    let z = glrlm_internal(x, omit_zeros, mask_array.as_ref().map(|m| m.view())).into_dyn();
    Ok(z.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (image, attributions, angle, distance, omit_zeros, mask=None))]
fn glcm_saliency_map<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    attributions: PyReadonlyArrayDyn<'py, f32>,
    angle: u16,
    distance: usize,
    omit_zeros: bool,
    mask: Option<&Bound<'py, PyAny>>
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let x = image.as_array();
    let a = attributions.as_array();
    let mask_array = parse_optional_mask(mask, x.shape())?;
    let z = glcm_saliency_map_internal(x, a, angle, distance, omit_zeros, mask_array.as_ref().map(|m| m.view())).into_dyn();
    Ok(z.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (image, attributions, omit_zeros, mask=None))]
fn glrlm_saliency_map<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    attributions: PyReadonlyArrayDyn<'py, f32>,
    omit_zeros: bool,
    mask: Option<&Bound<'py, PyAny>>
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let x = image.as_array();
    let a = attributions.as_array();
    let mask_array = parse_optional_mask(mask, x.shape())?;
    let z = glrlm_saliency_map_internal(x, a, omit_zeros, mask_array.as_ref().map(|m| m.view())).into_dyn();
    Ok(z.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (image, attributions, omit_zeros, mask=None))]
fn glszm_saliency_map<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    attributions: PyReadonlyArrayDyn<'py, f32>,
    omit_zeros: bool,
    mask: Option<&Bound<'py, PyAny>>
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let x = image.as_array();
    let a = attributions.as_array();
    let mask_array = parse_optional_mask(mask, x.shape())?;
    let z = glszm_saliency_map_internal(x, a, omit_zeros, mask_array.as_ref().map(|m| m.view())).into_dyn();
    Ok(z.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (image, attributions, alpha, delta, omit_zeros, mask=None))]
fn gldm_saliency_map<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    attributions: PyReadonlyArrayDyn<'py, f32>,
    alpha: u8,
    delta: usize,
    omit_zeros: bool,
    mask: Option<&Bound<'py, PyAny>>
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let x = image.as_array();
    let a = attributions.as_array();
    let mask_array = parse_optional_mask(mask, x.shape())?;
    let z = gldm_saliency_map_internal(x, a, alpha, delta, omit_zeros, mask_array.as_ref().map(|m| m.view())).into_dyn();
    Ok(z.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (image, omit_zeros, max_size, mask=None))]
fn glszm<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    omit_zeros: bool,
    max_size: usize,
    mask: Option<&Bound<'py, PyAny>>
) -> PyResult<Bound<'py, PyArrayDyn<u16>>> {
    let x = image.as_array();
    let mask_array = parse_optional_mask(mask, x.shape())?;
    let (z, _) = glszm_internal(x, omit_zeros, max_size, mask_array.as_ref().map(|m| m.view()));
    Ok(z.into_dyn().into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (image, alpha, delta, omit_zeros, mask=None))]
fn gldm<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    alpha: u8,
    delta: usize,
    omit_zeros: bool,
    mask: Option<&Bound<'py, PyAny>>
) -> PyResult<Bound<'py, PyArrayDyn<u16>>> {
    let x = image.as_array();
    let mask_array = parse_optional_mask(mask, x.shape())?;
    let z = gldm_internal(x, alpha, delta, omit_zeros, mask_array.as_ref().map(|m| m.view()));
    Ok(z.into_dyn().into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (image, delta, omit_zeros, mask=None))]
fn ngtdm<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    delta: usize,
    omit_zeros: bool,
    mask: Option<&Bound<'py, PyAny>>
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let x = image.as_array();
    let mask_array = parse_optional_mask(mask, x.shape())?;
    let z = ngtdm_internal(x, delta, omit_zeros, mask_array.as_ref().map(|m| m.view()));
    Ok(z.into_dyn().into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (image, attributions, delta, omit_zeros, mask=None))]
fn ngtdm_saliency_map<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    attributions: PyReadonlyArrayDyn<'py, f32>,
    delta: usize,
    omit_zeros: bool,
    mask: Option<&Bound<'py, PyAny>>
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let x = image.as_array();
    let mask_array = parse_optional_mask(mask, x.shape())?;
    let z = ngtdm_saliency_map_internal(x, attributions.as_array(), delta, omit_zeros, mask_array.as_ref().map(|m| m.view()));
    Ok(z.into_dyn().into_pyarray_bound(py))
}

#[pymodule]
#[pyo3(name="radiomics_rs")]
fn radiomics_rs<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_wrapped(wrap_pyfunction!(glcm))?;
    m.add_wrapped(wrap_pyfunction!(glrlm))?;
    m.add_wrapped(wrap_pyfunction!(glszm))?;
    m.add_wrapped(wrap_pyfunction!(gldm))?;
    m.add_wrapped(wrap_pyfunction!(ngtdm))?;
    m.add_wrapped(wrap_pyfunction!(glcm_saliency_map))?;
    m.add_wrapped(wrap_pyfunction!(glrlm_saliency_map))?;
    m.add_wrapped(wrap_pyfunction!(glszm_saliency_map))?;
    m.add_wrapped(wrap_pyfunction!(gldm_saliency_map))?;
    m.add_wrapped(wrap_pyfunction!(ngtdm_saliency_map))?;
    Ok(())
}
