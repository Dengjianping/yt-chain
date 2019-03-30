#![cfg(feature = "cuda")]

use libc::{ c_char, c_int };
use std::ffi::{ CStr, CString };
use std::ptr;

pub(crate) type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync + 'static>>;


#[link(name="cuda_mining", kind="static")]
extern {
    fn MD5(a: *const c_char) -> *const c_char;
    fn SHA256(a: *const c_char) -> *const c_char;
    fn SHA384(a: *const c_char) -> *const c_char;
    fn SHA512(a: *const c_char) -> *const c_char;
    fn get_gpu_props() -> *const GPU_Props;
}


#[inline]
// There's bug that generate wrong md5 value, under debug now.
// pub fn cuda_md5<T: AsRef<str>>(t: T) -> Result<String> {
fn cuda_md5<T: AsRef<str>>(t: T) -> Result<String> {
    let c_str = CString::new(t.as_ref())?;
    let r_ptr = unsafe { MD5(c_str.as_ptr()) }; // c_char
    let r_str: &CStr = unsafe { CStr::from_ptr(r_ptr) }; // &CStr
    let str_slice = r_str.to_str()?; // &str
    Ok(str_slice.to_string())
}

#[inline]
pub fn cuda_sha256<T: AsRef<str>>(t: T) -> Result<String> {
    let c_str = CString::new(t.as_ref())?;
    let r_ptr = unsafe { SHA256(c_str.as_ptr()) }; // c_char
    let r_str: &CStr = unsafe { CStr::from_ptr(r_ptr) }; // &CStr
    let str_slice = r_str.to_str()?; // &str
    Ok(str_slice.to_string())
}

#[inline]
pub fn cuda_sha384<T: AsRef<str>>(t: T) -> Result<String> {
    let c_str = CString::new(t.as_ref())?;
    let r_ptr = unsafe { SHA384(c_str.as_ptr()) }; // c_char
    let r_str: &CStr = unsafe { CStr::from_ptr(r_ptr) }; // &CStr
    let str_slice = r_str.to_str()?; // &str
    Ok(str_slice.to_string())
}

#[inline]
pub fn cuda_sha512<T: AsRef<str>>(t: T) -> Result<String> {
    let c_str = CString::new(t.as_ref())?;
    let r_ptr = unsafe { SHA512(c_str.as_ptr()) }; // c_char
    let r_str: &CStr = unsafe { CStr::from_ptr(r_ptr) }; // &CStr
    let str_slice = r_str.to_str()?; // &str
    Ok(str_slice.to_string())
}

#[inline]
pub fn gpu_properties() -> Result<GPU_Props> {
    let props = unsafe {
        let raw_props = get_gpu_props();
        ptr::read(raw_props) 
    };
    Ok(props)
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct GPU_Props {
    pub concurrentKernels: c_int,
    pub maxThreadsPerMultiProcessor: c_int,
    pub multiProcessorCount: c_int,
    pub warpSize: c_int,
}