use std::ffi;

use log::{Level, log};

#[cfg(feature = "no_model_log")]
const NO_LOG: bool = true;
#[cfg(not(feature = "no_model_log"))]
const NO_LOG: bool = false;

unsafe extern "C" fn log_cb(
    c_level: std::os::raw::c_uint,
    c_text: *const ::std::os::raw::c_char,
    _user_data: *mut ::std::os::raw::c_void,
) {
    if NO_LOG {
        return;
    }
    /*
     Log levels:
    pub const ggml_log_level_GGML_LOG_LEVEL_NONE: ggml_log_level = 0;
    pub const ggml_log_level_GGML_LOG_LEVEL_DEBUG: ggml_log_level = 1;
    pub const ggml_log_level_GGML_LOG_LEVEL_INFO: ggml_log_level = 2;
    pub const ggml_log_level_GGML_LOG_LEVEL_WARN: ggml_log_level = 3;
    pub const ggml_log_level_GGML_LOG_LEVEL_ERROR: ggml_log_level = 4;
     */
    let level = match c_level {
        0 => {
            // skip loging
            return;
        }
        1 => Level::Debug,
        2 => Level::Info,
        3 => Level::Warn,
        4 => Level::Error,
        _ => Level::Trace,
    };

    // SAFETY: `c_text` is a valid c string and `text` contains an owned copy of the string and can be safely used for logging.
    let text = unsafe { ffi::CStr::from_ptr(c_text).to_string_lossy().into_owned() };

    log!(
        target: "whisper.cpp",
        level, "{}", text.trim_end());
}

/// Set the logger callback for whisper.
pub fn setup_whisper_logger() {
    static ONCE_LOGGER: std::sync::Once = std::sync::Once::new();

    ONCE_LOGGER.call_once(|| {
        // SAFETY: the callback fn does not panic/unwind.
        unsafe {
            whisper_rs::set_log_callback(Some(log_cb), std::ptr::null_mut());
        }
    });
}
