use std::error::Error;

pub type ASRResult<T> = Result<T, ASRError>;

#[derive(Debug, thiserror::Error)]
pub enum ASRError {
    #[error("{model} Model was not found at {path}. Download it from {url}")]
    ModelNotFound {
        model: String,
        path: String,
        url: String,
    },

    #[error("Failed to initialize {model}: {error}")]
    ModelInit {
        model: String,
        error: Box<dyn Error + Sync + Send>,
    },

    #[error("Failed to transcribe audio: {0}")]
    Transcribe(whisper_rs::WhisperError),

    #[error("Failed to run vad on audio: {0}")]
    Vad(whisper_rs::WhisperError),

    #[error("Failed to initialize asr backend")]
    BackendInit,

    #[error("Failed to get segment")]
    SegmentParseFailed,
}
