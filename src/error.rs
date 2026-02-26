use std::error::Error as StdError;

pub type Result<T> = std::result::Result<T, Error>;
type AnyError = Box<dyn StdError + Sync + Send>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An error occured during model initialization.
    #[error("Failed to initialize {model}: {error}")]
    ModelInit { model: String, error: AnyError },

    /// The transcription attempt failed
    #[error("Failed to transcribe audio: {0}")]
    Transcribe(whisper_rs::WhisperError),

    /// vad failed
    #[error("Failed to run vad on audio: {0}")]
    Vad(whisper_rs::WhisperError),

    /// ASR backend initializion failed due to an unknown reason.
    /// This should not happen unless the asr thread exits unexpectedly.
    #[error("Failed to initialize asr backend")]
    BackendInit,

    /// Failed to get the speech segments from whisper.
    #[error("Failed to get segment")]
    SegmentParseFailed,

    /// Model download failed
    #[error("Failed to download model: {0}")]
    Download(AnyError),

    /// Worker has been shutdown.
    /// Only returned by AsyncStreamTranscriber
    #[error("Worker has been shutdown: {0}")]
    WorkerShutdown(AnyError),

    /// The model was not found at the path and model downloading is disabled ('model_download' feature is not enabled)
    #[error("{model} Model was not found at {path}. Download it from {url}")]
    ModelNotFound {
        model: String,
        path: String,
        url: String,
    },
}
