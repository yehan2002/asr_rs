#![warn(clippy::pedantic)]
#![deny(clippy::unwrap_used)]

pub mod backend;
mod config;
mod error;
mod segment;
mod stream;

#[cfg(feature = "async")]
mod stream_async;
#[cfg(feature = "async")]
pub use stream_async::*;

pub mod whisper {
    use crate::backend;

    #[deprecated = "use backend::Whisper instead"]
    pub type Config = backend::whisper::Whisper;
    #[deprecated = "use backend::WhisperModel instead"]
    pub type WhisperModel = backend::whisper::WhisperModel;
    #[deprecated = "use backend::WhisperVadModel instead"]
    pub type VadModel = backend::whisper::VadModel;
}

pub(crate) use backend::Backend;
pub use error::{Error, Result};
pub(crate) mod models;
pub use config::{BackendConfig, Config};
pub use segment::*;
pub use stream::*;

#[derive(Debug, Clone)]
pub struct Transcriber {
    cfg: Config,
}

impl Transcriber {
    /// Creates a new transcriber for the given config.
    ///
    /// # Errors
    /// Currently does not return an error.
    pub fn new(cfg: Config) -> Result<Self> {
        Ok(Transcriber { cfg })
    }

    /// Attempt to download the models given in the config.
    /// This will always fail if `model_download` feature is not enabled.
    ///
    /// # Errors
    ///
    /// This will return an error if model downloading fails or if the model is not downloaded and the `model_download` feature is disabled.
    pub fn download_models(&self) -> Result<()> {
        self.cfg.backend.download_models()
    }

    /// Transcribe the given audio.
    /// The audio must be sampled at 16,000hz.
    /// For transcribing live audio use a `StreamTranscriber`.
    ///
    /// # Errors
    /// Returns an error if transcription fails.
    pub fn transcribe(&self, audio: Vec<f32>) -> Result<Transcription> {
        let mut stream = self.create_stream()?;
        stream.transcribe_audio(audio)?;
        stream.finish_transcribing(None)
    }

    /// Create a transcriber that can trascribe a stream of audio chunks.
    ///
    /// # Errors
    ///
    /// This function will return an error if initializing the backend fails.
    pub fn create_stream(&self) -> Result<StreamTranscriber> {
        StreamTranscriber::create(self.cfg.clone())
    }

    #[cfg(feature = "async")]
    /// Create an async transcriber that can trascribe a stream of audio chunks.
    ///
    /// # Errors
    ///
    /// This function will return an error if initializing the backend fails.
    pub async fn create_async_stream(&self) -> Result<AsyncStreamTranscriber> {
        AsyncStreamTranscriber::create(self.cfg.clone()).await
    }
}
