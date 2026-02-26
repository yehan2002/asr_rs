#![warn(clippy::pedantic)]

mod backend;
mod config;
mod error;
mod segment;
mod stream;

#[cfg(feature = "async")]
mod stream_async;
#[cfg(feature = "async")]
pub use stream_async::*;

pub mod util;
pub mod whisper;

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
    pub fn new(cfg: Config) -> Result<Self> {
        return Ok(Transcriber { cfg });
    }

    /// Attempt to download the models given in the config.
    /// This will always fail if `model_download` feature is not enabled.
    pub fn download_models(&self) -> Result<()> {
        self.cfg.backend.download_models()
    }

    /// Transcribe a chunk of audio.
    pub fn transcribe(&self, audio: Vec<f32>) -> Result<Transcription> {
        let mut stream = self.create_stream()?;
        stream.transcribe_audio(audio)?;
        stream.finish_transcribing(None)
    }

    /// Create a transcriber that can trascribe a stream of audio chunks.
    pub fn create_stream(&self) -> Result<StreamTranscriber> {
        return StreamTranscriber::create(self.cfg.clone());
    }

    #[cfg(feature = "async")]
    pub async fn create_async_stream(&self) -> Result<AsyncStreamTranscriber> {
        return AsyncStreamTranscriber::create(self.cfg.clone()).await;
    }
}
