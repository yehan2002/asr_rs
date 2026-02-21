#![warn(clippy::pedantic)]

mod backend;
mod config;
mod error;
mod segment;
pub mod util;
pub mod whisper;

pub(crate) use backend::Backend;
pub use error::{Error, Result};
pub(crate) mod models;
pub use config::{BackendConfig, Config};
pub use segment::*;

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
        stream.finish_transcribing()
    }

    /// Create a transcriber that can trascribe a stream of audio chunks.
    pub fn create_stream(&self) -> Result<StreamTranscriber> {
        return StreamTranscriber::create(self.cfg.clone());
    }
}

pub struct StreamTranscriber {
    backend: Backend,
}

impl StreamTranscriber {
    pub fn create(cfg: Config) -> Result<Self> {
        let result = match cfg.backend {
            BackendConfig::Whisper(config) => whisper::WhisperBackend::new(config),
        }?;

        Ok(Self { backend: result })
    }

    pub fn transcribe_audio(&mut self, vec: Vec<f32>) -> Result<Transcription> {
        self.backend.transcribe_chunk(vec)
    }

    pub fn finish_transcribing(self) -> Result<Transcription> {
        self.backend.finish_transcribing()
    }
}
