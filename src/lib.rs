#![warn(clippy::pedantic)]

mod backend;
mod error;
mod segment;
pub mod whisper;

pub mod util;

pub use backend::Backend;
pub(crate) use backend::BackendImpl;
pub use error::{Error, Result};
pub(crate) mod models;
pub use segment::*;

pub struct StreamTranscriber {
    backend: BackendImpl,
}

impl StreamTranscriber {
    pub fn create(backend: Backend) -> Result<Self> {
        let result = match backend {
            Backend::Whisper(config) => whisper::WhisperBackend::new(config),
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
