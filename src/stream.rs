use crate::error::Result;
use crate::{Backend, Config, Transcription};

pub struct StreamTranscriber {
    backend: Backend,
}

impl StreamTranscriber {
    pub fn create(cfg: Config) -> Result<Self> {
        Ok(Self {
            backend: Backend::from_config(cfg.backend)?,
        })
    }

    pub fn transcribe_audio(&mut self, vec: Vec<f32>) -> Result<Transcription> {
        self.backend.transcribe_chunk(vec)
    }

    pub fn finish_transcribing(self, last_chunk: Option<Vec<f32>>) -> Result<Transcription> {
        self.backend.finish_transcribing(last_chunk)
    }
}
