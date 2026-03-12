use crate::error::Result;
use crate::{Backend, Config, Transcription};

pub struct StreamTranscriber {
    backend: Backend,
}

impl StreamTranscriber {
    /// Creates a new audio transciber for the given backend.
    ///
    /// # Errors
    ///
    /// This function will return an error if initializing the backend fails.
    pub fn create(cfg: Config) -> Result<Self> {
        Ok(Self {
            backend: Backend::from_config(cfg.backend)?,
        })
    }

    /// Transcribes the given audio chunk into text.
    /// The audio chunk must be sampled at 16,000hz.
    /// The returned result has all text that has been transcribed upto now (including from previous calls).
    ///
    /// # Errors
    ///
    /// This function will return an error if transcription fails.
    /// The reason for the error depends on the backend type.
    pub fn transcribe_audio(&mut self, vec: Vec<f32>) -> Result<Transcription> {
        self.backend.transcribe_chunk(vec)
    }

    /// Finishes the transcription and returns the total transcription.
    ///
    /// # Errors
    ///
    /// Same as `transcribe_audio`.
    pub fn finish_transcribing(self, last_chunk: Option<Vec<f32>>) -> Result<Transcription> {
        self.backend.finish_transcribing(last_chunk)
    }
}
