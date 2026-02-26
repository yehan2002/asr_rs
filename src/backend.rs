use std::pin::Pin;

use crate::{Result, Transcription, whisper};

pub(crate) enum Backend {
    Whisper(Pin<Box<whisper::WhisperBackend>>),
}

impl Backend {
    pub fn transcribe_chunk(&mut self, audio_chunk: Vec<f32>) -> Result<Transcription> {
        match self {
            Backend::Whisper(w) => w.transcribe_chunk(audio_chunk),
        }
    }

    pub fn finish_transcribing(self, last_chunk: Option<Vec<f32>>) -> Result<Transcription> {
        match self {
            Backend::Whisper(mut w) => w.finish_transcribing(last_chunk),
        }
    }
}
