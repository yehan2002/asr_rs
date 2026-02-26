use std::pin::Pin;

use crate::{Result, Transcription, whisper};

pub enum Backend {
    Whisper(whisper::Config),
}

pub(crate) enum BackendImpl {
    Whisper(Pin<Box<whisper::WhisperBackend>>),
}

impl BackendImpl {
    pub fn transcribe_chunk(&mut self, audio_chunk: Vec<f32>) -> Result<Transcription> {
        match self {
            BackendImpl::Whisper(w) => w.transcribe_chunk(audio_chunk),
        }
    }

    pub fn finish_transcribing(self, last_chunk: Option<Vec<f32>>) -> Result<Transcription> {
        match self {
            BackendImpl::Whisper(mut w) => w.finish_transcribing(last_chunk),
        }
    }
}
