use crate::{Result, Transcription, whisper};

pub enum Backend {
    Whisper(whisper::Config),
}

pub(crate) enum BackendImpl {
    Whisper(whisper::WhisperBackend),
}

impl BackendImpl {
    pub fn transcribe_chunk(&mut self, audio_chunk: Vec<f32>) -> Result<Transcription> {
        match self {
            BackendImpl::Whisper(w) => w.transcribe_chunk(audio_chunk),
        }
    }

    pub fn finish_transcribing(self) -> Result<Transcription> {
        match self {
            BackendImpl::Whisper(w) => w.finish_transcribing(),
        }
    }
}
