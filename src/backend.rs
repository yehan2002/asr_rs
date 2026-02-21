use crate::{Result, Transcription, whisper};

pub(crate) enum Backend {
    Whisper(whisper::WhisperBackend),
}

impl Backend {
    pub fn transcribe_chunk(&mut self, audio_chunk: Vec<f32>) -> Result<Transcription> {
        match self {
            Backend::Whisper(w) => w.transcribe_chunk(audio_chunk),
        }
    }

    pub fn finish_transcribing(self) -> Result<Transcription> {
        match self {
            Backend::Whisper(w) => w.finish_transcribing(),
        }
    }
}
