use crate::{BackendConfig, Result, Transcription};

pub(crate) mod whisper;

pub use whisper::{VadModel as WhisperVadModel, Whisper, WhisperModel};

pub(crate) enum Backend {
    Whisper(whisper::WhisperBackend),
}

impl Backend {
    pub fn from_config(cfg: BackendConfig) -> Result<Self> {
        match cfg {
            BackendConfig::Whisper(config) => {
                whisper::WhisperBackend::new(config).map(Backend::Whisper)
            }
        }
    }

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
