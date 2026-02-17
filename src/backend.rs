use crate::{Result, Transcription, whisper};
use tokio::sync::mpsc;

pub enum Backend {
    Whisper(whisper::Config),
}

pub(crate) enum BackendImpl {
    Whisper(whisper::WhisperBackend),
}

impl BackendImpl {
    pub(crate) fn process_stream(self, stream: AudioReceiver) {
        match self {
            BackendImpl::Whisper(w) => w.run(stream),
        }
    }
}

pub(crate) struct AudioReceiver {
    pub(crate) audio_rx: mpsc::Receiver<Vec<f32>>,
    pub(crate) transcribe_tx: mpsc::Sender<Result<Transcription>>,
}

impl AudioReceiver {
    pub(crate) fn next_chunk(&mut self) -> Option<Vec<f32>> {
        self.audio_rx.blocking_recv()
    }

    pub(crate) fn send_segment(&self, s: Result<Transcription>) -> Option<()> {
        self.transcribe_tx.blocking_send(s).ok()
    }
}
