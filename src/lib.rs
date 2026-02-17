use std::{
    error::Error,
    sync::{
        self,
        mpsc::{Receiver, Sender},
    },
    thread::{self, JoinHandle},
};

use serde::Serialize;

pub mod util;
pub mod whisper;

pub type ASRResult<T> = Result<T, ASRError>;

#[derive(Debug, thiserror::Error)]
pub enum ASRError {
    #[error("{model} Model was not found at {path}. Download it from {url}")]
    ModelNotFound {
        model: String,
        path: String,
        url: String,
    },

    #[error("Failed to initialize {model}: {error}")]
    ModelInit {
        model: String,
        error: Box<dyn Error + Sync + Send>,
    },

    #[error("Failed to transcribe audio: {0}")]
    Transcribe(whisper_rs::WhisperError),

    #[error("Failed to run vad on audio: {0}")]
    Vad(whisper_rs::WhisperError),
}

#[derive(Debug, Serialize)]
pub enum Segment {
    Partial { start: f64, end: f64, text: String },
    Full { text: String },
}

pub struct StreamTranscriber {
    transcribe_thread: JoinHandle<()>,
    pub audio_tx: Sender<Vec<f32>>,
    pub transcribe_rx: Receiver<ASRResult<Segment>>,
}

pub enum Backend {
    Whisper(whisper::Config),
}

impl StreamTranscriber {
    pub fn new(backend: Backend) -> ASRResult<Self> {
        let (init_tx, init_rx) = sync::mpsc::channel::<ASRResult<()>>();
        let (audio_tx, audio_rx) = sync::mpsc::channel::<Vec<f32>>();
        let (transcribe_tx, transcribe_rx) = sync::mpsc::channel::<ASRResult<Segment>>();

        let asr_thread = thread::spawn(move || {
            let result = match backend {
                Backend::Whisper(config) => whisper::WhisperBackend::new(config),
            };

            let backend = match result {
                Err(e) => {
                    init_tx
                        .send(Err(e))
                        .expect("send on init_tx should succeed");
                    return;
                }
                Ok(v) => {
                    init_tx
                        .send(Ok(()))
                        .expect("send on init_tx should succeed");
                    v
                }
            };

            backend.run(audio_rx, transcribe_tx);
        });

        init_rx.recv().expect("init message should be received")?;

        Ok(Self {
            transcribe_thread: asr_thread,
            audio_tx,
            transcribe_rx,
        })
    }

    pub fn send_audio(&self, vec: &[f32]) {
        self.audio_tx.send(vec.to_vec()).unwrap()
    }

    pub fn finish(self) {
        drop(self.audio_tx);
        self.transcribe_thread.join().unwrap();
    }
}
