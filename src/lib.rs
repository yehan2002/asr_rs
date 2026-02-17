#![warn(clippy::pedantic)]

use std::{sync, thread};

mod backend;
mod error;
mod segment;
pub mod util;
pub mod whisper;

pub use backend::Backend;
pub(crate) use backend::{AudioReceiver, BackendImpl};
pub use error::{Error, Result};
pub(crate) mod models;
pub use segment::*;

use tokio::sync::mpsc;

pub struct StreamTranscriber {
    pub audio_tx: mpsc::Sender<Vec<f32>>,
    pub transcribe_rx: mpsc::Receiver<Result<Transcription>>,
}

impl StreamTranscriber {
    pub fn create(backend: Backend) -> Result<Self> {
        // channel used indicate that the asr backend was created successully.
        let (init_tx, init_rx) = sync::mpsc::channel::<Result<()>>();

        let (audio_tx, audio_rx) = mpsc::channel::<Vec<f32>>(1);
        let (transcribe_tx, transcribe_rx) = mpsc::channel::<Result<Transcription>>(100);

        thread::spawn(move || {
            let result = match backend {
                Backend::Whisper(config) => whisper::WhisperBackend::new(config),
            };

            let backend = match result {
                Err(e) => {
                    let _ = init_tx.send(Err(e));
                    return;
                }
                Ok(v) => {
                    // notify successfull startup
                    init_tx
                        .send(Ok(()))
                        .expect("send on init_tx should succeed");

                    v
                }
            };

            backend.process_stream(AudioReceiver {
                audio_rx,
                transcribe_tx,
            });
        });

        init_rx.recv().map_err(|_e| Error::BackendInit).flatten()?;

        Ok(Self {
            audio_tx,
            transcribe_rx,
        })
    }

    pub fn transcribe_audio(&mut self, vec: Vec<f32>) -> Result<Transcription> {
        self.audio_tx.blocking_send(vec).unwrap();
        self.transcribe_rx.blocking_recv().unwrap()
    }

    pub fn finish(self) {
        drop(self.audio_tx);
    }
}
