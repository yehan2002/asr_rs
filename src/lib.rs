use colored::Colorize;
use std::{fmt::Display, fmt::Write, sync, thread};

mod backend;
mod error;
pub mod util;
pub mod whisper;
pub use backend::Backend;
pub(crate) use backend::{AudioReceiver, BackendImpl};
pub use error::{ASRError, ASRResult};

use tokio::sync::mpsc;

#[derive(Debug, serde::Serialize, Clone)]
pub struct PartialSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Debug, serde::Serialize)]
pub enum Segment {
    Silence {
        start: f64,
        end: f64,
    },
    Partial {
        finalized: Vec<PartialSegment>,
        current: Vec<PartialSegment>,
    },
    Full {
        text: String,
    },
}

impl Display for PartialSegment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "[{0:.2} -> {1:.2}] {2}", self.start, self.end, self.text)
    }
}

impl PartialSegment {
    fn format_styled(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        is_finalized: bool,
    ) -> std::fmt::Result {
        let text = format!("[{0:.2} -> {1:.2}] {2}", self.start, self.end, self.text);
        let text = if is_finalized {
            text.green()
        } else {
            text.bright_white()
        };

        writeln!(f, "{text}")
    }
}

impl Display for Segment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Segment::Silence { start, end } => write!(f, "[{start:.2} -> {end:.2}] [Silence]"),
            Segment::Partial { finalized, current } => {
                for part in finalized {
                    let _ = part.format_styled(f, true);
                }

                for part in current {
                    let _ = part.format_styled(f, false);
                }

                Ok(())
            }
            Segment::Full { text } => write!(f, "{text}"),
        }
    }
}

pub struct StreamTranscriber {
    pub audio_tx: mpsc::Sender<Vec<f32>>,
    pub transcribe_rx: mpsc::Receiver<ASRResult<Segment>>,
}

impl StreamTranscriber {
    pub fn create(backend: Backend) -> ASRResult<Self> {
        // channel used indicate that the asr backend was created successully.
        let (init_tx, init_rx) = sync::mpsc::channel::<ASRResult<()>>();

        let (audio_tx, audio_rx) = mpsc::channel::<Vec<f32>>(1);
        let (transcribe_tx, transcribe_rx) = mpsc::channel::<ASRResult<Segment>>(100);

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

        init_rx
            .recv()
            .map_err(|_e| ASRError::BackendInit)
            .flatten()?;

        Ok(Self {
            audio_tx,
            transcribe_rx,
        })
    }

    pub fn transcribe_audio(&mut self, vec: Vec<f32>) -> ASRResult<Segment> {
        self.audio_tx.blocking_send(vec).unwrap();
        self.transcribe_rx.blocking_recv().unwrap()
    }

    pub fn finish(self) {
        drop(self.audio_tx);
    }
}
