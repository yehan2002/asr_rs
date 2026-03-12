use std::thread;

use crate::{Backend, Config, Error, Transcription, error::Result};
use tokio::sync::{mpsc::Sender, oneshot};

enum Task {
    Transcribe(Vec<f32>, oneshot::Sender<Result<Transcription>>),
    Finish(Option<Vec<f32>>, oneshot::Sender<Result<Transcription>>),
}

pub struct AsyncStreamTranscriber {
    sender: Sender<Task>,
}

impl AsyncStreamTranscriber {
    pub(crate) async fn create(cfg: Config) -> Result<Self> {
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);

        let (init_tx, init_rx) = oneshot::channel::<Result<()>>();

        thread::spawn(move || {
            let mut transcriber = match Backend::from_config(cfg.backend) {
                Ok(t) => t,
                Err(e) => {
                    let _ = init_tx.send(Err(e));
                    return;
                }
            };

            if init_tx.send(Ok(())).is_err() {
                return;
            }

            while let Some(req) = rx.blocking_recv() {
                match req {
                    Task::Transcribe(vec, resp_tx) => {
                        let res = transcriber.transcribe_chunk(vec);
                        let _ = resp_tx.send(res);
                    }
                    Task::Finish(last_chunk, resp_tx) => {
                        let res = transcriber.finish_transcribing(last_chunk);
                        let _ = resp_tx.send(res);
                        break;
                    }
                }
            }
        });

        init_rx
            .await
            .map_err(|e| Error::WorkerShutdown(Box::new(e)))??;

        Ok(Self { sender: tx })
    }

    /// Transcribes the given audio chunk into text.
    /// The audio chunk must be sampled at 16,000hz.
    /// The returned result has all text that has been transcribed upto now (including from previous calls).
    ///
    /// # Errors
    ///
    /// This function will return an error if transcription fails.
    /// The reason for the error depends on the backend type.
    pub async fn transcribe_audio(&self, vec: Vec<f32>) -> Result<Transcription> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.sender
            .send(Task::Transcribe(vec, resp_tx))
            .await
            .map_err(|e| Error::WorkerShutdown(Box::new(e)))?;

        resp_rx
            .await
            .map_err(|e| Error::WorkerShutdown(Box::new(e)))?
    }

    /// Finishes the transcription and returns the total transcription.
    ///
    /// # Errors
    ///
    /// Same as `transcribe_audio`.
    pub async fn finish_transcribing(self, last_chunk: Option<Vec<f32>>) -> Result<Transcription> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.sender
            .send(Task::Finish(last_chunk, resp_tx))
            .await
            .map_err(|e| Error::WorkerShutdown(Box::new(e)))?;

        resp_rx
            .await
            .map_err(|e| Error::WorkerShutdown(Box::new(e)))?
    }
}
