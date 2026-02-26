use std::thread;

use crate::{Backend, Config, Error, Transcription, error::Result};
use tokio::sync::{mpsc::Sender, oneshot};

enum Task {
    Transcribe(Vec<f32>, oneshot::Sender<Result<Transcription>>),
    Finish(Option<Vec<f32>>, oneshot::Sender<Result<Transcription>>),
}

pub struct AsyncStreamTranscriber {
    sender: Sender<Task>,
    handle: thread::JoinHandle<()>,
}

impl AsyncStreamTranscriber {
    pub fn create(cfg: Config) -> Result<Self> {
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);

        let (init_tx, init_rx) = oneshot::channel::<Result<()>>();

        let handle = thread::spawn(move || {
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
            .blocking_recv()
            .map_err(|e| Error::WorkerShutdown(Box::new(e)))??;

        Ok(Self {
            sender: tx,
            handle: handle,
        })
    }

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
