mod config;
mod logger;

use std::time::{self, Duration};

use crate::{
    ASRError, ASRResult, AudioReceiver, BackendImpl, PartialSegment, Segment,
    whisper::config::ModelInfo,
};

pub use crate::whisper::config::{Config, VadModel, WhisperModel};

pub(crate) struct WhisperBackend {
    vad: whisper_rs::WhisperVadContext,
    state: whisper_rs::WhisperState,
}

impl WhisperBackend {
    pub fn new(config: Config) -> ASRResult<BackendImpl> {
        let model_path = config.model.resolve_path(&config.model_dir)?;
        let vad_path = config.vad.resolve_path(&config.model_dir)?;

        logger::set_whisper_logger();

        let params = whisper_rs::WhisperContextParameters {
            use_gpu: true,
            gpu_device: 0,
            ..Default::default()
        };

        let engine =
            whisper_rs::WhisperContext::new_with_params(&model_path, params).map_err(|e| {
                ASRError::ModelInit {
                    model: "Whisper".to_owned(),
                    error: Box::new(e),
                }
            })?;

        log::info!("Whisper model initilized from {model_path}");

        let state = engine.create_state().map_err(|e| ASRError::ModelInit {
            model: "Whisper".to_owned(),
            error: Box::new(e),
        })?;

        let vad = whisper_rs::WhisperVadContext::new(
            &vad_path,
            whisper_rs::WhisperVadContextParams::new(),
        )
        .map_err(|e| ASRError::ModelInit {
            model: "VAD".to_owned(),
            error: Box::new(e),
        })?;

        Ok(BackendImpl::Whisper(WhisperBackend { state, vad }))
    }

    pub(crate) fn run(mut self, mut stream: AudioReceiver) {
        if let Err(err) = self.run_inner(&mut stream) {
            log::error!("Failed to transcribe due to error: {err}");
            let _ = stream.send_segment(Err(err));
        }
    }

    fn run_inner(&mut self, stream: &mut AudioReceiver) -> ASRResult<()> {
        let mut params = whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::BeamSearch {
            beam_size: 5,
            patience: 1.0,
        });

        params.set_no_context(true);
        params.set_single_segment(false);
        params.set_print_progress(false);
        params.set_print_special(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_token_timestamps(true);

        let vad_params = whisper_rs::WhisperVadParams::new();

        let start = time::Instant::now();

        let mut next_start = Duration::from_secs(0);
        let mut full_text = String::new();

        while let Some(audio_chunk) = stream.next_chunk() {
            let chunk_start = next_start;
            next_start = time::Instant::now().duration_since(start);

            let vad_result = self
                .vad
                .segments_from_samples(vad_params, &audio_chunk)
                .map_err(ASRError::Vad)?;

            let vad_segments = vad_result.num_segments();
            if vad_segments <= 0 {
                log::debug!("No Vad segments. Skipping...");
                let result = stream.send_segment(Ok(Segment::Silence {
                    start: chunk_start.as_secs_f64(),
                    end: next_start.as_secs_f64(),
                }));
                if result.is_none() {
                    break;
                }
                continue;
            }
            log::debug!("Found Vad segments: {}", vad_segments);

            let mut params = params.clone();
            if full_text.is_empty() {
                params.set_no_context(false);
                params.set_initial_prompt(&full_text.to_owned());
            }

            self.state
                .full(params, &audio_chunk)
                .map_err(ASRError::Transcribe)?;

            let n_segments = self.state.full_n_segments();
            let mut partials = Vec::with_capacity(n_segments as usize);
            for idx in 0..n_segments {
                let Some(segment) = self.state.get_segment(idx) else {
                    break;
                };

                let start_time =
                    chunk_start.as_secs_f64() + segment.start_timestamp() as f64 / 100.0;
                let end_time = chunk_start.as_secs_f64() + segment.end_timestamp() as f64 / 100.0;
                let text = segment.to_str().map_err(ASRError::Transcribe)?;
                full_text.push_str(text);

                let part = PartialSegment {
                    start: start_time,
                    end: end_time,
                    text: text.to_string(),
                };
                partials.push(part);
            }

            let segment = Segment::Partial(partials);
            log::debug!("Partial segment: {:?}", segment);
            let send_result = stream.send_segment(Ok(segment));

            if send_result.is_none() {
                log::info!("Stopping asr due to reciever closing");
                break;
            }
        }

        let _ = stream.send_segment(Ok(Segment::Full { text: full_text }));
        Ok(())
    }
}
