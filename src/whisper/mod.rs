mod config;
mod logger;

use whisper_rs::WhisperSegment;

use crate::{
    ASRError, ASRResult, AudioReceiver, BackendImpl, PartialSegment, Segment,
    whisper::config::ModelInfo,
};

pub use crate::whisper::config::{Config, VadModel, WhisperModel};

pub(crate) struct WhisperBackend {
    vad: whisper_rs::WhisperVadContext,
    state: whisper_rs::WhisperState,

    time_offset: f64,
    audio_buffer: Vec<f32>,
    finalized_segments: Vec<PartialSegment>,
    silence_duration: f64,

    config: Config,
}

impl WhisperBackend {
    pub fn new(config: Config) -> ASRResult<BackendImpl> {
        let model_path = config.model.resolve_path(&config.model_dir)?;
        let vad_path = config.vad.resolve_path(&config.model_dir)?;

        logger::setup_whisper_logger();

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

        Ok(BackendImpl::Whisper(WhisperBackend {
            state,
            vad,
            audio_buffer: Vec::new(),
            time_offset: 0.0,
            finalized_segments: Vec::new(),
            silence_duration: 0.0,
            config,
        }))
    }

    /// the params to use for ` WhisperState.full`
    fn whisper_full_params<'a>(&self) -> whisper_rs::FullParams<'a, 'a> {
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
        params
    }

    pub(crate) fn run(mut self, mut stream: AudioReceiver) {
        if let Err(err) = self.run_inner(&mut stream) {
            log::error!("Failed to transcribe due to error: {err}");
            let _ = stream.send_segment(Err(err));
        }
    }
    fn run_inner(&mut self, stream: &mut AudioReceiver) -> ASRResult<()> {
        while let Some(audio_chunk) = stream.next_chunk() {
            let result = self.add_chunk(audio_chunk);
            let should_stop = stream.transcribe_tx.blocking_send(result).is_err();
            if should_stop {
                break;
            }
        }

        Ok(())
    }

    fn add_chunk(&mut self, mut audio_chunk: Vec<f32>) -> ASRResult<Segment> {
        let vad_params = whisper_rs::WhisperVadParams::new();
        let mut whisper_params = self.whisper_full_params();

        let vad_result = self
            .vad
            .segments_from_samples(vad_params, &audio_chunk)
            .map_err(ASRError::Vad)?;

        let vad_segments = vad_result.num_segments();
        if vad_segments <= 0 {
            if self.audio_buffer.is_empty() || self.silence_duration < 3.0 {
                log::debug!("No Vad segments. Skipping...");

                let silence_length = audio_chunk.len() as f64 / 16000.0;
                self.time_offset += silence_length;
                self.silence_duration += silence_length;

                return Ok(Segment::Silence {
                    start: self.time_offset - self.silence_duration,
                    end: self.time_offset,
                });
            }

            println!("No Vad segments. Flushing buffer...");
        } else {
            self.silence_duration = 0.0;
            log::debug!("Found Vad segments: {}", vad_segments);
        }

        let flush = vad_segments == 0;

        self.audio_buffer.append(&mut audio_chunk);

        // if !self.finalized_segments.is_empty() {
        //     whisper_params.set_initial_prompt(&self.finalized_segments.last().unwrap().text);
        // }

        self.state
            .full(whisper_params, &self.audio_buffer)
            .map_err(ASRError::Transcribe)?;
        let n_segments = self.state.full_n_segments();
        let mut partials_start_idx = 0;

        let time_offset = self.time_offset;

        if n_segments > self.config.segment_buffer || flush {
            partials_start_idx = if flush {
                n_segments
            } else {
                n_segments - self.config.segment_buffer
            };

            if flush {
                self.time_offset += self.audio_buffer.len() as f64 / 16000.0;
                self.audio_buffer.clear();
                log::debug!("cleared audio buffer",);
            } else {
                let segment = self
                    .state
                    .get_segment(partials_start_idx - 1)
                    .ok_or(ASRError::SegmentParseFailed)?;

                let sample_end = 160 * segment.end_timestamp() as usize;

                if sample_end < self.audio_buffer.len() {
                    let orig_size = self.audio_buffer.len();
                    let new_size = self.audio_buffer.len() - sample_end;
                    self.audio_buffer.copy_within(sample_end..orig_size, 0);
                    self.audio_buffer.truncate(new_size);
                    self.time_offset += segment.end_timestamp() as f64 / 100.0;
                    log::debug!("resized audio buffer {orig_size} -> {new_size}",);
                }
            }

            for idx in 0..partials_start_idx {
                let segment = self
                    .state
                    .get_segment(idx)
                    .ok_or(ASRError::SegmentParseFailed)?;

                let part = parse_segment(segment, time_offset)?;
                log::info!("Finalized partial segment:\n {}", part);

                self.finalized_segments.push(part);
            }
        }

        let mut current = Vec::with_capacity(n_segments as usize);
        for idx in partials_start_idx..n_segments {
            let Some(segment) = self.state.get_segment(idx) else {
                break;
            };

            current.push(parse_segment(segment, time_offset)?);
        }

        let segment = Segment::Partial {
            finalized: self.finalized_segments.to_owned(),
            current: current,
        };
        log::info!("Partial segment:\n {}", segment);
        Ok(segment)
    }
}

fn parse_segment(segment: WhisperSegment, time_offset: f64) -> ASRResult<PartialSegment> {
    let start_time = time_offset + segment.start_timestamp() as f64 / 100.0;
    let end_time = time_offset + segment.end_timestamp() as f64 / 100.0;
    let text = segment.to_str().map_err(ASRError::Transcribe)?;

    let part = PartialSegment {
        start: start_time,
        end: end_time,
        text: text.to_string(),
    };

    Ok(part)
}
