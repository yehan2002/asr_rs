mod config;
mod logger;

use whisper_rs::WhisperSegment;

use crate::{
    AudioReceiver, BackendImpl, Error, PartialTranscription, Result, Segment, Silence, Token,
    Transcription, models::Model,
};

pub use crate::whisper::config::{Config, VadModel, WhisperModel};

const SAMPLE_RATE: usize = 16000;

pub(crate) struct WhisperBackend {
    vad: whisper_rs::WhisperVadContext,
    whisper: whisper_rs::WhisperState,

    time_offset: f64,
    audio_buffer: Vec<f32>,
    silence_duration: f64,

    state: PartialTranscription,

    config: Config,
}

impl WhisperBackend {
    pub fn new(config: Config) -> Result<BackendImpl> {
        let model_path = config.model.resolve_model(&config.model_dir)?;

        let vad_path = config.vad.resolve_model(&config.model_dir)?;

        logger::setup_whisper_logger();

        let params = whisper_rs::WhisperContextParameters {
            use_gpu: true,
            gpu_device: 0,
            ..Default::default()
        };

        let engine =
            whisper_rs::WhisperContext::new_with_params(&model_path, params).map_err(|e| {
                Error::ModelInit {
                    model: "Whisper".to_owned(),
                    error: Box::new(e),
                }
            })?;

        log::info!("Whisper model initilized from {model_path}");

        let state = engine.create_state().map_err(|e| Error::ModelInit {
            model: "Whisper".to_owned(),
            error: Box::new(e),
        })?;

        let vad = whisper_rs::WhisperVadContext::new(
            &vad_path,
            whisper_rs::WhisperVadContextParams::new(),
        )
        .map_err(|e| Error::ModelInit {
            model: "VAD".to_owned(),
            error: Box::new(e),
        })?;

        Ok(BackendImpl::Whisper(WhisperBackend {
            whisper: state,
            vad,
            audio_buffer: Vec::new(),
            time_offset: 0.0,
            silence_duration: 0.0,
            config,
            state: PartialTranscription {
                finalized: vec![],
                processing: vec![],
                current_silence: None,
                silences: vec![],
                full_text: String::new(),
            },
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
    fn run_inner(&mut self, stream: &mut AudioReceiver) -> Result<()> {
        while let Some(audio_chunk) = stream.next_chunk() {
            let result = self.add_chunk(audio_chunk);
            let should_stop = stream.transcribe_tx.blocking_send(result).is_err();
            if should_stop {
                break;
            }
        }

        Ok(())
    }

    fn add_chunk(&mut self, mut audio_chunk: Vec<f32>) -> Result<Transcription> {
        let vad_params = whisper_rs::WhisperVadParams::new();
        let whisper_params = self.whisper_full_params();

        let vad_result = self
            .vad
            .segments_from_samples(vad_params, &audio_chunk)
            .map_err(Error::Vad)?;

        let vad_segments = vad_result.num_segments();
        if vad_segments <= 0 {
            if self.audio_buffer.is_empty() || self.silence_duration < 3.0 {
                log::debug!("No Vad segments. Skipping...");

                // update silence durations
                let silence_length = samples_to_duration(audio_chunk.len());
                self.time_offset += silence_length;
                self.silence_duration += silence_length;
                if let Some(ref mut silence) = self.state.current_silence {
                    silence.end += silence_length;
                } else {
                    self.state.current_silence = Some(Silence {
                        start: self.time_offset - self.silence_duration,
                        end: self.time_offset,
                    })
                }

                if !self.audio_buffer.is_empty() {
                    // add the empty audio to break sentences
                    self.audio_buffer.append(&mut audio_chunk);
                }

                return Ok(Transcription::Partial(self.state.clone()));
            }

            println!("No Vad segments. Flushing buffer...");
        } else {
            if self.silence_duration > 0.0 {
                if let Some(silence) = self.state.current_silence.take() {
                    self.state.silences.push(silence);
                }
                self.silence_duration = 0.0;
            }
            log::debug!("Found Vad segments: {}", vad_segments);
        }

        let flush = vad_segments == 0;

        self.audio_buffer.append(&mut audio_chunk);

        self.whisper
            .full(whisper_params, &self.audio_buffer)
            .map_err(Error::Transcribe)?;
        let n_segments = self.whisper.full_n_segments();
        let mut partials_start_idx = 0;

        let time_offset = self.time_offset;

        if n_segments > self.config.segment_buffer || flush {
            partials_start_idx = if flush {
                n_segments
            } else {
                n_segments - self.config.segment_buffer
            };

            if flush {
                self.time_offset += samples_to_duration(self.audio_buffer.len());
                self.audio_buffer.clear();
                log::debug!("cleared audio buffer",);
            } else {
                let segment = self
                    .whisper
                    .get_segment(partials_start_idx - 1)
                    .ok_or(Error::SegmentParseFailed)?;

                let sample_end = centiseconds_to_samples(segment.end_timestamp());
                if sample_end < self.audio_buffer.len() {
                    let orig_size = self.audio_buffer.len();
                    let new_size = self.audio_buffer.len() - sample_end;

                    self.audio_buffer.copy_within(sample_end..orig_size, 0);
                    self.audio_buffer.truncate(new_size);

                    self.time_offset += samples_to_duration(sample_end);
                    log::debug!("resized audio buffer {orig_size} -> {new_size}",);
                }
            }

            for idx in 0..partials_start_idx {
                let segment = self
                    .whisper
                    .get_segment(idx)
                    .ok_or(Error::SegmentParseFailed)?;

                let part = parse_segment(segment, time_offset)?;
                log::info!("Finalized partial segment:\n {}", part);

                self.state.full_text.push_str(&part.text);
                self.state.finalized.push(part);
            }
        }

        self.state.processing.clear();
        for idx in partials_start_idx..n_segments {
            let Some(segment) = self.whisper.get_segment(idx) else {
                break;
            };

            self.state
                .processing
                .push(parse_segment(segment, time_offset)?);
        }

        Ok(Transcription::Partial(self.state.clone()))
    }
}

/// converts the number of audio samples to a duration.
/// This assumes that the audio is single chanel and is sampled at `SAMPLE_RATE`.
#[inline(always)]
fn samples_to_duration(n: usize) -> f64 {
    n as f64 / SAMPLE_RATE as f64
}

/// converts the centisecond timestamp to the number of samples.
/// This has the same assumptions as `samples_to_duration`
#[inline(always)]
fn centiseconds_to_samples(c: i64) -> usize {
    (c as usize * SAMPLE_RATE) / 100
}

/// parses a whisper segment into the segment type used in this crate
fn parse_segment(segment: WhisperSegment, time_offset: f64) -> Result<Segment> {
    let start_time = time_offset + segment.start_timestamp() as f64 / 100.0;
    let end_time = time_offset + segment.end_timestamp() as f64 / 100.0;
    let text = segment.to_str().map_err(Error::Transcribe)?;

    let n_tokens = segment.n_tokens();
    let mut tokens = Vec::with_capacity(n_tokens as usize);

    let mut total_probability: f32 = 0.0;
    for idx in 0..n_tokens {
        if let Some(token) = segment.get_token(idx) {
            let probability = token.token_probability();
            tokens.push(Token {
                probability,
                text: token
                    .to_str_lossy()
                    .map_err(|e| Error::Transcribe(e))?
                    .into_owned(),
            });
            total_probability += probability;
        }
    }

    let part = Segment {
        start: start_time,
        end: end_time,
        text: text.to_string(),
        tokens,
        probability: total_probability / n_tokens as f32,
    };

    Ok(part)
}
