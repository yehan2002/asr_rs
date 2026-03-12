use std::sync::mpsc;

use anyhow::{Context, bail};
use cpal::{
    SampleFormat, Stream,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};

const BUFFER_SIZE: u32 = 1;

pub(crate) fn mic_input() -> anyhow::Result<(Stream, mpsc::Receiver<Vec<f32>>)> {
    let (audio_tx, audio_rx) = mpsc::channel::<Vec<f32>>();

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .context("Failed to get default output device")?;

    let config = if device.supports_input() {
        device.default_input_config()
    } else {
        device.default_output_config()
    }
    .context("Failed to get default input/output config")?;

    if config.sample_format() != SampleFormat::F32 {
        bail!("Device does not support f32 sample format")
    }

    let err_fn = move |err| {
        panic!("an error occurred on stream: {err}");
    };

    let stream = device
        .build_input_stream(
            &cpal::StreamConfig {
                channels: 1,
                sample_rate: 16000,
                buffer_size: cpal::BufferSize::Fixed(16000 * BUFFER_SIZE),
            },
            move |data, _: &_| {
                audio_tx
                    .send(data.to_vec())
                    .expect("channel should be open");
            },
            err_fn,
            None,
        )
        .context("Failed to create audio stream")?;

    stream.play().context("Failed to start stream")?;

    Ok((stream, audio_rx))
}
