use crate::StreamTranscriber;
use cpal::{
    SampleFormat,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};

pub fn mic_input(mut ts: StreamTranscriber) {
    let host = cpal::default_host();
    let device = host.default_output_device().unwrap();

    let config = if device.supports_input() {
        device.default_input_config()
    } else {
        device.default_output_config()
    }
    .expect("Failed to get default input/output config");
    assert!(config.sample_format() == SampleFormat::F32);

    let err_fn = move |err| {
        eprintln!("an error occurred on stream: {err}");
    };

    let stream = device
        .build_input_stream(
            &cpal::StreamConfig {
                channels: 1,
                sample_rate: 16000,
                buffer_size: cpal::BufferSize::Fixed(16000 * 1),
            },
            move |data, _: &_| {
                let result = ts.transcribe_audio(data.to_vec()).unwrap();
                println!("{result}");
            },
            err_fn,
            None,
        )
        .unwrap();

    stream.play().unwrap();
    std::thread::sleep(std::time::Duration::from_secs(10000));
}
