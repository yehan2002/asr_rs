#[cfg(feature = "microphone")]
fn main() {
    use asr_rs::{Backend, StreamTranscriber, mic_util, whisper};

    let config = whisper::Config {
        model: whisper::WhisperModel::Medium,
        ..Default::default()
    };
    let ts = StreamTranscriber::create(Backend::Whisper(config)).unwrap();

    mic_util::mic_input(ts);
}

#[cfg(not(feature = "microphone"))]
fn main() {
    println!("cpal_mic feature is not enabled")
}
