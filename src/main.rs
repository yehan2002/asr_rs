use asr_rs::{Backend, StreamTranscriber, util, whisper};

fn main() {
    let config = whisper::Config {
        model: whisper::WhisperModel::Medium,
        ..Default::default()
    };
    let ts = StreamTranscriber::create(Backend::Whisper(config)).unwrap();

    util::mic_input(ts);
}
