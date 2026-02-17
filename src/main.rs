use std::path::PathBuf;

use asr_rs::{Backend, StreamTranscriber, util, whisper};

fn main() {
    let config = whisper::Config {
        model: whisper::WhisperModel::Medium,
        vad: whisper::VadModel::Silero,
        model_dir: PathBuf::from("./models"),
    };
    let ts = StreamTranscriber::create(Backend::Whisper(config)).unwrap();

    util::mic_input(ts);
}
