#[cfg(all(feature = "microphone", not(feature = "tui")))]
fn main() {
    use asr_rs::util;
    use asr_rs::{Backend, StreamTranscriber, whisper};

    let config = whisper::Config {
        model: whisper::WhisperModel::Medium,
        ..Default::default()
    };

    let mut ts = StreamTranscriber::create(Backend::Whisper(config)).unwrap();
    let (stream, audio_rx) = util::mic_input();

    while let Ok(chunk) = audio_rx.recv() {
        let result = ts.transcribe_audio(chunk).unwrap();
        println!("{result}");
    }

    drop(stream);
}

#[cfg(all(feature = "microphone", feature = "tui"))]
fn main() {
    asr_rs::util::tui::run().unwrap();
}

#[cfg(not(feature = "microphone"))]
fn main() {
    println!("microphone feature is not enabled")
}
