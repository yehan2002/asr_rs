use std::path::PathBuf;

use crate::{ASRError, ASRResult};
const BASE_URL: &'static str = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/";

pub struct Config {
    pub model: WhisperModel,
    pub vad: VadModel,
    pub model_dir: PathBuf,
}

pub enum WhisperModel {
    Small,
    Tiny,
    Base,
    Medium,
}

pub enum VadModel {
    Silero,
}

pub trait ModelInfo
where
    Self: Sized,
{
    fn get_model_name(&self) -> String;
    fn model_type(&self) -> &'static str;

    fn get_path(&self, model_dir: &PathBuf) -> ASRResult<String> {
        let name = self.get_model_name();
        let model_type = self.model_type().to_string();

        let output = model_dir.join(&name);
        if output.exists() {
            return Ok(output
                .to_str()
                .expect("path should be a valid string")
                .to_owned());
        }

        let url = format!("{BASE_URL}{name}");

        Err(ASRError::ModelNotFound(model_type, name, url))
    }
}

impl ModelInfo for VadModel {
    fn get_model_name(&self) -> String {
        match self {
            VadModel::Silero => "ggml-silero-v6.2.0.bin".to_owned(),
        }
    }

    fn model_type(&self) -> &'static str {
        "Vad"
    }
}

impl ModelInfo for WhisperModel {
    fn get_model_name(&self) -> String {
        let name = match self {
            WhisperModel::Small => "small.en",
            WhisperModel::Tiny => "tiny.en",
            WhisperModel::Base => "base.en",
            WhisperModel::Medium => "medium.en",
        };

        format!("ggml-{name}.bin")
    }

    fn model_type(&self) -> &'static str {
        "Whisper"
    }
}
