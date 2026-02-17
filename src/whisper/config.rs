use std::path::PathBuf;

use crate::{ASRError, ASRResult};
const BASE_URL: &'static str = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/";

pub struct Config {
    pub model: WhisperModel,
    pub vad: VadModel,
    pub model_dir: PathBuf,
    pub segment_buffer: i32,
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
    fn file_name(&self) -> String;
    fn model_type(&self) -> &'static str;

    fn resolve_path(&self, model_dir: &PathBuf) -> ASRResult<String> {
        let file_name = self.file_name();
        let model = self.model_type().to_string();

        let model_path = model_dir.join(&model).join(&file_name);
        let model_path_str = model_path
            .to_str()
            .expect("path should be a valid string")
            .to_owned();

        if model_path.exists() {
            return Ok(model_path_str);
        }

        let url = format!("{BASE_URL}{file_name}");

        Err(ASRError::ModelNotFound {
            model,
            path: model_path_str,
            url,
        })
    }
}

impl ModelInfo for VadModel {
    fn file_name(&self) -> String {
        match self {
            VadModel::Silero => "ggml-silero-v6.2.0.bin".to_owned(),
        }
    }

    fn model_type(&self) -> &'static str {
        "silero"
    }
}

impl ModelInfo for WhisperModel {
    fn file_name(&self) -> String {
        let name = match self {
            WhisperModel::Small => "small.en",
            WhisperModel::Tiny => "tiny.en",
            WhisperModel::Base => "base.en",
            WhisperModel::Medium => "medium.en",
        };

        format!("ggml-{name}.bin")
    }

    fn model_type(&self) -> &'static str {
        "whisper"
    }
}
