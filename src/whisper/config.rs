use std::path::PathBuf;

use crate::models::Model;

/// Config for the whisper ASR backend
pub struct Config {
    /// The model to use.
    ///  The model must be available in the `model_dir`/whisper/ directory.
    pub model: WhisperModel,
    /// The vad model to use.
    /// Currently only supports silero.
    pub vad: VadModel,
    /// The directory that contains the models.
    pub model_dir: PathBuf,
    /// The amount of segments to buffer when processing audio chunks.
    /// Higher values give better transcription quality at the cost of processing time.
    pub segment_buffer: i32,
    /// If set automatically downloads missing models.
    pub auto_download_models: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: WhisperModel::Medium,
            vad: VadModel::Silero,
            model_dir: PathBuf::from("./models"),
            segment_buffer: 1,
            auto_download_models: false,
        }
    }
}

/// Supported whisper models
pub enum WhisperModel {
    Small,
    Tiny,
    Base,
    Medium,
    Large,
    Turbo,
    Custom { name: String, base_url: String },
}

pub enum VadModel {
    Silero,
    Custom { name: String, base_url: String },
}

const VAD_BASE_URL: &'static str = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/";
const WHISPER_BASE_URL: &'static str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/";

impl Model for VadModel {
    fn model_info(&self) -> crate::models::ModelInfo {
        match self {
            VadModel::Silero => crate::models::ModelInfo {
                file_name: "ggml-silero-v6.2.0.bin".to_owned(),
                model_type: "silero".to_owned(),
                base_url: VAD_BASE_URL.to_owned(),
            },
            VadModel::Custom { name, base_url } => crate::models::ModelInfo {
                file_name: name.to_owned(),
                model_type: "custom".to_owned(),
                base_url: base_url.to_owned(),
            },
        }
    }
}

impl Model for WhisperModel {
    fn model_info(&self) -> crate::models::ModelInfo {
        let name = match self {
            WhisperModel::Custom { name, base_url } => {
                return crate::models::ModelInfo {
                    file_name: name.to_owned(),
                    model_type: "custom".to_owned(),
                    base_url: base_url.to_owned(),
                };
            }
            WhisperModel::Small => "small.en",
            WhisperModel::Tiny => "tiny.en",
            WhisperModel::Base => "base.en",
            WhisperModel::Medium => "medium.en",
            WhisperModel::Large => "large-v3",
            WhisperModel::Turbo => "large-v3-turbo",
        };
        let file_name = format!("ggml-{name}.bin");
        return crate::models::ModelInfo {
            file_name: file_name,
            model_type: "whisper".to_owned(),
            base_url: WHISPER_BASE_URL.to_owned(),
        };
    }
}
