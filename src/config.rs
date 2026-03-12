use crate::{error::Result, models::Model, whisper};

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(tag = "backend", rename_all = "snake_case")]
pub enum BackendConfig {
    Whisper(whisper::Config),
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    #[serde(flatten)]
    pub backend: BackendConfig,
}

impl BackendConfig {
    /// Downloads the models used for this config.
    /// Skips download if the models are already in the `model_dir`.
    ///
    /// # Errors
    ///
    /// This function will return an error if downloading fails or if the `model_download` feature is not enabled.
    pub fn download_models(&self) -> Result<()> {
        match self {
            Self::Whisper(cfg) => {
                cfg.model.resolve_model(&cfg.model_dir)?;
                cfg.vad.resolve_model(&cfg.model_dir)?;
            }
        }
        Ok(())
    }
}
