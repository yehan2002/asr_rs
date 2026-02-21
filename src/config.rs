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
