use std::path::{Path, PathBuf};

use crate::{Error, Result};

pub(crate) struct ModelInfo {
    pub file_name: String,
    pub model_type: String,
    pub base_url: String,
}

pub(crate) trait Model {
    fn model_info(&self) -> ModelInfo;

    fn resolve_model(&self, model_dir: &Path) -> Result<String> {
        let model = self.model_info();

        let model_path = model_dir.join(&model.model_type).join(&model.file_name);

        let model_path_str = model_path
            .to_str()
            .expect("path should be a valid string")
            .to_owned();

        if !model_path.exists() {
            let url = format!("{}{}", model.base_url, model.file_name);
            download_model(&model.file_name, model_path, url)?;
        }

        Ok(model_path_str)
    }
}

#[cfg(not(feature = "model_download"))]
fn download_model(model_name: &str, out_path: PathBuf, url: String) -> Result<()> {
    Err(Error::ModelNotFound {
        model: model_name.to_owned(),
        path: out_path.to_string_lossy().into_owned(),
        url: url,
    })
}

#[cfg(feature = "model_download")]
fn download_model(model_name: &str, out_path: PathBuf, url: String) -> Result<()> {
    // create parent dir
    std::fs::create_dir_all(out_path.parent().expect("path should have parent"))
        .map_err(|e| Error::Download(Box::new(e)))?;

    let mut tmp_path = out_path.clone();
    tmp_path.add_extension(".tmp");

    log::info!("Downloading {model_name} from {url}");

    let response = ureq::get(url)
        .call()
        .map_err(|e| Error::Download(Box::new(e)))?;

    let total_size = response
        .headers()
        .get("Content-Length")
        .and_then(|ct_len| ct_len.to_str().ok())
        .and_then(|ct_len| ct_len.parse::<u64>().ok())
        .unwrap_or(0);

    // progress bar
    let pb = indicatif::ProgressBar::new(total_size);
    pb.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("{bytes}/{total_bytes} {bar:40.cyan/blue}] [{elapsed_precise}] ({eta})")
            .expect("Template should be valid"),
    );

    let reader = response.into_body().into_reader();
    let mut pb_reader = pb.wrap_read(reader);

    let file = std::fs::File::create(&tmp_path).map_err(|e| Error::Download(Box::new(e)))?;
    let mut writer = std::io::BufWriter::new(file);

    std::io::copy(&mut pb_reader, &mut writer).map_err(|e| Error::Download(Box::new(e)))?;

    std::fs::rename(tmp_path, out_path).map_err(|e| Error::Download(Box::new(e)))?;
    Ok(())
}
