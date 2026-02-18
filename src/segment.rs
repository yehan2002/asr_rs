use colored::Colorize;
use std::fmt::{Debug, Display};

#[derive(Debug, serde::Serialize, Clone)]
pub struct Token {
    pub text: String,
    pub probability: f32,
}

/// A part of the transcibed text.
/// This is typically 1-2 sentence fragments.
#[derive(Debug, serde::Serialize, Clone)]
pub struct Segment {
    /// The start time of the segment as seconds.
    pub start: f64,
    /// The end time of the segment as seconds.
    pub end: f64,
    /// The text of the segment.
    /// This may or may not be a full sentence.
    pub text: String,

    pub tokens: Vec<Token>,
    pub probability: f32,
}

/// A silent segment in the audio
#[derive(Debug, serde::Serialize, Clone)]
pub struct Silence {
    /// The start time of the segment as seconds.
    pub start: f64,
    /// The end time of the segment as seconds.
    pub end: f64,
}

#[derive(Debug, serde::Serialize, Clone)]
pub struct Transcription {
    pub finalized: Vec<Segment>,
    pub silences: Vec<Silence>,
    pub processing: Vec<Segment>,
    pub current_silence: Option<Silence>,
    pub full_text: String,
}

impl Display for Segment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "[{0:.2} -> {1:.2}] {2}", self.start, self.end, self.text)
    }
}

impl Segment {
    fn format_styled(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        is_finalized: bool,
    ) -> std::fmt::Result {
        let text = format!("[{0:.2} -> {1:.2}] {2}", self.start, self.end, self.text);
        let text = if is_finalized {
            if self.probability > 0.6 {
                text.green()
            } else {
                text.red()
            }
        } else {
            if self.probability > 0.6 {
                text.bright_white()
            } else {
                text.yellow()
            }
        };

        writeln!(f, "{text}")
    }
}

impl Display for Transcription {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for part in &self.finalized {
            let _ = part.format_styled(f, true);
        }

        for part in &self.processing {
            let _ = part.format_styled(f, false);
        }

        if let Some(ref current_silence) = self.current_silence {
            let _ = current_silence.format_styled(f);
        }

        Ok(())
    }
}

impl Silence {
    fn format_styled(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = format!("[{0:.2} -> {1:.2}] [Silence]", self.start, self.end).yellow();

        write!(f, "{text}")
    }
}
