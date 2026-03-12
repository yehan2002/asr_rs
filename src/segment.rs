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
    /// The text of the segment.
    /// This may or may not be a full sentence.
    pub text: String,

    pub tokens: Vec<Token>,
    pub probability: f32,
    pub timestamp: Timestamp,
}

/// A silent segment in the audio
#[derive(Debug, serde::Serialize, Clone)]
pub struct Silence {
    pub timestamp: Timestamp,
}

/// A timestamp
#[derive(Debug, serde::Serialize, Clone)]
pub struct Timestamp {
    /// The start time of the segment as seconds.
    pub start: f64,
    /// The end time of the segment as seconds.
    pub end: f64,
}

impl Timestamp {
    #[must_use]
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }
}

#[derive(Debug, serde::Serialize, Clone, Default)]
pub struct Transcription {
    pub finalized: Vec<Segment>,
    pub silences: Vec<Silence>,
    pub processing: Vec<Segment>,
    pub current_silence: Option<Silence>,
    pub full_text: String,
    pub is_complete: bool,
}

#[derive(Debug, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Line {
    Complete(Segment),
    Partial(Segment),
    Silence(Silence),
}

impl Line {
    #[must_use]
    pub fn text(&self) -> &str {
        match self {
            Line::Complete(segment) | Line::Partial(segment) => &segment.text,
            Line::Silence(_) => "[Silence]",
        }
    }

    #[must_use]
    pub fn timestamp(&self) -> &Timestamp {
        match self {
            Line::Complete(segment) | Line::Partial(segment) => &segment.timestamp,
            Line::Silence(silence) => &silence.timestamp,
        }
    }
}

impl Transcription {
    #[must_use]
    pub fn into_lines(self) -> Vec<Line> {
        let mut lines = Vec::with_capacity(self.finalized.len() + self.processing.len() + 1);

        for seg in self.finalized {
            lines.push(Line::Complete(seg));
        }

        for seg in self.processing {
            lines.push(Line::Partial(seg));
        }

        for seg in self.silences {
            lines.push(Line::Silence(seg));
        }

        lines.sort_by(|a, b| a.timestamp().start.total_cmp(&b.timestamp().start));

        if let Some(seg) = self.current_silence {
            lines.push(Line::Silence(seg));
        }

        lines
    }
}

impl Display for Segment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "[{0:.2} -> {1:.2}] {2}",
            self.timestamp.start, self.timestamp.end, self.text
        )
    }
}

impl Segment {
    fn format_styled(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        is_finalized: bool,
    ) -> std::fmt::Result {
        let text = format!(
            "[{0:.2} -> {1:.2}] {2}",
            self.timestamp.start, self.timestamp.end, self.text
        );
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
        let text = format!(
            "[{0:.2} -> {1:.2}] [Silence]",
            self.timestamp.start, self.timestamp.end
        )
        .yellow();

        write!(f, "{text}")
    }
}
