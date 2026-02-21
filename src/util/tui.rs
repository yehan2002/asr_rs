use std::sync::mpsc::{self, TryRecvError};
use std::time::Duration;
use std::{io, thread};

use crate::util::mic_input;
use crate::{BackendConfig, StreamTranscriber, Transcription, whisper};
use ratatui::style::{Color, Style};
use ratatui::text::Text;
use ratatui::widgets::{self, List, ListState};
use ratatui::{
    buffer::Buffer,
    crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind},
    layout::Rect,
    style::Stylize,
    symbols::border,
    text::Line,
    widgets::{Block, Widget},
};

#[derive(Debug, Default)]
struct App {
    transcript: Transcription,
    exit: bool,

    list_state: ListState,
    auto_scroll: bool,
}

pub fn run() -> std::io::Result<()> {
    ratatui::run(|terminal| App::default().run(terminal))
}

impl App {
    pub fn run(&mut self, terminal: &mut ratatui::DefaultTerminal) -> io::Result<()> {
        let config = crate::Config {
            backend: BackendConfig::Whisper(whisper::Config {
                model: whisper::WhisperModel::Medium,
                ..Default::default()
            }),
        };

        let (ts_tx, ts_rx) = mpsc::sync_channel(0);

        thread::spawn(move || {
            let mut ts = StreamTranscriber::create(config).unwrap();
            let (stream, audio_rx) = mic_input();

            while let Ok(chunk) = audio_rx.recv() {
                if ts_tx.send(ts.transcribe_audio(chunk).unwrap()).is_err() {
                    return;
                }
            }
            drop(stream);
        });

        self.auto_scroll = true;

        while !self.exit {
            terminal.draw(|frame| self.draw(frame))?;

            loop {
                let has_event = event::poll(Duration::from_millis(100))?;
                if has_event {
                    self.handle_events()?;
                    break;
                };

                match ts_rx.try_recv() {
                    Ok(data) => self.transcript = data,
                    Err(TryRecvError::Disconnected) => self.exit(),
                    Err(TryRecvError::Empty) => continue,
                }

                break;
            }
        }

        Ok(())
    }

    fn draw(&self, frame: &mut ratatui::Frame) {
        frame.render_widget(self, frame.area());
    }

    fn handle_events(&mut self) -> io::Result<()> {
        match event::read()? {
            Event::Key(key_event) if key_event.kind == KeyEventKind::Press => {
                self.handle_key_event(key_event)
            }
            _ => {}
        };
        Ok(())
    }

    fn handle_key_event(&mut self, key_event: KeyEvent) {
        match key_event.code {
            KeyCode::Char('q') => self.exit(),
            KeyCode::Up => {
                self.list_state.scroll_up_by(1);
                self.auto_scroll = false;
            }
            KeyCode::Down => {
                self.list_state.scroll_down_by(1);
                self.auto_scroll = false;
            }
            KeyCode::Char('a') => self.auto_scroll = true,
            _ => {}
        }
    }

    fn exit(&mut self) {
        self.exit = true;
    }
}

impl Widget for &App {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let title = Line::from("Audio transcription".bold());
        let instructions = Line::from(vec![
            " Auto Scroll ".into(),
            "<A> ".blue().bold(),
            " Quit ".into(),
            "<Q> ".blue().bold(),
        ]);
        let block = Block::bordered()
            .title(title.centered())
            .title_bottom(instructions.centered())
            .border_set(border::THICK);

        let items = self
            .transcript
            .clone()
            .into_lines()
            .iter()
            .map(line_to_list_item)
            .collect::<Vec<_>>();

        let mut list = List::new(items).block(block);

        let mut state = self.list_state;
        if self.auto_scroll {
            state.select_last();
        } else {
            list = list.highlight_style(Style::default().bg(Color::DarkGray));
        }

        widgets::StatefulWidget::render(list, area, buf, &mut state);
    }
}

fn line_to_list_item(line: &crate::Line) -> Text<'static> {
    let timestamp = line.timestamp();
    let text = format!(
        "[{0:.2} -> {1:.2}] {2}",
        timestamp.start,
        timestamp.end,
        line.text()
    );
    let style = Style::new();

    let style = match line {
        crate::Line::Complete(segment) => {
            if segment.probability > 0.85 {
                style.green()
            } else {
                style.light_yellow()
            }
        }
        crate::Line::Partial(segment) => {
            if segment.probability > 0.85 {
                style.white()
            } else {
                style.gray()
            }
        }
        crate::Line::Silence(_) => style.magenta(),
    };

    Text::styled(text, style)
}
