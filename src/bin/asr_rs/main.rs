mod mic;
mod tui;

pub fn main() {
    ratatui::run(|terminal| tui::App::default().run(terminal)).expect("Failed to start tui");
}
