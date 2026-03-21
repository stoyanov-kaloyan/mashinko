use std::{
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

use eframe::egui::{self, Color32, Pos2, Stroke, vec2};

#[derive(Clone)]
pub struct UiSnapshot {
    pub run_name: String,
    pub status: String,
    pub running: bool,
    pub epoch: usize,
    pub total_epochs: usize,
    pub batch: usize,
    pub total_batches: usize,
    pub current_loss: f32,
    pub avg_loss: f32,
    pub test_accuracy: Option<f32>,
    pub loss_history: Vec<f32>,
}

pub struct UiState {
    pub run_name: String,
    pub status: String,
    pub running: bool,
    pub epoch: usize,
    pub total_epochs: usize,
    pub batch: usize,
    pub total_batches: usize,
    pub current_loss: f32,
    pub avg_loss: f32,
    pub test_accuracy: Option<f32>,
    pub loss_history: Vec<f32>,
}

pub type SharedUiState = Arc<Mutex<UiState>>;

impl UiState {
    pub fn new(run_name: impl Into<String>) -> Self {
        Self {
            run_name: run_name.into(),
            status: "initializing".to_owned(),
            running: true,
            epoch: 0,
            total_epochs: 0,
            batch: 0,
            total_batches: 0,
            current_loss: 0.0,
            avg_loss: 0.0,
            test_accuracy: None,
            loss_history: Vec::new(),
        }
    }

    pub fn snapshot(&self) -> UiSnapshot {
        UiSnapshot {
            run_name: self.run_name.clone(),
            status: self.status.clone(),
            running: self.running,
            epoch: self.epoch,
            total_epochs: self.total_epochs,
            batch: self.batch,
            total_batches: self.total_batches,
            current_loss: self.current_loss,
            avg_loss: self.avg_loss,
            test_accuracy: self.test_accuracy,
            loss_history: self.loss_history.clone(),
        }
    }
}

pub fn run_training_ui<F>(run_name: impl Into<String>, train_fn: F)
where
    F: FnOnce(SharedUiState) + Send + 'static,
{
    let state = Arc::new(Mutex::new(UiState::new(run_name)));
    let worker_state = Arc::clone(&state);
    let worker = thread::spawn(move || {
        train_fn(worker_state);
    });

    let options = eframe::NativeOptions::default();
    let ui_result = eframe::run_native(
        "Mashinko Trainer",
        options,
        Box::new(move |_cc| Ok(Box::new(TrainingUiApp::new(state)))),
    );
    if let Err(err) = ui_result {
        eprintln!(
            "UI failed to start or exited with error ({err}). Continuing training without interactive UI."
        );
    }

    worker.join().expect("training worker thread panicked");
}

pub fn with_ui_state<F>(state: &SharedUiState, f: F)
where
    F: FnOnce(&mut UiState),
{
    let mut guard = state
        .lock()
        .expect("training UI state mutex poisoned while updating");
    f(&mut guard);
}

struct TrainingUiApp {
    state: SharedUiState,
}

impl TrainingUiApp {
    fn new(state: SharedUiState) -> Self {
        Self { state }
    }

    fn draw_loss_history(ui: &mut egui::Ui, history: &[f32]) {
        let desired_size = vec2(ui.available_width(), 140.0);
        let (response, painter) = ui.allocate_painter(desired_size, egui::Sense::hover());

        if history.len() < 2 {
            painter.text(
                response.rect.center(),
                egui::Align2::CENTER_CENTER,
                "waiting for loss values...",
                egui::FontId::proportional(13.0),
                Color32::GRAY,
            );
            return;
        }

        let min = history
            .iter()
            .copied()
            .fold(f32::INFINITY, |acc, x| if x < acc { x } else { acc });
        let max = history
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, |acc, x| if x > acc { x } else { acc });
        let range = (max - min).max(1e-6);
        let width = response.rect.width().max(1.0);
        let height = response.rect.height().max(1.0);
        let n = history.len().saturating_sub(1) as f32;

        let points: Vec<Pos2> = history
            .iter()
            .enumerate()
            .map(|(i, value)| {
                let x = response.rect.left() + width * (i as f32 / n);
                let y_norm = (*value - min) / range;
                let y = response.rect.bottom() - y_norm * height;
                Pos2::new(x, y)
            })
            .collect();

        painter.rect_stroke(
            response.rect,
            2.0,
            Stroke::new(1.0, Color32::DARK_GRAY),
            egui::StrokeKind::Outside,
        );
        painter.add(egui::Shape::line(
            points,
            Stroke::new(2.0, Color32::LIGHT_BLUE),
        ));
    }
}

impl eframe::App for TrainingUiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let snapshot = {
            let guard = self
                .state
                .lock()
                .expect("training UI state mutex poisoned while reading");
            guard.snapshot()
        };

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading(format!("Run: {}", snapshot.run_name));
            ui.separator();

            ui.label(format!("Status: {}", snapshot.status));
            ui.label(format!(
                "Training: {}",
                if snapshot.running { "running" } else { "finished" }
            ));
            ui.label(format!(
                "Epoch: {}/{}",
                snapshot.epoch, snapshot.total_epochs
            ));
            ui.label(format!(
                "Batch: {}/{}",
                snapshot.batch, snapshot.total_batches
            ));
            ui.label(format!("Current loss: {:.6}", snapshot.current_loss));
            ui.label(format!("Avg loss (epoch): {:.6}", snapshot.avg_loss));

            match snapshot.test_accuracy {
                Some(acc) => {
                    ui.label(format!("Test accuracy: {:.4}", acc));
                }
                None => {
                    ui.label("Test accuracy: pending");
                }
            }

            ui.separator();
            ui.label("Loss history");
            Self::draw_loss_history(ui, &snapshot.loss_history);
        });

        ctx.request_repaint_after(Duration::from_millis(100));
    }
}
