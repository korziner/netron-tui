use anyhow::{Context, Result};
use clap::Parser;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    text::Line,
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
    Frame, Terminal,
};
use std::io;
use std::path::PathBuf;

#[cfg(feature = "onnx")]
use tract_onnx::prelude::*;

#[cfg(feature = "jld2")]
use hdf5_metno as hdf5;

/// A TUI viewer for ONNX and JLD2 (nanoChat GPT) model files.
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the model file (.onnx or .jld2)
    #[arg(value_name = "MODEL")]
    model_path: PathBuf,

    /// Non‑interactive CLI mode: print structure and exit
    #[arg(long)]
    cli: bool,

    /// Show this help message
    #[arg(short, long)]
    help: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.help {
        print_help();
        return Ok(());
    }

    let model_path = &args.model_path;

    if !model_path.exists() {
        eprintln!("Error: File '{}' not found.", model_path.display());
        std::process::exit(1);
    }

    let ext = model_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    // Feature availability checks
    if ext == "onnx" && !cfg!(feature = "onnx") {
        eprintln!("Error: ONNX support not enabled. Recompile with --features=onnx");
        std::process::exit(1);
    }
    if ext == "jld2" && !cfg!(feature = "jld2") {
        eprintln!("Error: JLD2 support not enabled. Recompile with --features=jld2");
        std::process::exit(1);
    }

    // Collect model information as strings
    let info_lines = collect_model_info(model_path, &ext)?;

    if args.cli {
        // Print directly to stdout
        for line in info_lines {
            println!("{}", line);
        }
        Ok(())
    } else {
        // Interactive TUI mode
        run_tui(info_lines)
    }
}

/// Print detailed help message
fn print_help() {
    println!(
        "Netron TUI – Model Structure Viewer for ONNX and JLD2 (nanoChat GPT)\n\
         \n\
         USAGE:\n\
             netron-tui [OPTIONS] <MODEL>\n\
         \n\
         ARGS:\n\
             <MODEL>    Path to the model file (.onnx or .jld2)\n\
         \n\
         OPTIONS:\n\
             --cli          Non‑interactive mode: print structure and exit\n\
             -h, --help     Print this help message\n\
         \n\
         EXAMPLES:\n\
             netron-tui model.onnx               # Interactive TUI\n\
             netron-tui --cli model.onnx         # Print ONNX structure\n\
             netron-tui --cli checkpoint.jld2    # Print JLD2 structure\n\
         \n\
         SUPPORTED FORMATS:\n\
             .onnx  – requires 'onnx' feature (enabled by default)\n\
             .jld2  – requires 'jld2' feature (enabled by default)"
    );
}

/// Extract model structure into a vector of strings (one per line)
fn collect_model_info(model_path: &PathBuf, ext: &str) -> Result<Vec<String>> {
    let mut lines = Vec::new();
    lines.push(format!("File: {}", model_path.display()));

    match ext {
        #[cfg(feature = "onnx")]
        "onnx" => {
            let model = tract_onnx::onnx()
                .model_for_path(model_path)
                .with_context(|| format!("Failed to load ONNX model from {}", model_path.display()))?
                .into_optimized()
                .context("Failed to optimize ONNX model")?;

            lines.push("├── Inputs".to_string());
            for input in &model.inputs {
                let fact = model.outlet_fact(*input)?;
                lines.push(format!("│   └── {}: {:?}", input.node, fact));
            }

            lines.push("├── Nodes / Layers".to_string());
            for node in model.nodes() {
                lines.push(format!("│   ├── {} ({})", node.op.name(), node.name));
            }

            lines.push("├── Outputs".to_string());
            for output in &model.outputs {
                let fact = model.outlet_fact(*output)?;
                lines.push(format!("│   └── {}: {:?}", output.node, fact));
            }
        }

        #[cfg(feature = "jld2")]
        "jld2" => {
            let file = match hdf5::File::open(model_path) {
                Ok(f) => f,
                Err(e) => {
                    lines.push(format!("│   ✗ Failed to open JLD2 file: {}", e));
                    return Ok(lines);
                }
            };

            lines.push("├── JLD2 Contents".to_string());

            let members = match file.member_names() {
                Ok(names) => names,
                Err(e) => {
                    lines.push(format!("│   ✗ Failed to read members: {}", e));
                    return Ok(lines);
                }
            };

            if members.is_empty() {
                lines.push("│   └── (empty)".to_string());
            } else {
                for (idx, name) in members.iter().enumerate() {
                    let is_last = idx == members.len() - 1;
                    let prefix = if is_last { "│   └── " } else { "│   ├── " };

                    // Try dataset first
                    if let Ok(dset) = file.dataset(name) {
                        let shape = dset.shape();
                        lines.push(format!("{}{} (dataset, shape: {:?})", prefix, name, shape));
                    } else if let Ok(group) = file.group(name) {
                        lines.push(format!("{}{} (group)", prefix, name));
                        // Optionally show common keys inside group (indented)
                        for subkey in ["weight", "bias", "ln_1", "ln_2", "attn", "mlp"] {
                            if group.link_exists(subkey) {
                                lines.push(format!("│   │   └── {}", subkey));
                            }
                        }
                    } else {
                        lines.push(format!("{}{} (unknown type)", prefix, name));
                    }
                }
            }
        }

        _ => {
            lines.push(format!("Unsupported format '{}'. Supported: .onnx, .jld2", ext));
        }
    }

    Ok(lines)
}

/// Run the interactive TUI using the collected lines
fn run_tui(info_lines: Vec<String>) -> Result<()> {
    // Convert each line to a ratatui ListItem
    let items: Vec<ListItem> = info_lines
        .into_iter()
        .map(|line| ListItem::new(line))
        .collect();

    // Add navigation help at the bottom
    let mut list_state = ListState::default();
    list_state.select(Some(0));

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = (|| {
        loop {
            terminal.draw(|f| ui(f, &items, &mut list_state))?;

            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                        KeyCode::Down => {
                            let i = list_state.selected().unwrap_or(0);
                            if i < items.len().saturating_sub(1) {
                                list_state.select(Some(i + 1));
                            }
                        }
                        KeyCode::Up => {
                            let i = list_state.selected().unwrap_or(0);
                            if i > 0 {
                                list_state.select(Some(i - 1));
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    })();

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

/// TUI drawing function
fn ui(f: &mut Frame, items: &[ListItem], list_state: &mut ListState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(3)])
        .split(f.area());

    let list = List::new(items.to_vec())
        .block(
            Block::default()
                .title("Netron TUI (Ratatui) • ONNX + JLD2/nanoChat GPT • Vertical Scroll")
                .borders(Borders::ALL),
        )
        .highlight_style(Style::default().fg(Color::Yellow))
        .highlight_symbol(">> ");

    f.render_stateful_widget(list, chunks[0], list_state);

    let help = Paragraph::new(Line::from(
        "q=quit  ↑↓=scroll  (CLI mode: --cli for non‑interactive output)",
    ))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(help, chunks[1]);
}
