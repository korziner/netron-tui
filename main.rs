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
use similar::{Algorithm, ChangeTag, TextDiff, DiffOp};
use std::io;
use std::path::PathBuf;

#[cfg(feature = "onnx")]
use tract_onnx::prelude::*;

#[cfg(feature = "jld2")]
use hdf5_metno as hdf5;

#[derive(Debug, Clone, PartialEq)]
enum OutputFormat {
    Unified,
    SideBySide,
}

/// A TUI viewer and CLI comparator for ONNX and JLD2 model files.
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the first model file (.onnx or .jld2)
    #[arg(value_name = "MODEL1", required_unless_present = "help")]
    model_path: Option<PathBuf>,

    /// Second model file for comparison (requires --cli)
    #[arg(long, value_name = "MODEL2")]
    compare: Option<PathBuf>,

    /// Diff algorithm for comparison: myers, patience, histogram (default: patience)
    #[arg(long, value_name = "ALGO", requires = "compare")]
    diff_algorithm: Option<String>,

    /// Output format: unified (default) or side-by-side
    #[arg(long, value_name = "FORMAT", requires = "compare", default_value = "unified")]
    output_format: String,

    /// Suppress lines that are identical (only show differences)
    #[arg(long, requires = "compare")]
    suppress_common: bool,

    /// Non‑interactive CLI mode (required for comparison)
    #[arg(long)]
    cli: bool,

    /// Skip ONNX model optimization (use raw model)
    #[arg(long)]
    no_optimize: bool,

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

    let model_path = match args.model_path {
        Some(p) => p,
        None => {
            eprintln!("Error: missing MODEL1 argument.\nTry 'netron-tui --help' for usage.");
            std::process::exit(1);
        }
    };

    // Comparison mode
    if let Some(model2) = args.compare {
        if !args.cli {
            eprintln!(
                "\n\x1b[1;31mError:\x1b[0m --compare requires --cli (non‑interactive mode).\n\
                 Use: netron-tui --cli --compare <MODEL1> <MODEL2> [options]\n"
            );
            std::process::exit(1);
        }
        if !model_path.exists() || !model2.exists() {
            eprintln!("Error: one or both files not found.");
            std::process::exit(1);
        }
        let ext1 = model_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        let ext2 = model2
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        // Feature checks
        if (ext1 == "onnx" || ext2 == "onnx") && !cfg!(feature = "onnx") {
            eprintln!("Error: ONNX support not enabled. Recompile with --features=onnx");
            std::process::exit(1);
        }
        if (ext1 == "jld2" || ext2 == "jld2") && !cfg!(feature = "jld2") {
            eprintln!("Error: JLD2 support not enabled. Recompile with --features=jld2");
            std::process::exit(1);
        }

        let algorithm = if let Some(algo_str) = args.diff_algorithm {
            parse_diff_algorithm(&algo_str)?
        } else {
            Algorithm::Patience
        };

        let format = parse_output_format(&args.output_format)?;

        compare_models(
            &model_path,
            &model2,
            &ext1,
            &ext2,
            algorithm,
            format,
            args.suppress_common,
            args.no_optimize,
        )?;
        return Ok(());
    }

    // Single file mode
    if !model_path.exists() {
        eprintln!("Error: File '{}' not found.", model_path.display());
        std::process::exit(1);
    }
    let ext = model_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    if ext == "onnx" && !cfg!(feature = "onnx") {
        eprintln!("Error: ONNX support not enabled. Recompile with --features=onnx");
        std::process::exit(1);
    }
    if ext == "jld2" && !cfg!(feature = "jld2") {
        eprintln!("Error: JLD2 support not enabled. Recompile with --features=jld2");
        std::process::exit(1);
    }

    let info_lines = collect_model_info(&model_path, &ext, args.no_optimize)?;

    if args.cli {
        for line in info_lines {
            println!("{}", line);
        }
        Ok(())
    } else {
        run_tui(info_lines)
    }
}

fn parse_diff_algorithm(algo: &str) -> Result<Algorithm> {
    match algo.to_lowercase().as_str() {
        "myers" => Ok(Algorithm::Myers),
        "patience" => Ok(Algorithm::Patience),
        "histogram" => Ok(Algorithm::Histogram),
        _ => Err(anyhow::anyhow!(
            "Unknown diff algorithm: {}. Use 'myers', 'patience', or 'histogram'.",
            algo
        )),
    }
}

fn parse_output_format(format: &str) -> Result<OutputFormat> {
    match format.to_lowercase().as_str() {
        "unified" => Ok(OutputFormat::Unified),
        "side-by-side" | "sidebyside" | "side" => Ok(OutputFormat::SideBySide),
        _ => Err(anyhow::anyhow!(
            "Unknown output format: {}. Use 'unified' or 'side-by-side'.",
            format
        )),
    }
}

fn print_help() {
    println!(
        "Netron TUI – Model Structure Viewer & Comparator\n\
         \n\
         USAGE:\n\
             netron-tui [OPTIONS] <MODEL1> [--compare <MODEL2>]\n\
         \n\
         ARGS:\n\
             <MODEL1>    Path to model file (.onnx or .jld2)\n\
         \n\
         OPTIONS:\n\
             --compare <MODEL2>          Compare two models (requires --cli)\n\
             --diff-algorithm <ALGO>     Diff algorithm: myers, patience, histogram (default: patience)\n\
             --output-format <FORMAT>    Output format: unified (default) or side-by-side\n\
             --suppress-common           When comparing, only show lines that differ\n\
             --cli                       Non‑interactive mode (print to stdout, no TUI)\n\
             --no-optimize               Skip ONNX model optimization (use raw model)\n\
             -h, --help                  Print this help message\n\
         \n\
         EXAMPLES:\n\
             # Interactive TUI\n\
             netron-tui model.onnx\n\
             netron-tui checkpoint.jld2\n\
         \n\
             # Non‑interactive (just print structure)\n\
             netron-tui --cli model.onnx\n\
             netron-tui --cli checkpoint.jld2 | less\n\
             netron-tui --cli --no-optimize model.onnx\n\
         \n\
             # Compare two models (unified diff)\n\
             netron-tui --cli --compare model1.onnx model2.onnx\n\
         \n\
             # Compare side‑by‑side, suppress identical lines\n\
             netron-tui --cli --compare old.jld2 new.jld2 \\\n\
                 --output-format side-by-side --suppress-common\n\
         \n\
         SUPPORTED FORMATS:\n\
             .onnx  – ONNX neural network models\n\
             .jld2  – JLD2 / HDF5 checkpoints (e.g., nanoChat GPT)\n\
         \n\
         NOTE: The --compare flag only works with --cli (non‑interactive)."
    );
}

fn collect_model_info(model_path: &PathBuf, ext: &str, no_optimize: bool) -> Result<Vec<String>> {
    let mut lines = Vec::new();
    lines.push(format!("File: {}", model_path.display()));

    match ext {
        #[cfg(feature = "onnx")]
        "onnx" => {
            let model = tract_onnx::onnx()
                .model_for_path(model_path)
                .with_context(|| format!("Failed to load ONNX model from {}", model_path.display()))?;

            let model = if !no_optimize {
                match model.into_optimized() {
                    Ok(opt) => opt,
                    Err(e) => {
                        eprintln!(
                            "\x1b[33mWarning: Optimization failed ({}), using unoptimized model.\x1b[0m",
                            e
                        );
                        model
                    }
                }
            } else {
                model
            };

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
                    if let Ok(dset) = file.dataset(name) {
                        let shape = dset.shape();
                        lines.push(format!("{}{} (dataset, shape: {:?})", prefix, name, shape));
                    } else if let Ok(group) = file.group(name) {
                        lines.push(format!("{}{} (group)", prefix, name));
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
            lines.push(format!(
                "Unsupported format '{}'. Supported: .onnx, .jld2",
                ext
            ));
        }
    }

    Ok(lines)
}

/// Unified diff (original behaviour)
fn print_unified_diff(lines1: &[String], lines2: &[String], algorithm: Algorithm) {
    let text1 = lines1.join("\n");
    let text2 = lines2.join("\n");

    let diff = TextDiff::configure()
        .algorithm(algorithm)
        .diff_lines(&text1, &text2);

    for change in diff.iter_all_changes() {
        let sign = match change.tag() {
            ChangeTag::Delete => "-",
            ChangeTag::Insert => "+",
            ChangeTag::Equal => " ",
        };
        let line = change.to_string();
        if change.tag() == ChangeTag::Delete {
            println!("\x1b[31m{}{}\x1b[0m", sign, line);
        } else if change.tag() == ChangeTag::Insert {
            println!("\x1b[32m{}{}\x1b[0m", sign, line);
        } else {
            println!("{}{}", sign, line);
        }
    }
}

/// Side‑by‑side diff with optional common suppression (using correct DiffOp fields)
fn print_side_by_side_diff(
    lines1: &[String],
    lines2: &[String],
    algorithm: Algorithm,
    suppress_common: bool,
) {
    let text1 = lines1.join("\n");
    let text2 = lines2.join("\n");

    let diff = TextDiff::configure()
        .algorithm(algorithm)
        .diff_lines(&text1, &text2);

    let mut pairs = Vec::new();
    let mut i = 0; // index in lines1
    let mut j = 0; // index in lines2

    for op in diff.ops() {
        match op {
            DiffOp::Equal { len, .. } => {
                for _ in 0..*len {
                    pairs.push((lines1[i].clone(), lines2[j].clone()));
                    i += 1;
                    j += 1;
                }
            }
            DiffOp::Delete { old_len, .. } => {
                for _ in 0..*old_len {
                    pairs.push((lines1[i].clone(), String::new()));
                    i += 1;
                }
            }
            DiffOp::Insert { new_len, .. } => {
                for _ in 0..*new_len {
                    pairs.push((String::new(), lines2[j].clone()));
                    j += 1;
                }
            }
            DiffOp::Replace { old_len, new_len, .. } => {
                let old_len = *old_len;
                let new_len = *new_len;
                for k in 0..old_len.max(new_len) {
                    let left = if k < old_len { lines1[i + k].clone() } else { String::new() };
                    let right = if k < new_len { lines2[j + k].clone() } else { String::new() };
                    pairs.push((left, right));
                }
                i += old_len;
                j += new_len;
            }
        }
    }

    // Filter if suppress_common
    let filtered: Vec<_> = if suppress_common {
        pairs
            .into_iter()
            .filter(|(l, r)| l != r || (l.is_empty() && !r.is_empty()) || (!l.is_empty() && r.is_empty()))
            .collect()
    } else {
        pairs
    };

    if filtered.is_empty() {
        println!("(no differences)");
        return;
    }

    let max_left_len = filtered
        .iter()
        .map(|(l, _)| l.len())
        .max()
        .unwrap_or(0)
        .min(80);
    let separator = " │ ";

    for (left, right) in filtered {
        let left_display = if left.is_empty() { "" } else { &left };
        let right_display = if right.is_empty() { "" } else { &right };
        if left != right {
            println!(
                "\x1b[31m{:<width$}\x1b[0m{sep}\x1b[32m{}\x1b[0m",
                left_display,
                right_display,
                width = max_left_len,
                sep = separator
            );
        } else {
            println!(
                "{:<width$}{sep}{}",
                left_display,
                right_display,
                width = max_left_len,
                sep = separator
            );
        }
    }
}

/// Main comparison dispatcher
fn compare_models(
    model1: &PathBuf,
    model2: &PathBuf,
    ext1: &str,
    ext2: &str,
    algorithm: Algorithm,
    format: OutputFormat,
    suppress_common: bool,
    no_optimize: bool,
) -> Result<()> {
    let lines1 = collect_model_info(model1, ext1, no_optimize)?;
    let lines2 = collect_model_info(model2, ext2, no_optimize)?;

    let name1 = model1.file_name().unwrap_or_default().to_string_lossy();
    let name2 = model2.file_name().unwrap_or_default().to_string_lossy();

    match format {
        OutputFormat::Unified => {
            println!("diff --git a/{} b/{}", name1, name2);
            println!("--- a/{}", name1);
            println!("+++ b/{}", name2);
            print_unified_diff(&lines1, &lines2, algorithm);
        }
        OutputFormat::SideBySide => {
            if suppress_common {
                println!("Only differing lines (common lines suppressed):");
            } else {
                println!("Side‑by‑side diff of {} and {}", name1, name2);
            }
            print_side_by_side_diff(&lines1, &lines2, algorithm, suppress_common);
        }
    }
    Ok(())
}

/// Interactive TUI mode (unchanged)
fn run_tui(info_lines: Vec<String>) -> Result<()> {
    let items: Vec<ListItem> = info_lines
        .into_iter()
        .map(|line| ListItem::new(line))
        .collect();

    let mut list_state = ListState::default();
    list_state.select(Some(0));

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

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}

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
        "q=quit  ↑↓=scroll  (CLI mode: --cli for non‑interactive output; --compare for diff)",
    ))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(help, chunks[1]);
}
