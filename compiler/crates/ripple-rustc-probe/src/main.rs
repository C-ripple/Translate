//! Pinned-rustc adapter: inventory and fail-closed import into the owned IR.
//! Backend SPMD legality and target execution are separate gates.
#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_span;

mod import;

use ripple_ir::{Code, Diagnostic};
use rustc_driver::{Callbacks, Compilation};
use rustc_hir::def::DefKind;
use rustc_interface::interface;
use rustc_middle::mir::{StatementKind, TerminatorKind};
use rustc_middle::ty::{self, Instance, TyCtxt};
use rustc_span::Symbol;
use serde::Serialize;
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    stage: &'static str,
    traversal_scope: &'static str,
    ripple_codegen_validated: bool,
    target: String,
    target_cpu: String,
    pointer_width: u64,
    data_layout: String,
    mir_query: &'static str,
    mir_opt_level: usize,
    entry: String,
    lane_marker: String,
    borrow_checked_local_bodies: usize,
    functions: Vec<Function>,
    owned_ir: ripple_ir::Module,
}

#[derive(Serialize)]
struct Function {
    instance: String,
    source: String,
    argument_types: Vec<String>,
    return_type: String,
    basic_blocks: usize,
    assertions: usize,
    drops: usize,
    statement_inventory: BTreeMap<String, usize>,
    calls: Vec<Call>,
}

#[derive(Serialize)]
struct Call {
    callee: String,
    source: String,
    classification: &'static str,
}

struct Probe {
    entry: String,
    report: Option<Report>,
    failed: bool,
}

struct Passthrough;
impl Callbacks for Passthrough {}

impl Callbacks for Probe {
    fn after_analysis<'tcx>(
        &mut self,
        _compiler: &interface::Compiler,
        tcx: TyCtxt<'tcx>,
    ) -> Compilation {
        match inspect(tcx, &self.entry) {
            Ok(report) => self.report = Some(report),
            Err(message) => {
                eprintln!(
                    "RIPPLE_DIAGNOSTIC: {}",
                    serde_json::to_string(&message).expect("diagnostic serialization")
                );
                tcx.dcx().err(message.to_string());
                self.failed = true;
            }
        }
        // This is a check-only workspace wrapper. Cargo may finish metadata
        // emission; no target executable or Ripple output is produced.
        Compilation::Continue
    }
}

fn inspect(tcx: TyCtxt<'_>, entry: &str) -> Result<Report, Diagnostic> {
    if tcx.sess.target.arch.to_string() != "hexagon" || tcx.data_layout.pointer_size().bits() != 32
    {
        return Err("the probe requires the 32-bit Hexagon kernel target".into());
    }
    if tcx.sess.opts.unstable_opts.mir_opt_level != Some(0) {
        return Err("use -Zmir-opt-level=0; other MIR configurations are not qualified".into());
    }
    let mut checked = 0;
    let mut entries = Vec::new();
    for &id in tcx.mir_keys(()) {
        if matches!(
            tcx.def_kind(id),
            DefKind::Fn | DefKind::AssocFn | DefKind::Closure
        ) {
            tcx.mir_borrowck(id)
                .map_err(|_| "Rust borrow checking failed")?;
            checked += 1;
            if tcx.def_path_str(id.to_def_id()) == entry {
                entries.push(id.to_def_id());
            }
        }
    }
    if entries.len() != 1 {
        return Err(Diagnostic::new(
            Code::Entry,
            format!(
                "expected one non-generic entry path {entry:?}; found {}",
                entries.len()
            ),
        ));
    }
    let root = entries[0];
    if tcx.generics_of(root).count() != 0 {
        return Err("generic entry points are not supported by this probe".into());
    }
    let marker = tcx
        .get_diagnostic_item(Symbol::intern("ripple_lane_id_v1"))
        .ok_or("ripple-kernel lane marker metadata is missing")?;
    if tcx.crate_name(marker.krate).as_str() != "ripple_kernel" {
        return Err("lane marker must come from the pinned ripple-kernel crate".into());
    }
    let env = ty::TypingEnv::fully_monomorphized();
    let mut queue = VecDeque::from([Instance::mono(tcx, root)]);
    let mut seen = HashSet::new();
    let mut functions = Vec::new();
    let mut instances = Vec::new();
    let mut marker_calls = 0;
    while let Some(instance) = queue.pop_front() {
        if !seen.insert(instance) {
            continue;
        }
        if seen.len() > 256 {
            return Err("reachable instance limit (256) exceeded".into());
        }
        let body = tcx.instance_mir(instance.def);
        instances.push(instance);
        let normalize = |t| {
            instance.instantiate_mir_and_normalize_erasing_regions(
                tcx,
                env,
                ty::EarlyBinder::bind(t),
            )
        };
        let mut function = Function {
            instance: instance.to_string(),
            source: tcx.sess.source_map().span_to_diagnostic_string(body.span),
            argument_types: body
                .args_iter()
                .map(|i| normalize(body.local_decls[i].ty).to_string())
                .collect(),
            return_type: normalize(body.return_ty()).to_string(),
            basic_blocks: body.basic_blocks.len(),
            assertions: 0,
            drops: 0,
            statement_inventory: BTreeMap::new(),
            calls: Vec::new(),
        };
        for block in body.basic_blocks.iter() {
            for statement in &block.statements {
                let kind = match statement.kind {
                    StatementKind::Assign(_) => "assign",
                    StatementKind::StorageLive(_) => "storage_live",
                    StatementKind::StorageDead(_) => "storage_dead",
                    _ => "other",
                };
                *function.statement_inventory.entry(kind.into()).or_default() += 1;
            }
            let terminator = block.terminator();
            match &terminator.kind {
                TerminatorKind::Assert { .. } => function.assertions += 1,
                TerminatorKind::Drop { .. } => function.drops += 1,
                TerminatorKind::Call { func, .. } => {
                    let source = tcx
                        .sess
                        .source_map()
                        .span_to_diagnostic_string(terminator.source_info.span);
                    let callee_ty = normalize(func.ty(body, tcx));
                    let ty::FnDef(def_id, args) = *callee_ty.kind() else {
                        return Err(Diagnostic::new(
                            Code::UnsupportedCall,
                            "indirect call is outside the probe contract",
                        )
                        .at(source));
                    };
                    if def_id == marker {
                        marker_calls += 1;
                        function.calls.push(Call {
                            callee: tcx.def_path_str(def_id),
                            source,
                            classification: "lane_marker",
                        });
                        continue;
                    }
                    let resolved = Instance::try_resolve(tcx, env, def_id, args)
                        .map_err(|_| format!("cannot resolve {}", tcx.def_path_str(def_id)))?
                        .ok_or_else(|| format!("unresolved call at {source}"))?;
                    let classification = match resolved.def {
                        ty::InstanceKind::Intrinsic(_) => "rust_intrinsic_inventory_only",
                        ty::InstanceKind::Item(id) if tcx.is_mir_available(id) => {
                            queue.push_back(resolved);
                            "mir_body"
                        }
                        _ => {
                            return Err(Diagnostic::new(
                                Code::UnsupportedCall,
                                format!("unavailable or unsupported callee {resolved}"),
                            )
                            .at(source));
                        }
                    };
                    function.calls.push(Call {
                        callee: resolved.to_string(),
                        source,
                        classification,
                    });
                }
                TerminatorKind::TailCall { .. } => {
                    return Err("tail calls are outside the probe contract".into());
                }
                _ => {}
            }
        }
        functions.push(function);
    }
    if marker_calls == 0 {
        return Err("no surviving lane marker in the reachable MIR".into());
    }
    functions.sort_by(|a, b| a.instance.cmp(&b.instance));
    let owned_ir = import::module(tcx, &instances, marker)?;
    Ok(Report {
        schema_version: 1,
        stage: "typed_owned_ir",
        traversal_scope: "direct FnDef calls; drop glue and intrinsic bodies are not traversed; statement inventory is not semantic IR",
        ripple_codegen_validated: false,
        target: tcx.sess.opts.target_triple.to_string(),
        target_cpu: tcx
            .sess
            .opts
            .cg
            .target_cpu
            .clone()
            .unwrap_or_else(|| tcx.sess.target.cpu.to_string()),
        pointer_width: tcx.data_layout.pointer_size().bits(),
        data_layout: tcx.sess.target.data_layout.to_string(),
        mir_query: "after_analysis -> local mir_borrowck -> instance_mir (optimized_mir for Item)",
        mir_opt_level: 0,
        entry: entry.into(),
        lane_marker: tcx.def_path_str(marker),
        borrow_checked_local_bodies: checked,
        functions,
        owned_ir,
    })
}

fn publish(path: &Path, report: &Report) -> Result<(), Box<dyn std::error::Error>> {
    let parent = path.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    let tmp = parent.join(format!(".ripple-report-{}.tmp", std::process::id()));
    let result = (|| {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)?;
        serde_json::to_writer_pretty(&mut file, report)?;
        writeln!(file)?;
        file.sync_all()?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(tmp);
    }
    result
}

fn main() {
    // Cargo workspace-wrapper protocol: argv[1] is the real rustc executable.
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("use as RUSTC_WORKSPACE_WRAPPER with RIPPLE_PROBE_ENTRY and RIPPLE_PROBE_REPORT");
        std::process::exit(2);
    }
    // Keep extraction settings out of build-std dependencies. An initial debug
    // build with a global MIR override failed with duplicate fma symbols;
    // a release build did not. Only the fixture needs level 0 for this probe;
    // dependencies retain their normal configuration, recorded by the harness.
    if !args
        .windows(2)
        .any(|a| a[0] == "--crate-name" && a[1] == "ripple_probe_fixture")
    {
        let code = rustc_driver::catch_with_exit_code(|| {
            rustc_driver::run_compiler(&args, &mut Passthrough)
        });
        std::process::exit(code);
    }
    args.push("-Zmir-opt-level=0".into());
    // A failed re-check must not leave an earlier report looking current.
    if let Some(path) = std::env::var_os("RIPPLE_PROBE_REPORT") {
        match std::fs::remove_file(path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                eprintln!("RIPPLE-P003: cannot invalidate previous report: {e}");
                std::process::exit(2);
            }
        }
    }
    let mut probe = Probe {
        entry: std::env::var("RIPPLE_PROBE_ENTRY").unwrap_or_else(|_| "copy".into()),
        report: None,
        failed: false,
    };
    let code = rustc_driver::catch_with_exit_code(|| rustc_driver::run_compiler(&args, &mut probe));
    if code != 0 || probe.failed {
        std::process::exit(if code == 0 { 1 } else { code });
    }
    if let Some(report) = &probe.report {
        let Some(path) = std::env::var_os("RIPPLE_PROBE_REPORT").map(PathBuf::from) else {
            eprintln!("RIPPLE-P002: RIPPLE_PROBE_REPORT must name an output file");
            std::process::exit(2);
        };
        if let Err(error) = publish(&path, report) {
            eprintln!("RIPPLE-P003: cannot publish report: {error}");
            std::process::exit(2);
        }
    }
}
