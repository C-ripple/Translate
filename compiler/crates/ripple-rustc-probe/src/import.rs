//! Fail-closed import of a deliberately small concrete MIR dialect.
use ir::{Code, Diagnostic};
use ripple_ir as ir;
use rustc_hir::def_id::DefId;
use rustc_middle::{
    mir,
    ty::{self, Instance, Ty, TyCtxt},
};

fn validate_unwind(action: mir::UnwindAction, source: String) -> Result<(), Diagnostic> {
    if matches!(
        action,
        mir::UnwindAction::Unreachable | mir::UnwindAction::Terminate(_)
    ) {
        Ok(())
    } else {
        Err(Diagnostic::new(Code::Unwind, "unwinding calls are unsupported").at(source))
    }
}

fn ty(t: Ty<'_>) -> Result<ir::Type, Diagnostic> {
    Ok(match t.kind() {
        ty::Tuple(fields) if fields.is_empty() => ir::Type::Unit,
        ty::Bool => ir::Type::Bool,
        ty::Uint(ty::UintTy::U32) => ir::Type::U32,
        ty::Uint(ty::UintTy::Usize) => ir::Type::Usize,
        ty::Float(ty::FloatTy::F32) => ir::Type::F32,
        ty::RawPtr(pointee, mutability)
            if matches!(pointee.kind(), ty::Float(ty::FloatTy::F32)) =>
        {
            ir::Type::F32Pointer {
                mutable: mutability.is_mut(),
            }
        }
        _ => {
            return Err(Diagnostic::new(
                Code::UnsupportedType,
                format!("unsupported IR type {t}"),
            ));
        }
    })
}

fn place(p: mir::Place<'_>) -> Result<ir::Place, Diagnostic> {
    match p.projection.as_ref() {
        [] => Ok(ir::Place::Local(p.local.as_usize())),
        [mir::ProjectionElem::Deref] => Ok(ir::Place::Deref(p.local.as_usize())),
        _ => Err(Diagnostic::new(
            Code::UnsupportedOperation,
            format!("unsupported place projection {p:?}"),
        )),
    }
}

struct Import<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    instances: &'a [Instance<'tcx>],
    marker: DefId,
}

impl<'tcx> Import<'_, 'tcx> {
    fn normalize<T: ty::TypeFoldable<TyCtxt<'tcx>>>(&self, value: T) -> T {
        self.instance.instantiate_mir_and_normalize_erasing_regions(
            self.tcx,
            ty::TypingEnv::fully_monomorphized(),
            ty::EarlyBinder::bind(value),
        )
    }

    fn operand(&self, operand: &mir::Operand<'tcx>) -> Result<ir::Operand, Diagnostic> {
        Ok(match operand {
            mir::Operand::Copy(p) | mir::Operand::Move(p) => ir::Operand::Read(place(*p)?),
            mir::Operand::Constant(c) => {
                let c = self.normalize(c.const_);
                let t = ty(c.ty())?;
                if t == ir::Type::Unit {
                    return Ok(ir::Operand::Unit);
                }
                let bits = c
                    .try_eval_bits(self.tcx, ty::TypingEnv::fully_monomorphized())
                    .ok_or("constant could not be evaluated")?;
                match t {
                    ir::Type::Bool if bits <= 1 => ir::Operand::Bool(bits != 0),
                    ir::Type::U32 | ir::Type::Usize => ir::Operand::Integer {
                        value: u32::try_from(bits).map_err(|_| "integer exceeds target width")?,
                        ty: t,
                    },
                    _ => {
                        return Err(Diagnostic::new(
                            Code::UnsupportedOperation,
                            format!("unsupported constant type {t:?}"),
                        ));
                    }
                }
            }
            _ => return Err("unsupported operand".into()),
        })
    }

    fn value(&self, value: &mir::Rvalue<'tcx>) -> Result<ir::Value, Diagnostic> {
        Ok(match value {
            mir::Rvalue::Use(operand) => ir::Value::Use(self.operand(operand)?),
            mir::Rvalue::Cast(mir::CastKind::IntToInt, operand, t) => ir::Value::IntegerCast {
                operand: self.operand(operand)?,
                to: ty(self.normalize(*t))?,
            },
            mir::Rvalue::BinaryOp(op, operands) => ir::Value::Binary {
                op: match op {
                    mir::BinOp::Add => ir::Binary::Add,
                    mir::BinOp::Sub => ir::Binary::Subtract,
                    mir::BinOp::Mul => ir::Binary::Multiply,
                    mir::BinOp::Lt => ir::Binary::LessThan,
                    mir::BinOp::Eq => ir::Binary::Equal,
                    mir::BinOp::Offset => ir::Binary::PointerOffset,
                    _ => {
                        return Err(Diagnostic::new(
                            Code::UnsupportedOperation,
                            format!("unsupported binary operator {op:?}"),
                        ));
                    }
                },
                lhs: self.operand(&operands.0)?,
                rhs: self.operand(&operands.1)?,
            },
            _ => {
                return Err(Diagnostic::new(
                    Code::UnsupportedOperation,
                    format!("unsupported rvalue {value:?}"),
                ));
            }
        })
    }

    fn function(&self) -> Result<ir::Function, Diagnostic> {
        let body = self.tcx.instance_mir(self.instance.def);
        let source = |span| self.tcx.sess.source_map().span_to_diagnostic_string(span);
        // Diagnose unsupported loops before unused never-typed MIR temporaries
        // obscure the control-flow limitation. This does not erase any blocks.
        let mut indegree = vec![0usize; body.basic_blocks.len()];
        for block in body.basic_blocks.iter() {
            for next in block.terminator().successors() {
                indegree[next.as_usize()] += 1;
            }
        }
        let mut queue: std::collections::VecDeque<_> = indegree
            .iter()
            .enumerate()
            .filter_map(|(i, n)| (*n == 0).then_some(i))
            .collect();
        let mut visited = 0;
        while let Some(id) = queue.pop_front() {
            visited += 1;
            for next in body.basic_blocks[mir::BasicBlock::from_usize(id)]
                .terminator()
                .successors()
            {
                indegree[next.as_usize()] -= 1;
                if indegree[next.as_usize()] == 0 {
                    queue.push_back(next.as_usize());
                }
            }
        }
        if visited != body.basic_blocks.len() {
            return Err(Diagnostic::new(
                Code::ControlFlow,
                "cyclic MIR is outside the acyclic dialect",
            )
            .at(source(body.span)));
        }
        // Reject implicit effects before processing locals/assignments, including
        // inside reachable helpers. None of these terminators may be erased.
        for block in body.basic_blocks.iter() {
            let term = block.terminator();
            let rejected = match term.kind {
                mir::TerminatorKind::Assert { .. } => Some((
                    Code::Assertion,
                    "MIR assertions are unsupported; the compiler will not erase runtime checks",
                )),
                mir::TerminatorKind::Drop { .. } => Some((Code::Drop, "drop glue is unsupported")),
                _ => None,
            };
            if let Some((code, message)) = rejected {
                return Err(Diagnostic::new(code, message).at(source(term.source_info.span)));
            }
        }
        let mut blocks = Vec::new();
        for block in body.basic_blocks.iter() {
            let mut assignments = Vec::new();
            for statement in &block.statements {
                match &statement.kind {
                    mir::StatementKind::Assign(pair) => assignments.push(ir::Assignment {
                        destination: place(pair.0)
                            .map_err(|e| e.at(source(statement.source_info.span)))?,
                        value: self
                            .value(&pair.1)
                            .map_err(|e| e.at(source(statement.source_info.span)))?,
                        source: source(statement.source_info.span),
                    }),
                    // This dialect has no address-taking, references, or drops.
                    mir::StatementKind::StorageLive(_)
                    | mir::StatementKind::StorageDead(_)
                    | mir::StatementKind::Nop => {}
                    other => {
                        return Err(Diagnostic::new(
                            Code::UnsupportedOperation,
                            format!("unsupported statement {other:?}"),
                        )
                        .at(source(statement.source_info.span)));
                    }
                }
            }
            let terminator = block.terminator();
            let term = match &terminator.kind {
                mir::TerminatorKind::Assert { .. } => return Err(Diagnostic::new(
                    Code::Assertion,
                    "MIR assertions are unsupported; the compiler will not erase runtime checks",
                )
                .at(source(terminator.source_info.span))),
                mir::TerminatorKind::Drop { .. } => {
                    return Err(Diagnostic::new(Code::Drop, "drop glue is unsupported")
                        .at(source(terminator.source_info.span)));
                }
                mir::TerminatorKind::Return => ir::Terminator::Return,
                mir::TerminatorKind::Goto { target } => ir::Terminator::Goto(target.as_usize()),
                mir::TerminatorKind::SwitchInt { discr, targets } => {
                    if ty(self.normalize(discr.ty(body, self.tcx)))? != ir::Type::Bool {
                        return Err("only boolean switches are supported".into());
                    }
                    ir::Terminator::Branch {
                        condition: self.operand(discr)?,
                        on_true: targets.target_for_value(1).as_usize(),
                        on_false: targets.target_for_value(0).as_usize(),
                    }
                }
                mir::TerminatorKind::Call {
                    func,
                    args,
                    destination,
                    target,
                    unwind,
                    ..
                } => {
                    validate_unwind(*unwind, source(terminator.source_info.span))?;
                    let callee_ty = self.normalize(func.ty(body, self.tcx));
                    let ty::FnDef(id, substitutions) = *callee_ty.kind() else {
                        return Err("indirect call".into());
                    };
                    let callee = if id == self.marker {
                        ir::Callee::LaneId
                    } else {
                        let instance = Instance::try_resolve(
                            self.tcx,
                            ty::TypingEnv::fully_monomorphized(),
                            id,
                            substitutions,
                        )
                        .map_err(|_| "callee resolution failed")?
                        .ok_or("unresolved callee")?;
                        ir::Callee::Function(
                            self.instances
                                .iter()
                                .position(|i| *i == instance)
                                .ok_or_else(|| format!("callee {instance} has no imported body"))?,
                        )
                    };
                    let ir::Place::Local(destination) = place(*destination)? else {
                        return Err("call destination must be a local".into());
                    };
                    ir::Terminator::Call {
                        callee,
                        arguments: args
                            .iter()
                            .map(|arg| self.operand(&arg.node))
                            .collect::<Result<_, _>>()?,
                        destination,
                        next: target.ok_or("diverging calls are unsupported")?.as_usize(),
                    }
                }
                other => {
                    return Err(Diagnostic::new(
                        Code::UnsupportedOperation,
                        format!("unsupported terminator {other:?}"),
                    )
                    .at(source(terminator.source_info.span)));
                }
            };
            blocks.push(ir::Block {
                assignments,
                terminator: term,
                source: source(terminator.source_info.span),
            });
        }
        Ok(ir::Function {
            name: self.instance.to_string(),
            source: source(body.span),
            locals: body
                .local_decls
                .iter()
                .map(|local| {
                    ty(self.normalize(local.ty)).map_err(|e| e.at(source(local.source_info.span)))
                })
                .collect::<Result<_, _>>()?,
            argument_count: body.arg_count,
            blocks,
        })
    }
}

pub fn module<'tcx>(
    tcx: TyCtxt<'tcx>,
    instances: &[Instance<'tcx>],
    marker: DefId,
) -> Result<ir::Module, Diagnostic> {
    let functions = instances
        .iter()
        .map(|&instance| {
            Import {
                tcx,
                instance,
                instances,
                marker,
            }
            .function()
            .map_err(|e| {
                e.in_function(instance.to_string()).at(tcx
                    .sess
                    .source_map()
                    .span_to_diagnostic_string(tcx.instance_mir(instance.def).span))
            })
        })
        .collect::<Result<_, _>>()?;
    let module = ir::Module {
        schema_version: 1,
        target: tcx.sess.opts.target_triple.to_string(),
        pointer_bits: 32,
        functions,
    };
    Ok(ir::verify(module)?.module().clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unwind_propagation_and_cleanup_with_call_site() {
        for action in [
            mir::UnwindAction::Continue,
            mir::UnwindAction::Cleanup(mir::START_BLOCK),
        ] {
            let error = validate_unwind(action, "test.rs:3:5".into()).unwrap_err();
            assert_eq!(error.code, Code::Unwind);
            assert_eq!(error.source.as_deref(), Some("test.rs:3:5"));
        }
        assert!(validate_unwind(mir::UnwindAction::Unreachable, "test.rs:3:5".into()).is_ok());
    }
}
