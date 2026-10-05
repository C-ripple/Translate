//! First, deliberately narrow SPMD legality pass and Ripple C emitter.
//!
//! Symbolically inline acyclic IR with bounded path expansion. Accept only
//! lane-indexed transfers from the immutable input to the disjoint output,
//! guarded by lane < length. Every memory address has a lane-shaped index;
//! scalar effects, unchecked bounds, offset chains and general memory aliasing
//! are rejected. This is a subset compiler, not general Rust code generation.
use ripple_ir::*;

#[derive(Clone, Debug, PartialEq)]
enum Expr {
    Unit,
    Bool(bool),
    Integer(u32),
    Lane,
    Length,
    Binary(&'static str, Box<Expr>, Box<Expr>),
    Pointer {
        output: bool,
        index: Option<Box<Expr>>,
    },
    Load(Box<Expr>),
}

#[derive(Clone)]
struct State {
    locals: Vec<Option<Expr>>,
    guards: Vec<(Expr, bool)>,
}

struct Store {
    guards: Vec<(Expr, bool)>,
    value: Expr,
}
struct Lower<'a> {
    module: &'a Module,
    stores: Vec<Store>,
    steps: usize,
}

fn bounds() -> Expr {
    Expr::Binary("<", Box::new(Expr::Lane), Box::new(Expr::Length))
}

fn safe_memory(pointer: &Expr, state: &State, output: bool) -> Result<(), Diagnostic> {
    if !matches!(pointer, Expr::Pointer { output: actual, index: Some(index) }
        if *actual == output && **index == Expr::Lane)
    {
        return Err(Diagnostic::new(
            Code::Address,
            "memory access must use the lane index of the appropriate ABI buffer",
        ));
    }
    if !state
        .guards
        .iter()
        .any(|(expr, truth)| *truth && *expr == bounds())
    {
        return Err(Diagnostic::new(
            Code::Bounds,
            "memory access is not dominated by lane < length",
        ));
    }
    Ok(())
}

fn operand(operand: &Operand, state: &State) -> Result<Expr, Diagnostic> {
    Ok(match operand {
        Operand::Unit => Expr::Unit,
        Operand::Bool(value) => Expr::Bool(*value),
        Operand::Integer { value, .. } => Expr::Integer(*value),
        Operand::Read(Place::Local(local)) => state.locals[*local]
            .clone()
            .ok_or("missing symbolic local")?,
        Operand::Read(Place::Deref(local)) => {
            let pointer = state.locals[*local]
                .as_ref()
                .ok_or("missing symbolic pointer")?;
            safe_memory(pointer, state, false)?;
            Expr::Load(Box::new(pointer.clone()))
        }
    })
}

fn value(value: &Value, state: &State) -> Result<Expr, Diagnostic> {
    let result = match value {
        Value::Use(op) | Value::IntegerCast { operand: op, .. } => operand(op, state)?,
        Value::Binary { op, lhs, rhs } => {
            let lhs = operand(lhs, state)?;
            let rhs = operand(rhs, state)?;
            match op {
                Binary::PointerOffset => {
                    let Expr::Pointer {
                        output,
                        index: None,
                    } = lhs
                    else {
                        return Err(Diagnostic::new(
                            Code::Address,
                            "chained or non-buffer pointer offsets are unsupported",
                        ));
                    };
                    if rhs != Expr::Lane || !state.guards.iter().any(|(e, t)| *t && *e == bounds())
                    {
                        return Err(Diagnostic::new(
                            Code::Bounds,
                            "pointer offset requires a proven in-range lane",
                        ));
                    }
                    Expr::Pointer {
                        output,
                        index: Some(Box::new(rhs)),
                    }
                }
                _ => Expr::Binary(
                    match op {
                        Binary::Add => "+",
                        Binary::Subtract => "-",
                        Binary::Multiply => "*",
                        Binary::LessThan => "<",
                        Binary::Equal => "==",
                        _ => unreachable!(),
                    },
                    Box::new(lhs),
                    Box::new(rhs),
                ),
            }
        }
    };
    let mut pending = vec![&result];
    let mut nodes = 0;
    while let Some(expr) = pending.pop() {
        nodes += 1;
        if nodes > 128 {
            return Err(Diagnostic::new(
                Code::Expansion,
                "symbolic expression size limit exceeded",
            ));
        }
        match expr {
            Expr::Binary(_, lhs, rhs) => {
                pending.push(lhs);
                pending.push(rhs);
            }
            Expr::Pointer {
                index: Some(index), ..
            }
            | Expr::Load(index) => pending.push(index),
            _ => {}
        }
    }
    Ok(result)
}

impl Lower<'_> {
    fn walk(
        &mut self,
        function: usize,
        block: usize,
        mut state: State,
    ) -> Result<Vec<State>, Diagnostic> {
        self.steps += 1;
        if self.steps > 256 || self.stores.len() > 64 {
            return Err(Diagnostic::new(
                Code::Expansion,
                "SPMD path expansion limit exceeded",
            ));
        }
        let current = self.module.functions[function].blocks[block].clone();
        for assignment in current.assignments {
            self.steps += 1;
            if self.steps > 256 {
                return Err(Diagnostic::new(
                    Code::Expansion,
                    "SPMD path expansion limit exceeded",
                ));
            }
            let result = value(&assignment.value, &state).map_err(|e| e.at(&assignment.source))?;
            match assignment.destination {
                Place::Local(local) => state.locals[local] = Some(result),
                Place::Deref(local) => {
                    if self.stores.len() >= 64 {
                        return Err(Diagnostic::new(
                            Code::Expansion,
                            "store expansion limit exceeded",
                        ));
                    }
                    safe_memory(
                        state.locals[local]
                            .as_ref()
                            .ok_or("missing store pointer")?,
                        &state,
                        true,
                    )
                    .map_err(|e| e.at(&assignment.source))?;
                    if !matches!(result, Expr::Load(_)) {
                        return Err(Diagnostic::new(
                            Code::UnsupportedOperation,
                            "initial backend supports f32 buffer transfers only",
                        ));
                    }
                    self.stores.push(Store {
                        guards: state.guards.clone(),
                        value: result,
                    });
                }
            }
        }
        match current.terminator {
            Terminator::Return => Ok(vec![state]),
            Terminator::Goto(next) => self.walk(function, next, state),
            Terminator::Branch {
                condition,
                on_true,
                on_false,
            } => {
                let condition = operand(&condition, &state)?;
                let mut result = Vec::new();
                for (truth, next) in [(true, on_true), (false, on_false)] {
                    if matches!(condition, Expr::Bool(v) if v != truth) {
                        continue;
                    }
                    if state
                        .guards
                        .iter()
                        .any(|(expr, prior)| *expr == condition && *prior != truth)
                    {
                        continue;
                    }
                    let mut branch = state.clone();
                    branch.guards.push((condition.clone(), truth));
                    result.extend(self.walk(function, next, branch)?);
                }
                Ok(result)
            }
            Terminator::Call {
                callee,
                arguments,
                destination,
                next,
            } => match callee {
                Callee::LaneId => {
                    state.locals[destination] = Some(Expr::Lane);
                    self.walk(function, next, state)
                }
                Callee::Function(id) => {
                    let mut child = State {
                        locals: vec![None; self.module.functions[id].locals.len()],
                        guards: state.guards.clone(),
                    };
                    for (i, argument) in arguments.iter().enumerate() {
                        child.locals[i + 1] = Some(operand(argument, &state)?);
                    }
                    let returns = self.walk(id, 0, child)?;
                    let mut results = Vec::new();
                    for returned in returns {
                        let mut continuation = state.clone();
                        continuation.guards = returned.guards;
                        continuation.locals[destination] =
                            returned.locals[0].clone().or_else(|| {
                                (self.module.functions[id].locals[0] == Type::Unit)
                                    .then_some(Expr::Unit)
                            });
                        results.extend(self.walk(function, next, continuation)?);
                    }
                    Ok(results)
                }
            },
        }
    }
}

fn expression(expr: &Expr) -> Result<String, Diagnostic> {
    Ok(match expr {
        Expr::Bool(v) => if *v { "1" } else { "0" }.into(),
        Expr::Integer(v) => format!("{v}U"),
        Expr::Lane => "lane".into(),
        Expr::Length => "remaining".into(),
        Expr::Binary(op, lhs, rhs) => {
            let text = format!("({} {op} {})", expression(lhs)?, expression(rhs)?);
            if matches!(*op, "+" | "-" | "*") {
                format!("((uint32_t){text})")
            } else {
                text
            }
        }
        Expr::Load(_) => "src[safe_lane]".into(),
        _ => {
            return Err(Diagnostic::new(
                Code::UnsupportedOperation,
                "non-scalar expression in emitted predicate",
            ));
        }
    })
}

/// ABI v1 requires disjoint, 128-byte-aligned buffers of at least `length`
/// initialized f32 elements; each block owns its lane-indexed output elements.
/// The wrapper validates the descriptor, but cannot validate allocation bounds.
pub fn emit(module: VerifiedModule) -> Result<String, Diagnostic> {
    let module = module.module();
    let root = &module.functions[0];
    if root.argument_count != 3
        || root.locals[..4]
            != [
                Type::Unit,
                Type::F32Pointer { mutable: false },
                Type::F32Pointer { mutable: true },
                Type::U32,
            ]
    {
        return Err(Diagnostic::new(
            Code::Entry,
            "entry must have signature (*const f32, *mut f32, u32) -> ()",
        ));
    }
    let mut state = State {
        locals: vec![None; root.locals.len()],
        guards: vec![],
    };
    state.locals[1] = Some(Expr::Pointer {
        output: false,
        index: None,
    });
    state.locals[2] = Some(Expr::Pointer {
        output: true,
        index: None,
    });
    state.locals[3] = Some(Expr::Length);
    let mut lowering = Lower {
        module,
        stores: vec![],
        steps: 0,
    };
    lowering
        .walk(0, 0, state)
        .map_err(|e| e.at(&root.source).in_function(&root.name))?;
    if lowering.stores.is_empty() {
        return Err(Diagnostic::new(
            Code::Entry,
            "entry has no supported buffer effects",
        ));
    }
    let mut result = String::from(
        "/* Generated from verified owned Rust MIR. Initial ABI v1 subset. */\n#include <stddef.h>\n#include <stdint.h>\n#include <ripple.h>\n#include \"launch.h\"\n_Static_assert(sizeof(void *) == 4, \"pointer width\");\n_Static_assert(sizeof(ripple_launch_v1) == 16, \"descriptor size\");\n_Static_assert(offsetof(ripple_launch_v1, block_offset) == 12, \"descriptor layout\");\nstatic void kernel_block(const float *src, float *dst, uint32_t remaining) {\n  ripple_block_t block = ripple_set_block_shape(0, 32);\n  uint32_t lane = ripple_id(block, 0);\n  size_t safe_lane = lane;\n",
    );
    let mut body = String::new();
    for store in lowering.stores {
        let guard = store
            .guards
            .iter()
            .map(|(expr, truth)| {
                expression(expr).map(|text| if *truth { text } else { format!("!({text})") })
            })
            .collect::<Result<Vec<_>, _>>()?
            .join(" && ");
        body.push_str(&format!(
            "  if ({guard}) {{ dst[safe_lane] = {}; }}\n",
            expression(&store.value)?
        ));
        if body.len() > 512 * 1024 {
            return Err(Diagnostic::new(
                Code::Expansion,
                "generated source size limit exceeded",
            ));
        }
    }
    result.push_str(&body);
    result.push_str("}\nstatic void kernel_tail(const float *src, float *dst, uint32_t remaining) {\n  for (uint32_t lane = 0; lane < remaining; ++lane) {\n    size_t safe_lane = lane;\n");
    result.push_str(&body);
    result.push_str("  }\n}\nint ripple_probe_copy_v1(const ripple_launch_v1 *launch, const float *src, float *dst) {\n  if (!launch || launch->abi_version != 1 || launch->struct_bytes != sizeof(*launch)) return RIPPLE_BAD_ABI;\n  if (launch->block_offset > launch->length || launch->block_offset % 32 != 0) return RIPPLE_BAD_LAUNCH;\n  uint32_t remaining = launch->length - launch->block_offset;\n  if (!remaining) return RIPPLE_OK;\n  if (!src || !dst) return RIPPLE_BAD_LAUNCH;\n  if (remaining < 32) kernel_tail(src + launch->block_offset, dst + launch->block_offset, remaining);\n  else kernel_block(src + launch->block_offset, dst + launch->block_offset, remaining);\n  return RIPPLE_OK;\n}\n");
    if result.len() > 1024 * 1024 {
        return Err(Diagnostic::new(
            Code::Expansion,
            "generated source size limit exceeded",
        ));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn read(local: usize) -> Operand {
        Operand::Read(Place::Local(local))
    }
    fn assignment(destination: Place, value: Value) -> Assignment {
        Assignment {
            destination,
            value,
            source: "test.rs:1".into(),
        }
    }
    fn binary(local: usize, op: Binary, lhs: usize, rhs: usize) -> Assignment {
        assignment(
            Place::Local(local),
            Value::Binary {
                op,
                lhs: read(lhs),
                rhs: read(rhs),
            },
        )
    }
    fn block(assignments: Vec<Assignment>, terminator: Terminator) -> Block {
        Block {
            assignments,
            terminator,
            source: "test.rs:1".into(),
        }
    }
    fn transfer() -> Module {
        Module {
            schema_version: 1,
            target: "hexagon-unknown-none-elf".into(),
            pointer_bits: 32,
            functions: vec![Function {
                name: "transfer".into(),
                source: "test.rs:1".into(),
                locals: vec![
                    Type::Unit,
                    Type::F32Pointer { mutable: false },
                    Type::F32Pointer { mutable: true },
                    Type::U32,
                    Type::U32,
                    Type::Bool,
                    Type::F32Pointer { mutable: false },
                    Type::F32Pointer { mutable: true },
                    Type::F32,
                    Type::U32,
                ],
                argument_count: 3,
                blocks: vec![
                    block(
                        vec![],
                        Terminator::Call {
                            callee: Callee::LaneId,
                            arguments: vec![],
                            destination: 4,
                            next: 1,
                        },
                    ),
                    block(
                        vec![binary(5, Binary::LessThan, 4, 3)],
                        Terminator::Branch {
                            condition: read(5),
                            on_true: 2,
                            on_false: 3,
                        },
                    ),
                    block(
                        vec![
                            binary(6, Binary::PointerOffset, 1, 9),
                            assignment(Place::Local(8), Value::Use(Operand::Read(Place::Deref(6)))),
                            binary(7, Binary::PointerOffset, 2, 9),
                            assignment(Place::Deref(7), Value::Use(read(8))),
                        ],
                        Terminator::Goto(3),
                    ),
                    block(vec![], Terminator::Return),
                ],
            }],
        }
    }
    fn valid_transfer() -> Module {
        let mut m = transfer();
        m.functions[0].locals[9] = Type::Usize;
        m.functions[0].blocks[2].assignments.insert(
            0,
            assignment(
                Place::Local(9),
                Value::IntegerCast {
                    operand: read(4),
                    to: Type::Usize,
                },
            ),
        );
        m
    }
    #[test]
    fn accepts_guarded_transfer_and_rejects_unguarded() {
        assert!(emit(verify(valid_transfer()).unwrap()).is_ok());
        let mut m = valid_transfer();
        m.functions[0].blocks[1].terminator = Terminator::Goto(2);
        assert!(
            emit(verify(m).unwrap())
                .unwrap_err()
                .message
                .contains("in-range")
        );
    }
    #[test]
    fn rejects_output_loads_and_scalar_stores() {
        let mut m = valid_transfer();
        let assignments = &mut m.functions[0].blocks[2].assignments;
        assignments.last_mut().unwrap().destination = Place::Deref(2);
        assert!(
            emit(verify(m).unwrap())
                .unwrap_err()
                .message
                .contains("lane index")
        );
        let mut m = valid_transfer();
        m.functions[0].locals[6] = Type::F32Pointer { mutable: true };
        m.functions[0].blocks[2].assignments[1] = binary(6, Binary::PointerOffset, 2, 9);
        assert!(
            emit(verify(m).unwrap())
                .unwrap_err()
                .message
                .contains("lane index")
        );
    }
    #[test]
    fn bounds_expression_growth_inside_one_block() {
        let mut m = valid_transfer();
        let assignments = &mut m.functions[0].blocks[1].assignments;
        assignments.push(assignment(Place::Local(4), Value::Use(read(3))));
        for _ in 0..20 {
            assignments.push(binary(4, Binary::Add, 4, 4));
        }
        assert!(
            emit(verify(m).unwrap())
                .unwrap_err()
                .message
                .contains("expression size")
        );
    }
    #[test]
    fn bounds_assignments_inside_one_block() {
        let mut m = valid_transfer();
        for _ in 0..300 {
            m.functions[0].blocks[1]
                .assignments
                .push(assignment(Place::Local(0), Value::Use(Operand::Unit)));
        }
        assert!(
            emit(verify(m).unwrap())
                .unwrap_err()
                .message
                .contains("expansion limit")
        );
    }
}
