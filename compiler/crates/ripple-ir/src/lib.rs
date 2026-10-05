//! Owned, typed IR for the initial acyclic, scalar/f32-pointer kernel subset.
//! This crate has no dependency on rustc internals or a target code generator.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
pub mod diagnostic;
pub use diagnostic::{Code, Diagnostic};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Type {
    Unit,
    Bool,
    U32,
    Usize,
    F32,
    F32Pointer { mutable: bool },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Module {
    pub schema_version: u32,
    pub target: String,
    pub pointer_bits: u32,
    pub functions: Vec<Function>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Function {
    pub name: String,
    pub source: String,
    /// Local 0 is the return slot; locals 1..=argument_count are parameters.
    pub locals: Vec<Type>,
    pub argument_count: usize,
    pub blocks: Vec<Block>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Block {
    pub assignments: Vec<Assignment>,
    pub terminator: Terminator,
    pub source: String,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Place {
    Local(usize),
    Deref(usize),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Operand {
    Read(Place),
    Integer { value: u32, ty: Type },
    Bool(bool),
    Unit,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Binary {
    /// Unsigned arithmetic wraps modulo 2^32. Checked and unchecked MIR
    /// operations must not be imported as these operations.
    Add,
    /// Unsigned subtraction wraps modulo 2^32.
    Subtract,
    /// Unsigned multiplication wraps modulo 2^32.
    Multiply,
    LessThan,
    Equal,
    /// Rust in-bounds pointer offset. Backend must account for inactive lanes.
    PointerOffset,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Value {
    Use(Operand),
    Binary {
        op: Binary,
        lhs: Operand,
        rhs: Operand,
    },
    IntegerCast {
        operand: Operand,
        to: Type,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Assignment {
    pub destination: Place,
    pub value: Value,
    pub source: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Callee {
    Function(usize),
    LaneId,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Terminator {
    Return,
    Goto(usize),
    Branch {
        condition: Operand,
        on_true: usize,
        on_false: usize,
    },
    Call {
        callee: Callee,
        arguments: Vec<Operand>,
        destination: usize,
        next: usize,
    },
}

pub struct VerifiedModule(Module);

impl VerifiedModule {
    pub fn module(&self) -> &Module {
        &self.0
    }
}

fn integer(ty: Type) -> bool {
    matches!(ty, Type::U32 | Type::Usize)
}

fn place_type(function: &Function, place: Place, write: bool) -> Result<Type, Diagnostic> {
    let (Place::Local(local) | Place::Deref(local)) = place;
    let ty = *function
        .locals
        .get(local)
        .ok_or_else(|| Diagnostic::new(Code::InvalidIr, "invalid local reference"))?;
    match place {
        Place::Local(_) => Ok(ty),
        Place::Deref(_) => match ty {
            Type::F32Pointer { mutable } if !write || mutable => Ok(Type::F32),
            _ => Err(Diagnostic::new(
                Code::Type,
                "invalid pointer dereference or write through const pointer",
            )),
        },
    }
}

fn operand_type(function: &Function, operand: &Operand) -> Result<Type, Diagnostic> {
    match operand {
        Operand::Read(place) => place_type(function, *place, false),
        Operand::Integer { ty, .. } if integer(*ty) => Ok(*ty),
        Operand::Integer { .. } => Err(Diagnostic::new(
            Code::Type,
            "integer constant has non-integer type",
        )),
        Operand::Bool(_) => Ok(Type::Bool),
        Operand::Unit => Ok(Type::Unit),
    }
}

fn value_type(function: &Function, value: &Value) -> Result<Type, Diagnostic> {
    match value {
        Value::Use(operand) => operand_type(function, operand),
        Value::IntegerCast { operand, to }
            if integer(operand_type(function, operand)?) && integer(*to) =>
        {
            Ok(*to)
        }
        Value::IntegerCast { .. } => Err(Diagnostic::new(Code::Type, "unsupported integer cast")),
        Value::Binary { op, lhs, rhs } => {
            let lhs = operand_type(function, lhs)?;
            let rhs = operand_type(function, rhs)?;
            match op {
                Binary::PointerOffset
                    if matches!(lhs, Type::F32Pointer { .. }) && rhs == Type::Usize =>
                {
                    Ok(lhs)
                }
                Binary::LessThan | Binary::Equal if lhs == rhs && integer(lhs) => Ok(Type::Bool),
                Binary::Add | Binary::Subtract | Binary::Multiply if lhs == rhs && integer(lhs) => {
                    Ok(lhs)
                }
                _ => Err(Diagnostic::new(Code::Type, "invalid binary operand types")),
            }
        }
    }
}

fn read(operand: &Operand, initialized: &BTreeSet<usize>) -> Result<(), Diagnostic> {
    if let Operand::Read(Place::Local(local) | Place::Deref(local)) = operand
        && !initialized.contains(local)
    {
        return Err(Diagnostic::new(
            Code::Initialization,
            format!("read of uninitialized local {local}"),
        ));
    }
    Ok(())
}

fn read_value(value: &Value, initialized: &BTreeSet<usize>) -> Result<(), Diagnostic> {
    match value {
        Value::Use(operand) | Value::IntegerCast { operand, .. } => read(operand, initialized),
        Value::Binary { lhs, rhs, .. } => {
            read(lhs, initialized)?;
            read(rhs, initialized)
        }
    }
}

fn successors(terminator: &Terminator) -> Vec<usize> {
    match *terminator {
        Terminator::Return => vec![],
        Terminator::Goto(next) | Terminator::Call { next, .. } => vec![next],
        Terminator::Branch {
            on_true, on_false, ..
        } => vec![on_true, on_false],
    }
}

/// Validate the deliberately restricted first IR dialect. This verifies types,
/// acyclic control flow, definite initialization and direct call signatures. It
/// does NOT establish SPMD/predication legality; that is a separate backend gate.
pub fn verify(module: Module) -> Result<VerifiedModule, Diagnostic> {
    if module.schema_version != 1
        || module.pointer_bits != 32
        || module.target != "hexagon-unknown-none-elf"
        || module.functions.is_empty()
    {
        return Err(Diagnostic::new(
            Code::InvalidIr,
            "unsupported IR schema/target or empty module",
        ));
    }
    let mut operations = 0usize;
    if module.functions.len() > 256 {
        return Err(Diagnostic::new(Code::IrLimit, "function limit exceeded"));
    }
    for function in &module.functions {
        if function.blocks.len() > 4096 || function.locals.len() > 4096 {
            return Err(
                Diagnostic::new(Code::IrLimit, "block/local limit exceeded").at(&function.source)
            );
        }
        for block in &function.blocks {
            operations = operations
                .saturating_add(block.assignments.len())
                .saturating_add(1);
            if operations > 16384 {
                return Err(
                    Diagnostic::new(Code::IrLimit, "module operation limit exceeded")
                        .at(&block.source),
                );
            }
        }
    }
    for function in &module.functions {
        verify_function(&module, function)
            .map_err(|e| e.at(&function.source).in_function(&function.name))?;
    }
    // Recursion is outside the initial dialect, even if each CFG is acyclic.
    fn visit(module: &Module, id: usize, state: &mut [u8]) -> Result<(), Diagnostic> {
        if state[id] == 1 {
            return Err(Diagnostic::new(
                Code::ControlFlow,
                "recursive call graph is unsupported",
            ));
        }
        if state[id] == 2 {
            return Ok(());
        }
        state[id] = 1;
        for block in &module.functions[id].blocks {
            if let Terminator::Call {
                callee: Callee::Function(next),
                ..
            } = block.terminator
            {
                visit(module, next, state)
                    .map_err(|e| e.at(&block.source).in_function(&module.functions[id].name))?;
            }
        }
        state[id] = 2;
        Ok(())
    }
    let mut state = vec![0; module.functions.len()];
    for id in 0..state.len() {
        visit(&module, id, &mut state)?;
    }
    Ok(VerifiedModule(module))
}

fn verify_function(module: &Module, function: &Function) -> Result<(), Diagnostic> {
    if function.locals.is_empty()
        || function.argument_count >= function.locals.len()
        || function.blocks.is_empty()
    {
        return Err(Diagnostic::new(
            Code::InvalidIr,
            "missing return slot, invalid arguments, or empty CFG",
        ));
    }
    let mut predecessors = vec![Vec::new(); function.blocks.len()];
    for (id, block) in function.blocks.iter().enumerate() {
        for next in successors(&block.terminator) {
            predecessors
                .get_mut(next)
                .ok_or_else(|| {
                    Diagnostic::new(Code::ControlFlow, "invalid basic-block reference")
                        .at(&block.source)
                })?
                .push(id);
        }
    }
    let mut indegree: Vec<_> = predecessors.iter().map(Vec::len).collect();
    if indegree[0] != 0 {
        return Err(Diagnostic::new(
            Code::ControlFlow,
            "backedge to entry is unsupported",
        ));
    }
    let mut queue = std::collections::VecDeque::from([0]);
    let mut output = vec![BTreeSet::new(); function.blocks.len()];
    let mut visited = 0;
    while let Some(id) = queue.pop_front() {
        visited += 1;
        let mut initialized: BTreeSet<usize> = if id == 0 {
            (1..=function.argument_count).collect()
        } else {
            let mut values = output[predecessors[id][0]].clone();
            for &predecessor in &predecessors[id][1..] {
                values = values.intersection(&output[predecessor]).copied().collect();
            }
            values
        };
        let block = &function.blocks[id];
        if block.assignments.len() > 4096 {
            return Err(
                Diagnostic::new(Code::IrLimit, "assignment limit exceeded").at(&block.source)
            );
        }
        for assignment in &block.assignments {
            let result: Result<(), Diagnostic> = (|| {
                if place_type(function, assignment.destination, true)?
                    != value_type(function, &assignment.value)?
                {
                    return Err(Diagnostic::new(Code::Type, "assignment type mismatch")
                        .at(&assignment.source));
                }
                read_value(&assignment.value, &initialized)?;
                match assignment.destination {
                    Place::Local(local) => {
                        initialized.insert(local);
                    }
                    Place::Deref(local) if !initialized.contains(&local) => {
                        return Err(Diagnostic::new(
                            Code::Initialization,
                            "store through uninitialized pointer",
                        ));
                    }
                    _ => {}
                }
                Ok(())
            })();
            result.map_err(|e| e.at(&assignment.source))?;
        }
        let result: Result<(), Diagnostic> = (|| {
            match &block.terminator {
                Terminator::Return
                    if function.locals[0] != Type::Unit && !initialized.contains(&0) =>
                {
                    return Err(Diagnostic::new(
                        Code::Initialization,
                        "uninitialized return slot",
                    ));
                }
                Terminator::Branch { condition, .. } => {
                    read(condition, &initialized)?;
                    if operand_type(function, condition)? != Type::Bool {
                        return Err(Diagnostic::new(Code::Type, "branch condition is not bool"));
                    }
                }
                Terminator::Call {
                    callee,
                    arguments,
                    destination,
                    ..
                } => {
                    let (parameters, returned) = match callee {
                        Callee::LaneId => (vec![], Type::U32),
                        Callee::Function(id) => {
                            let callee = module.functions.get(*id).ok_or_else(|| {
                                Diagnostic::new(Code::Call, "invalid function reference")
                            })?;
                            if callee.argument_count >= callee.locals.len() {
                                return Err(Diagnostic::new(
                                    Code::Call,
                                    "invalid callee signature",
                                ));
                            }
                            (
                                callee.locals[1..=callee.argument_count].to_vec(),
                                callee.locals[0],
                            )
                        }
                    };
                    if arguments.len() != parameters.len() {
                        return Err(Diagnostic::new(Code::Call, "call arity mismatch"));
                    }
                    for (argument, expected) in arguments.iter().zip(parameters) {
                        read(argument, &initialized)?;
                        if operand_type(function, argument)? != expected {
                            return Err(Diagnostic::new(Code::Call, "call argument type mismatch"));
                        }
                    }
                    if place_type(function, Place::Local(*destination), true)? != returned {
                        return Err(Diagnostic::new(Code::Call, "call result type mismatch"));
                    }
                    initialized.insert(*destination);
                }
                _ => {}
            }
            Ok(())
        })();
        result.map_err(|e| e.at(&block.source))?;
        output[id] = initialized;
        for next in successors(&block.terminator) {
            indegree[next] -= 1;
            if indegree[next] == 0 {
                queue.push_back(next);
            }
        }
    }
    if visited != function.blocks.len() {
        return Err(Diagnostic::new(
            Code::ControlFlow,
            "cyclic or unreachable basic blocks are outside the initial dialect",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module(blocks: Vec<Block>) -> Module {
        Module {
            schema_version: 1,
            target: "hexagon-unknown-none-elf".into(),
            pointer_bits: 32,
            functions: vec![Function {
                name: "test".into(),
                source: "test.rs:1".into(),
                locals: vec![Type::U32, Type::Bool, Type::U32],
                argument_count: 1,
                blocks,
            }],
        }
    }
    fn block(assignments: Vec<Assignment>, terminator: Terminator) -> Block {
        Block {
            assignments,
            terminator,
            source: "test.rs:1".into(),
        }
    }
    fn assign(local: usize, operand: Operand) -> Assignment {
        Assignment {
            destination: Place::Local(local),
            value: Value::Use(operand),
            source: "test.rs:1".into(),
        }
    }
    fn constant(value: u32) -> Operand {
        Operand::Integer {
            value,
            ty: Type::U32,
        }
    }

    #[test]
    fn rejects_value_defined_on_only_one_branch() {
        let m = module(vec![
            block(
                vec![],
                Terminator::Branch {
                    condition: Operand::Read(Place::Local(1)),
                    on_true: 1,
                    on_false: 2,
                },
            ),
            block(vec![assign(2, constant(7))], Terminator::Goto(3)),
            block(vec![], Terminator::Goto(3)),
            block(
                vec![assign(0, Operand::Read(Place::Local(2)))],
                Terminator::Return,
            ),
        ]);
        assert!(
            verify(m)
                .err()
                .unwrap()
                .message
                .contains("uninitialized local 2")
        );
    }

    #[test]
    fn accepts_value_defined_on_both_branches() {
        let m = module(vec![
            block(
                vec![],
                Terminator::Branch {
                    condition: Operand::Read(Place::Local(1)),
                    on_true: 1,
                    on_false: 2,
                },
            ),
            block(vec![assign(2, constant(7))], Terminator::Goto(3)),
            block(vec![assign(2, constant(9))], Terminator::Goto(3)),
            block(
                vec![assign(0, Operand::Read(Place::Local(2)))],
                Terminator::Return,
            ),
        ]);
        assert!(verify(m).is_ok()); // Type/CFG correctness alone is not SPMD legality.
    }

    #[test]
    fn rejects_invalid_target_and_cfg() {
        let mut m = module(vec![block(vec![], Terminator::Goto(99))]);
        assert!(
            verify(m.clone())
                .err()
                .unwrap()
                .message
                .contains("basic-block")
        );
        m.pointer_bits = 64;
        assert!(verify(m).err().unwrap().message.contains("target"));
    }

    #[test]
    fn rejects_call_signature_mismatch() {
        let m = module(vec![
            block(
                vec![],
                Terminator::Call {
                    callee: Callee::LaneId,
                    arguments: vec![constant(2)],
                    destination: 0,
                    next: 1,
                },
            ),
            block(vec![], Terminator::Return),
        ]);
        assert!(verify(m).err().unwrap().message.contains("arity"));
    }
    #[test]
    fn malformed_assignment_diagnostics_preserve_context() {
        for (operand, code) in [
            (Operand::Read(Place::Local(99)), Code::InvalidIr),
            (Operand::Bool(true), Code::Type),
            (Operand::Read(Place::Local(2)), Code::Initialization),
        ] {
            let error = verify(module(vec![block(
                vec![assign(0, operand)],
                Terminator::Return,
            )]))
            .err()
            .unwrap();
            assert_eq!(error.code, code);
            assert_eq!(error.source.as_deref(), Some("test.rs:1"));
            assert_eq!(error.function.as_deref(), Some("test"));
        }
    }

    #[test]
    fn limits_fail_before_graph_traversal() {
        let mut m = module(vec![block(
            vec![assign(0, constant(1))],
            Terminator::Return,
        )]);
        m.functions[0].locals.resize(4097, Type::U32);
        assert_eq!(verify(m).err().unwrap().code, Code::IrLimit);
    }

    #[test]
    fn bounded_reference_mutations_never_panic() {
        for index in 0..512 {
            let m = module(vec![block(
                vec![assign(index, constant(1))],
                Terminator::Goto(index),
            )]);
            assert!(verify(m).is_err());
        }
    }
}
