use crate::lexer::{
    TokenType,
    Token
};

/*
Visitors in Rust seem to be frowned upon. I'm not sure what to think
about it, because the Rust compiler uses the visitor pattern [1].

There is an interesting discussion on reddit [2] about the visitor
pattern and AST data structures.

The Rust pattern website has a small example of visitors [3].

The Molten compiler/language uses the visitor pattern for typechecking [4],
which may be interesting to peruse further.

[1] https://doc.rust-lang.org/beta/nightly-rustc/rustc_ast/visit/trait.Visitor.html
[2] https://www.reddit.com/r/rust/comments/11q7l8m/best_practices_for_ast_design_in_rust/
[3] https://rust-unofficial.github.io/patterns/patterns/behavioural/visitor.html
[4] https://github.com/transistorfet/molten/blob/8487b4018f1eecbc1cb15e32c2652a6f0fed0941/src/analysis/typecheck.rs#L10
 */

// pub mod parse_tree {
//    use super::*;
#[derive(PartialEq, Debug)]
pub enum BinOp {
    Plus, Minus, Times, Divide, Power,
    // relational operators
    Eq, Ne, Le, Lt, Ge, Gt,
    // logical operators
    Eqv, Neqv, And, Or,
    // string operators
    Concatenate,
}

pub fn token_to_binop(token: Token) -> BinOp {
    match token.token_type {
        TokenType::Star => BinOp::Times,
        TokenType::Slash => BinOp::Divide,
        TokenType::Plus => BinOp::Plus,
        TokenType::Minus => BinOp::Minus,
        TokenType::Concatenation => BinOp::Concatenate,
        TokenType::Eq => BinOp::Eq,
        TokenType::NotEqual => BinOp::Ne,
        TokenType::Leq => BinOp::Le,
        TokenType::Less => BinOp::Lt,
        TokenType::Greater => BinOp::Gt,
        TokenType::Geq => BinOp::Ge,
        TokenType::And => BinOp::And,
        TokenType::Or => BinOp::Or,
        TokenType::Equiv => BinOp::Eqv,
        TokenType::NotEquiv => BinOp::Neqv,
        _ => {
            panic!("Trying to transform an invalid token into a binop: {}", token);
        }
    }
}

#[derive(PartialEq, Debug)]
pub enum UnOp {
    Plus, Minus,
    Not
}

pub fn token_to_unary_op(token: Token) -> UnOp {
    match token.token_type {
        TokenType::Plus => UnOp::Plus,
        TokenType::Minus => UnOp::Minus,
        TokenType::Not => UnOp::Not,
        _ => {
            panic!("Trying to transform an invalid token into a unary op: {}", token);
        }
    }
}

/*
At the level of the parse tree, we don't know the difference between a
function call and an array access [unless the array access involves
slicing the array].

We will later, during semantic analysis, replace variables by
"flavored de Bruijn indices" depending on if it refers to a function,
a subroutine, or a local variable. Its value is precisely the index for
the `Program.functions`, `Program.subroutines`, or `ProgramUnit::*.spec`
entry.

The grammar for arrays and function calls are represented by the
`NameDataRef` node.

```ebnf
Name ::= Identifier
NameDataRef ::= Name ComplexDataRefTail*
ComplexDataRefTail ::= "(" SectionSubscriptList ")"
SectionSubscriptList ::= SectionSubscript
                      | SectionSubscript "," SectionSubscriptList
SectionSubscript ::= Expr SubscriptTripletTail?
                  |  SubscriptTripletTail
SubscriptTripletTail ::= ":" Expr?
FunctionCallExpr ::= Name "(" ")"
```
 */
/*
TODO: It would be nice to parametrize the Expr structure by the type of
variable representation, so we could effectively write something along
the lines of:
```rust
pub enum Expr<T> {
   // ....
   Variable(T),
   FunCall(T, Vec<Expr>),
   ArrayElement(T, Vec<Expr>),
   ArraySection(T, Vec<Expr>),
   // ...
}
```
This would have been nice, so I could have then disambiguated named data
references in a function taking an `Expr<String>` and returning a `Expr<usize>`.

However, this would require propagating the generic parameter `<T>` to
statements, array specifications, variable declarations, program units,
and programs (and possibly quite a bit more than I realize at the moment).
This requires a substantial rewrite, compared to adding three or six new
lines to the enumeration as it stands.
 */
#[derive(PartialEq, Debug)]
pub enum Expr {
    // literals
    Character(Vec<char>),
    Float32(f32),
    Float64(f64),
    Int32(i32),
    Int64(i64),
    Logical(bool),
    Variable(String),
    // de Bruijn indices for various named data reference disambiguation
    VariableIndex(usize),
    FunctionIndex(usize),
    SubroutineIndex(usize),
    // TODO: consider adding a Subscript(Box<Expr>) to remind myself
    //       of a lingering burden to check during typechecking?
    // array slice section: start, stop, stride
    Section((Option<Box<Expr>>, Option<Box<Expr>>, Option<Box<Expr>>)),
    // composite expressions
    Binary(Box<Expr>, BinOp, Box<Expr>),
    Unary(UnOp, Box<Expr>),
    Grouping(Box<Expr>),
    NamedDataRef(String, Vec<Expr>), // function call or array element or array section?
    FunCall(String, Vec<Expr>),
    ArrayElement(String, Vec<Expr>), // e.g., "MYARRAY(3,65,2)"
    ArraySection(String, Vec<Expr>), // e.g., "MYARRAY(3:65)"
    // composite expressions transformed to use de Bruijn indices
    IndexedFunCall(usize, Vec<Expr>),
    IndexedArrayElement(usize, Vec<Expr>),
    IndexedArraySection(usize, Vec<Expr>),
    // placeholder, should never be forced to arrive here
    ErrorExpr,
}

#[derive(PartialEq, Debug)]
pub enum Command<E: std::cmp::PartialEq> {
    Continue,
    Goto(i32),
    Write(Vec<E>),
    Read(Vec<E>),
    IfBlock {test: E,
             true_branch: Vec<Statement<E>>,
             false_branch: Vec<Statement<E>>},
    ArithIf {test: E,
             negative: i32,
             zero: i32,
             positive: i32},
    IfStatement {test: E,
                 true_branch: Box::<Statement<E>>},
    LabelDo {target_label: i32,
             var: E,
             start: E,
             stop: E,
             stride: Option<Expr>,
             body: Vec<Statement<E>>,
             terminal: Box<Statement<E>>},
    CallSubroutine {
        subroutine: E, // identifier
        args: Vec<E>
    },
    ExprStatement(E),
    Assignment {
        lhs: E,
        rhs: E
    },
    Stop,
    Return,
    End,
    Illegal // should never be reached
}

// 5 digit label with values <= (10^5) - 1 <= 0b11000011010011111 < 2^17
#[derive(PartialEq, Debug)]
pub struct Statement<E: std::cmp::PartialEq> {
    pub label: Option<i32>,
    pub command: Command<E>,
}

impl<E: std::cmp::PartialEq> Statement<E> {
    pub fn is_continue(&mut self) -> bool {
        matches!(self.command, Command::<E>::Continue)
    }
    pub fn is_end(&mut self) -> bool {
        matches!(self.command, Command::<E>::End)
    }
    pub fn is_stop(&mut self) -> bool {
        matches!(self.command, Command::<E>::Stop)
    }
    pub fn is_return(&mut self) -> bool {
        matches!(self.command, Command::<E>::Return)
    }
}

/*
We could use the fact that the Real mask equals `0x0d` and the Integer
mask equals `0x0e`, but also we could note that `Type::Real & kind` is
nonzero for `Type::Real`, `Type::Real64`, and `Type::Real128`. (Similar
remarks apply for `Type::Integer & kind`.)

These flags are chosen for compatibility with `lexer::BaseType` flags.

Arguably, this can be abused for external functions by treating
`Type::External` as a mask instead of a flag.
 */
#[derive(PartialEq, Debug, Copy, Clone)]
#[repr(u8)]
pub enum Type {
    Logical    = 0x10,  // 0b0001_0000
    Real       = 0x01,  // 0b0000_0001
    Integer    = 0x02,  // 0b0000_0010
    Character  = 0x20,  // 0b0010_0000
    Real64     = 0x05,  // 0b0000_0101
    Real128    = 0x09,  // 0b0000_1001
    Integer64  = 0x06,  // 0b0000_0110
    Integer128 = 0x0a,  // 0b0000_1010
    External   = 0x40,  // 0b0100_0000 function reference
}

/*
Section 5.1 of the Fortran 77 Standard does not actually give bounds on
the dimensions of an array (just that they must be arithmetic
expressions containing integer expressions).

Table 6.1 in the Fortran 90 standard gives examples of indexing in
Fortran, and how it's interpreted. If we were to follow this Standard,
we should have `upper_bound - lower_bound` be a `usize`. It therefore
makes sense to use `isize` for the ArrayIndex. We can use
`upper_bound.abs_diff(lower_bound)` to get the size of the dimension.
 */
pub type ArrayIndex = isize;

/*
array_spec = explicit_shape_spec_list
           | assumed_shape_spec_list
           | assumed_size_spec;

explicit_shape_spec_list = explicit_spec {"," explicit_spec};
explicit_spec = [lower_bound ":"] upper_bound;

assumed_shape_spec_list = assumed_shape {"," assumed_shape};
assumed_shape = [lower_bound] ":";

assumed_size_spec = [explicit_shape_spec_list ","] [lower_bound ":"] "*";
 */
#[derive(PartialEq, Debug)]
pub enum ArraySpec {
    ExplicitShape(Vec<(Option<Expr>, Expr)>),
    AssumedShape(Vec<Option<Expr>>),
    // assumed_size_spec stored as AssumedSize(ExplicitShape, Some(lower_bound))
    AssumedSize(Vec<(Option<Expr>, Expr)>, Option<Expr>),
    Scalar,
}

impl ArraySpec {
    pub fn rank(self) -> usize {
        match self {
            ArraySpec::ExplicitShape(v) => v.len(),
            ArraySpec::AssumedShape(v) => v.len(),
            ArraySpec::AssumedSize(v, e) => v.len() + 1,
            ArraySpec::Scalar => 0,
        }
    }
}

/*
Specification statements include parameter declarations, type
declarations, etc. See section 8 of the Fortran 77 Standard.

```ebnf
specification_statement = type_declaration
                        | parameter;

unsupported_specification_statement = 
            equivalence (* no support *)
          | common
          | dimension
          | implicit
          | external
          | intrinsic
          | save;
 */
#[derive(PartialEq, Debug)]
pub struct VarDeclaration {
    pub kind: Type,
    pub name: String,
    pub array: ArraySpec,
    /*
    rank: u8, // F77 says this must be less than 7
    dims: Vec<(ArrayIndex,ArrayIndex)>,
     */
}
#[derive(PartialEq, Debug)]
pub enum Specification {
    TypeDeclaration (VarDeclaration),
    Param (String, Expr),
}

impl VarDeclaration {
    pub fn rank(self) -> usize {
        self.array.rank()
    }
}

#[derive(PartialEq, Debug)]
pub enum ProgramUnit<E: std::cmp::PartialEq> {
    Program {
        name: String,
        spec: Vec<Specification>,
        body: Vec<Statement<E>>,
    },
    Function {
        name: String,
        return_type: Type,
        params: Vec<String>,
        spec: Vec<Specification>,
        body: Vec<Statement<E>>,
    },
    Subroutine {
        name: String,
        params: Vec<String>,
        spec: Vec<Specification>,
        body: Vec<Statement<E>>,
    },
    Empty,
}

#[derive(PartialEq, Debug, Copy, Clone)]
pub enum ProgramUnitKind {
    Program,
    Subroutine,
    Function,
    Empty,
}

impl<E: std::cmp::PartialEq> ProgramUnit<E> {
    pub fn kind(&self) -> ProgramUnitKind {
        match *self {
            ProgramUnit::<E>::Program {..} => ProgramUnitKind::Program,
            ProgramUnit::<E>::Function {..} => ProgramUnitKind::Function,
            ProgramUnit::<E>::Subroutine {..} => ProgramUnitKind::Subroutine,
            ProgramUnit::<E>::Empty => ProgramUnitKind::Empty,
        }
    }

    pub fn is_empty(self) -> bool {
        matches!(self, ProgramUnit::<E>::Empty)
    }

    pub fn is_named(&self, the_name: &str) -> bool {
        return matches!(self, ProgramUnit::<E>::Program {name: the_name, ..})
            || matches!(self, ProgramUnit::<E>::Function {name: the_name, ..})
            || matches!(self, ProgramUnit::<E>::Subroutine {name: the_name, ..});
    }

    pub fn get_name(&self) -> String {
        match self {
            ProgramUnit::<E>::Program {name, ..} => name.clone(),
            ProgramUnit::<E>::Function {name, ..} => name.clone(),
            ProgramUnit::<E>::Subroutine {name, ..} => name.clone(),
            _ => String::from("")
        }
    }

    pub fn shares_name(&self, other: &ProgramUnit<E>) -> bool {
        match (self, other) {
            (ProgramUnit::<E>::Program {name: a,..},
             ProgramUnit::<E>::Program {name: b,..})
                | (ProgramUnit::<E>::Program {name: a,..},
                   ProgramUnit::<E>::Function {name: b,..})
                | (ProgramUnit::<E>::Program {name: a,..},
                   ProgramUnit::<E>::Subroutine {name: b,..})
                | (ProgramUnit::<E>::Function {name: a,..},
                   ProgramUnit::<E>::Program {name: b,..})
                | (ProgramUnit::<E>::Function {name: a,..},
                   ProgramUnit::<E>::Function {name: b,..})
                | (ProgramUnit::<E>::Function {name: a,..},
                   ProgramUnit::<E>::Subroutine {name: b,..})
                | (ProgramUnit::<E>::Subroutine {name: a,..},
                   ProgramUnit::<E>::Program {name: b,..})
                | (ProgramUnit::<E>::Subroutine {name: a,..},
                   ProgramUnit::<E>::Function {name: b,..})
                | (ProgramUnit::<E>::Subroutine {name: a,..},
                   ProgramUnit::<E>::Subroutine {name: b,..}) => a == b,
            _ => false,
        }
    }
}

#[derive(PartialEq, Debug)]
pub struct Program<E: std::cmp::PartialEq> {
    pub program: ProgramUnit<E>,
    pub functions: Vec<ProgramUnit<E>>,
    pub subroutines: Vec<ProgramUnit<E>>,
}

impl<E: std::cmp::PartialEq> Program<E> {
    pub fn new() -> Self {
        Self {
            program: ProgramUnit::<E>::Empty,
            functions: Vec::<ProgramUnit<E>>::with_capacity(8),
            subroutines: Vec::<ProgramUnit<E>>::with_capacity(8),
        }
    }

    /*
    1. TODO (F90+): if you want to allow function overloading, then you need to
    reconsider this bit of code. The FORTRAN 77 Standard didn't even
    consider this possibility, because that was ahead of its time. But
    I think Fortran 90+ supports function overloading, and you'd need to
    change this logic to reflect that possibility.

    2. TODO (performance): this is the simplest/worst implementation, but I
    want to get this done quickly. We can profile the code and see if
    hashmaps are better (they usually aren't in Rust).
     */
    pub fn has_unit_sharing_name(&self, unit: &ProgramUnit<E>) -> bool {
        if self.program.shares_name(unit) {
            return true;
        }
        for f in &self.functions {
            if f.shares_name(unit) {
                return true;
            }
        }
        for sub in &self.subroutines {
            if sub.shares_name(unit) {
                return true;
            }
        }
        return false;
    }

    /**
    Given a name, return the `FunctionIndex` or `SubroutineIndex` for
    the name (if there's a matching function or subroutine). When
    there's no match, `None` is returned.
     */
    pub fn index_for(self, name: String) -> Option<Expr> {
        for (idx, f) in self.functions.iter().enumerate() {
            if f.is_named(&name) {
                return Some(Expr::FunctionIndex(idx));
            }
        }
        for (idx, sub) in self.subroutines.iter().enumerate() {
            if sub.is_named(&name) {
                return Some(Expr::SubroutineIndex(idx));
            }
        }
        return None;
    }

    /**
    Adds a new program unit to the representation of the
    program. If the new program unit is `ProgramUnit<E>::Empty`, then
    nothing is done.

    Mutates the `Program` structure.

    # Panics
    
    1. Panics if there are two program units with the same name.

    2. Panics if you try to add a `ProgramUnit<E>::Program` when one has
    already been pushed.

    # Returns
    
    Returns nothing.
     */
    pub fn push(&mut self, unit: ProgramUnit<E>) {
        if self.has_unit_sharing_name(&unit) {
            // panic if there are two units sharing the same name
            panic!("Two program units share name '{}'",
                   unit.get_name());
        }
        // let kind = unit.kind();
        match unit.kind() {
            ProgramUnitKind::Program => {
                if ProgramUnit::<E>::Empty != self.program {
                    // panic
                    panic!("Trying to have two main programs");
                } else {
                    self.program = unit;
                }
            },
            ProgramUnitKind::Function => {
                self.functions.push(unit);
            },
            ProgramUnitKind::Subroutine => {
                self.subroutines.push(unit);
            },
            ProgramUnitKind::Empty => {},
        }
    }
}
