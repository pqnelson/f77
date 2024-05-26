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

The grammar for arrays and function calls are represented by the
`NameDataRef` node.

```
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
    // placeholder, should never be forced to arrive here
    ErrorExpr,
}

#[derive(PartialEq, Debug)]
pub enum Command {
    Continue,
    Goto(i32),
    Write(Vec<Expr>),
    Read(Vec<Expr>),
    IfBlock {test: Expr,
             true_branch: Vec<Statement>,
             false_branch: Vec<Statement>},
    ArithIf {test: Expr,
             negative: i32,
             zero: i32,
             positive: i32},
    IfStatement {test: Expr,
                 true_branch: Box::<Statement>},
    LabelDo {target_label: i32,
             var: Expr,
             start: Expr,
             stop: Expr,
             stride: Option<Expr>,
             body: Vec<Statement>,
             terminal: Box<Statement>},
    CallSubroutine {
        subroutine: Expr, // identifier
        args: Vec<Expr>
    },
    ExprStatement(Expr),
    Assignment {
        lhs: Expr,
        rhs: Expr
    },
    Stop,
    Return,
    End,
    Illegal // should never be reached
}

// 5 digit label with values <= (10^5) - 1 <= 0b11000011010011111 < 2^17
#[derive(PartialEq, Debug)]
pub struct Statement {
    pub label: Option<i32>,
    pub command: Command,
}

impl Statement {
    pub fn is_continue(&mut self) -> bool {
        matches!(self.command, Command::Continue)
    }
    pub fn is_end(&mut self) -> bool {
        matches!(self.command, Command::End)
    }
    pub fn is_stop(&mut self) -> bool {
        matches!(self.command, Command::Stop)
    }
    pub fn is_return(&mut self) -> bool {
        matches!(self.command, Command::Return)
    }
}

#[derive(PartialEq, Debug)]
pub enum ProgramUnitKind {
    Program,
    Subroutine,
    Function,
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
                        | parameter; (* not yet *)

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
pub enum ProgramUnit {
    Program {
        name: String,
        spec: Vec<Specification>,
        body: Vec<Statement>,
    },
    Function {
        name: String,
        return_type: Type,
        params: Vec<String>,
        spec: Vec<Specification>,
        body: Vec<Statement>,
    },
    Subroutine {
        name: String,
        params: Vec<String>,
        spec: Vec<Specification>,
        body: Vec<Statement>,
    },
    Empty,
}

impl ProgramUnit {
    pub fn is_named(self, the_name: &String) -> bool {
        match self {
            ProgramUnit::Program {name, ..} => *the_name == name,
            ProgramUnit::Function {name, ..} => *the_name == name,
            ProgramUnit::Subroutine {name, ..} => *the_name == name,
            _ => false
        }
    }
}
