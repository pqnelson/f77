use crate::lexer::{
    BaseType,
    TokenType,
    Token,
    Lexer
};

pub mod parse_tree {
    use super::*;
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
            TokenType::Star => return BinOp::Times,
            TokenType::Slash => return BinOp::Divide,
            TokenType::Plus => return BinOp::Plus,
            TokenType::Minus => return BinOp::Minus,
            TokenType::Concatenation => return BinOp::Concatenate,
            TokenType::Eq => return BinOp::Eq,
            TokenType::NotEqual => return BinOp::Ne,
            TokenType::Leq => return BinOp::Le,
            TokenType::Less => return BinOp::Lt,
            TokenType::Greater => return BinOp::Gt,
            TokenType::Geq => return BinOp::Ge,
            TokenType::And => return BinOp::And,
            TokenType::Or => return BinOp::Or,
            TokenType::Equiv => return BinOp::Eqv,
            TokenType::NotEquiv => return BinOp::Neqv,
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
            TokenType::Plus => return UnOp::Plus,
            TokenType::Minus => return UnOp::Minus,
            TokenType::Not => return UnOp::Not,
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
            match self.command {
                Command::Continue => return true,
                _ => return false,
            }
        }
    }
}
