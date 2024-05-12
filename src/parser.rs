use crate::lexer::{
    BaseType,
    TokenType,
    Token,
    Lexer
};

/*
Current plans:
1. Statements
  - Simple statements (assignment?)
  - If-then-else statements
  - Labels and goto statements
  - do-loops
  - We should also note that the 77 Standard says, "a statement must
    contain no more than 1320 characters." (3.3)
2. Program Units

I found it useful to write the grammar rules as comments before each
function. This is a simple recursive descent parser, so production rules
correspond to function names.

Also, it may be useful to refactor out an `info` struct in the lexer to
store the line and column numbers (and file name? and lexeme?). This
would be useful to include when generating assembly code.
 */

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

fn token_to_binop(token: Token) -> BinOp {
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

fn token_to_unary_op(token: Token) -> UnOp {
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

/*
The EBNF for statements:

statement = [label] "continue"
          | [label] "goto" label
*/
#[derive(PartialEq, Debug)]
pub enum StatementType {
    Continue,
    Goto(i32),
    Write(Vec<Expr>),
    Read(Vec<Expr>),
    Illegal // should never be reached
}

// 6 digit label <= (10^6) - 1 < 20^19
#[derive(PartialEq, Debug)]
pub struct Statement {
    label: Option<i32>,
    sort: StatementType,
}

pub struct Parser {
    scanner: Lexer,
    current: Option<Token>,
    continuation_count: u8,
}

/* The Fortran 77 Standard explicitly states:

>          3.2.3  Continuation_Line.  A continuation line  is  any
>          line   that  contains  any  character  of  the  FORTRAN
>          character set other than the  character  blank  or  the
>          digit  0 in column 6 and contains only blank characters
>          in columns 1 through 5.  A statement must not have more
>          than nineteen continuation lines.

The Fortran 90 Standard also explicitly states: "A fixed form statement
must not have more than 19 continuation lines." (3.3.2.4)

Free-form F90 extends this to 39 lines (see section 3.3.1.4 of the
Fortran 90 Standard).
*/
pub const MAX_CONTINUATIONS : u8 = 19;

impl Parser {
    pub fn new(scanner: Lexer) -> Self {
        Self {
            scanner: scanner,
            current: None,
            continuation_count: 0,
        }
    }

    // to be invoked in `statement()`
    fn reset_continuation_count(&mut self) {
        self.continuation_count = 0;
    }

    fn inc_continuation_count(&mut self) {
        self.continuation_count += 1;
        if self.continuation_count > MAX_CONTINUATIONS {
            // I am too nice to throw an error here, a warning suffices.
            eprintln!("Warning: found {} continuations as of line {}",
                      self.continuation_count,
                      self.scanner.line_number());
        }
    }

    fn is_finished(&mut self) -> bool {
        return None == self.current && self.scanner.is_finished();
    }

    /*
    Skips continuation tokens, while tracking how many we have seen
    parsing the current statement.

    This is THE source of tokens in the parser, DO NOT access the
    scanner's next token directly.
    
    Requires: nothing.
    Assigns: updates the scanner.
    Ensures: result is not a continuation.
    */
    fn next_token(&mut self) -> Token {
        let mut t = self.scanner.next_token();
        while t.token_type.is_continuation() {
            self.inc_continuation_count();
            t = self.scanner.next_token();
        }
        return t;
    }

    fn populate_current(&mut self) -> () {
        if None == self.current && !self.is_finished() {
            self.current = Some(self.next_token());
        }
    }

    fn peek(&mut self) -> &Option<Token> {
        self.populate_current();
        return &self.current;
    }

    fn advance(&mut self) -> Token {
        if let Some(v) = self.current.take() {
            assert!(self.current == None);
            return v;
        } else {
            assert!(self.current == None);
            return self.next_token();
        }
    }

    // DOES NOT ADVANCE the scanner
    fn matches(&mut self, types: &[TokenType]) -> bool {
        for token_type in types.iter() {
            match self.peek() {
                None => return false,
                Some(v) => if v.token_type == *token_type {
                    return true;
                },
            }
        }
        return false;
    }

    fn check(&mut self, token_type: TokenType) -> bool {
        match self.peek() {
            Some(v) => return v.token_type == token_type,
            None => return false,
        }
    }

    fn consume(&mut self, expected: TokenType, msg: &str) {
        if self.check(expected) {
            self.advance();
            return;
        } else {
            panic!("{}", msg);
        }
    }

    /* *****************************************************************
    Statements
     ********************************************************** */

    fn io_list(&mut self) -> Vec<Expr> {
        let mut args = Vec::<Expr>::with_capacity(32);
        args.push(self.expr());
        while self.matches(&[TokenType::Comma]) {
            self.advance(); // consume the comma
            args.push(self.expr());
        }
        args.shrink_to_fit();
        return args;
    }
    
    /*
    read-statement = "read (*,*)" io-list
    */
    fn read(&mut self, label: Option<i32>) -> Statement {
        assert!(TokenType::Read == self.advance().token_type);
        
        self.consume(TokenType::LeftParen,
                     "Expected left paren in READ statement's UNIT");
        self.consume(TokenType::Star,
                     "Expected dummy arg in READ statement's UNIT");
        self.consume(TokenType::Comma,
                     "Expected comma in READ statement's UNIT");
        self.consume(TokenType::Star,
                     "Expected dummy arg in READ statement's UNIT");
        self.consume(TokenType::RightParen,
                     "Expected right paren in READ statement's UNIT");
        return Statement {
            label: label,
            sort: StatementType::Read(self.io_list()),
        };
    }
    
    /*
    write-statement = "write (*,*)" io-list
    */
    fn write(&mut self, label: Option<i32>) -> Statement {
        assert!(TokenType::Write == self.advance().token_type);
        self.consume(TokenType::LeftParen,
                     "Expected left paren in WRITE statement's UNIT");
        self.consume(TokenType::Star,
                     "Expected dummy arg in WRITE statement's UNIT");
        self.consume(TokenType::Comma,
                     "Expected comma in WRITE statement's UNIT");
        self.consume(TokenType::Star,
                     "Expected dummy arg in WRITE statement's UNIT");
        self.consume(TokenType::RightParen,
                     "Expected right paren in WRITE statement's UNIT");
        return Statement {
            label: label,
            sort: StatementType::Write(self.io_list()),
        };
    }
    
    fn goto_statement(&mut self, label: Option<i32>) -> Statement {
        assert!(TokenType::Goto == self.advance().token_type);
        let target: i32;
        let t = self.next_token();
        match t.token_type {
            TokenType::Integer(v) => {
                target = v.iter().collect::<String>().parse::<i32>().unwrap();
            },
            other => {
                panic!("Line {} GOTO expected a label, found {}",
                       t.line,
                       other);
            },
        }
        return Statement {
            label: label,
            sort: StatementType::Goto(target),
        };
    }

    fn continue_statement(&mut self, label: Option<i32>) -> Statement {
        if let Some(token) = self.peek() {
            assert!(TokenType::Continue == token.token_type);
            self.advance();
            return Statement {
                label: label,
                sort: StatementType::Continue,
            };
        } else {
            panic!("parser: continue statement has the current token is NONE?");
        }
    }
    
    /*
    requires: start of statement
    assigns: current if it is a label token
    ensures: self.peek() is not a label
     */
    fn statement_label(&mut self) -> Option<i32> {
        if let Some(token) = self.peek() {
            match &token.token_type {
                TokenType::Label(v) => {
                    let label = Some(v.iter().collect::<String>().parse::<i32>().unwrap());
                    self.advance();
                    return label;
                },
                _ => return None,
            };
        } else {
            return None;
        }
    }

    fn illegal_statement(&mut self, label: Option<i32>) -> Statement {
        return Statement {
            label: label,
            sort: StatementType::Illegal,
        };
    }
    
    pub fn statement(&mut self) -> Statement {
        self.reset_continuation_count();
        let label: Option<i32> = self.statement_label();
        // assert!(!self.peek().token_type.is_label());
        if let Some(token) = self.peek() {
            match token.token_type {
                TokenType::Goto => return self.goto_statement(label),
                TokenType::Continue => return self.continue_statement(label),
                TokenType::Write => return self.write(label),
                TokenType::Read => return self.read(label),
                _ => {
                    eprintln!("Parser::statement() illegal statement starting line {} with Token: #{:?}",
                              self.scanner.line_number(),
                              self.peek());
                    self.advance();
                    return self.illegal_statement(label);
                }
            }
        } else {
            return self.illegal_statement(None);
        }
    }
    
    /* *****************************************************************
    Expressions
     ********************************************************** */

    pub fn expr(&mut self) -> Expr {
        let e = self.level_5_expr();
        return e;
    }

    /*
    level_5_expr ::= equiv_operand (equiv_op equiv_operand)*
    equiv_op ::= ".eqv." | ".neqv."
     */
    fn level_5_expr(&mut self) -> Expr {
        let mut e = self.equiv_operand();
        while self.matches(&[TokenType::Equiv, TokenType::NotEquiv]) {
            let conjoin = token_to_binop(self.advance());
            let rhs = self.equiv_operand();
            e = Expr::Binary(Box::new(e), conjoin, Box::new(rhs));
        }
        return e;
    }

    /*
    equiv_operand ::= or_operand (".or." or_operand)*
     */
    fn equiv_operand(&mut self) -> Expr {
        let mut e = self.or_operand();
        while self.matches(&[TokenType::Or]) {
            let conjoin = token_to_binop(self.advance());
            let rhs = self.or_operand();
            e = Expr::Binary(Box::new(e), conjoin, Box::new(rhs));
        }
        return e;
    }

    /*
    or_operand ::= and_operand (".and." and_operand)*
     */
    fn or_operand(&mut self) -> Expr {
        let mut e = self.and_operand();
        while self.matches(&[TokenType::And]) {
            let conjoin = token_to_binop(self.advance());
            let rhs = self.and_operand();
            e = Expr::Binary(Box::new(e), conjoin, Box::new(rhs));
        }
        return e;
    }

    /*
    and_operand ::= [".not."] level_4_expr
     */
    fn and_operand(&mut self) -> Expr {
        if self.matches(&[TokenType::Not]) {
            let rator = token_to_unary_op(self.advance());
            let rand = self.level_4_expr();
            return Expr::Unary(rator, Box::new(rand));
        } else {
            return self.level_4_expr();
        }
    }

    /*
    level_4_expr ::= level_3_expr (rel_op level_3_expr)*
    rel_op ::= ".eq." | ".neq." | ".le."
            |  ".lt." | ".ge." | ".gt."
     */
    fn level_4_expr(&mut self) -> Expr {
        let mut e = self.level_3_expr();
        while self.matches(&[TokenType::Eq, TokenType::NotEqual, TokenType::Leq,
                             TokenType::Less, TokenType::Greater,
                             TokenType::Geq]) {
            let operator = token_to_binop(self.advance());
            let rhs = self.level_3_expr();
            e = Expr::Binary(Box::new(e), operator, Box::new(rhs));
        }
        return e;
    }
    
    /*
    level_3_expr ::= level_2_expr
                  | level_3_expr "//' level_2_expr
     */
    fn level_3_expr(&mut self) -> Expr {
        let mut e = self.level_2_expr();
        while self.matches(&[TokenType::Concatenation]) {
            let operator = token_to_binop(self.advance());
            let rhs = self.level_2_expr();
            e = Expr::Binary(Box::new(e), operator, Box::new(rhs));
        }
        return e;
    }

    // XXX: this is dangerous for a large number of exponentiations
    /*
    mult_operand ::= level_1_expr
                  | level_1_expr "**" mult_operand
    */
    fn mult_operand(&mut self) -> Expr {
        let b = self.level_1_expr();
        if self.check(TokenType::Pow) {
            self.advance();
            let e = self.mult_operand();
            self.current = None;
            return Expr::Binary(Box::new(b), BinOp::Power, Box::new(e));
        }
        return b;
    }

    /*
    add_operand ::= mult_operand
                 |  mult_operand mult_op add_operand;
    mult_op ::= '*' | '/'
    */
    fn add_operand(&mut self) -> Expr {
        let mut e = self.mult_operand();
        while self.matches(&[TokenType::Star, TokenType::Slash]) {
            let binop = token_to_binop(self.advance());
            let rhs = self.mult_operand();
            e = Expr::Binary(Box::new(e), binop, Box::new(rhs));
        }
        return e;
    }

    /*
    level_2_expr ::= [sign] add_operand
                  |  [sign] add_operand add_op level_2_expr;
    add_op ::= '+' | '-'
    */
    fn level_2_expr(&mut self) -> Expr {
        let mut e;
        if self.matches(&[TokenType::Plus, TokenType::Minus]) {
            let sign = token_to_unary_op(self.advance());
            let operand = self.add_operand();
            e = Expr::Unary(sign, Box::new(operand));
        } else {
            e = self.add_operand();
        }
        while self.matches(&[TokenType::Plus, TokenType::Minus]) {
            let operator = token_to_binop(self.advance());
            let rhs = self.add_operand();
            e = Expr::Binary(Box::new(e), operator, Box::new(rhs));
        }
        return e;
    }
    
    /*
    level_1_expr ::= primary
     */
    fn level_1_expr(&mut self) -> Expr {
        let e = self.primary();
        return e;
    }

    /*
    subscript ::= int_scalar_expr
     */
    fn subscript(&mut self) -> Expr {
        // TODO: check that it's really an integer scalar quantity
        return self.expr();
    }
    /*
    section_triplet_tail = ":" [expr] [":" expr]
    --- equivalently ---
    section_triplet_tail = ":"
                         | ":" expr
                         | ":" ":" expr
                         | ":" expr ":" expr
     */
    fn section_triplet_tail(&mut self) -> Option<(Option<Box<Expr>>,Option<Box<Expr>>)> {
        if !self.matches(&[TokenType::Colon]) {
            return None;
        }
        self.advance(); // eat the colon
        let stop;
        let stride;
        if self.matches(&[TokenType::Comma,
                          TokenType::RightParen]) {
            // case 1: matches `":"`
            stop = None;
            stride = None;
        } else if self.matches(&[TokenType::Colon]) {
            // case 3: matches `":" ":" expr`
            self.advance();
            stop = None;
            stride = Some(Box::new(self.subscript()));
        } else {
            // case 2 or 4: matches `expr [":" expr]`
            stop = Some(Box::new(self.subscript()));
            
            // [":" expr]
            if self.matches(&[TokenType::Colon]) {
                self.advance();
                stride = Some(Box::new(self.subscript()));
            } else {
                stride = None;
            }
        }
        return Some((stop, stride));
    }
    
    /*
    section_subscript ::= subscript
                       |  [subscript] section_triplet_tail
     */
    fn section_subscript(&mut self) -> Expr {
        // if [subscript] is omitted
        if self.matches(&[TokenType::Colon]) {
            if let Some((stop, stride)) = self.section_triplet_tail() {
                // ":stop[:stride]" matched
                return Expr::Section((None, stop, stride));
            } else {
                // ":" matched
                return Expr::Section((None, None, None));
            }
        } else { // section_subscript = subscript + stuff
            let e = self.subscript();
            if let Some((stop, stride)) = self.section_triplet_tail() {
                // "e:stop[:stride]" matched
                return Expr::Section((Some(Box::new(e)), stop, stride));
            } else {
                // subscript matched
                return e;
            }
        }
    }

    fn array_section_or_fn_call(&mut self, identifier: String) -> Expr {
        let mut args = Vec::<Expr>::with_capacity(64);
        let mut is_array_section = false;
        /* parses section_subscript {"," section_subscript} */
        loop {
            let e = self.section_subscript();
            if !is_array_section {
                match e {
                    Expr::Section(_) => is_array_section = true,
                    _ => {}
                }
            }
            args.push(e);
            if self.matches(&[TokenType::Comma]) {
                self.advance();
            } else {
                break;
            }
        }
        args.shrink_to_fit();
        if is_array_section {
            return Expr::ArraySection(identifier, args);
        } else {
            return Expr::NamedDataRef(identifier, args);
        }
        // it's an array, or a function reference, or an array slice
        // we do not know until we get more information
    }
    
    /*
named_data_ref = identifier
               | function_call
               | identifier "(" section_subscript {"," section_subscript} ")"
     */
    fn named_data_ref(&mut self, identifier: Token) -> Expr {
        if let TokenType::Identifier(v) = identifier.token_type {
            // CASE 1: an identifier
            if !self.matches(&[TokenType::LeftParen]) {
                /* Note: into_iter moves the characters from the token
                   into the expression */
                return Expr::Variable(v.into_iter().collect());
            }
            self.consume(TokenType::LeftParen, "Expected '(' in array access or function call");
            // CASE 2: a function call
            if self.matches(&[TokenType::RightParen]) {
                self.advance();
                return Expr::FunCall(v.into_iter().collect(),
                                     Vec::new());
            } else {
                // CASE 3: array section, array element, or function
                // call with arguments
                return self.array_section_or_fn_call(v.into_iter().collect());
            }
        } else {
            panic!("This should never be reached! Expected an identifier, received {}", identifier);
        }
        return Expr::ErrorExpr;
    }

    /*
    primary ::= int_constant
             |  real_constant
             |  string_constant
             |  named_data_ref
             |  "(" expr ")"
     */
    fn primary(&mut self) -> Expr {
        let token = self.advance();
        match token.token_type {
            TokenType::Integer(i) => {
                return Expr::Int64(i.iter().collect::<String>().parse::<i64>().unwrap());
            },
            TokenType::Float(f) => {
                return Expr::Float64(f.iter().collect::<String>().parse::<f64>().unwrap());
            },
            TokenType::String(s) => {
                return Expr::Character(s.to_vec());
            },
            TokenType::Identifier(_) => {
                return self.named_data_ref(token);
            },
            TokenType::LeftParen => {
                let e = self.expr();
                self.consume(TokenType::RightParen,
                             "Expected ')' after expression.");
                return Expr::Grouping(Box::new(e));
            },
            _ => {
                return Expr::ErrorExpr;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod stmt {
        use super::*;
        #[macro_export]
        macro_rules! should_parse_stmt {
            ( $text:expr, $expected:expr ) => {
                let l = Lexer::new($text.chars().collect());
                let mut parser = Parser::new(l);
                let actual = parser.statement();
                assert_eq!($expected, actual);
            }
        }

        #[test]
        fn labeled_write_one_variable() {
            let mut args = Vec::<Expr>::new();
            args.push(Expr::Variable(String::from("X")));
            args.shrink_to_fit();
            let expected = Statement {
                label: Some(10 as i32),
                sort: StatementType::Write(args),
            };
            should_parse_stmt!(" 10   WRITE (*,*) X",
                               expected);
        }

        #[test]
        fn labeled_write_three_variable() {
            let mut args = Vec::<Expr>::new();
            args.push(Expr::Variable(String::from("X")));
            args.push(Expr::Variable(String::from("Y")));
            args.push(Expr::Variable(String::from("Z")));
            args.shrink_to_fit();
            let expected = Statement {
                label: Some(10 as i32),
                sort: StatementType::Write(args),
            };
            should_parse_stmt!(" 10   WRITE (*,*) X, Y,Z",
                               expected);
        }
        
        #[test]
        fn labeled_read_one_variable() {
            let mut args = Vec::<Expr>::new();
            args.push(Expr::Variable(String::from("X")));
            args.shrink_to_fit();
            let expected = Statement {
                label: Some(10 as i32),
                sort: StatementType::Read(args),
            };
            should_parse_stmt!(" 10   READ (*,*) X",
                               expected);
        }

        #[test]
        fn labeled_read_three_variable() {
            let mut args = Vec::<Expr>::new();
            args.push(Expr::Variable(String::from("X")));
            args.push(Expr::Variable(String::from("Y")));
            args.push(Expr::Variable(String::from("Z")));
            args.shrink_to_fit();
            let expected = Statement {
                label: Some(10 as i32),
                sort: StatementType::Read(args),
            };
            should_parse_stmt!(" 10   READ (*,*) X, Y,Z",
                               expected);
        }

        #[test]
        fn labeled_continue() {
            let expected = Statement {
                label: Some(10 as i32),
                sort: StatementType::Continue
            };
            should_parse_stmt!(" 10   continue",
                               expected);
        }

        #[test]
        fn unlabeled_continue() {
            let expected = Statement {
                label: None,
                sort: StatementType::Continue
            };
            should_parse_stmt!("      continue",
                               expected);
        }

        #[test]
        fn labeled_goto() {
            let expected = Statement {
                label: Some(100 as i32),
                sort: StatementType::Goto(325)
            };
            should_parse_stmt!(" 100  goto 325",
                               expected);
        }

        #[test]
        fn unlabeled_goto() {
            let expected = Statement {
                label: None,
                sort: StatementType::Goto(321)
            };
            should_parse_stmt!("      goto 321",
                               expected);
        }

    }
    
    mod expr {
        use super::*;
        
        #[macro_export]
        macro_rules! should_parse_expr {
            ( $text:expr, $expected:expr ) => {
                let prefix = "       ";
                let input = [prefix, $text].concat();
                let l = Lexer::new(input.chars().collect());
                let mut parser = Parser::new(l);
                let actual = parser.expr();
                assert_eq!($expected, actual);
            }
        }

        #[test]
        fn parse_array_section() {
            let start = Expr::Int64(1);
            let stop = Expr::Int64(3);
            let sec = Expr::Section((Some(Box::new(start)),
                                     Some(Box::new(stop)),
                                     None));
            let mut args = Vec::<Expr>::new();
            args.push(sec);
            args.shrink_to_fit();
            should_parse_expr!("A(1:3)",
                               Expr::ArraySection(String::from("A"),
                                                  args));
        }

        #[test]
        fn parse_array_section_without_start() {
            let stop = Expr::Int64(3);
            let sec = Expr::Section((None,
                                     Some(Box::new(stop)),
                                     None));
            let mut args = Vec::<Expr>::new();
            args.push(sec);
            args.shrink_to_fit();
            should_parse_expr!("A(:3)",
                               Expr::ArraySection(String::from("A"),
                                                  args));
        }

        #[test]
        fn parse_array_section_without_stop() {
            let start = Expr::Int64(3);
            let sec = Expr::Section((Some(Box::new(start)),
                                     None,
                                     None));
            let mut args = Vec::<Expr>::new();
            args.push(sec);
            args.shrink_to_fit();
            should_parse_expr!("A(3:)",
                               Expr::ArraySection(String::from("A"),
                                                  args));
        }

        #[test]
        fn f90_array_section_ex3() {
            let mut args = Vec::<Expr>::new();
            args.push(Expr::Section((None,None,None)));
            args.push(Expr::Section((None,None,None)));
            args.push(Expr::Section((None,None,None)));
            args.shrink_to_fit();
            should_parse_expr!("B(:,:,:)",
                               Expr::ArraySection(String::from("B"),
                                                  args));
        }
        
        #[test]
        fn parse_array_section_with_only_start_and_stride() {
            let start = Expr::Int64(7);
            let stride = Expr::Int64(4);
            let sec = Expr::Section((Some(Box::new(start)),
                                     None,
                                     Some(Box::new(stride))));
            let mut args = Vec::<Expr>::new();
            args.push(sec);
            args.shrink_to_fit();
            should_parse_expr!("A(7::4)",
                               Expr::ArraySection(String::from("A"),
                                                  args));
        }
        
        #[test]
        fn parse_array_section_with_start_and_stop_and_stride() {
            let start = Expr::Int64(7);
            let stop = Expr::Int64(36);
            let stride = Expr::Int64(4);
            let sec = Expr::Section((Some(Box::new(start)),
                                     Some(Box::new(stop)),
                                     Some(Box::new(stride))));
            let mut args = Vec::<Expr>::new();
            args.push(sec);
            args.shrink_to_fit();
            should_parse_expr!("A(7:36:4)",
                               Expr::ArraySection(String::from("A"),
                                                  args));
        }
        
        #[test]
        fn parse_array_section_with_only_stop_and_stride() {
            let stop = Expr::Int64(36);
            let stride = Expr::Int64(4);
            let sec = Expr::Section((None,
                                     Some(Box::new(stop)),
                                     Some(Box::new(stride))));
            let mut args = Vec::<Expr>::new();
            args.push(sec);
            args.shrink_to_fit();
            should_parse_expr!("A(:36:4)",
                               Expr::ArraySection(String::from("A"),
                                                  args));
        }
        
        #[test]
        fn parse_array_section_with_only_stride() {
            let stride = Expr::Int64(4);
            let sec = Expr::Section((None,
                                     None,
                                     Some(Box::new(stride))));
            let mut args = Vec::<Expr>::new();
            args.push(sec);
            args.shrink_to_fit();
            should_parse_expr!("A(::4)",
                               Expr::ArraySection(String::from("A"),
                                                  args));
        }
        
        #[test]
        fn parse_fn_with_args() {
            let arg1 = Expr::Int64(1);
            let arg2 = Expr::Int64(3);
            let mut args = Vec::<Expr>::new();
            args.push(arg1);
            args.push(arg2);
            args.shrink_to_fit();
            should_parse_expr!("AREA(1, 3)",
                               Expr::NamedDataRef(String::from("AREA"),
                                                  args));
        }

        #[test]
        fn parse_simple_fn_call() {
            should_parse_expr!("f()",
                               Expr::FunCall(String::from("f"),
                                             Vec::new()));
        }
        
        #[test]
        fn parse_polysyllabic_variable() {
            should_parse_expr!("humidity",
                               Expr::Variable(String::from("humidity")));
        }

        #[test]
        fn parse_variable() {
            should_parse_expr!("x", Expr::Variable(String::from("x")));
        }

        #[test]
        fn parse_variable_with_numeric_suffix() {
            should_parse_expr!("alpha13",
                               Expr::Variable(String::from("alpha13")));
        }

        #[test]
        fn parse_int_without_sign() {
            let val: i64 = 51;
            should_parse_expr!("51", Expr::Int64(val));
        }

        #[test]
        fn parse_positive_int() {
            let val: i64 = 51;
            should_parse_expr!("+51", Expr::Unary(UnOp::Plus, Box::new(Expr::Int64(val))));
        }

        #[test]
        fn parse_negative_int() {
            let val: i64 = 51;
            should_parse_expr!("-51", Expr::Unary(UnOp::Minus, Box::new(Expr::Int64(val))));
        }

        #[test]
        fn parse_int_pow_int() {
            let x: i64 = 3;
            let y: i64 = 4;
            should_parse_expr!("3 ** 4",
                               Expr::Binary(Box::new(Expr::Int64(x)),
                                            BinOp::Power,
                                            Box::new(Expr::Int64(y))));
        }

        #[test]
        fn parse_level_2_expr_subtraction() {
            let x: i64 = 3;
            let y: i64 = 4;
            should_parse_expr!("3 - 4",
                               Expr::Binary(Box::new(Expr::Int64(x)),
                                            BinOp::Minus,
                                            Box::new(Expr::Int64(y))));
        }

        #[test]
        fn parse_level_2_expr_add() {
            let x: i64 = 3;
            let y: i64 = 4;
            should_parse_expr!("3 + 4",
                               Expr::Binary(Box::new(Expr::Int64(x)),
                                            BinOp::Plus,
                                            Box::new(Expr::Int64(y))));
        }

        #[test]
        fn parse_iterated_powers() {
            let x: i64 = 3;
            let y: i64 = 4;
            let z: i64 = 5;
            should_parse_expr!("3 ** 4 ** 5",
                               Expr::Binary(Box::new(Expr::Int64(x)),
                                            BinOp::Power,
                                            Box::new(
                                                Expr::Binary(Box::new(Expr::Int64(y)),
                                                             BinOp::Power,
                                                             Box::new(Expr::Int64(z))))));
        }

        #[test]
        fn subtraction_is_left_associative() {
            let x: i64 = 3;
            let y: i64 = 4;
            let z: i64 = 5;
            should_parse_expr!("3 - 4 - 5",
                               Expr::Binary(Box::new(
                                                Expr::Binary(Box::new(Expr::Int64(x)),
                                                             BinOp::Minus,
                                                             Box::new(Expr::Int64(y)))),
                                            BinOp::Minus,
                                            Box::new(Expr::Int64(z))));
        }

        #[test]
        fn division_is_left_associative() {
            let x: i64 = 3;
            let y: i64 = 4;
            let z: i64 = 5;
            should_parse_expr!("3 / 4 / 5",
                               Expr::Binary(Box::new(
                                                Expr::Binary(Box::new(Expr::Int64(x)),
                                                             BinOp::Divide,
                                                             Box::new(Expr::Int64(y)))),
                                            BinOp::Divide,
                                            Box::new(Expr::Int64(z))));
        }

    }

}
