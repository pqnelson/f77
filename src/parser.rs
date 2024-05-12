use crate::lexer::{
    BaseType,
    TokenType,
    Token,
    Lexer
};

/*
Current plans:
1. "Named data references" expression
2. Fix line number problems
  - When advancing through the tokens, we should skip over continuation
    tokens when parsing an expression or a statement. 
  - If we want to adhere to the "letter of the law", we should track how
    many lines the current statement has, since the 77 Standard says that
    a statement consists "of an initial line and as many as nineteen
    continuation lines." This could be flagged with a warning, or we
    could enter panic mode.
  - The Fortran 90 Standard explicitly says:
    "A fixed form statement must not have more than 19 continuation lines."
    (3.3.2.4) The free-form Fortran 90 is more generous, a free form
    statement must not have more than 39 continuation lines (3.3.1.4).
    Again, how to handle this? Issue a warning? Panic? 
3. Statements
  - Simple statements (assignment?)
  - If-then-else statements
  - Labels and goto statements
  - do-loops
  - We should also note that the 77 Standard says, "a statement must
    contain no more than 1320 characters." (3.3)
4. Program Units

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

pub struct Parser {
    scanner: Lexer,
    current: Option<Token>
}

impl Parser {
    pub fn new(scanner: Lexer) -> Self {
        Self {
            scanner: scanner,
            current: None
        }
    }

    fn is_finished(&mut self) -> bool {
        return None == self.current && self.scanner.is_finished();
    }

    fn populate_current(&mut self) -> () {
        if None == self.current && !self.is_finished() {
            self.current = Some(self.scanner.next_token());
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
            return self.scanner.next_token();
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
            // matches `":"`
            stop = None;
            stride = None;
        } else if self.matches(&[TokenType::Colon]) {
            // matches `":" ":" expr`
            self.advance();
            stop = None;
            stride = Some(Box::new(self.subscript()));
        } else {
            // matches `expr [":" expr]`
            stop = Some(Box::new(self.subscript()));
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
    
    /*
named_data_ref = identifier
               | function_call
               | identifier "(" section_subscript {"," section_subscript} ")"
     */
    fn named_data_ref(&mut self, identifier: Token) -> Expr {
        if let TokenType::Identifier(v) = identifier.token_type {
            if !self.matches(&[TokenType::LeftParen]) {
                /* Note: into_iter moves the characters from the token
                   into the expression */
                return Expr::Variable(v.into_iter().collect());
            }
            self.consume(TokenType::LeftParen, "Expected '(' in array access or function call");
            if self.matches(&[TokenType::RightParen]) {
                self.advance();
                return Expr::FunCall(v.into_iter().collect(),
                                     Vec::new());
            }
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
                return Expr::ArraySection(v.into_iter().collect(), args);
            } else {
                return Expr::NamedDataRef(v.into_iter().collect(), args);
            }
            // it's an array, or a function reference, or an array slice
            // we do not know until we get more information
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
