use crate::lexer::{
    BaseType,
    TokenType,
    Token,
    Lexer
};

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

#[derive(PartialEq, Debug)]
pub enum Expr {
    // literals
    Character(Vec<char>),
    Float32(f32),
    Float64(f64),
    Int32(i32),
    Int64(i64),
    Logical(bool),
    // composite expressions
    Binary(Box<Expr>, BinOp, Box<Expr>),
    Unary(UnOp, Box<Expr>),
    Grouping(Box<Expr>),
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
            return v;
        } else {
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
            None => return false,
            Some(v) => return v.token_type == token_type,
        }
    }

    fn consume(&mut self, expected: TokenType, msg: &str) {
    }

    pub fn expr(&mut self) -> Expr {
        return self.level_5_expr();
    }

    fn level_5_expr(&mut self) -> Expr {
        let mut e = self.equiv_operand();
        while self.matches(&[TokenType::Equiv, TokenType::NotEquiv]) {
            let conjoin = token_to_binop(self.advance());
            let rhs = self.equiv_operand();
            e = Expr::Binary(Box::new(e), conjoin, Box::new(rhs));
        }
        return e;
    }

    fn equiv_operand(&mut self) -> Expr {
        let mut e = self.or_operand();
        while self.matches(&[TokenType::Or]) {
            let conjoin = token_to_binop(self.advance());
            let rhs = self.or_operand();
            e = Expr::Binary(Box::new(e), conjoin, Box::new(rhs));
        }
        return e;
    }

    fn or_operand(&mut self) -> Expr {
        let mut e = self.and_operand();
        while self.matches(&[TokenType::And]) {
            let conjoin = token_to_binop(self.advance());
            let rhs = self.and_operand();
            e = Expr::Binary(Box::new(e), conjoin, Box::new(rhs));
        }
        return e;
    }

    fn and_operand(&mut self) -> Expr {
        if self.matches(&[TokenType::Not]) {
            let rator = token_to_unary_op(self.advance());
            let rand = self.level_4_expr();
            return Expr::Unary(rator, Box::new(rand));
        } else {
            return self.level_4_expr();
        }
    }

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
    fn mult_operand(&mut self) -> Expr {
        let b = self.level_1_expr();
        if self.check(TokenType::Pow) {
            self.advance();
            return Expr::Binary(Box::new(b), BinOp::Power, Box::new(self.mult_operand()));
        }
        return b;
    }

    fn add_operand(&mut self) -> Expr {
        let mut e = self.mult_operand();
        while self.matches(&[TokenType::Star, TokenType::Slash]) {
            let binop = token_to_binop(self.advance());
            let rhs = self.mult_operand();
            e = Expr::Binary(Box::new(e), binop, Box::new(rhs));
        }
        return e;
    }    

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

    fn level_1_expr(&mut self) -> Expr {
        return self.primary();
    }
    
    fn primary(&mut self) -> Expr {
        match self.peek() {
            Some(v) => match &v.token_type {
                TokenType::Integer(i) => {
                    // TODO: something more sophisticated here
                    return Expr::Int64(i.iter().collect::<String>().parse::<i64>().unwrap());
                },
                TokenType::Float(f) => {
                    // TODO: try to figure out if we are working with
                    // f32 or f128 or f256?
                    return Expr::Float64(f.iter().collect::<String>().parse::<f64>().unwrap());
                },
                TokenType::String(s) => {
                    return Expr::Character(s.to_vec());
                },
                TokenType::LeftParen => {
                    self.advance();
                    let e = self.expr();
                    self.consume(TokenType::RightParen,
                                 "Expected ')' after expression.");
                    return Expr::Grouping(Box::new(e));
                },
                _ => {
                    return Expr::ErrorExpr;
                }
            }
            None => return Expr::ErrorExpr,
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
    }

}
