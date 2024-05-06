#[derive(PartialEq, Debug)] 
pub enum BaseTypeToken {
    Integer, Real, Character, Logical
}

#[derive(PartialEq, Debug)] 
pub enum TokenType {
    // single-character tokens
    LeftParen, RightParen,
    Comma,
    Minus,
    Plus,
    Slash,
    Star,
    Equal,

    Continuation(char), // TODO
    
    // literals
    Identifier(Vec<char>),
    Integer(Vec<char>),
    Label(Vec<char>),
    Float(Vec<char>),
    // String(Vec<char>),

    Type(BaseTypeToken),
    
    // keywords
    Program,
    If, Then, Else, EndIf,
    Do,
    Continue,
    True, False,
    Stop, End, Function, Return, Subroutine,
    Less, Leq, Eq, NotEqual, Greater, Geq,
    Not, And, Or, Equiv, NotEquiv, Xor,
    Goto,

    // Provided primitive functions
    Write, Read,
    
    // The End
    Illegal, Eof
}

fn get_keyword_token(ident: &Vec<char>) -> Result<TokenType, String> {
    let identifier: String = ident.into_iter().collect();
    match &identifier.to_lowercase()[..] {
        "program" => Ok(TokenType::Program),
        "do" => Ok(TokenType::Do),
        "continue" => Ok(TokenType::Continue),
        "if" => Ok(TokenType::If),
        "then" => Ok(TokenType::Then),
        "else" => Ok(TokenType::Else),
        "endif" => Ok(TokenType::EndIf),
        "end" => Ok(TokenType::End),
        "function" => Ok(TokenType::Function),
        "return" => Ok(TokenType::Return),
        "subroutine" => Ok(TokenType::Subroutine),
        "stop" => Ok(TokenType::Stop),
        "goto" => Ok(TokenType::Goto),
        "write" => Ok(TokenType::Write),
        "read" => Ok(TokenType::Read),
        ".true." => Ok(TokenType::True),
        ".false." => Ok(TokenType::False),
        ".lt." => Ok(TokenType::Less),
        ".le." => Ok(TokenType::Leq),
        ".eq." => Ok(TokenType::Eq),
        ".ne." => Ok(TokenType::NotEqual),
        ".gt." => Ok(TokenType::Greater),
        ".geq." => Ok(TokenType::Geq),
        ".not." => Ok(TokenType::Not),
        ".and." => Ok(TokenType::And),
        ".or." => Ok(TokenType::Or),
        ".neqv." => Ok(TokenType::NotEquiv),
        ".eqv." => Ok(TokenType::Equiv),
        ".xor." => Ok(TokenType::Xor),
        _ => Err(String::from("Not a keyword"))
    }
}

pub struct Lexer {
    input: Vec<char>,           // Source code
    pub position: usize,        // Reading position
    pub read_position: usize,   // Current moving reading position
    pub ch: char,               // Current read character
    line: i64,                  // Line number
    offset: i64                 // Offset from the start of the line
}

fn is_letter(ch: char) -> bool {
    return 'a' <= ch && ch <= 'z' || 'A' <= ch && ch <= 'Z';
}

fn is_digit(ch: char) -> bool {
    return '0' <= ch && ch <= '9';
}

fn implies(antecedent: bool, consequent: bool) -> bool {
    return !antecedent || consequent;
}

fn is_id_start(c: char) -> bool {
    return c.is_alphabetic() || '.' == c;
}

fn is_identifier(c: char) -> bool {
    return c.is_alphanumeric();
}

impl Lexer {
    pub fn new(input : Vec<char>) -> Self {
        Self {
            input: input,
            position: 0,
            read_position: 0,
            ch: '\0',
            line: 1,
            offset: 1
        }
    }

    pub fn is_finished(&mut self) -> bool {
        return self.read_position >= self.input.len();
    }
    
    pub fn read_char(&mut self) {
        if self.is_finished() {
            self.position = self.read_position;
            self.ch = '\0';
        } else {
            self.ch = self.input[self.read_position];
            self.position = self.read_position;
            self.read_position += 1;
            if '\n' == self.ch {
                self.line += 1;
                self.offset = 1;
            } else {
                self.offset += 1;

                // TODO: if self.offset > 72, issue a warning?
            }
        }
    }

    fn skip_whitespace(&mut self) {
        while self.ch.is_whitespace() {
            self.read_char();
        }
    }

    // We should check if this is a 'C' or a '*' (or maybe '!')
    // but we are generous in accepting comments...
    fn is_comment(&mut self) -> bool {
        if 1 != self.offset { return false; }
        self.read_char();
        return !self.ch.is_whitespace();
    }

    // Try to parse the label for the line, if it is present; if not,
    // return None.
    fn try_label(&mut self) -> Option<TokenType> {
        if 2 <= self.offset && self.offset <= 5 {
            let mut l: Vec<char> = Vec::new();
            for _i in 2..5 {
                if !self.ch.is_whitespace() {
                    l.push(self.ch);
                } else if !l.is_empty() {
                    return Some(TokenType::Label(l));
                }
                self.read_char();
            }
        }
        return None;
    }

    fn is_continuation(&mut self) -> bool {
        return 6 == self.offset && !self.ch.is_whitespace();
    }

    fn skip_rest_of_line(&mut self) {
        let line_number = self.line;
        while line_number == self.line && !self.is_finished() {
            self.read_char();
        }
        self.read_char(); // move past the '\n' character
    }

    fn skip_comments(&mut self) {
        while self.is_comment() {
            self.skip_rest_of_line();
        }
    }

    fn peek(&mut self) -> char {
        if self.position >= self.input.len() {
            return '\0';
        }
        return self.input[self.position + 1];
    }

    // TODO: handle exponents, '1.234e7' as well as '1e5', 1e-5,
    // 1.234e-5, etc.
    fn lex_number(&mut self) -> TokenType {
        let position = self.position;
        while self.position < self.input.len() && is_digit(self.ch) {
            self.read_char();
        }
        if '.' == self.ch && is_digit(self.peek()) {
            self.read_char();
            while self.position < self.input.len() && is_digit(self.ch) {
                self.read_char();
            }
            return TokenType::Float(self.input[position..self.position].to_vec());
        } else {
            return TokenType::Integer(self.input[position..self.position].to_vec());
        }
    }
    
    pub fn next_token(&mut self) -> TokenType {
        // TODO: this should raise an error if the identifier is more
        // than six characters long, but I'm a generous soul, so...
        let read_identifier = |l: &mut Lexer| -> Vec<char> {
            assert!(is_id_start(l.ch));
            let position = l.position;
            if l.position < l.input.len() {
                l.read_char();
            }
            while l.position < l.input.len() && is_identifier(l.ch) {
                l.read_char();
            }
            if '.' == l.ch {
                l.read_char();
            }
            // starts with a dot, ends with a dot
            assert!(implies('.' == l.input[position], '.' == l.input[l.position-1]));
            l.input[position..l.position].to_vec()
        };

        // start of the method proper
        self.skip_comments();
        self.skip_whitespace();
        let tok: TokenType;

        if self.is_continuation() {
            tok = TokenType::Continuation(self.ch);
            self.read_char();
            return tok;
        }
        
        match self.try_label() {
            Some(v) => { return v; },
            None => {},
        }
        
        match self.ch {
            '=' => {
                tok = TokenType::Equal;
            },
            '+' => {
                tok = TokenType::Plus;
            },
            '(' => {
                tok = TokenType::LeftParen;
            },
            ')' => {
                tok = TokenType::RightParen;
            },
            ',' => {
                tok = TokenType::Comma;
            },
            '-' => {
                tok = TokenType::Minus;
            },
            '/' => {
                tok = TokenType::Slash; // todo
            },
            '*' => {
                tok = TokenType::Star; // todo
            },
            '\0' => {
                tok = TokenType::Eof;
            }
            _ => {
                if is_letter(self.ch) || '.' == self.ch {
                    let ident: Vec<char> = read_identifier(self);
                    match get_keyword_token(&ident) {
                        Ok(keyword_token) => {
                            return keyword_token;
                        },
                        Err(_err) => {
                            if '.' == ident[0] {
                                return TokenType::Illegal;
                            } else {
                                return TokenType::Identifier(ident);
                            }
                        }
                    }
                } else if is_digit(self.ch) {
                    return self.lex_number();
                } else {
                    return TokenType::Illegal;
                }
            }
        }
        self.read_char();
        tok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_lex_screaming_program_kw() {
        let input = String::from("      PROGRAM main\n      do stuff;\n      end");
        let mut l = Lexer::new(input.chars().collect());
        l.read_char();
        let token = l.next_token();
        assert_eq!(token, TokenType::Program);
    }

    #[test]
    fn should_lex_program_kw() {
        let input = String::from("      program main\n      do stuff;\n      end");
        let mut l = Lexer::new(input.chars().collect());
        l.read_char();
        let token = l.next_token();
        assert_eq!(token, TokenType::Program);
    }

    #[test]
    fn should_lex_program_name() {
        let input  = String::from("      program main\n  10  do stuff;\n      end");
        let mut l = Lexer::new(input.chars().collect());
        l.read_char();
        let token = l.next_token();
        assert_eq!(token, TokenType::Program);
        let token = l.next_token();
        assert_eq!(token, TokenType::Identifier("main".chars().collect()));
    }

    #[test]
    fn should_match_label() {
        let input  = String::from("      program main\n  10  do stuff;\n      end");
        let mut l = Lexer::new(input.chars().collect());
        l.read_char();
        let token = l.next_token();
        assert_eq!(token, TokenType::Program);
        let token = l.next_token();
        assert_eq!(token, TokenType::Identifier("main".chars().collect()));
        let token = l.next_token();
        assert_eq!(token, TokenType::Label("10".chars().collect()));
    }

    #[test]
    fn should_lex_do() {
        let input  = String::from("      program main\n  10  do stuff;\n      end");
        let mut l = Lexer::new(input.chars().collect());
        l.read_char();
        l.next_token(); // program
        l.next_token(); // main
        l.next_token(); // label
        let token = l.next_token();
        assert_eq!(token, TokenType::Do);
    }

    #[test]
    fn should_not_lex_invalid_kw() {
        let input  = String::from("      enddo program main\n  10  do stuff;\n      end");
        let mut l = Lexer::new(input.chars().collect());
        l.read_char();
        let token = l.next_token();
        assert_eq!(token, TokenType::Identifier("enddo".chars().collect()));
    }

    #[test]
    fn should_lex_if_kw() {
        let input = String::from("      program main\n      if (.true.) then\n       do stuff\n      end\n      stop\n      end");
        let mut l = Lexer::new(input.chars().collect());
        l.read_char();
        l.next_token(); // program
        l.next_token(); // main
        let token = l.next_token();
        assert_eq!(token, TokenType::If);
    }

    #[test]
    fn should_lex_lparen_kw() {
        let input = String::from("      program main\n      if (.true.) then\n       do stuff\n      end\n      stop\n      end");
        let mut l = Lexer::new(input.chars().collect());
        l.read_char();
        l.next_token(); // program
        l.next_token(); // main
        l.next_token(); // if
        let token = l.next_token();
        assert_eq!(token, TokenType::LeftParen);
    }

    #[test]
    fn should_lex_screaming_true_kw() {
        let input = String::from("      program main\n      if (.TRUE.) then\n       do stuff\n      end\n      stop\n      end");
        let mut l = Lexer::new(input.chars().collect());
        l.read_char();
        l.next_token(); // program
        l.next_token(); // main
        l.next_token(); // if
        l.next_token(); // (
        let token = l.next_token();
        assert_eq!(token, TokenType::True);
    }

    #[test]
    fn should_lex_true_kw() {
        let input = String::from("      program main\n      if (.true.) then\n       do stuff\n      end\n      stop\n      end");
        let mut l = Lexer::new(input.chars().collect());
        l.read_char();
        l.next_token(); // program
        l.next_token(); // main
        l.next_token(); // if
        l.next_token(); // (
        let token = l.next_token();
        assert_eq!(token, TokenType::True);
    }
    
    #[test]
    fn should_lex_dot_prefix_invalid() {
        let input = String::from("      program main\n      if (.breaks.) then\n       do stuff\n      end\n      stop\n      end");
        let mut l = Lexer::new(input.chars().collect());
        l.read_char();
        l.next_token(); // program
        l.next_token(); // main
        l.next_token(); // if
        l.next_token(); // (
        let token = l.next_token();
        assert_eq!(token, TokenType::Illegal);
    }

    #[test]
    #[should_panic]
    fn should_lex_dot_prefix_without_dot_suffix_panics() {
        let input = String::from("      program main\n      if (.breaks) then\n       do stuff\n      end\n      stop\n      end");
        let mut l = Lexer::new(input.chars().collect());
        l.read_char();
        l.next_token(); // program
        l.next_token(); // main
        l.next_token(); // if
        l.next_token(); // (
        l.next_token(); // .breaks
    }
    
    #[test]
    fn should_lex_then_kw() {
        let input = String::from("      program main\n      if (.true.) then\n       do stuff\n      endif\n      stop\n      end");
        let mut l = Lexer::new(input.chars().collect());
        l.read_char();
        l.next_token(); // program
        l.next_token(); // main
        l.next_token(); // if
        l.next_token(); // (
        l.next_token(); // .true.
        let token = l.next_token();
        assert_eq!(token, TokenType::RightParen);
        let token = l.next_token();
        assert_eq!(token, TokenType::Then);
    }
    
    #[test]
    fn should_lex_endif_kw() {
        let input = String::from("      program main\n      if (.true.) then\n       do stuff\n      endif\n      stop\n      end");
        let mut l = Lexer::new(input.chars().collect());
        l.read_char();
        l.next_token(); // program
        l.next_token(); // main
        l.next_token(); // if
        l.next_token(); // (
        l.next_token(); // .true.
        l.next_token(); // )
        l.next_token(); // then
        l.next_token(); // do
        l.next_token(); // stuff
        let token = l.next_token();
        assert_eq!(token, TokenType::EndIf);
    }
    
    #[test]
    fn should_lex_pi_as_float() {
        let input = String::from("      3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679");
        let mut l = Lexer::new(input.chars().collect());
        l.read_char();
        let token = l.next_token();
        assert_eq!(token, TokenType::Float("3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679".chars().collect()));
    }
    
    #[test]
    fn should_lex_grothendieck_prime() {
        let input = String::from("      51");
        let mut l = Lexer::new(input.chars().collect());
        l.read_char();
        let token = l.next_token();
        assert_eq!(token, TokenType::Integer("51".chars().collect()));
    }

    #[macro_export]
    macro_rules! should_lex {
        ( $text:expr, $expected:expr ) => {
            {
                let prefix = "       ";
                let input = [prefix, $text].concat();
                let mut l = Lexer::new(input.chars().collect());
                l.read_char();
                let token = l.next_token();
                assert_eq!(token, $expected);
            }
        };
    }

    #[test]
    fn lex_continue() {
        should_lex!("continue", TokenType::Continue);
    }

    #[test]
    fn lex_else() {
        should_lex!("else", TokenType::Else);
    }

    #[test]
    fn lex_endif() {
        should_lex!("endif", TokenType::EndIf);
    }

    #[test]
    fn lex_end() {
        should_lex!("end", TokenType::End);
    }

    #[test]
    fn lex_function() {
        should_lex!("function", TokenType::Function);
    }

    #[test]
    fn lex_return() {
        should_lex!("return", TokenType::Return);
    }

    #[test]
    fn lex_subroutine() {
        should_lex!("subroutine", TokenType::Subroutine);
    }

    #[test]
    fn lex_stop() {
        should_lex!("stop", TokenType::Stop);
    }

    #[test]
    fn lex_goto() {
        should_lex!("goto", TokenType::Goto);
    }

    #[test]
    fn lex_write() {
        should_lex!("write", TokenType::Write);
    }

    #[test]
    fn lex_read() {
        should_lex!("read", TokenType::Read);
    }

    #[test]
    fn lex_true() {
        should_lex!(".true.", TokenType::True);
    }

    #[test]
    fn lex_false() {
        should_lex!(".false.", TokenType::False);
    }

    #[test]
    fn lex_lt() {
        should_lex!(".LT.", TokenType::Less);
    }

    #[test]
    fn lex_le() {
        should_lex!(".LE.", TokenType::Leq);
    }

    #[test]
    fn lex_eq() {
        should_lex!(".EQ.", TokenType::Eq);
    }

    #[test]
    fn lex_ne() {
        should_lex!(".NE.", TokenType::NotEqual);
    }

    #[test]
    fn lex_gt() {
        should_lex!(".GT.", TokenType::Greater);
    }

    #[test]
    fn lex_geq() {
        should_lex!(".GEQ.", TokenType::Geq);
    }
    
    #[test]
    fn lex_not() {
        should_lex!(".NOT.", TokenType::Not);
    }
    
    #[test]
    fn lex_and() {
        should_lex!(".AND.", TokenType::And);
    }
    
    #[test]
    fn lex_or() {
        should_lex!(".OR.", TokenType::Or);
    }

    #[test]
    fn lex_eqv() {
        should_lex!(".EQV.", TokenType::Equiv);
    }

    #[test]
    fn lex_neqv() {
        should_lex!(".NEQV.", TokenType::NotEquiv);
    }
    
    #[test]
    fn lex_xor() {
        should_lex!(".XOR.", TokenType::Xor);
    }
    
    #[test]
    fn lex_equal() {
        should_lex!("=", TokenType::Equal);
    }
    
    #[test]
    fn lex_plus() {
        should_lex!("+", TokenType::Plus);
    }
    
    #[test]
    fn lex_left_paren() {
        should_lex!("(", TokenType::LeftParen);
    }
    
    #[test]
    fn lex_right_paren() {
        should_lex!(")", TokenType::RightParen);
    }
    
    #[test]
    fn lex_comma() {
        should_lex!(",", TokenType::Comma);
    }
    
    #[test]
    fn lex_minus() {
        should_lex!("-", TokenType::Minus);
    }
    
    #[test]
    fn lex_slash() {
        should_lex!("/", TokenType::Slash);
    }
    
    #[test]
    fn lex_star() {
        should_lex!("*", TokenType::Star);
    }
    
    #[test]
    fn dot_should_start_literal() {
        assert!(is_id_start('.'));
    }
}