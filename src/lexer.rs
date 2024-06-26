use std::fmt;

// TODO: consider switching back to using String instead of Vec<char>,
//       or at least switch to Vec<u8>...

/*
Compare to `parse_tree::Type`, where the flags are chosen for
compatibility and the ability to mask properly.
 */
#[derive(PartialEq, Debug, Copy, Clone)] 
#[repr(u8)]
pub enum BaseType {
    Logical   = 0x10,
    Real      = 0x01,
    Integer   = 0x02,
    Character = 0x20,
}

impl fmt::Display for BaseType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BaseType::Integer => write!(f, "INTEGER"),
            BaseType::Real => write!(f, "REAL"),
            BaseType::Character => write!(f, "CHARACTER"),
            BaseType::Logical => write!(f, "LOGICAL"),
        }
    }
}

#[derive(PartialEq, Debug)] 
pub enum TokenType {
    // single-character tokens
    LeftParen, RightParen,
    Comma,
    Minus,
    Plus,
    Slash,
    Concatenation, // i.e., "//"
    Star,
    Pow, // "**"
    Equal, // "=" for assignment, do-loop initialization
    Colon, 

    Continuation(char),
    
    // literals
    Identifier(Vec<char>),
    Integer(Vec<char>),
    Label(Vec<char>),
    Float(Vec<char>),
    String(Vec<char>),

    Type(BaseType),
    
    // keywords
    Program,
    Parameter,
    If, Then, Else, EndIf,
    Do,
    Continue,
    True, False,
    Stop, End, Function, Return, Subroutine,
    Less, Leq, Eq, NotEqual, Greater, Geq,
    Not, And, Or, Equiv, NotEquiv, Xor,
    Goto, Call,

    // Provided primitive functions
    Write, Read,
    
    // The End
    Illegal, Eof
}

impl TokenType {
    pub fn is_continuation(&mut self) -> bool {
        matches!(self, TokenType::Continuation(_))
    }
    
    pub fn is_label(&mut self) -> bool {
        matches!(self, TokenType::Label(_))
    }
}

impl fmt::Display for TokenType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            // single-character tokens
            TokenType::LeftParen => write!(f, "("),
            TokenType::RightParen => write!(f, ")"),
            TokenType::Comma => write!(f, ","),
            TokenType::Colon => write!(f, ":"),
            TokenType::Minus => write!(f, "-"),
            TokenType::Plus => write!(f, "+"),
            TokenType::Slash => write!(f, "/"),
            TokenType::Concatenation => write!(f, "//"), // i.e., "//"
            TokenType::Star => write!(f, "*"),
            TokenType::Pow => write!(f, "**"), // "**"
            TokenType::Equal => write!(f, "="),

            TokenType::Continuation(c) => write!(f, "{}", c),
    
            // literals
            TokenType::Identifier(i) |
            TokenType::Integer(i) |
            TokenType::Label(i) |
            TokenType::Float(i) |
            TokenType::String(i) => {
                let s: String = i.iter().collect();
                write!(f, "{}", s)
            },
            TokenType::Type(base) => write!(f, "{}", base),
    
            // keywords
            TokenType::Program => write!(f, "program"),
            TokenType::Parameter => write!(f, "parameter"),
            TokenType::If => write!(f, "if"),
            TokenType::Then => write!(f, "then"),
            TokenType::Else => write!(f, "else"),
            TokenType::EndIf => write!(f, "endif"),
            TokenType::Do => write!(f, "do"),
            TokenType::Continue => write!(f, "continue"),
            TokenType::Call => write!(f, "call"),
            TokenType::True => write!(f, ".TRUE."),
            TokenType::False => write!(f, ".FALSE."),
            TokenType::Stop => write!(f, "STOP"),
            TokenType::End => write!(f, "end"),
            TokenType::Function => write!(f, "function"),
            TokenType::Return => write!(f, "return"),
            TokenType::Subroutine => write!(f, "subroutine"),
            TokenType::Less => write!(f, ".LT."),
            TokenType::Leq => write!(f, ".LE."),
            TokenType::Eq => write!(f, ".EQ."),
            TokenType::NotEqual => write!(f, ".NEQ."),
            TokenType::Greater => write!(f, ".GT."),
            TokenType::Geq => write!(f, ".GEQ."),
            TokenType::Not => write!(f, ".NOT."),
            TokenType::And => write!(f, ".AND."),
            TokenType::Or => write!(f, ".OR."),
            TokenType::Equiv => write!(f, ".EQV."),
            TokenType::NotEquiv => write!(f, ".NEQV."),
            TokenType::Xor => write!(f, ".XOR."),
            TokenType::Goto => write!(f, "Goto"),
            TokenType::Write => write!(f, "Write"),
            TokenType::Read => write!(f, "Read"),
            TokenType::Illegal => write!(f, "Illegal"),
            TokenType::Eof => write!(f, "Eof"),
        }
    }
}

#[derive(PartialEq, Debug)] 
pub struct Token {
    pub token_type: TokenType,
    pub line: usize,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.token_type)
    }
}


impl Token {
    pub fn new(t: TokenType, l: usize) -> Self {
        Self {
            token_type: t,
            line: l
        }
    }
}

fn get_keyword_token(ident: &[char]) -> Result<TokenType, String> {
    let identifier: String = ident.iter().collect();
    match &identifier.to_lowercase()[..] {
        "program" => Ok(TokenType::Program),
        "parameter" => Ok(TokenType::Parameter),
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
        "call" => Ok(TokenType::Call),
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
        ".ge." => Ok(TokenType::Geq),
        ".not." => Ok(TokenType::Not),
        ".and." => Ok(TokenType::And),
        ".or." => Ok(TokenType::Or),
        ".neqv." => Ok(TokenType::NotEquiv),
        ".eqv." => Ok(TokenType::Equiv),
        ".xor." => Ok(TokenType::Xor),
        // types
        "integer" => Ok(TokenType::Type(BaseType::Integer)),
        "real" => Ok(TokenType::Type(BaseType::Real)),
        "character" => Ok(TokenType::Type(BaseType::Character)),
        "logical" => Ok(TokenType::Type(BaseType::Logical)),
        
        _ => Err(String::from("Not a keyword"))
    }
}

pub struct Lexer {
    input: Vec<char>,       // Source code
    position: usize,        // Reading position
    read_position: usize,   // Current moving reading position
    ch: char,               // Current read character
    line: usize,            // Line number
    offset: i64             // Offset from the start of the line
}

fn is_letter(ch: char) -> bool {
    ch.is_ascii_lowercase() || ch.is_ascii_uppercase()
}

fn implies(antecedent: bool, consequent: bool) -> bool {
    !antecedent || consequent
}

fn is_id_start(c: char) -> bool {
    c.is_alphabetic() || '.' == c
}

fn is_identifier(c: char) -> bool {
    c.is_alphanumeric()
}

fn is_sign(c: char) -> bool {
    '-' == c || '+' == c
}

fn is_exponent(c: char) -> bool {
    'e' == c || 'd' == c || 'E' == c || 'D' == c
}

impl Lexer {
    // INVARIANT: 1 <= Lexer.offset 
    pub fn new(input : Vec<char>) -> Self {
        let c = if input.is_empty() {
            '\0'
        } else {
            input[0]
        };
        Self {
            input,
            position: 0,
            read_position: 0,
            ch: c,
            line: 1,
            offset: 1
        }
    }

    pub fn line_number(&mut self) -> usize {
        self.line
    }
    
    pub fn is_finished(&mut self) -> bool {
        self.read_position >= self.input.len()
    }
    
    pub fn read_char(&mut self) {
        if self.is_finished() {
            self.position = self.read_position;
            self.ch = '\0';
        } else {
            self.ch = self.input[self.read_position];
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

    // bool is returned so we can continue skipping comments and whitespace
    fn skipping_whitespace(&mut self) -> bool {
        let result = self.peek().is_whitespace();
        while self.peek().is_whitespace() {
            self.read_char();
        }
        result

    }

    // We should check if this is a 'C' or a '*' (or maybe '!')
    // but we are generous in accepting comments...
    fn is_comment(&mut self) -> bool {
        if 1 != self.offset { return false; }
        !self.peek().is_whitespace()
    }

    /*
    The Standard (3.2.2) says that "Columns 1 through 5 may contain a
    statement label (3.4), or each of the columns 1 through 5 must
    contain the blank character."
     */
    // Try to parse the label for the line, if it is present; if not,
    // return None.
    fn try_label(&mut self) -> Option<TokenType> {
        println!("\n\ntry_label() self.offset = {}", self.offset);
        if 1 <= self.offset && self.offset <= 6 {
            let mut l: Vec<char> = Vec::new();
            let stop = 7 - self.offset;
            for _i in 0..stop {
                if !self.input[self.read_position].is_whitespace() {
                    l.push(self.input[self.read_position]);
                } else if !l.is_empty() {
                    return Some(TokenType::Label(l));
                }
                self.read_char();
            }
        }
        None
    }

    fn is_continuation(&mut self) -> bool {
        6 == self.offset && !self.input[self.position].is_whitespace()
    }

    fn skip_rest_of_line(&mut self) {
        let line_number = self.line;
        while line_number == self.line && !self.is_finished() {
            self.read_char();
        }
        assert!(self.line != line_number);
    }

    // bool is returned so we can continue skipping comments and whitespace
    fn skipping_comment(&mut self) -> bool {
        if self.is_comment() {
            self.skip_rest_of_line();
            return true;
        }
        false
    }

    fn peek(&mut self) -> char {
        if self.read_position >= self.input.len() {
            return '\0';
        }
        self.input[self.read_position]
    }

    fn peek_next(&mut self) -> char {
        if self.read_position + 1 >= self.input.len() {
            return '\0';
        }
        self.input[self.read_position + 1]
    }

    // 4.3 of 77 Standard for Integer constants
    // 4.4 of 77 Standard for Real constants
    /**
    Integer constants in FORTRAN 77 look like:
    ```
    integer = [sign] digit+
    digit = 0 | 1 | 2 | ... | 9
    sign = "+" | "-"
    ```
    
    Real constants in the FORTAN 77 Standard in EBNF looks like:
    ```
    real = [sign] integer "." fractional_part ["E" integer]
    ```

    Fixed-form Fortran 90 gives a more well-thought out specification
    for real constants in rules 412--416 in Section 4.3.1.2 of the 1990
    Standard. 
    */
    fn lex_number(&mut self) -> TokenType {
        let position = self.position;
        assert!(self.ch.is_ascii_digit());
        while !self.is_finished() && self.peek().is_ascii_digit() {
            self.read_char();
        }
        if '.' == self.peek() && self.peek_next().is_ascii_digit() {
            self.read_char();
            while !self.is_finished() && self.peek().is_ascii_digit() {
                self.read_char();
            }
            if is_exponent(self.peek()) {
                self.read_char();
                if is_sign(self.peek()) {
                    self.read_char();
                }
                while !self.is_finished() && self.peek().is_ascii_digit() {
                    self.read_char();
                }
            }
            
            TokenType::Float(self.input[position..self.read_position].to_vec())
        } else {
            TokenType::Integer(self.input[position..self.read_position].to_vec())
        }
    }

    // TODO: this should raise an error if the identifier is more
    // than six characters long, but I'm a generous soul, so...
    fn read_identifier(&mut self) -> Vec<char> {
        assert!(is_id_start(self.ch));
        let position = self.position;
        while !self.is_finished() && is_identifier(self.peek()) {
            self.read_char();
        }
        if '.' == self.input[position] && '.' == self.input[self.read_position] {
            self.read_char();
        }
        // starts with a dot, ends with a dot
        assert!(implies('.' == self.input[position],
                        '.' == self.input[self.read_position-1]));
        self.input[position..self.read_position].to_vec()
    }

    // TODO: consider a command-line option supporting lexing C-like strings?
    fn lex_string(&mut self) -> TokenType {
        assert!('\'' == self.ch);
        self.read_char(); // gobble the "'"
        let start = self.position;
        while !self.is_finished() {
            println!("lex_string() read_position = {}",
                     self.read_position);
            self.read_char();
            // for escaped "'"
            if '\'' == self.peek() {
                if '\'' == self.peek_next() {
                    self.read_char();
                    assert!('\'' == self.ch);
                    assert!('\'' != self.peek_next());
                } else {
                    break;
                }
            }
        }
        self.read_char();
        // unterminated string
        if '\'' != self.ch && self.is_finished() {
            panic!("Unterminated string on line {} starting at column {}",
                   self.line, start);
        }
        // else, terminated string
        // assert!('\'' == self.ch);
        // assert!('\'' != self.peek());
        let value = self.input[start+1..self.read_position-1].to_vec();
        TokenType::String(value)
    }
    
    pub fn next_token_type(&mut self) -> TokenType {
        while self.skipping_comment() || self.skipping_whitespace() {
            continue;
        }
        self.position = self.read_position;
        let tok: TokenType;
        if self.is_continuation() {
            self.read_char();
            tok = TokenType::Continuation(self.ch);
            return tok;
        }

        if let Some(v) = self.try_label() { return v; }
        
        self.read_char();
        println!("next_token_type = {}", self.ch);
        match self.ch {
            '=' => {
                tok = TokenType::Equal;
            },
            ':' => {
                tok = TokenType::Colon;
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
                if '/' == self.peek() {
                    self.read_char();
                    tok = TokenType::Concatenation;
                } else {
                    tok = TokenType::Slash;
                }
            },
            '*' => {
                if '*' == self.peek() {
                    self.read_char();
                    tok = TokenType::Pow;
                } else {
                    tok = TokenType::Star;
                }
            },
            '\'' => {
                tok = self.lex_string();
            },
            '\0' => {
                tok = TokenType::Eof;
            },
            _ => {
                println!("Falling through default, char = {}",
                         self.ch);
                if self.ch.is_ascii_alphabetic() || '.' == self.ch {
                    let ident: Vec<char> = self.read_identifier();
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
                } else if self.ch.is_ascii_digit() {
                    return self.lex_number();
                } else {
                    return TokenType::Illegal;
                }
            }
        }
        // self.read_char();
        tok
    }
    
    pub fn next_token(&mut self) -> Token {
        let t: TokenType = self.next_token_type();
        let l = self.line;
        Token::new(t, l)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_lex_screaming_program_kw() {
        let input = String::from("      PROGRAM main\n      do stuff;\n      end");
        let mut l = Lexer::new(input.chars().collect());
        let token = l.next_token_type();
        assert_eq!(token, TokenType::Program);
    }

    #[test]
    fn should_lex_program_kw() {
        let input = String::from("      program main\n      do stuff;\n      end");
        let mut l = Lexer::new(input.chars().collect());
        let token = l.next_token_type();
        assert_eq!(token, TokenType::Program);
    }

    #[test]
    fn should_lex_program_name() {
        let input  = String::from("      program main\n  10  do stuff;\n      end");
        let mut l = Lexer::new(input.chars().collect());
        let token = l.next_token_type();
        assert_eq!(token, TokenType::Program);
        let token = l.next_token_type();
        assert_eq!(token,
                   TokenType::Identifier("main".chars().collect()));
    }

    #[test]
    fn should_match_label() {
        let input  = String::from("      program main\n  10  do stuff;\n      end");
        let mut l = Lexer::new(input.chars().collect());
        let token = l.next_token_type();
        assert_eq!(token, TokenType::Program);
        let token = l.next_token_type();
        assert_eq!(token,
                   TokenType::Identifier("main".chars().collect()));
        let token = l.next_token_type();
        assert_eq!(token, TokenType::Label("10".chars().collect()));
    }

    #[test]
    fn should_lex_label_10() {
        let input  = String::from("\n  10  do stuff;\n      end");
        let mut l = Lexer::new(input.chars().collect());
        let token = l.next_token_type(); // label
        assert_eq!(token, TokenType::Label("10".chars().collect()));
    }
    
    #[test]
    fn should_lex_doo() {
        let input  = String::from("      program main\n  10  do stuff;\n      end");
        let mut l = Lexer::new(input.chars().collect());
        let token = l.next_token_type(); // program
        assert_eq!(token, TokenType::Program);
        l.next_token_type(); // main
        l.next_token_type(); // label
        let token = l.next_token_type();
        assert_eq!(token, TokenType::Do);
    }

        #[test]
    fn should_lex_end_after_comment() {
        let input  = String::from("      program main\nC  10  do stuff;\n      end");
        let mut l = Lexer::new(input.chars().collect());
        l.next_token(); // program
        l.next_token(); // main
        let token = l.next_token_type();
        assert_eq!(token, TokenType::End);
    }

    #[test]
    fn should_not_lex_invalid_kw() {
        let input  = String::from("      enddo program main\n  10  do stuff;\n      end");
        let mut l = Lexer::new(input.chars().collect());
        let token = l.next_token_type();
        assert_eq!(token, TokenType::Identifier("enddo".chars().collect()));
    }

    #[test]
    fn should_lex_if_kw() {
        let input = String::from("      program main\n      if (.true.) then\n       do stuff\n      end\n      stop\n      end");
        let mut l = Lexer::new(input.chars().collect());
        l.next_token_type(); // program
        l.next_token_type(); // main
        let token = l.next_token_type();
        assert_eq!(token, TokenType::If);
    }

    #[test]
    fn should_lex_lparen_kw() {
        let input = String::from("      program main\n      if (.true.) then\n       do stuff\n      end\n      stop\n      end");
        let mut l = Lexer::new(input.chars().collect());
        l.next_token_type(); // program
        l.next_token_type(); // main
        l.next_token_type(); // if
        let token = l.next_token_type();
        assert_eq!(token, TokenType::LeftParen);
    }

    #[test]
    fn should_lex_screaming_true_kw() {
        let input = String::from("      program main\n      if (.TRUE.) then\n       do stuff\n      end\n      stop\n      end");
        let mut l = Lexer::new(input.chars().collect());
        l.next_token_type(); // program
        l.next_token_type(); // main
        l.next_token_type(); // if
        l.next_token_type(); // (
        let token = l.next_token_type();
        assert_eq!(token, TokenType::True);
    }

    #[test]
    fn should_lex_true_kw() {
        let input = String::from("      program main\n      if (.true.) then\n       do stuff\n      end\n      stop\n      end");
        let mut l = Lexer::new(input.chars().collect());
        l.next_token_type(); // program
        l.next_token_type(); // main
        l.next_token_type(); // if
        l.next_token_type(); // (
        let token = l.next_token_type();
        assert_eq!(token, TokenType::True);
    }
    
    #[test]
    fn should_lex_dot_prefix_invalid() {
        let input = String::from("      program main\n      if (.breaks.) then\n       do stuff\n      end\n      stop\n      end");
        let mut l = Lexer::new(input.chars().collect());
        l.next_token_type(); // program
        l.next_token_type(); // main
        l.next_token_type(); // if
        l.next_token_type(); // (
        let token = l.next_token_type();
        assert_eq!(token, TokenType::Illegal);
    }

    #[test]
    #[should_panic]
    fn should_lex_dot_prefix_without_dot_suffix_panics() {
        let input = String::from("      program main\n      if (.breaks) then\n       do stuff\n      end\n      stop\n      end");
        let mut l = Lexer::new(input.chars().collect());
        l.next_token_type(); // program
        l.next_token_type(); // main
        l.next_token_type(); // if
        l.next_token_type(); // (
        l.next_token_type(); // .breaks
    }
    
    #[test]
    fn should_lex_then_kw() {
        let input = String::from("      program main\n      if (.true.) then\n       do stuff\n      endif\n      stop\n      end");
        let mut l = Lexer::new(input.chars().collect());
        l.next_token_type(); // program
        l.next_token_type(); // main
        l.next_token_type(); // if
        l.next_token_type(); // (
        l.next_token_type(); // .true.
        let token = l.next_token_type();
        assert_eq!(token, TokenType::RightParen);
        let token = l.next_token_type();
        assert_eq!(token, TokenType::Then);
    }
    
    #[test]
    fn should_lex_endif_kw() {
        let input = String::from("      program main\n      if (.true.) then\n       do stuff\n      endif\n      stop\n      end");
        let mut l = Lexer::new(input.chars().collect());
        l.next_token_type(); // program
        l.next_token_type(); // main
        l.next_token_type(); // if
        l.next_token_type(); // (
        l.next_token_type(); // .true.
        l.next_token_type(); // )
        l.next_token_type(); // then
        l.next_token_type(); // do
        l.next_token_type(); // stuff
        let token = l.next_token_type();
        assert_eq!(token, TokenType::EndIf);
    }
    
    #[test]
    fn should_lex_pi_as_float() {
        let input = String::from("      3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679");
        let mut l = Lexer::new(input.chars().collect());
        let token = l.next_token_type();
        assert_eq!(token, TokenType::Float("3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679".chars().collect()));
    }
    
    #[test]
    fn should_lex_grothendieck_prime() {
        let input = String::from("      51");
        let mut l = Lexer::new(input.chars().collect());
        let token = l.next_token_type();
        assert_eq!(token, TokenType::Integer("51".chars().collect()));
    }

    #[test]
    fn should_lex_continuation() {
        let input = String::from("     2 + 7 = 9");
        let mut l = Lexer::new(input.chars().collect());
        let token = l.next_token_type();
        assert_eq!(token, TokenType::Continuation('2'));
    }

    #[test]
    fn should_lex_int_pow_int() {
        let input = String::from("      3 ** 4");
        let mut l = Lexer::new(input.chars().collect());
        let token = l.next_token_type();
        assert_eq!(token, TokenType::Integer("3".chars().collect()));
        let token = l.next_token_type();
        assert_eq!(token, TokenType::Pow);
        let token = l.next_token_type();
        assert_eq!(token, TokenType::Integer("4".chars().collect()));
    }

    #[macro_export]
    macro_rules! should_lex {
        ( $text:expr, $expected:expr ) => {
            {
                let prefix = "       ";
                let input = [prefix, $text].concat();
                let mut l = Lexer::new(input.chars().collect());
                let token = l.next_token_type();
                assert_eq!(token, $expected);
            }
        };
    }
    
    #[test]
    fn lexes_param_kw() {
        should_lex!("PARAMETER (x=5)", TokenType::Parameter);
    }

    #[test]
    fn lex_fn_name() {
        should_lex!("f()", TokenType::Identifier("f".chars().collect()));
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
        should_lex!(".GE.", TokenType::Geq);
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
    fn lex_concatenation() {
        should_lex!("//", TokenType::Concatenation);
    }
    
    #[test]
    fn lex_pow() {
        should_lex!("**", TokenType::Pow);
    }
    
    #[test]
    fn lex_star() {
        should_lex!("*", TokenType::Star);
    }

    #[test]
    fn lex_pie2_as_float() {
        should_lex!("3.14159e2", TokenType::Float("3.14159e2".chars().collect()));
    }

    #[test]
    fn lex_pi_e_minus2_as_float() {
        should_lex!("3.14159e-2", TokenType::Float("3.14159e-2".chars().collect()));
    }

    #[test]
    fn lex_pi_e_plus2_as_float() {
        should_lex!("3.14159e+2", TokenType::Float("3.14159e+2".chars().collect()));
    }

    #[test]
    fn lex_pi_screaming_e_2_as_float() {
        should_lex!("3.14159E2", TokenType::Float("3.14159E2".chars().collect()));
    }

    #[test]
    fn lex_pi_as_float() {
        should_lex!("3.14159", TokenType::Float("3.14159".chars().collect()));
    }

    #[test]
    fn lex_simple_string() {
        should_lex!("'This is a string'",
                    TokenType::String("This is a string".chars().collect()));
    }

    #[test]
    fn lex_simple_string_with_apostrophe() {
        should_lex!("'This isn''t a number'",
                    TokenType::String("This isn''t a number".chars().collect()));
    }

    #[test]
    #[should_panic]
    fn lex_runaway_string() {
        should_lex!("'This is an unterminated string",
                    TokenType::Illegal);
    }
    
    #[test]
    fn lex_character_type() {
        should_lex!("CHARACTER", TokenType::Type(BaseType::Character));
    }
    
    #[test]
    fn lex_logical_type() {
        should_lex!("LOGICAL", TokenType::Type(BaseType::Logical));
    }
    
    #[test]
    fn lex_real_type() {
        should_lex!("real", TokenType::Type(BaseType::Real));
    }
    
    #[test]
    fn lex_integer_type() {
        should_lex!("INTEGER", TokenType::Type(BaseType::Integer));
    }
    
    #[test]
    fn dot_should_start_literal() {
        assert!(is_id_start('.'));
    }

    #[test]
    fn should_not_match_f_as_exponent() {
        assert!(!is_exponent('f'));
    }

    #[test]
    fn should_match_e_as_exponent() {
        assert!(is_exponent('e'));
    }
}
