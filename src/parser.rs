use crate::lexer::{
    BaseType,
    TokenType,
    Token,
    Lexer
};
use crate::parse_tree::*;

/*
Current plans:
1. Program Units

I found it useful to write the grammar rules as comments before each
function. This is a simple recursive descent parser, so production rules
correspond to function names.

Also, it may be useful to refactor out an `info` struct in the lexer to
store the line and column numbers (and file name? and lexeme?). This
would be useful to include when generating assembly code.
 */

/*
Lingering TODOs:
- [ ] Handle bizarre types like "LOGICAL*3"
- [ ] Handle "CHARACTER*n" type
 */

pub struct Parser {
    scanner: Lexer,
    current: Option<Token>,
    continuation_count: i16,
    line: usize,
    is_panicking: bool,
    pub support_f90: bool,
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
pub const MAX_CONTINUATIONS : i16 = 19;

impl Parser {
    pub fn new(scanner: Lexer) -> Self {
        Self {
            scanner,
            current: None,
            continuation_count: -1,
            line: 0,
            is_panicking: false,
            support_f90: true,
        }
    }

    /*
    Invoked in `statement()`

    Continuation count is negative for "initial lines" (i.e., new
    statements).
     */
    fn reset_continuation_count(&mut self) {
        self.continuation_count = -1;
        self.line = self.scanner.line_number();
    }

    fn inc_continuation_count(&mut self) {
        if self.continuation_count < 0 {
            self.continuation_count = 1;
        } else {
            self.continuation_count += 1;
        }
        if self.continuation_count > MAX_CONTINUATIONS {
            // I am too nice to throw an error here, a warning suffices.
            eprintln!("Warning: found {} continuations as of line {}",
                      self.continuation_count,
                      self.scanner.line_number());
        }
    }

    fn is_finished(&mut self) -> bool {
        self.current.is_none() && self.scanner.is_finished()
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
        let mut continues = false;
        while t.token_type.is_continuation() {
            self.inc_continuation_count();
            t = self.scanner.next_token();
            continues = true;
        }
        if t.line != self.line {
            if continues || self.continuation_count < 0 {
                self.line = t.line;
            } else {
                panic!("Line continuation needed to start line {}",
                       t.line);
            }
        }
        t
    }

    fn populate_current(&mut self) {
        if self.current.is_none() && !self.is_finished() {
            self.current = Some(self.next_token());
        }
    }

    fn peek(&mut self) -> &Option<Token> {
        self.populate_current();
        &self.current
    }

    fn push_back(&mut self, token: Token) {
        self.current = Some(token);
    }

    fn advance(&mut self) -> Token {
        if let Some(v) = self.current.take() {
            assert!(self.current.is_none());
            v
        } else {
            assert!(self.current.is_none());
            self.next_token()
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
        false
    }

    fn check(&mut self, token_type: TokenType) -> bool {
        match self.peek() {
            Some(v) => v.token_type == token_type,
            None => false,
        }
    }

    fn consume(&mut self, expected: TokenType, msg: &str) {
        if self.check(expected) {
            self.advance();
        } else {
            let token = self.advance();
            panic!("ERROR on Line {}: {}, found {token}", self.scanner.line_number(), msg);
        }
    }

    /*
    The next statement is determined by finding the next "initial line".
    This is the term the Fortran 77 Standard uses for the start of a
    statement:

    > 3.2.2  Initial Line.  An initial line is any line  that
    > is  not a comment line and contains the character blank
    > or the digit 0 in column 6.  Columns 1  through  5  may
    > contain a statement label (3.4), or each of the columns
    > 1 through 5 must contain the character blank.
     */
    fn is_at_next_statement(&mut self) -> bool {
        if let Some(token) = self.peek() {
            matches!(token.token_type, TokenType::Label(_))
        } else {
            self.is_finished()
        }
    }

    /*
    Skips ahead to the next statement.
    
    requires: self.is_panicking = true;
    assigns: self.current is updated until we're at the next statement;
    ensures: self.peek() is a label token or a new statement token
     */
    fn synchronize(&mut self) {
        while !self.is_finished() {
            if self.is_at_next_statement() {
                return;
            } else {
                self.advance();
            }
        }
    }

    /* *****************************************************************
    Program Unit
     ********************************************************** */
    fn program_unit(&mut self) -> ProgramUnit {
        if let Some(token) = self.peek() {
            match token.token_type {
                TokenType::Program => return self.program(),
                TokenType::Subroutine => panic!("Subroutine parsing not supported yet"),
                TokenType::Function => return self.function(),
                TokenType::Type(kind) => return self.function(),
                _ => panic!("Expected a program unit, found {token}"),
            }
        }
        return ProgramUnit::Empty;
    }

    /*
    Section 8 of the FORTRAN 77 Standard says there are 9 specification
    statements. The full grammar is reproduced below, but I annotated
    which ones are not supported.
    
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

    (* 8.4 of the 1977 spec... *)
    type_declaration = non_char_type_spec entity_decl_list
    | "CHARACTER" ["*" len [","]] char_entity_decl_list;
    
    non_char_type_spec = non_char_base_type ["*" size];

    non_char_base_type = "INTEGER" | "REAL" | "LOGICAL";

    char_entity_decl = name ["*" len]
                     | array_name "(" array_spec ")";

    len = size
        | "(" integer_const_expr ")"
        | "(*)";
    ```
     */
    fn specification(&mut self) -> Vec<Specification> {
        let mut result = Vec::<Specification>::with_capacity(16);
        while !self.is_finished() {
            if let Some(token) = self.peek() {
                match token.token_type {
                    TokenType::Type(t) => {
                        for d in self.type_declarations() {
                            result.push(Specification::TypeDeclaration(d));
                        }
                    },
                    TokenType::Parameter => {
                        for (lhs,rhs) in self.parameter_spec() {
                            result.push(Specification::Param(lhs,rhs));
                        }
                    },
                    _ => {
                        // Others are left "TODO"
                        break;
                    },
                }
            }
        }
        result.shrink_to_fit();
        return result;
    }

    /*
    R538 and R539 in the 1990 Standard:
    ```ebnf
    parameter = "PARAMETER (" name "=" expr {"," name "=" expr} ")"
    ```
     */
    fn parameter_spec(&mut self) -> Vec<(String, Expr)> {
        let mut params = Vec::<(String,Expr)>::with_capacity(8);
        self.consume(TokenType::Parameter, "");
        self.consume(TokenType::LeftParen, "Parameter statement expects '('");
        while !self.is_finished() {
            let name;
            let id = self.advance();
            match id.token_type {
                TokenType::Identifier(v) => name = v.iter().collect(),
                other => {
                    panic!("Expected identifier, found {:?}",
                           other);
                },
            }
            self.consume(TokenType::Equal, "Expected equality in parameter statement");
            let rhs = self.expr();
            params.push((name,rhs));
            if let Some(next) = self.peek() {
                eprintln!("\n\n\nNext token type {:?}\n\n\n", next.token_type);
            }
            if self.check(TokenType::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        self.consume(TokenType::RightParen, "Parameter statement should end with ')'");
        params.shrink_to_fit();
        return params;
    }
    
    /*
    The type declaration is, well, a hot mess. Although Fortran 90 did a
    crackerjack job cleaning up FORTRAN 77's tangled allowances, we are
    trying to implement a FORTRAN 77 compiler.

    See:
    - Section 5 of the Fortran 90 specification
    - Section 8.4 of the FORTRAN 77 specification (and 5.1 for array declarations)

    The grammar we implement:
    ```ebnf
    (* R501 et seq. terminology stolen *)
    type_declaration = non_char_type_declaration
    | char_type_declaration;
    
    non_char_type_declaration = non_char_type_spec entity_decl_list;
    
    entity_decl_list = entity_decl {"," entity_decl};

    type_spec = base_type ["*"
    ```
    fn entity_decl_list(&mut self, type_spec) {
        
    }
     */
    fn determine_type(&mut self) -> Type {
        // get the basic type
        let token = self.advance();
        let kind: BaseType;
        match token.token_type {
            TokenType::Type(b) => {
                kind = b;
            },
            other => {
                panic!("Line {}: type declaration expects type, found {:?}",
                       token.line,
                       other);
            }
        }
        let result;
        // determine if the next character is a "*"
        if self.check(TokenType::Star) {
            self.advance();
            // then get the modified size
            let size = self.advance();
            let width;
            match size.token_type {
                TokenType::Integer(x) => {
                    // parse the integer
                    width = x.iter().collect::<String>().parse::<i32>().unwrap();
                },
                other => {
                    panic!("Line {}: type declaration expects integer size, found {:?}",
                           size.line,
                           other);
                },
            }
            match kind {
                BaseType::Real => {
                    match width {
                        1..=3 => {
                            eprintln!("Line {} Warning: REAL*{} invalid width, rounding up to REAL*4",
                                      self.line,
                                      width);
                            result = Type::Real;
                        },
                        1..=4 => result = Type::Real,
                        5..=7 => {
                            eprintln!("Line {} Warning: REAL*{} invalid width, rounding up to REAL*8",
                                      self.line,
                                      width);
                            result = Type::Real64;
                        },
                        8 => result = Type::Real64,
                        9..=15 => {
                            eprintln!("Line {} Warning: REAL*{} invalid width, rounding up to REAL*16",
                                      self.line,
                                      width);
                            result = Type::Real128;
                        },
                        16 => result = Type::Real128,
                        other => {
                            eprintln!("Line {} Warning: REAL*{} invalid width, defaulting to REAL*8",
                                      self.line,
                                      width);
                            result = Type::Real64;
                        },
                    }
                },
                BaseType::Integer => { 
                    match width {
                        1..=3 => {
                            eprintln!("Line {} Warning: INTEGER*{} invalid width, rounding to INTEGER*4",
                                      self.line,
                                      width);
                            result = Type::Integer;
                        },
                        4 => {
                            result = Type::Integer;
                        },
                        5..=7 => {
                            eprintln!("Line {} Warning: INTEGER*{} invalid width, rounding to INTEGER*8",
                                      self.line,
                                      width);
                            result = Type::Integer64;
                        },
                        8 => {
                            result = Type::Integer64;
                        },
                        9..=15 => {
                            eprintln!("Line {} Warning: INTEGER*{} invalid width, rounding to INTEGER*16",
                                      self.line,
                                      width);
                            result = Type::Integer128;
                        },
                        16 => {
                            result = Type::Integer128;
                        },
                        other => {
                            eprintln!("Line {} Warning: INTEGER*{} invalid width, defaulting to INTEGER*8",
                                      self.line,
                                      width);
                            result = Type::Integer64;
                        },
                    }
                },
                BaseType::Character => { result = Type::Character; },
                BaseType::Logical => { result = Type::Logical; },
            }
            // weird quirk of Fortran 77 standard...
            if self.check(TokenType::Comma) {
                self.advance();
            }
        } else {
            match kind {
                BaseType::Real => result = Type::Real,
                BaseType::Integer => result = Type::Integer,
                BaseType::Character => result = Type::Character,
                BaseType::Logical => result = Type::Logical,
            };
        }
        return result;
    }
    
    fn type_declarations(&mut self) -> Vec<VarDeclaration> {
        // assert (current token is a basic type)
        // determine the type
        let kind: Type = self.determine_type();
        let mut results = Vec::<VarDeclaration>::with_capacity(8);
        while !self.is_finished() {
            let name;
            let id = self.advance();
            match id.token_type {
                TokenType::Identifier(v) => name = v.iter().collect(),
                other => {
                    panic!("Expected identifier, found {:?}",
                           other);
                },
            }
            let array = self.array_spec();
            // add variable declaration to statement
            results.push(VarDeclaration {
                kind: kind,
                name: name,
                array: array
            });
            // keep iterating while it's a list
            if self.check(TokenType::Comma) {
                self.advance();
            } else {
                break;
            }
        };
        results.shrink_to_fit();
        return results;
    }
    
    /*
    Using the F90 Standard terminology and grammar, I omit the deferred
    shape spec (since that seems to be new to F90).
    
    ```ebnf
    array_spec = explicit_shape_spec_list
               | assumed_shape_spec_list
               | assumed_size_spec;
    
    explicit_shape_spec_list = explicit_shape_spec {"," explicit_shape_spec};

    explicit_shape_spec = [lower_bound ":"] upper_bound;

    assumed_shape_spec_list = assumed_shape_spec {"," assumed_shape_spec};

    assumed_shape_spec = [lower_bound] ":";

    assumed_size_spec = [explicit_shape_spec_list ","] [lower_bound ":"] "*";
    ```

    ASSUMES: the leading left parentheses has NOT been processed
     */
    // TODO: test thoroughly
    fn array_spec(&mut self) -> ArraySpec {
        if !self.check(TokenType::LeftParen) {
            return ArraySpec::Scalar;
        } else {
            self.advance();
        }
        if self.matches(&[TokenType::Star]) {
            self.advance();
            self.consume(TokenType::RightParen, "'*' expected to end array spec");
            return ArraySpec::AssumedSize(Vec::<(Option::<Expr>, Expr)>::new(), None);
        } else if self.matches(&[TokenType::Colon]) {
            self.advance();
            if self.matches(&[TokenType::Comma]) {
                let mut indices = Vec::<Option::<Expr>>::with_capacity(8);
                indices.push(None);
                return self.assumed_shape(indices);
            } else if self.matches(&[TokenType::RightParen]) {
                self.advance();
                let mut indices = Vec::<Option::<Expr>>::with_capacity(1);
                indices.push(None);
                return ArraySpec::AssumedShape(indices);
            } else {
                // illegal form
                let token = self.advance();
                panic!("Line {}: Array spec found {} following ':', expected with ',' or ')'",
                       token.line,
                       token.token_type);
            }
        } else {
            // expr;
            let lower = self.expr();
            if self.matches(&[TokenType::Colon]) {
                // "(7:...)"
                self.advance();
                if self.matches(&[TokenType::Comma]) {
                    // "(7:, ...)"
                    let mut indices = Vec::<Option::<Expr>>::with_capacity(8);
                    indices.push(Some(lower));
                    return self.assumed_shape(indices);
                } else if self.matches(&[TokenType::RightParen]) {
                    // "(7:)"
                    let mut indices = Vec::<Option::<Expr>>::with_capacity(1);
                    indices.push(Some(lower));
                    return ArraySpec::AssumedShape(indices);
                } else if self.matches(&[TokenType::Star]) {
                    // "(7:*)"
                    self.advance();
                    self.consume(TokenType::RightParen, "array spec should end with *");
                    let mut indices = Vec::<(Option::<Expr>, Expr)>::new();
                    return ArraySpec::AssumedSize(indices, Some(lower));
                } else {
                    // "(7:15...)"
                    let upper = self.expr();
                    if self.matches (&[TokenType::Comma]) {
                        // "(7:15, ...)"
                        let mut indices = Vec::<(Option::<Expr>,Expr)>::with_capacity(8);
                        indices.push((Some(lower),upper));
                        self.advance();
                        return self.explicit_shape_or_assumed_size(indices);
                    } else if self.matches (&[TokenType::RightParen]) {
                        // "(7:15)"
                        self.advance();
                        let mut indices = Vec::<(Option::<Expr>,Expr)>::with_capacity(1);
                        indices.push((Some(lower),upper));
                        return ArraySpec::ExplicitShape(indices);
                    } else {
                        // "(7:???)", i.e., illegal form...
                        let token = self.advance();
                        panic!("Line {}: array specifier malformed, expected ',' or ')', found: {}",
                               self.line,
                               token.token_type);
                    }
                }
            } else if self.matches(&[TokenType::Comma]) {
                // "(7, ...)"
                self.advance();
                let mut indices = Vec::<(Option::<Expr>, Expr)>::with_capacity(8);
                indices.push((None, lower));
                return self.explicit_shape_or_assumed_size(indices);
            } else if self.matches(&[TokenType::RightParen]) {
                // "(7)"
                self.advance();
                let mut indices = Vec::<(Option::<Expr>, Expr)>::with_capacity(1);
                indices.push((None, lower));
                return ArraySpec::ExplicitShape(indices);
            } else {
                // (expr<something unexpected>);
                let token = self.advance();
                panic!("Line {}: Array spec...shouldn't be in this state, but found unexpected token {}",
                       token.line,
                       token.token_type);
            }
        }
    }

    /*
Requires: self.peek() == Some(TokenType::Comma)
Requires: indices is not empty
     */
    fn assumed_shape(&mut self, mut indices: Vec<(Option<Expr>)>) -> ArraySpec {
        let line = self.line;
        while !self.is_finished() {
            // each iteration processes if it should continue, or if
            // it's done, or if it should panic...
            if self.matches(&[TokenType::RightParen]) {
                self.advance();
                indices.shrink_to_fit();
                return ArraySpec::AssumedShape(indices);
            } else {
                self.consume(TokenType::Comma,
                             "Assumed shape spec separates dimensions by commas");
            }
            
            if self.matches(&[TokenType::Colon]) {
                self.advance();
                indices.push(None);
            } else {
                let bound = self.expr();
                self.consume(TokenType::Colon, "Assumed shape spec expected a colon");
                indices.push(Some(bound));
            }
        }
        panic!("Runaway Array spec starting on line {}", line);
    }

    /*
    We don't know if we're processing an explicit_shaped or assumed_size
    specification, so we bundle them together in a single place.
     */
    fn explicit_shape_or_assumed_size(&mut self, mut indices: Vec<(Option<Expr>,Expr)>) -> ArraySpec {
        /*
        This processes the following grammar:

        - expr "," rest...
        - expr ")"
        - expr ":" expr "," rest...
        - expr ":" expr ")"
        - expr ":*)"
        - "*)"

        If there is a `rest...`, it will be handled in the next iteration.
         */
        let line = self.line;
        while !self.is_finished() {
            if self.matches(&[TokenType::Star]) {
                // "*)" case
                self.advance();
                self.consume(TokenType::RightParen, "array spec must end with '*'");
                indices.shrink_to_fit();
                return ArraySpec::AssumedSize(indices, None);
            }
            let bound = self.expr();
            if self.matches(&[TokenType::Comma]) {
                // expr "," rest...
                self.advance();
                indices.push((None, bound));
            } else if self.matches(&[TokenType::RightParen]) {
                // expr ")"
                self.advance();
                indices.push((None, bound));
                indices.shrink_to_fit();
                return ArraySpec::ExplicitShape(indices);
            } else if self.matches(&[TokenType::Colon]) {
                self.advance();
                if self.matches(&[TokenType::Star]) {
                    // "expr:*)"
                    self.advance();
                    self.consume(TokenType::RightParen, "array spec must end with '*'");
                    indices.shrink_to_fit();
                    return ArraySpec::AssumedSize(indices, Some(bound));
                } else {
                    let upper = self.expr();
                    if self.matches(&[TokenType::Comma]) {
                        // "expr:expr, ...)"
                        self.consume(TokenType::Comma, "array spec expects comma separating dimensions");
                        indices.push((Some(bound), upper));
                    } else if self.matches(&[TokenType::RightParen]) {
                        // "expr:expr)"
                        self.advance();
                        indices.push((Some(bound), upper));
                        indices.shrink_to_fit();
                        return ArraySpec::ExplicitShape(indices);
                    } else {
                        // illegal
                        let token = self.advance();
                        panic!("Line {} unexpected token following '{:?}:' found: {}",
                               token.line,
                               bound,
                               token.token_type);
                    }
                }
            } else {
                // illegal
                let token = self.advance();
                panic!("Line {} unexpected token following {:?} found: {}",
                       token.line,
                       bound,
                       token.token_type);
            }
        }
        panic!("Line {}: runaway unterminated array specifier encountered",
               line);
    }

    fn dummy_args(&mut self) -> Vec<String> {
        let mut args = Vec::<String>::with_capacity(8);
        self.consume(TokenType::LeftParen, "Dummy arguments expected to start with '('");
        while !self.is_finished() {
            let token = self.advance();
            match token.token_type {
                TokenType::Identifier(v) => args.push(v.iter().collect()),
                TokenType::RightParen => {
                    args.shrink_to_fit();
                    return args;
                },
                other => panic!("Unknown token in dummy arguments: {other}"),
            }
            if self.check(TokenType::Comma) {
                self.advance();
            }
        }
        self.consume(TokenType::RightParen, "Dummy arguments expected to end with ')'");
        args.shrink_to_fit();
        return args;
    }

    fn determine_type_from_spec(&mut self, var_name: &String, spec: &Vec<Specification>)
                                -> Type {
        for entry in spec {
            if let Specification::TypeDeclaration(VarDeclaration{kind,
                                                                 name, ..}) = entry {
                if name == var_name {
                    return *kind;
                }
            }
        }
        panic!("Return type for {var_name} not declared anywhere");
    }

    /*
    Following the style of the Sun F77 compiler, the declared type for
    the function can be something like `REAL*16` (or some other
    specified width).

    Also, allow support for F90-style declarations where the return type
    is among the variable specification.
     */
    fn function(&mut self) -> ProgramUnit {
        let line = self.line;
        let return_type;
        let mut type_prefix: Option<Type> = None;
        // parse the return type and eat the "function" keyword
        if let Some(token) = self.peek() {
            match &token.token_type {
                TokenType::Type(kind) => {
                    type_prefix = Some(self.determine_type());
                    self.consume(TokenType::Function, "");
                },
                TokenType::Function => {
                    self.advance();
                },
                other => {
                    panic!("Expected 'function' or return type, found {other}");
                },
            }
        }
        // get the name of the function
        let token = self.advance();
        let name = match token.token_type {
            TokenType::Identifier(v) => v.into_iter().collect(),
            other => panic!("Program expected name, found {other}"),
        };
        // get the "dummy arguments" [parameters]
        let params = self.dummy_args();
        // parse the body of the function
        let spec = self.specification();
        if let Some(kind) = type_prefix {
            return_type = kind;
        } else /* if F90-style function declaration */ {
            return_type = self.determine_type_from_spec(&name, &spec);
        }
        let mut body = Vec::<Statement>::with_capacity(32);
        let mut has_return = false;
        while !self.is_finished() {
            let mut stmt = self.statement();
            has_return = has_return || stmt.is_return();
            let is_done = stmt.is_end();
            body.push(stmt);
            if is_done {
                break;
            }
        }
        if !has_return {
            panic!("LINE {} function has no return statement",
                   line);
        }
        return ProgramUnit::Function { name, return_type, params, spec, body };
    }

    /* Subroutines *must* include a `return` statement.

    ```ebnf
    subroutine = "subroutine" name dummy-args newline
                  specification
                  statements
                 "return" newline
                 "end"
    ```
     */
    fn subroutine(&mut self) -> ProgramUnit {
        let line = self.line;
        self.consume(TokenType::Subroutine, "");
        // get the name of the function
        let token = self.advance();
        let name = match token.token_type {
            TokenType::Identifier(v) => v.into_iter().collect(),
            other => panic!("Program expected name, found {other}"),
        };
        // get the "dummy arguments" [parameters]
        let params = self.dummy_args();
        // start parsing the body of the subroutine
        let spec = self.specification();
        let mut body = Vec::<Statement>::with_capacity(32);
        let mut has_return = false;
        while !self.is_finished() {
            let mut stmt = self.statement();
            has_return = has_return || stmt.is_return();
            let is_done = stmt.is_end();
            body.push(stmt);
            if is_done {
                break;
            }
        }
        if !has_return {
            panic!("LINE {} subroutine has no return statement",
                   line);
        }
        return ProgramUnit::Subroutine { name, params, spec, body };
    }

    /*
    ```ebnf
    program = "program" name [specification] {statement} "end";
    ```
     */
    fn program(&mut self) -> ProgramUnit {
        self.consume(TokenType::Program, "");
        let token = self.advance();
        let name = match token.token_type {
            TokenType::Identifier(v) => v.into_iter().collect(),
            other => panic!("Program expected name, found {other}"),
        };
        let spec = self.specification();
        let mut body = Vec::<Statement>::with_capacity(32);
        while !self.is_finished() {
            let mut stmt = self.statement();
            if stmt.is_stop() {
                body.push(stmt);
                let mut stmt2 = self.statement();
                if !stmt2.is_end() {
                    panic!("main program 'stop' should follow by 'end', found {:?}", stmt2);
                } else {
                    body.push(stmt2);
                    break;
                }
            } else if stmt.is_end() {
                panic!("WARNING: main program 'end' must be preceded by 'stop'");
            } else {
                body.push(stmt);
            }
        }
        return ProgramUnit::Program { name, spec, body };
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
        args
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
        Statement {
            label,
            command: Command::Read(self.io_list()),
        }
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
        Statement {
            label,
            command: Command::Write(self.io_list()),
        }
    }

    /*
    We only handle "unconditional GOTO" statement (11.1 of 77 Standard).

    The grammar for unconditional goto statements which we support:
    ```ebnf
    (* R836 *)
    goto-statement = "GOTO" label
    ```
     */
    fn goto_statement(&mut self, label: Option<i32>) -> Statement {
        assert!(TokenType::Goto == self.advance().token_type);
        let target: i32;
        let t = self.advance();
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
        Statement {
            label,
            command: Command::Goto(target),
        }
    }

    /*
    The 77 Standard has:
    ```ebnf
    continue-statement = "continue"
    ```
     */
    fn continue_statement(&mut self, label: Option<i32>) -> Statement {
        self.consume(TokenType::Continue,
                     "Continue statement expected to, well, 'continue'");
        Statement {
            label,
            command: Command::Continue,
        }
    }

    /*
    If statements are one of the two control statements in Fortran 77,
    and comes in three delicious flavours.
    
    # Arithmetic If Statement
    
    There are arithmetic `if` statements in Fortran 77 (11.4), which
    looks like:

    ```ebnf
    arithmetic-if = "if" "(" scalar-numeric-expr ")" label "," label "," label
    ```

    It is executed by looking at `if(s) l1, l2, l3` and performs the
    following:
    - if `s < 0` then goto l1,
    - else if s = 0 then goto l2,
    - else if s > 0 then goto l3.

    # Logical If Statements

    A logical if statement (11.5) in Fortran 77 looks like:
    
    ```ebnf
    (* R807 *)
    logical-if-statement = "if" "(" scalar-logical-expr ")" statement;
    ```

    Here the `statement` is any executable statement except a do,
    block-if, else-if, else, end-if, end, or another logical-if
    statement.

    Executation of logical-if statements are the same as:

    ```fortran
    if (test) statement
    
    ! syntactic sugar for

    if (test) then
        statement
    else
        continue
    end if
    ```

    # Block-If Statements
    This is discussed in the Fortran 77 Standard (11.6 et seq).
    
    The Fortran 90 Standard succinctly gives us the grammar for
    if-statements, which we **intentionally modify** to omit the
    `[if-name-construct]` labels.

    ```ebnf
    (* R802 *)
    if-construct = if-then-statement newline
                   block newline
                   {else-if-statement newline block newline}
                   [else-statement newline block newline]
                   end-if-statement;
    (* R803 *)
    if-then-statement = "if" "(" scalar-logical-expr ")" "then";
    (* R804 *)
    else-if-statement = "else" "if" "(" scalar-logical-expr ")" "then";
    (* R805 *)
    else-statement = "else";
    (* R806 *)
    end-if-statement = "end" "if";
    ```
    We can represent this using a `Statement::IfElse`.

    See Kernighan and Plauger's _The Elements of Programming Style_
    for more about avoiding arithmetic if statements.
     */
    fn if_construct(&mut self, label: Option<i32>) -> Statement {
        assert!(TokenType::If == self.advance().token_type);
        self.consume(TokenType::LeftParen,
                     "Expected '(' to start if statement.");
        let test = self.expr();
        self.consume(TokenType::RightParen,
                     "Expected ')' to close test condition in if statement");
        
        if let Some(token) = self.peek() {
            match token.token_type {
                TokenType::Then => self.block_if(label, test),
                TokenType::Integer(_) => {
                    /* TODO: issue warning here, since arithmetic-if is
                     * bad style */
                    return self.arithmetic_if(label, test);
                },
                // warn about "if (...) goto wherever"?
                _ => self.if_statement(label, test),
            }
        } else {
            panic!("Line {}: Unexpected termination of incomplete if statement",
                   self.line);
        }
        // self.illegal_statement(label)
    }

    fn end_if(&mut self, label: Option<i32>, test: Expr, true_branch: Vec<Statement>, false_branch: Vec<Statement>) -> Statement {
        if self.check(TokenType::End) {
            self.advance();
            self.consume(TokenType::If,
                         "If statement terminated by 'end', expected 'end if'");
            Statement {
                label,
                command: Command::IfBlock {
                    test,
                    true_branch,
                    false_branch,
                },
            }
        } else if self.check(TokenType::EndIf) {
            return Statement {
                label,
                command: Command::IfBlock {
                    test,
                    true_branch,
                    false_branch,
                },
            };
        } else {
            return self.illegal_statement(label);
        }
    }

    fn block_if(&mut self, label: Option<i32>, test: Expr) -> Statement {
        assert!(TokenType::Then == self.advance().token_type);
        let mut true_branch = Vec::<Statement>::with_capacity(32);
        loop {
            true_branch.push(self.statement());
            if self.matches(&[TokenType::Else, TokenType::End,
                              TokenType::EndIf]) {
                break;
            }
        }
        true_branch.shrink_to_fit();
        if self.check(TokenType::Else) {
            self.consume(TokenType::Else, "");
            if self.check(TokenType::If) {
                // else if ...
                let else_if: Statement = self.if_construct(None);
                Statement {
                    label,
                    command: Command::IfBlock {
                        test,
                        true_branch,
                        false_branch: vec![else_if],
                    },
                }
            } else {
                let mut false_branch = Vec::<Statement>::with_capacity(32);
                loop {
                    false_branch.push(self.statement());
                    if self.matches(&[TokenType::End, TokenType::EndIf]) {
                        break;
                    }
                }
                false_branch.shrink_to_fit();
                self.end_if(label, test, true_branch, false_branch)
            }
        } else {
            self.end_if(label, test, true_branch, Vec::<Statement>::new())
        }
    }

    /*
    Note that Fortran 90 does not appear to have arithmetic-if
    statements.
    
    ```ebnf
    arithmetic_if = [label] "if (" test ")" label "," label "," label;
    ```
     */
    fn arithmetic_if(&mut self, label: Option<i32>, test: Expr) -> Statement {
        fn get_label(this: &mut Parser, case_name: &str) -> i32 {
            let token = this.advance();
            match token.token_type {
                TokenType::Integer(v) => return v.iter().collect::<String>().parse::<i32>().unwrap(),
                _ => panic!("Arithmetic-if expected label for {} test, found {}",
                            case_name, token),
            }
        }
        let negative = get_label(self, "negative");
        self.consume(TokenType::Comma,
                     "Arithmetic-if expects comma separating labels");
        let zero = get_label(self, "negative");
        self.consume(TokenType::Comma,
                     "Arithmetic-if expects comma separating labels");
        let positive = get_label(self, "negative");
        Statement {
            label,
            command: Command::ArithIf {
                test,
                negative,
                zero,
                positive,
            },
        }
    }

    /*
    ```ebnf
    if_statement = "if (" logical-scalar-expr ")" statement;
    */
    fn if_statement(&mut self, label: Option<i32>, test: Expr) -> Statement {
        Statement {
            label,
            command: Command::IfStatement {test,
                                           true_branch: Box::new(self.statement())},
        }
    }

    /*
    ```ebnf
    (* R817 *)
    block_do_construct = do_statement newline do_block newline end_do;
    (* R818 *)
    do_statement = label_do_statement
                 | nonlabel_do_statement;
    (* R819 *)
    label_do_statement = "do" label [loop_control];
    (* R820 *)
    nonlabel_do_statement = "do" [loop_control];
    (* R821 *)
    loop_control = do_var "=" scalar_numeric_expr "," scalar_numeric_expr ["," scalar_numeric_expr];

    do_block = {statement newline};

    end_do = "end do" | label continue;
    ```
     */
    fn block_do_construct(&mut self, label: Option<i32>) -> Statement {
        self.consume(TokenType::Do, "");
        let line = self.line;
        if let Some(token) = self.peek() {
            match &token.token_type {
                TokenType::Integer(v) => {
                    let end_label = v.iter().collect::<String>().parse::<i32>().unwrap();
                    self.advance();
                    self.label_do_statement(label, end_label)
                },
                TokenType::Identifier(_) => {
                    self.nonlabel_do_statement(label)
                },
                _ => {
                    panic!("DO statement on line {} expects either a label or a variable, found {:?}",
                           line,
                           token);
                }
            }
        } else {
            panic!("Do statement on line {} is runaway?", line);
        }
    }

    // parses `var "=" expr "," expr ["," expr]`
    fn loop_control(&mut self) -> (Expr, Expr, Expr, Option<Expr>) {
        let line = self.line;
        let var = self.expr();
        self.consume(TokenType::Equal, "Expected a '=' to initialize variable in do-loop");
        let start = self.expr();
        self.consume(TokenType::Comma, "Expected a ',' to separate start from stop expressions in do-loop");
        let stop = self.expr();
        let stride;
        if let Some(token) = self.peek() {
            match token.token_type {
                TokenType::Comma => {
                    self.advance();
                    stride = Some(self.expr());
                },
                _ => {
                    stride = None;
                    if line == token.line {
                        panic!("Found unexpected token {:?} in loop control on line {}",
                               token,
                               line);
                    }
                }
            }
        } else {
            panic!("Do-loop terminated unexpectedly in loop-control on line {}", line);
        }
        (var, start, stop, stride)
    }

    fn label_do_statement(&mut self, label: Option<i32>, target: i32) -> Statement {
        let line = self.line;
        // parse the loop_control
        let (var, start, stop, stride) = self.loop_control();
        // parse the do_block
        let mut target_statement;
        let mut do_block = Vec::<Statement>::with_capacity(32);
        loop {
            let statement = self.statement();
            if Some(target) == statement.label {
                target_statement = statement;
                if !target_statement.is_continue() {
                    eprintln!("WARNING: do-loop starting on line {} terminated on line {} by statement which is not a 'continue', found: {:?}",
                              line,
                              self.line,
                              target_statement);
                }
                break;
            }
            do_block.push(statement);
            if self.is_finished() {
                panic!("Runaway do-loop started on line {}", line);
            }
        }
        do_block.shrink_to_fit();
        Statement {
            label,
            command: Command::LabelDo {
                target_label: target,
                var,
                start,
                stop,
                stride,
                body: do_block,
                terminal: Box::new(target_statement)
            },
        }
    }

    /*
     */
    fn nonlabel_do_statement(&mut self, label: Option<i32>) -> Statement {
        let line = self.line;
        if !self.support_f90 {
            eprintln!("WARNING: trying to parse do-loop on line {} using Fortran 90 syntax",
                      line);
        }
        fn found_end(this: &mut Parser, start_line: usize) -> bool {
            if let Some(token) = this.peek() {
                if TokenType::End == token.token_type {
                    this.advance(); // eat the End
                    this.consume(TokenType::Do, "do-loop starting on line {start_line} terminated by END but not END DO");
                    true
                } else {
                    false
                }
            } else {
                panic!("do-loop starting on line {} runaway", start_line);
            }
        }
        // parse the loop_control
        let (var, start, stop, stride) = self.loop_control();
        // parse the do_block
        let mut do_block = Vec::<Statement>::with_capacity(32);
        while !self.is_finished() {
            if found_end(self, line) {
                break;
            }
            do_block.push(self.statement());
        }
        // hack
        let terminal_statement = Statement {
            label: None,
            command: Command::Continue,
        };

        do_block.shrink_to_fit();

        Statement {
            label,
            command: Command::LabelDo {
                target_label: -1,
                var,
                start,
                stop,
                stride,
                body: do_block,
                terminal: Box::new(terminal_statement)
            },
        }
    }

    fn call_subroutine(&mut self, label: Option<i32>) -> Statement {
        self.consume(TokenType::Call, "subroutine call expected 'CALL'");
        let line = self.line;
        
        let token = self.advance();
        let subroutine: Expr = match token.token_type {
            TokenType::Identifier(v) => Expr::Variable(v.into_iter().collect()),
            other => panic!("Line {}: Calling subroutine expected subroutine name, found {:?}",
                            token.line,
                            other),
        };

        self.consume(TokenType::LeftParen,
                     "Expected '(' after subroutine name in call statement");

        let mut args = Vec::<Expr>::with_capacity(32);
        loop {
            if let Some(token) = self.peek() {
                match token.token_type {
                    TokenType::RightParen => break,
                    TokenType::Comma => { _ = self.advance(); },
                    _ => {},
                }
            }
            if self.is_finished() {
                panic!("Line {line}: Terminated unexpectedly while parsing subroutine call");
            }
            args.push(self.expr());
        }
        args.shrink_to_fit();
        Statement {
            label,
            command: Command::CallSubroutine {
                subroutine,
                args,
            }
        }
    }

    fn assignment_or_expr(&mut self, label: Option<i32>) -> Statement {
        let e = self.expr();
        if self.check(TokenType::Equal) {
            self.consume(TokenType::Equal, "");
            let rhs = self.expr();
            Statement {
                label,
                command: Command::Assignment {
                    lhs: e,
                    rhs,
                }
            }
        } else {
            Statement {
                label,
                command: Command::ExprStatement(e),
            }
        }
    }
    
    /*
    requires: start of statement
    assigns: current if it is a label token
    ensures: self.peek() is not a label
    ensures: result == None if the initial token is not a TokenType::Label
             OR its TokenType::Label(value) has value == 0
    ensures: result == Some(value) if the token is TokenType::Label(value) 
             with value > 0 --- discards leading zeros.
     */
    fn statement_label(&mut self) -> Option<i32> {
        if let Some(token) = self.peek() {
            match &token.token_type {
                TokenType::Label(v) => {
                    let value = v.iter().collect::<String>().parse::<i32>().unwrap();
                    self.advance();
                    if 0 == value {
                        None
                    } else {
                        Some(value)
                        /*
                        let label = Some(value);
                        return label;
                        */
                    }
                },
                _ => None,
            }
        } else {
            None
        }
    }

    fn illegal_statement(&mut self, label: Option<i32>) -> Statement {
        Statement {
            label,
            command: Command::Illegal,
        }
    }
    /*
    (* Section 11.5 of F77 says an action-statement is any statement
    which is not a DO, block IF, ELSE IF, ELSE, END IF, END, or another
    logical-if statement. See also R807 of the Fortran 90 Standard (and
    R216 for action statements in Fortran 90). *)
    logical-if-statement = "if" if-test action-statement;
    if-test = "(" bool-expr ")";
    endif = "endif" | "end" "if";
    if-statement = "if" if-test "then" {statement} endif
                 | "if" if-test "then" {statement} "else" {statement} endif
                 | logical-if-statement
     */
    /*
    action-statement = goto-statement
                     | read-statement
                     | write-statement
                     | continue-statement
                     | assignment-statement
    statement = action-statement
              | do-statement
              | if-statement;
     */
    pub fn statement(&mut self) -> Statement {
        self.reset_continuation_count();
        let label: Option<i32> = self.statement_label();
        // assert!(!self.peek().token_type.is_label());
        if let Some(token) = self.peek() {
            match token.token_type {
                TokenType::Goto => self.goto_statement(label),
                TokenType::Continue => self.continue_statement(label),
                TokenType::Write => self.write(label),
                TokenType::Read => self.read(label),
                TokenType::Do => self.block_do_construct(label),
                TokenType::If => self.if_construct(label),
                TokenType::Call => self.call_subroutine(label),
                TokenType::Stop => {
                    self.advance();
                    Statement { label: label,
                                command: Command::Stop }
                },
                TokenType::End => {
                    self.advance();
                    Statement { label: label,
                                command: Command::End }
                },
                _ => {
                    self.assignment_or_expr(label)
                }
            }
        } else {
            self.illegal_statement(None)
        }
    }
    
    /* *****************************************************************
    Expressions
     ********************************************************** */

    pub fn expr(&mut self) -> Expr {
        
        self.level_5_expr()
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
        e
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
        e
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
        e
    }

    /*
    and_operand ::= [".not."] level_4_expr
     */
    fn and_operand(&mut self) -> Expr {
        if self.matches(&[TokenType::Not]) {
            let rator = token_to_unary_op(self.advance());
            let rand = self.level_4_expr();
            Expr::Unary(rator, Box::new(rand))
        } else {
            self.level_4_expr()
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
        e
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
        e
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
        b
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
        e
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
        e
    }
    
    /*
    level_1_expr ::= primary
     */
    fn level_1_expr(&mut self) -> Expr {
        
        self.primary()
    }

    /*
    subscript ::= int_scalar_expr
     */
    fn subscript(&mut self) -> Expr {
        // TODO: check that it's really an integer scalar quantity
        self.expr()
    }

    /*
    section_triplet_tail = ":" [expr] [":" expr]
    --- equivalently ---
    section_triplet_tail = ":"
                         | ":" expr
                         | ":" ":" expr
                         | ":" expr ":" expr
     */
    fn section_triplet_tail(&mut self) -> Option<(Option<Box<Expr>>,
                                                  Option<Box<Expr>>)> {
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
        Some((stop, stride))
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
                Expr::Section((None, stop, stride))
            } else {
                // ":" matched
                Expr::Section((None, None, None))
            }
        } else { // section_subscript = subscript + stuff
            let e = self.subscript();
            if let Some((stop, stride)) = self.section_triplet_tail() {
                // "e:stop[:stride]" matched
                Expr::Section((Some(Box::new(e)), stop, stride))
            } else {
                // subscript matched
                e
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
                if let Expr::Section(_) = e { is_array_section = true }
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
            Expr::ArraySection(identifier, args)
        } else {
            Expr::NamedDataRef(identifier, args)
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
                let e = self.array_section_or_fn_call(v.into_iter().collect());
                self.consume(TokenType::RightParen,
                             "Expected ')' closing array access or function call");
                return e;
            }
        } else {
            panic!("This should never be reached! Expected an identifier, received {}", identifier);
        }
    }

    fn intrinsic_real_conversion(&mut self) -> Expr {
        self.consume(TokenType::LeftParen,
                     "Left paren expected in 'REAL' function call conversion");
        let arg = self.expr();
        self.consume(TokenType::RightParen,
                     "Right paren expected in 'REAL' conversion function call");
        let mut args = Vec::<Expr>::with_capacity(1);
        args.push(arg);
        return Expr::FunCall(String::from("REAL"), args);
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
                Expr::Character(s.to_vec())
            },
            TokenType::Identifier(_) => {
                self.named_data_ref(token)
            },
            // carve out for intrinsic function
            TokenType::Type(BaseType::Real) => {
                self.intrinsic_real_conversion()
            },
            TokenType::LeftParen => {
                let e = self.expr();
                self.consume(TokenType::RightParen,
                             "Expected ')' after expression.");
                Expr::Grouping(Box::new(e))
            },
            _ => {
                Expr::ErrorExpr
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // test program unit parsing [PROGRAM, FUNCTION, SUBROUTINE]
    mod unit {
        use super::*;

        #[test]
        fn main_with_only_declarations_test() {
            let src = ["      PROGRAM min1\n",
                       "      INTEGER k\n",
                       "      REAL m\n",
                       "      STOP\n",
                       "      end"].concat();
            let l = Lexer::new(src.chars().collect());
            let mut parser = Parser::new(l);
            let actual = parser.program();
            let mut spec = Vec::<Specification>::with_capacity(2);
            spec.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Integer,
                    name: String::from("k"),
                    array: ArraySpec::Scalar,
                }
            ));
            spec.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Real,
                    name: String::from("m"),
                    array: ArraySpec::Scalar,
                }
            ));
            let mut body = Vec::<Statement>::with_capacity(32);
            body.push(Statement {
                label: None,
                command: Command::Stop,
            });
            body.push(Statement {
                label: None,
                command: Command::End,
            });
            spec.shrink_to_fit();
            body.shrink_to_fit();
            let expected = ProgramUnit::Program {
                name: String::from("min1"),
                spec,
                body
            };
            assert_eq!(expected, actual);
        }
        
        /* van Loan & Coleman, "Handbook for Matrix
         * Computations". SIAM, pg 44 */
        #[test]
        fn min1_test() {
            let src = ["      PROGRAM min1",
                       "      INTEGER k",
                       "      REAL m",
                       "C   ",
                       "C  Set m = f(0)",
                       "C   ",
                       "      m = -3.0e0",
                       "C   ",
                       "C  Now check f(1), ..., f(10)",
                       "C   ",
                       "      DO 10 k = 1,10",
                       "         x = real(k)",
                       "         m = min(m, ((x - 2.0)*x - 7.0)*x - 3.0)",
                       "   10 continue",
                       "      WRITE(*,*)m",
                       "      STOP",
                       "      end"].join("\n");
            let l = Lexer::new(src.chars().collect());
            let mut parser = Parser::new(l);
            let actual = parser.program();
            let mut spec = Vec::<Specification>::with_capacity(2);
            spec.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Integer,
                    name: String::from("k"),
                    array: ArraySpec::Scalar,
                }
            ));
            spec.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Real,
                    name: String::from("m"),
                    array: ArraySpec::Scalar,
                }
            ));
            let mut body = Vec::<Statement>::with_capacity(32);
            body.push(Statement {
                label: None,
                command: Command::Assignment {
                    lhs: Expr::Variable(String::from("m")),
                    rhs: Expr::Unary(UnOp::Minus,
                                     Box::new(Expr::Float64(3.0)))
                }
            });
            let mut do_body = Vec::<Statement>::with_capacity(2);
            let mut args = Vec::<Expr>::with_capacity(1);
            args.push(Expr::Variable(String::from("k")));
            do_body.push(Statement {
                label: None,
                command: Command::Assignment {
                    lhs: Expr::Variable(String::from("x")),
                    rhs: Expr::FunCall(String::from("REAL"),
                                       args)
                }
            });
            // m = min(m, ((x - 2.0)*x - 7.0)*x - 3.0)
            let mut args = Vec::<Expr>::with_capacity(2);
            args.push(Expr::Variable(String::from("m")));
            let x_minus_2 = Expr::Grouping(
                Box::new(Expr::Binary(Box::new(Expr::Variable(String::from("x"))),
                                      BinOp::Minus,
                                      Box::new(Expr::Float64(2.0)))));
            let x_minus_2_then_multiply_by_x_minus_7 = Expr::Grouping(
                Box::new(Expr::Binary(
                    Box::new(Expr::Binary(Box::new(x_minus_2),
                                          BinOp::Times,
                                          Box::new(Expr::Variable(String::from("x"))))),
                    BinOp::Minus,
                    Box::new(Expr::Float64(7.0)))));
            args.push(Expr::Binary(Box::new(Expr::Binary(
                Box::new(x_minus_2_then_multiply_by_x_minus_7),
                BinOp::Times,
                Box::new(Expr::Variable(String::from("x"))))),
                                   BinOp::Minus,
                                   Box::new(Expr::Float64(3.0))));
            do_body.push(Statement {
                label: None,
                command: Command::Assignment {
                    lhs: Expr::Variable(String::from("m")),
                    rhs: Expr::NamedDataRef(String::from("min"),
                                            args)
                }
            });
            let terminal = Statement {
                label: Some(10),
                command: Command::Continue,
            };
            body.push(Statement {
                label: None,
                command: Command::LabelDo {
                    target_label: 10,
                    var: Expr::Variable(String::from("k")),
                    start: Expr::Int64(1),
                    stop: Expr::Int64(10),
                    stride: None,
                    body: do_body,
                    terminal: Box::new(terminal),
                },
            });
            let mut args = Vec::<Expr>::with_capacity(1);
            args.push(Expr::Variable(String::from("m")));
            body.push(Statement {
                label: None,
                command: Command::Write(args),
            });
            body.push(Statement {
                label: None,
                command: Command::Stop,
            });
            body.push(Statement {
                label: None,
                command: Command::End,
            });
            spec.shrink_to_fit();
            body.shrink_to_fit();
            let expected = ProgramUnit::Program {
                name: String::from("min1"),
                spec,
                body
            };
            assert_eq!(expected, actual);
        }
    }

    // test all the specification statements parse correctly
    mod spec {
        use super::*;
        #[macro_export]
        macro_rules! should_parse_spec {
            ( $text:expr, $expected:expr ) => {
                let l = Lexer::new($text.chars().collect());
                let mut parser = Parser::new(l);
                let actual = parser.specification();
                assert_eq!($expected, actual);
            }
        }

        
        #[test]
        fn f90_standard_parameter_example() {
            let src = "      PARAMETER (MODULUS = MOD (28, 3), SENATORS = 100)";
            let mut expected = Vec::<Specification>::with_capacity(2);
            let mut args = Vec::<Expr>::with_capacity(2);
            args.push(Expr::Int64(28));
            args.push(Expr::Int64(3));
            expected.push(Specification::Param(
                String::from("MODULUS"),
                Expr::NamedDataRef(String::from("MOD"), args)));
            expected.push(Specification::Param(
                String::from("SENATORS"),
                Expr::Int64(100)));
            should_parse_spec!(src, expected);
        }

        #[test]
        fn explicit_array_spec_with_negative_start() {
            let src = "      REAL C(-3:12)";
            let mut expected = Vec::<Specification>::with_capacity(1);
            let mut c_spec = Vec::<(Option<Expr>, Expr)>::with_capacity(1);
            c_spec.push((Some(Expr::Unary(UnOp::Minus,
                                          Box::new(Expr::Int64(3)))),
                         Expr::Int64(12)));
            expected.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Real,
                    name: String::from("C"),
                    array: ArraySpec::ExplicitShape(c_spec),
                }
            ));
            should_parse_spec!(src, expected);
        }

        #[test]
        #[should_panic]
        fn malformed_array_spec() {
            let src = "      REAL C(-3:12, :*)";
            let mut expected = Vec::<Specification>::with_capacity(1);
            let mut c_spec = Vec::<(Option<Expr>, Expr)>::with_capacity(1);
            c_spec.push((Some(Expr::Unary(UnOp::Minus,
                                          Box::new(Expr::Int64(3)))),
                         Expr::Int64(12)));
            expected.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Real,
                    name: String::from("C"),
                    array: ArraySpec::AssumedSize(c_spec, None),
                }
            ));
            should_parse_spec!(src, expected);
        }

        #[test]
        fn assumed_size_array_with_lower_bound() {
            let src = "      REAL C(-3:12, *)";
            let mut expected = Vec::<Specification>::with_capacity(1);
            let mut c_spec = Vec::<(Option<Expr>, Expr)>::with_capacity(1);
            c_spec.push((Some(Expr::Unary(UnOp::Minus,
                                          Box::new(Expr::Int64(3)))),
                         Expr::Int64(12)));
            expected.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Real,
                    name: String::from("C"),
                    array: ArraySpec::AssumedSize(c_spec, None),
                }
            ));
            should_parse_spec!(src, expected);
        }

        #[test]
        fn explicit_array_with_lower_bound() {
            let src = "      REAL B(10:70)";
            let mut expected = Vec::<Specification>::with_capacity(1);
            let mut b_spec = Vec::<(Option<Expr>, Expr)>::with_capacity(1);
            b_spec.push((Some(Expr::Int64(10)), Expr::Int64(70)));
            expected.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Real,
                    name: String::from("B"),
                    array: ArraySpec::ExplicitShape(b_spec),
                }
            ));
            should_parse_spec!(src, expected);
        }

        #[test]
        fn scalar_and_array_params() {
            let src = "      REAL A,B(:)";
            let mut expected = Vec::<Specification>::with_capacity(2);
            expected.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Real,
                    name: String::from("A"),
                    array: ArraySpec::Scalar,
                }
            ));
            let mut b_spec = Vec::<(Option<Expr>)>::with_capacity(1);
            b_spec.push(None);
            expected.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Real,
                    name: String::from("B"),
                    array: ArraySpec::AssumedShape(b_spec),
                }
            ));
            should_parse_spec!(src, expected);
        }

        #[test]
        fn malformed_assumed_size_with_var_param() {
            let src = "      REAL S(N,:)";
            let mut expected = Vec::<Specification>::with_capacity(1);
            let mut s_spec = Vec::<(Option<Expr>,Expr)>::with_capacity(2);
            s_spec.push((None,Expr::Variable(String::from("N"))));
            s_spec.push((None,Expr::ErrorExpr));
            expected.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Real,
                    name: String::from("S"),
                    array: ArraySpec::ExplicitShape(s_spec),
                }
            ));
            should_parse_spec!(src, expected);
        }

        #[test]
        fn assumed_size_with_var_param() {
            let src = "      REAL S(N,*)";
            let mut expected = Vec::<Specification>::with_capacity(1);
            let mut s_spec = Vec::<(Option<Expr>,Expr)>::with_capacity(1);
            s_spec.push((None,Expr::Variable(String::from("N"))));
            expected.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Real,
                    name: String::from("S"),
                    array: ArraySpec::AssumedSize(s_spec, None),
                }
            ));
            should_parse_spec!(src, expected);
        }

        #[test]
        fn two_colon_indices_in_array_spec() {
            let src = "      REAL D(:,:)";
            let mut expected = Vec::<Specification>::with_capacity(1);
            let mut d_spec = Vec::<Option<Expr>>::with_capacity(2);
            d_spec.push(None);
            d_spec.push(None);
            expected.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Real,
                    name: String::from("D"),
                    array: ArraySpec::AssumedShape(d_spec),
                }
            ));
            should_parse_spec!(src, expected);
        }

        #[test]
        fn two_assumed_shape_arrays() {
            let src = "      REAL A(:), B(0:)";
            let mut expected = Vec::<Specification>::with_capacity(2);
            let mut a_spec = Vec::<Option<Expr>>::with_capacity(1);
            a_spec.push(None);
            expected.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Real,
                    name: String::from("A"),
                    array: ArraySpec::AssumedShape(a_spec),
                }
            ));
            let mut b_spec = Vec::<Option<Expr>>::with_capacity(1);
            b_spec.push(Some(Expr::Int64(0)));
            expected.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Real,
                    name: String::from("B"),
                    array: ArraySpec::AssumedShape(b_spec),
                }
            ));
            should_parse_spec!(src, expected);
        }

        #[test]
        fn explicit_array_with_var() {
            let src = "      REAL W(n, 10)";
            let mut expected = Vec::<Specification>::with_capacity(1);
            let mut spec = Vec::<(Option<Expr>, Expr)>::with_capacity(2);
            spec.push((None, Expr::Variable(String::from("n"))));
            spec.push((None, Expr::Int64(10)));
            
            expected.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Real,
                    name: String::from("W"),
                    array: ArraySpec::ExplicitShape(spec),
                }
            ));

            should_parse_spec!(src, expected);
        }

        #[macro_export]
        macro_rules! should_parse_scalar_spec {
            ( $text:expr, $expected_name:expr, $expected_type:expr ) => {
                let l = Lexer::new($text.chars().collect());
                let mut parser = Parser::new(l);
                let actual = parser.specification();
                let mut expected = Vec::<Specification>::with_capacity(1);
                expected.push(Specification::TypeDeclaration(
                    VarDeclaration {
                        kind: $expected_type,
                        name: String::from($expected_name),
                        array: ArraySpec::Scalar,
                    }
                ));
                assert_eq!(expected, actual);
            }
        }

        #[test]
        fn spec_real1() {
            let src = "      REAL*1 D";
            should_parse_scalar_spec!(src, "D", Type::Real);
        }

        #[test]
        fn spec_real2() {
            let src = "      REAL*2 D";
            should_parse_scalar_spec!(src, "D", Type::Real);
        }
        
        #[test]
        fn spec_real3() {
            let src = "      REAL*3 D";
            should_parse_scalar_spec!(src, "D", Type::Real);
        }
        
        #[test]
        fn spec_real4() {
            let src = "      REAL*4 D";
            should_parse_scalar_spec!(src, "D", Type::Real);
        }
        
        #[test]
        fn spec_real5() {
            let src = "      REAL*5 D";
            should_parse_scalar_spec!(src, "D", Type::Real64);
        }
        
        #[test]
        fn spec_real6() {
            let src = "      REAL*6 D";
            should_parse_scalar_spec!(src, "D", Type::Real64);
        }
        
        #[test]
        fn spec_real7() {
            let src = "      REAL*7 D";
            should_parse_scalar_spec!(src, "D", Type::Real64);
        }
        
        #[test]
        fn spec_real8() {
            let src = "      REAL*8 D";
            should_parse_scalar_spec!(src, "D", Type::Real64);
        }
        
        #[test]
        fn spec_real9() {
            let src = "      REAL*9 D";
            should_parse_scalar_spec!(src, "D", Type::Real128);
        }
        
        #[test]
        fn spec_real10() {
            let src = "      REAL*10 D";
            should_parse_scalar_spec!(src, "D", Type::Real128);
        }
        
        #[test]
        fn spec_real11() {
            let src = "      REAL*11 D";
            should_parse_scalar_spec!(src, "D", Type::Real128);
        }
        
        #[test]
        fn spec_real12() {
            let src = "      REAL*12 D";
            should_parse_scalar_spec!(src, "D", Type::Real128);
        }
        
        #[test]
        fn spec_real13() {
            let src = "      REAL*13 D";
            should_parse_scalar_spec!(src, "D", Type::Real128);
        }
        
        #[test]
        fn spec_real14() {
            let src = "      REAL*14 D";
            should_parse_scalar_spec!(src, "D", Type::Real128);
        }
        
        #[test]
        fn spec_real15() {
            let src = "      REAL*15 D";
            should_parse_scalar_spec!(src, "D", Type::Real128);
        }
        
        #[test]
        fn spec_real16() {
            let src = "      REAL*16 D";
            should_parse_scalar_spec!(src, "D", Type::Real128);
        }
        
        #[test]
        fn spec_int4() {
            let src = "      INTEGER*4 D";
            should_parse_scalar_spec!(src, "D", Type::Integer);
        }
        
        #[test]
        fn spec_int8() {
            let src = "      INTEGER*8 D";
            should_parse_scalar_spec!(src, "D", Type::Integer64);
        }
        
        #[test]
        fn spec_int16() {
            let src = "      INTEGER*16 D";
            should_parse_scalar_spec!(src, "D", Type::Integer128);
        }

        #[test]
        fn should_parse_multiple_decls() {
            let src = ["      CHARACTER id",
                       "      INTEGER windspeed, temp, rain"].concat();
            let l = Lexer::new(src.chars().collect());
            let mut parser = Parser::new(l);
            let actual = parser.specification();
            let mut expected = Vec::<Specification>::with_capacity(4);
            expected.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Character,
                    name: String::from("id"),
                    array: ArraySpec::Scalar,
                }
            ));
            expected.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Integer,
                    name: String::from("windspeed"),
                    array: ArraySpec::Scalar,
                }
            ));
            expected.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Integer,
                    name: String::from("temp"),
                    array: ArraySpec::Scalar,
                }
            ));
            expected.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Integer,
                    name: String::from("rain"),
                    array: ArraySpec::Scalar,
                }
            ));
            assert_eq!(expected, actual);
        }

        #[test]
        fn should_parse_three_int_decls() {
            let src = ["      INTEGER windspeed, temp, rain"].concat();
            let l = Lexer::new(src.chars().collect());
            let mut parser = Parser::new(l);
            let actual = parser.specification();
            let mut expected = Vec::<Specification>::with_capacity(3);
            expected.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Integer,
                    name: String::from("windspeed"),
                    array: ArraySpec::Scalar,
                }
            ));
            expected.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Integer,
                    name: String::from("temp"),
                    array: ArraySpec::Scalar,
                }
            ));
            expected.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Integer,
                    name: String::from("rain"),
                    array: ArraySpec::Scalar,
                }
            ));
            assert_eq!(expected, actual);
        }

        #[test]
        fn should_parse_single_logical_decl() {
            let src = ["      LOGICAL panicking"].concat();
            let l = Lexer::new(src.chars().collect());
            let mut parser = Parser::new(l);
            let actual = parser.specification();
            let mut expected = Vec::<Specification>::with_capacity(1);
            expected.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Logical,
                    name: String::from("panicking"),
                    array: ArraySpec::Scalar,
                }
            ));
            assert_eq!(expected, actual);
        }

        #[test]
        fn should_parse_single_real_rank_2_assumed_size_array_decl() {
            let src = ["      REAL windspeed(17,*)"].concat();
            let l = Lexer::new(src.chars().collect());
            let mut parser = Parser::new(l);
            let actual = parser.specification();
            let mut expected = Vec::<Specification>::with_capacity(1);
            let mut spec = Vec::<(Option<Expr>, Expr)>::with_capacity(1);
            spec.push((None, Expr::Int64(17)));
            expected.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Real,
                    name: String::from("windspeed"),
                    array: ArraySpec::AssumedSize(spec, None),
                }
            ));
            assert_eq!(expected, actual);
        }

        #[test]
        fn should_parse_single_real_rank_2_array_decl() {
            let src = ["      REAL windspeed(17,34)"].concat();
            let l = Lexer::new(src.chars().collect());
            let mut parser = Parser::new(l);
            let actual = parser.specification();
            let mut expected = Vec::<Specification>::with_capacity(1);
            let mut spec = Vec::<(Option<Expr>, Expr)>::with_capacity(2);
            spec.push((None, Expr::Int64(17)));
            spec.push((None, Expr::Int64(34)));
            expected.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Real,
                    name: String::from("windspeed"),
                    array: ArraySpec::ExplicitShape(spec),
                }
            ));
            assert_eq!(expected, actual);
        }

        #[test]
        fn should_parse_single_real_array_decl() {
            let src = ["      REAL windspeed(17)"].concat();
            let l = Lexer::new(src.chars().collect());
            let mut parser = Parser::new(l);
            let actual = parser.specification();
            let mut expected = Vec::<Specification>::with_capacity(1);
            let mut spec = Vec::<(Option<Expr>, Expr)>::with_capacity(1);
            spec.push((None, Expr::Int64(17)));
            expected.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Real,
                    name: String::from("windspeed"),
                    array: ArraySpec::ExplicitShape(spec),
                }
            ));
            assert_eq!(expected, actual);
        }

        #[test]
        fn should_parse_single_real_decl() {
            let src = ["      REAL windspeed"].concat();
            let l = Lexer::new(src.chars().collect());
            let mut parser = Parser::new(l);
            let actual = parser.specification();
            let mut expected = Vec::<Specification>::with_capacity(1);
            expected.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Real,
                    name: String::from("windspeed"),
                    array: ArraySpec::Scalar,
                }
            ));
            assert_eq!(expected, actual);
        }

        #[test]
        fn should_parse_single_int_decl() {
            let src = ["      INTEGER windspeed"].concat();
            let l = Lexer::new(src.chars().collect());
            let mut parser = Parser::new(l);
            let actual = parser.specification();
            let mut expected = Vec::<Specification>::with_capacity(1);
            expected.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Integer,
                    name: String::from("windspeed"),
                    array: ArraySpec::Scalar,
                }
            ));
            assert_eq!(expected, actual);
        }

        #[test]
        fn should_parse_single_char_decl() {
            let src = ["      CHARACTER x"].concat();
            let l = Lexer::new(src.chars().collect());
            let mut parser = Parser::new(l);
            let actual = parser.specification();
            let mut expected = Vec::<Specification>::with_capacity(1);
            expected.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Character,
                    name: String::from("x"),
                    array: ArraySpec::Scalar,
                }
            ));
            assert_eq!(expected, actual);
        }
    }

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
        fn should_parse_end_statement() {
            let expected = Statement {
                label: None,
                command: Command::End
            };
            should_parse_stmt!("      end",
                               expected);
        }
        
        #[test]
        fn assign_constant_to_var() {
            let lhs = Expr::Variable(String::from("X"));
            let rhs = Expr::Int64(51);
            let expected = Statement {
                label: None,
                command: Command::Assignment {
                    lhs,
                    rhs,
                },
            };
            should_parse_stmt!("      X = 51",
                               expected);
        }

        #[test]
        fn call_subroutine_with_two_args() {
            let mut args = Vec::<Expr>::with_capacity(1);
            args.push(Expr::Variable(String::from("X")));
            args.push(Expr::Binary(Box::new(Expr::Variable(String::from("X"))),
                                   BinOp::Plus,
                                   Box::new(Expr::Variable(String::from("Y")))));
            let expected = Statement {
                label: None,
                command: Command::CallSubroutine{
                    subroutine: Expr::Variable(String::from("mySubroutine")),
                    args,
                },
            };
            should_parse_stmt!("      CALL mySubroutine(X, X+Y)",
                               expected);
        }

        #[test]
        fn call_subroutine_with_one_args() {
            let mut args = Vec::<Expr>::with_capacity(1);
            args.push(Expr::Variable(String::from("X")));
            let expected = Statement {
                label: None,
                command: Command::CallSubroutine{
                    subroutine: Expr::Variable(String::from("mySubroutine")),
                    args,
                },
            };
            should_parse_stmt!("      CALL mySubroutine(X)",
                               expected);
        }

        #[test]
        fn call_subroutine_without_args() {
            let args = Vec::<Expr>::new();
            let expected = Statement {
                label: None,
                command: Command::CallSubroutine{
                    subroutine: Expr::Variable(String::from("mySubroutine")),
                    args,
                },
            };
            should_parse_stmt!("      CALL mySubroutine()",
                               expected);
        }

        #[test]
        fn f77_do_loop_example() {
            let mut args = Vec::<Expr>::new();
            args.push(Expr::Variable(String::from("X")));
            args.shrink_to_fit();
            let mut do_body = vec!(Statement {
                label: None,
                command: Command::Write(args),
            });
            do_body.shrink_to_fit();
            let terminal = Statement {
                label: Some(100),
                command: Command::Continue,
            };
            let expected = Statement {
                label: None,
                command: Command::LabelDo {
                    target_label: 100,
                    var: Expr::Variable(String::from("I")),
                    start: Expr::Int64(1),
                    stop: Expr::Int64(10),
                    stride: None,
                    body: do_body,
                    terminal: Box::new(terminal),
                },
            };
            should_parse_stmt!(["      DO 100 I = 1,10",
                                "      WRITE (*,*) X",
                                " 100  CONTINUE"
                                ].join("\n"),
                               expected);            
        }

        #[test]
        fn f77_do_loop_with_stride_example() {
            let mut args = Vec::<Expr>::new();
            args.push(Expr::Variable(String::from("X")));
            args.shrink_to_fit();
            let mut do_body = vec!(Statement {
                label: None,
                command: Command::Write(args),
            });
            do_body.shrink_to_fit();
            let terminal = Statement {
                label: Some(100),
                command: Command::Continue,
            };
            let expected = Statement {
                label: None,
                command: Command::LabelDo {
                    target_label: 100,
                    var: Expr::Variable(String::from("I")),
                    start: Expr::Int64(1),
                    stop: Expr::Int64(10),
                    stride: Some(Expr::Int64(3)),
                    body: do_body,
                    terminal: Box::new(terminal),
                },
            };
            should_parse_stmt!(["      DO 100 I = 1,10,3",
                                "      WRITE (*,*) X",
                                " 100  CONTINUE"
                                ].join("\n"),
                               expected);            
        }

        #[test]
        fn f77_do_loop_with_negative_stride_example() {
            let mut args = Vec::<Expr>::new();
            args.push(Expr::Variable(String::from("X")));
            args.shrink_to_fit();
            let mut do_body = vec!(Statement {
                label: None,
                command: Command::Write(args),
            });
            do_body.shrink_to_fit();
            let terminal = Statement {
                label: Some(100),
                command: Command::Continue,
            };
            let expected = Statement {
                label: None,
                command: Command::LabelDo {
                    target_label: 100,
                    var: Expr::Variable(String::from("I")),
                    start: Expr::Int64(10),
                    stop: Expr::Int64(1),
                    stride: Some(Expr::Unary(UnOp::Minus,
                                             Box::new(Expr::Int64(3)))),
                    body: do_body,
                    terminal: Box::new(terminal),
                },
            };
            should_parse_stmt!(["      DO 100 I = 10,1,-3",
                                "      WRITE (*,*) X",
                                " 100  CONTINUE"
                                ].join("\n"),
                               expected);            
        }

        #[test]
        fn f90_do_loop_example() {
            let mut args = Vec::<Expr>::new();
            args.push(Expr::Variable(String::from("X")));
            args.shrink_to_fit();
            let mut do_body = vec!(Statement {
                label: None,
                command: Command::Write(args),
            });
            do_body.shrink_to_fit();
            let terminal = Statement {
                label: None,
                command: Command::Continue,
            };
            let expected = Statement {
                label: None,
                command: Command::LabelDo {
                    target_label: -1,
                    var: Expr::Variable(String::from("I")),
                    start: Expr::Int64(1),
                    stop: Expr::Int64(10),
                    stride: None,
                    body: do_body,
                    terminal: Box::new(terminal),
                },
            };
            should_parse_stmt!(["      DO I = 1,10",
                                "      WRITE (*,*) X",
                                "      END DO"
                                ].join("\n"),
                               expected);            
        }

        #[test]
        fn if_block_statement() {
            let test = Expr::Binary(Box::new(Expr::Variable(String::from("X"))),
                                    BinOp::Eq,
                                    Box::new(Expr::Variable(String::from("Y"))));
            let mut args = Vec::<Expr>::new();
            args.push(Expr::Variable(String::from("X")));
            args.shrink_to_fit();
            let mut true_branch = vec!(Statement {
                label: None,
                command: Command::Write(args),
            });
            true_branch.shrink_to_fit();
            let expected = Statement {
                label: Some(10_i32),
                command: Command::IfBlock {
                    test,
                    true_branch,
                    false_branch: Vec::<Statement>::new(),
                },
            };
            should_parse_stmt!(["  10  IF (X.EQ.Y) THEN",
                                "      WRITE (*,*) X",
                                "      END IF",
                                ].join("\n"),
                               expected);
        }

        #[test]
        fn if_else_block_statement() {
            let test = Expr::Binary(Box::new(Expr::Variable(String::from("X"))),
                                    BinOp::Eq,
                                    Box::new(Expr::Variable(String::from("Y"))));
            let mut args = Vec::<Expr>::new();
            args.push(Expr::Variable(String::from("X")));
            args.shrink_to_fit();
            let mut true_branch = vec!(Statement {
                label: None,
                command: Command::Write(args),
            });
            true_branch.shrink_to_fit();

            let mut args = Vec::<Expr>::new();
            args.push(Expr::Variable(String::from("Y")));
            args.shrink_to_fit();
            let mut false_branch = vec!(Statement {
                label: None,
                command: Command::Write(args),
            });
            false_branch.shrink_to_fit();
            
            let expected = Statement {
                label: Some(10_i32),
                command: Command::IfBlock {
                    test,
                    true_branch,
                    false_branch,
                },
            };
            should_parse_stmt!(["  10  IF (X.EQ.Y) THEN",
                                "      WRITE (*,*) X",
                                "      ELSE",
                                "      WRITE (*,*) Y",
                                "      END IF",
                                ].join("\n"),
                               expected);
        }

        #[test]
        fn if_elseif_else_block_statement() {
            let test = Expr::Binary(Box::new(Expr::Variable(String::from("X"))),
                                    BinOp::Eq,
                                    Box::new(Expr::Variable(String::from("Y"))));
            let mut args = Vec::<Expr>::new();
            args.push(Expr::Variable(String::from("X")));
            args.shrink_to_fit();
            let mut true_branch = vec!(Statement {
                label: None,
                command: Command::Write(args),
            });
            true_branch.shrink_to_fit();
            // else if branch
            // else if true subranch
            let mut args = Vec::<Expr>::new();
            args.push(Expr::Variable(String::from("Y")));
            args.shrink_to_fit();
            let mut else_if_true_branch = vec!(Statement {
                label: None,
                command: Command::Write(args),
            });
            else_if_true_branch.shrink_to_fit();
            // else if false subbranch
            let mut args = Vec::<Expr>::new();
            args.push(Expr::Variable(String::from("Z")));
            args.shrink_to_fit();
            let mut else_if_false_branch = vec!(Statement {
                label: None,
                command: Command::Write(args),
            });
            else_if_false_branch.shrink_to_fit();
            let else_test = Expr::Binary(Box::new(Expr::Variable(String::from("Y"))),
                                    BinOp::Gt,
                                    Box::new(Expr::Variable(String::from("Z"))));
            let mut else_if_branch = vec!(Statement {
                label: None,
                command: Command::IfBlock {
                    test: else_test,
                    true_branch: else_if_true_branch,
                    false_branch: else_if_false_branch
                }
            });
            else_if_branch.shrink_to_fit();
            
            let expected = Statement {
                label: Some(10_i32),
                command: Command::IfBlock {
                    test,
                    true_branch,
                    false_branch: else_if_branch,
                },
            };
            should_parse_stmt!(["  10  IF (X.EQ.Y) THEN",
                                "      WRITE (*,*) X",
                                "      ELSE IF (Y.GT.Z) THEN",
                                "      WRITE (*,*) Y",
                                "      ELSE",
                                "      WRITE (*,*) Z",
                                "      END IF",
                                ].join("\n"),
                               expected);
        }

        #[test]
        fn labeled_if_statement() {
            let test = Expr::Binary(Box::new(Expr::Variable(String::from("X"))),
                                    BinOp::Eq,
                                    Box::new(Expr::Variable(String::from("Y"))));
            let expected = Statement {
                label: Some(10_i32),
                command: Command::IfStatement {
                    test,
                    true_branch: Box::new(Statement{
                        label: None,
                        command: Command::Goto(300),
                    }),
                },
            };
            should_parse_stmt!("  10  IF (X.EQ.Y) GOTO 300",
                               expected);
        }

        #[test]
        fn labeled_arith_if_example() {
            let test = Expr::Variable(String::from("X"));
            let expected = Statement {
                label: Some(10_i32),
                command: Command::ArithIf {
                    test,
                    negative: 30,
                    zero: 40,
                    positive: 100,
                },
            };
            should_parse_stmt!("  10  IF (X) 30, 40, 100",
                               expected);
        }

        #[test]
        fn labeled_write_one_variable() {
            let mut args = Vec::<Expr>::new();
            args.push(Expr::Variable(String::from("X")));
            args.shrink_to_fit();
            let expected = Statement {
                label: Some(10_i32),
                command: Command::Write(args),
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
                label: Some(10_i32),
                command: Command::Write(args),
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
                label: Some(10_i32),
                command: Command::Read(args),
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
                label: Some(10_i32),
                command: Command::Read(args),
            };
            should_parse_stmt!(" 10   READ (*,*) X, Y,Z",
                               expected);
        }

        #[test]
        fn zero_label_continue_is_unlabeled() {
            let expected = Statement {
                label: None,
                command: Command::Continue
            };
            should_parse_stmt!(" 0    continue",
                               expected);
        }

        #[test]
        fn labeled_continue_with_leading_zeros() {
            let expected = Statement {
                label: Some(10_i32),
                command: Command::Continue
            };
            should_parse_stmt!(" 010  continue",
                               expected);
        }

        #[test]
        fn labeled_continue() {
            let expected = Statement {
                label: Some(10_i32),
                command: Command::Continue
            };
            should_parse_stmt!(" 10   continue",
                               expected);
        }

        #[test]
        fn unlabeled_continue() {
            let expected = Statement {
                label: None,
                command: Command::Continue
            };
            should_parse_stmt!("      continue",
                               expected);
        }

        #[test]
        fn labeled_goto() {
            let expected = Statement {
                label: Some(100_i32),
                command: Command::Goto(325)
            };
            should_parse_stmt!(" 100  goto 325",
                               expected);
        }

        #[test]
        fn unlabeled_goto() {
            let expected = Statement {
                label: None,
                command: Command::Goto(321)
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
