use crate::parse_tree::{
    BinOp,
    UnOp,
    ArraySpec,
    Command,
    Specification,
    Statement,
    ProgramUnit,
    ProgramUnitKind,
    Program,
    Type,
    VarDeclaration,
};
use crate::parse_tree;

#[derive(PartialEq, Debug)]
pub enum Expr {
    // literals
    Character(Vec<char>),
    Float32(f32),
    Float64(f64),
    Int32(i32),
    Int64(i64),
    Logical(bool),
    Variable(usize),   // "de Bruijn index"
    Subroutine(usize), // for "call subroutine" statements

    // array slice section: start, stop, stride
    Section((Option<Box<Expr>>, Option<Box<Expr>>, Option<Box<Expr>>)),

    // composite expressions
    Binary(Box<Expr>, BinOp, Box<Expr>),
    Unary(UnOp, Box<Expr>),
    Grouping(Box<Expr>),
    FunCall(usize, Vec<Expr>),
    ArrayElement(usize, Vec<Expr>), // e.g., "MYARRAY(3,65,2)"
    ArraySection(usize, Vec<Expr>), // e.g., "MYARRAY(3:65)"
    ErrorExpr,
}

/*
All the functions for translating a parse tree into a disambiguated
syntax tree with de Bruijn indices.
 */
mod disambiguate {
    use super::*;
    use crate::ast::Expr;

    fn var_index<E: std::cmp::PartialEq>(name: &str,
                                         spec: &[Specification<E>]) -> Option<usize> {
        for (idx, s) in spec.iter().enumerate() {
            if s.has_name(name) {
                return Some(idx);
            }
        }
        None
    }

    fn fn_index<E: std::cmp::PartialEq>(name: &str,
                                        program: &Program<E>) -> Option<usize> {
        match program.index_for(name) {
            Some ((idx, ProgramUnitKind::Function)) => Some(idx),
            _ => None
        }
    }
    
    fn expr<E1: std::cmp::PartialEq,E2: std::cmp::PartialEq>(program: &Program<E1>, spec: &Vec<Specification<E2>>, e: &parse_tree::Expr) -> Expr {
        match e {
            parse_tree::Expr::Character(s) => Expr::Character(s.to_vec()),
            parse_tree::Expr::Float32(f) => Expr::Float32(*f),
            parse_tree::Expr::Float64(f) => Expr::Float64(*f),
            parse_tree::Expr::Int32(i) => Expr::Int32(*i),
            parse_tree::Expr::Int64(i) => Expr::Int64(*i),
            parse_tree::Expr::Logical(b) => Expr::Logical(*b),
            parse_tree::Expr::Variable(x) => {
                match var_index(x, spec) {
                    Some(i) => Expr::Variable(i),
                    None => panic!("No variable {x} found"),
                }
            },
            parse_tree::Expr::Section((start,stop,stride)) => {
                Expr::Section((start.as_ref().map(|e| Box::new(expr(program, spec, e))),
                               stop.as_ref().map(|e| Box::new(expr(program, spec, e))),
                               stride.as_ref().map(|e| Box::new(expr(program, spec, e)))))
            },
            parse_tree::Expr::Binary(lhs, op, rhs) => {
                Expr::Binary(Box::new(expr(program, spec, lhs)),
                             *op,
                             Box::new(expr(program, spec, rhs)))
            },
            parse_tree::Expr::Unary(op, e) => Expr::Unary(*op, Box::new(expr(program, spec, e))),
            parse_tree::Expr::Grouping(e) => expr(program, spec, e),
            parse_tree::Expr::NamedDataRef(name, args) => {
                let mut xs = Vec::<Expr>::with_capacity(args.len());
                for arg in args {
                    xs.push(expr(program, spec, arg));
                }
                // try looking up the name as an array
                if let Some(idx) = var_index(name, spec) {
                    return Expr::ArrayElement(idx, xs);
                }
                // or else go to the functions
                if let Some(idx) = fn_index(name, program) {
                    return Expr::FunCall(idx, xs);
                }
                panic!("Cannot disambiguate name {}", name);
            },
            parse_tree::Expr::FunCall(f, args) => {
                let mut xs = Vec::<Expr>::with_capacity(args.len());
                for arg in args {
                    xs.push(expr(program, spec, arg));
                }
                // look up the functions
                if let Some(idx) = fn_index(f, program) {
                    return Expr::FunCall(idx, xs);
                }
                panic!("Cannot disambiguate function {}", f);
            },
            parse_tree::Expr::ArrayElement(a, indices) => {
                let mut idxs = Vec::<Expr>::with_capacity(indices.len());
                for index in indices {
                    idxs.push(expr(program, spec, index));
                }
                // look up the array variable
                if let Some(idx) = var_index(a, spec) {
                    return Expr::ArrayElement(idx, idxs);
                }
                panic!("Cannot disambiguate array name {}", a);
            },
            parse_tree::Expr::ArraySection(a, indices) => {
                let mut idxs = Vec::<Expr>::with_capacity(indices.len());
                for index in indices {
                    idxs.push(expr(program, spec, index));
                }
                // look up the array variable
                if let Some(idx) = var_index(a, spec) {
                    return Expr::ArraySection(idx, idxs);
                }
                panic!("Cannot disambiguate array name {}", a);
            },
            parse_tree::Expr::ErrorExpr => Expr::ErrorExpr,
        }
    }

    /*
    When a type declaration is an array, parametrized by a variable like
    `REAL wind(N_latitude, N_longitude)`, we need
    to replace the variable parameters `N_latitude` and `N_longitude` with
    appropriate de Bruijn indices.
     */
    fn array_spec(program: &Program<parse_tree::Expr>,
                  prev_spec: &Vec<Specification<Expr>>,
                  spec: &ArraySpec<parse_tree::Expr>) -> ArraySpec<Expr> {
        fn optional_expr(e: &Option<parse_tree::Expr>,
                         p: &Program<parse_tree::Expr>,
                         ps: &Vec<Specification<Expr>>) -> Option<Expr> {
            e.as_ref().map(|ex| expr(p, ps, ex))
        }
        
        match spec {
            ArraySpec::ExplicitShape(v) => {
                // v: Vec<(Option<parse_tree::Expr>, parse_tree::Expr)>;
                let mut dims = Vec::<(Option<super::Expr>, super::Expr)>::with_capacity(v.len());
                for (start,stop) in v {
                    dims.push((optional_expr(start, program, prev_spec),
                               expr(program, prev_spec, stop)));
                }
                ArraySpec::ExplicitShape(dims)
            },
            ArraySpec::AssumedShape(v) => {
                // v: Vec<Option<parse_tree::Expr>>
                let mut dims = Vec::<Option<Expr>>::with_capacity(v.len());
                for d in v {
                    dims.push(optional_expr(d, program, prev_spec));
                }
                ArraySpec::AssumedShape(dims)
            },
            ArraySpec::AssumedSize(v, e) => {
                // v: Vec<(Option<parse_tree::Expr>, parse_tree::Expr)>
                // e: Option<parse_tree::Expr>
                let mut dims = Vec::<(Option<Expr>, Expr)>::with_capacity(v.len());
                for (start,stop) in v {
                    dims.push((optional_expr(start, program, prev_spec),
                               expr(program, prev_spec, stop)));
                }
                ArraySpec::AssumedSize(dims,
                                       optional_expr(e, program, prev_spec))
            },
            ArraySpec::Scalar => ArraySpec::<Expr>::Scalar,
        }
    }
    
    /*
    The specification statements can include parameters appearing elsewhere
    in the specification. So we need to have these be replaced by de Bruijn
    indices.
    
    NOTE: you cannot "forward reference" parameters in Fortran. For example,
    the following will throw an error because `n` is declared after its
    usage:
    
    ```fortran
          PROGRAM array1
          REAL wind(n)
          INTEGER i,n
          PARAMETER (n=5)

          do i=1, n
             wind(i) = 0.5*i
          end do

          write (*,*) wind
          STOP
          END
    ```
     */
    fn specification(program: &Program<parse_tree::Expr>,
            prev_spec: &Vec<Specification<Expr>>,
            spec: &Specification<parse_tree::Expr>) -> Specification<Expr> {
        match spec {
            Specification::TypeDeclaration(VarDeclaration {kind, name, array}) => {
                Specification::TypeDeclaration(VarDeclaration::<Expr> {
                    kind: *kind,
                    name: name.to_string(),
                    array: array_spec(program, prev_spec, array)
                })
            },
            Specification::Param(x, rhs) => 
                Specification::Param(String::from(x), expr(program, prev_spec, rhs)),
        }
    }
    
    /*
    In subroutine calls, replace subroutine name with de Bruijn index, and
    replaces its arguments with disambiguated expressions.
    
    In other statements, replace variables, function names, etc., with de
    Bruijn indices.
    
    Produces a new `Statement` object.
     */
    pub fn statement(program: &Program<parse_tree::Expr>,
                     spec: &Vec<Specification<Expr>>,
                     stmt: &Statement<parse_tree::Expr>) ->  Statement<Expr> {
        // STUB, replace with sensible version in a moment
        let label = stmt.label;
        Statement::<Expr> {
            label,
            command: 
            match &stmt.command {
                Command::Continue => Command::Continue,
                Command::Goto(loc) => Command::Goto(*loc),
                Command::Write(args) => {
                    let mut xs = Vec::<Expr>::with_capacity(args.len());
                    for arg in args {
                        xs.push(expr(program, spec, arg));
                    }
                    Command::Write(xs)
                },
                Command::Read(args) => {
                    let mut xs = Vec::<Expr>::with_capacity(args.len());
                    for arg in args {
                        xs.push(expr(program, spec, arg));
                    }
                    Command::Read(xs)
                },
                Command::IfBlock {test, true_branch, false_branch} => {
                    let t = expr(program, spec, test);
                    let mut t_branch = Vec::<Statement<Expr>>::with_capacity(true_branch.len());
                    for stmt in true_branch {
                        t_branch.push(statement(program, spec, stmt));
                    }
                    let mut f_branch = Vec::<Statement<Expr>>::with_capacity(false_branch.len());
                    for stmt in false_branch {
                        f_branch.push(statement(program, spec, stmt));
                    }
                    Command::IfBlock {
                        test: t,
                        true_branch: t_branch,
                        false_branch: f_branch
                    }
                },
                Command::ArithIf {test, negative, zero, positive} =>
                    Command::ArithIf {
                        test: expr(program, spec, test),
                        negative: *negative,
                        zero: *zero,
                        positive: *positive
                    },
                Command::IfStatement {test, true_branch} =>
                    Command::IfStatement {
                        test: expr(program, spec, test),
                        true_branch: Box::new(statement(program, spec, true_branch))
                    },
                Command::LabelDo {target_label, var, start, stop,
                                  stride, body, terminal} => {
                    let mut b = Vec::<Statement<Expr>>::with_capacity(body.len());
                    for line in body {
                        b.push(statement(program, spec, line));
                    }
                    Command::LabelDo {
                        target_label: *target_label,
                        var: expr(program, spec, var),
                        start: expr(program, spec, start),
                        stop: expr(program, spec, stop),
                        stride: stride.as_ref().map(|e| expr(program,
                                                             spec, e)),
                        body: b,
                        terminal: Box::new(statement(program, spec, terminal))
                    }
                },
                Command::CallSubroutine {subroutine, args} => {
                    let mut xs = Vec::<Expr>::with_capacity(args.len());
                    for arg in args {
                        xs.push(expr(program, spec, arg));
                    }
                    let sub = expr(program, spec, subroutine);
                    // assert!(matches!(sub, Expr::Subroutine(_)))
                    Command::CallSubroutine {
                        subroutine: sub,
                        args: xs
                    }
                },
                Command::ExprStatement(e) => Command::ExprStatement(expr(program,spec,e)),
                Command::Assignment {lhs, rhs} => Command::Assignment {
                    lhs: expr(program, spec, lhs),
                    rhs: expr(program, spec, rhs),
                },
                Command::Stop => Command::Stop,
                Command::Return => Command::Return,
                Command::End => Command::End,
                Command::Illegal => Command::Illegal,
            }
        }
    }
    
    /*
    Disambiguates the specification and statements for a program unit.
     */
    pub fn program_unit(program: &Program<parse_tree::Expr>,
                        unit: &ProgramUnit<parse_tree::Expr>) -> ProgramUnit<Expr> {
        match unit {
            ProgramUnit::Program {name, spec, body} => {
                let mut sp = Vec::<Specification<Expr>>::with_capacity(spec.len());
                for line in spec.iter() {
                    sp.push(specification(program, &sp, line));
                }
                let mut stmts = Vec::<Statement::<Expr>>::with_capacity(body.len());
                for line in body {
                    stmts.push(statement(program, &sp, line));
                }
                ProgramUnit::Program {
                    name: name.to_string(),
                    spec: sp,
                    body: stmts
                }
            },
            ProgramUnit::Function {name, return_type, params, spec, body } => {
                let mut sp = Vec::<Specification<Expr>>::with_capacity(spec.len());
                for line in spec.iter() {
                    sp.push(specification(program, &sp, line));
                }
                let mut stmts = Vec::<Statement::<Expr>>::with_capacity(body.len());
                for line in body {
                    stmts.push(statement(program, &sp, line));
                }
                ProgramUnit::Function {
                    name: name.to_string(),
                    return_type: *return_type,
                    params: params.to_vec(),
                    spec: sp,
                    body: stmts
                }
            },
            ProgramUnit::Subroutine {name, params, spec, body} => {
                let mut sp = Vec::<Specification<Expr>>::with_capacity(spec.len());
                for line in spec.iter() {
                    sp.push(specification(program, &sp, line));
                }
                let mut stmts = Vec::<Statement::<Expr>>::with_capacity(body.len());
                for line in body {
                    stmts.push(statement(program, &sp, line));
                }
                ProgramUnit::Subroutine {
                    name: name.to_string(),
                    params: params.to_vec(),
                    spec: sp,
                    body: stmts
                }
            },
            ProgramUnit::Empty => ProgramUnit::Empty,
        }
    }

    /*
    Disambiguate each program unit, returning a new `Program` object.
     */
    pub fn program(src: Program<parse_tree::Expr>) -> Program<super::Expr> {
        let main = program_unit(&src, &src.program);
        let mut fns = Vec::<ProgramUnit<super::Expr>>::with_capacity(src.functions.len());
        for fun in src.functions.iter() {
            fns.push(program_unit(&src, fun));
        }
        let mut subroutines = Vec::<ProgramUnit<Expr>>::with_capacity(src.subroutines.len());
        for sub in src.subroutines.iter() {
            subroutines.push(program_unit(&src, sub));
        }
        Program::<Expr> {
            program: main,
            functions: fns,
            subroutines
        }
    }


    mod tests {
        use super::*;

        /* van Loan and Coleman, pg 62 */
        #[test]
        fn subroutine_scale1() {
            let src = ["      subroutine scale1(c, m, n, B, bdim)",
                       "      INTEGER m, n, bdim",
                       "      REAL c, B(bdim, *)",
                       "      INTEGER i, j",
                       "C",
                       "C  Overwrites the m-by-n matrix with cB.",
                       "C  The array B has row dimension bdim.",
                       "C",
                       "      do 10 j = 1,n",
                       "        do 5 i=1,m",
                       "          B(i,j) = c*B(i,j)",
                       "   5    continue",
                       "  10  continue",
                       "       RETURN ",
                       "      end"].join("\n");
            let mut spec = Vec::<Specification<Expr>>::with_capacity(7);
            for var in ["m", "n", "bdim"] {
                spec.push(Specification::TypeDeclaration(
                    VarDeclaration {
                        kind: Type::Integer,
                        name: String::from(var),
                        array: ArraySpec::<Expr>::Scalar,
                    }
                ));
            }
            spec.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Real,
                    name: String::from("c"),
                    array: ArraySpec::<Expr>::Scalar,
                }
            ));
            let mut bdim_shape = Vec::<(Option<Expr>,Expr)>::with_capacity(1);
            bdim_shape.push((None, Expr::Variable(2_usize)));
            spec.push(Specification::TypeDeclaration(
                VarDeclaration {
                    kind: Type::Real,
                    name: String::from("B"),
                    array: ArraySpec::AssumedSize(bdim_shape, None),
                }
            ));
            for var in ["i", "j"] {
                spec.push(Specification::TypeDeclaration(
                    VarDeclaration {
                        kind: Type::Integer,
                        name: String::from(var),
                        array: ArraySpec::<Expr>::Scalar,
                    }
                ));
            }
            let mut inner_body = Vec::<Statement::<Expr>>::with_capacity(1);
            
            // rhs = c*B(i,j)
            let mut b_indices = Vec::<Expr>::with_capacity(2);
            b_indices.push(Expr::Variable(5_usize));
            b_indices.push(Expr::Variable(6_usize));
            let rhs = Expr::Binary(Box::new(Expr::Variable(3_usize)),
                                   BinOp::Times,
                                   Box::new(Expr::ArrayElement(4_usize, b_indices)));
            // lhs = B(i, j)
            let mut b_indices = Vec::<Expr>::with_capacity(2);
            b_indices.push(Expr::Variable(5_usize));
            b_indices.push(Expr::Variable(6_usize));
            let lhs = Expr::ArrayElement(4_usize, b_indices);
            inner_body.push(Statement::<Expr> {
                label: None,
                command: Command::Assignment { lhs, rhs },
            });
            let inner = Statement::<Expr> {
                label: None,
                command: Command::LabelDo {
                    target_label: 5,
                    var: Expr::Variable(5_usize),
                    start: Expr::Int64(1),
                    stop: Expr::Variable(0_usize),
                    stride: None,
                    body: inner_body,
                    terminal: Box::new(Statement {
                        label: Some(5),
                        command: Command::Continue,
                    })
                }
            };
            let mut outer_body = Vec::<Statement::<Expr>>::with_capacity(1);
            outer_body.push(inner);
            let outer = Statement::<Expr> {
                label: None,
                command: Command::LabelDo {
                    target_label: 10,
                    var: Expr::Variable(6_usize),
                    start: Expr::Int64(1),
                    stop: Expr::Variable(1_usize),
                    stride: None,
                    body: outer_body,
                    terminal: Box::new(Statement {
                        label: Some(10),
                        command: Command::Continue,
                    })
                }
            };
            let mut body = Vec::<Statement::<Expr>>::with_capacity(6);
            body.push(outer);
            body.push(Statement::<Expr> {
                label: None,
                command: Command::Return,
            });
            body.push(Statement::<Expr> {
                label: None,
                command: Command::End,
            });
            body.shrink_to_fit();
            let mut params = Vec::<String>::with_capacity(5); 
            for var in ["c", "m", "n", "B", "bdim"] {
                params.push(String::from(var));
            }
            let scale1 = ProgramUnit::<Expr>::Subroutine {
                name: String::from("scale1"),
                params,
                spec,
                body
            };
            let expected = Program {
                program: ProgramUnit::<Expr>::Empty,
                functions: Vec::<ProgramUnit::<Expr>>::new(),
                subroutines: vec![scale1]
            };
            // now ask the parser for what it gives with everything else
            let l = crate::lexer::Lexer::new(src.chars().collect());
            let mut parser = crate::parser::Parser::new(l);
            let actual = program(parser.parse_all());
            assert_eq!(actual, expected);
        }
    }
}
