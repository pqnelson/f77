This is a compiler for a large subset of Fortran 77, targeting a
hypothetical CPU assembly language. The output is a file containing
assembly code for the Fortran source code being compiled.

Right now, it's just a lexer for a fragment of Fortran 77. Soon I hope
to add a parser, then code optimizer, and finally a code generator.

# Subset of FORTRAN 77

For the subset of FORTRAN 77 supported, this is discussed in the [`notes/`](./notes/)
subdirectory documentation.

# But...Why? 

Why FORTRAN 77? Well, this is arguably some subset of fixed-form Fortran
90, but there are two reasons:

1. It is suitably simple enough to parse and compile, which means the
   possibility of having a bug in the compiler is small (compared to
   complicated languages like, say, Julia).
2. It's considered one of the top 10 algorithms of the 20th century
   according to [Computing in Science and Engineering](https://www.computer.org/csdl/magazine/cs/2000/01/c1022/13rRUxBJhBm)