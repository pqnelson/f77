This is a compiler for a large subset of Fortran 77, targeting a
hypothetical CPU assembly language. The output is a file containing
assembly code for the Fortran source code being compiled.

Right now, here's the progress:
- [X] Lexer
- [ ] Parser
  - [X] Expression parser
  - [ ] Statement parser
    - [X] GOTO
    - [X] Continue
    - [X] Read and write statements
    - [ ] If statement
    - [ ] Assignment
    - [ ] do-loop
    - [ ] Function call statement
    - [ ] Subroutine call statement
  - [ ] Program unit parser
- [ ] Code generation
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
   
# References

- Fortran 77 Standard [HTML](https://wg5-fortran.org/ARCHIVE/Fortran77.html)
  and [PDF](https://nvlpubs.nist.gov/nistpubs/Legacy/FIPS/fipspub69-1.pdf)
- Fortran 90 Standard [PDF](https://wg5-fortran.org/N001-N1100/N692.pdf) which has FORTRAN 77 as a subset
- [Jan Derricks' F90 Grammar](https://slebok.github.io/zoo/fortran/f90/derricks/extracted/index.html)
- William Waite and James Cordy's [Fortran 90 Grammar](https://slebok.github.io/zoo/fortran/f90/waite-cordy/extracted/index.html)
