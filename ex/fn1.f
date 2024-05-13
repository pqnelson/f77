C     gfortran fn1
C Then running ./a.out prints to the screen "1"      
      program fn1
      integer g
      g = 0
      call test(g)
C Substitution semantics says g = 2, copy/restore semantics says g = 1...
      write (*,*) g

      stop
      end

      subroutine test(x)
      integer x

      x = 1
C     At this point, x is not copied back into g, so g = 0 and x = 1
      x = g + 1
C     We have x = 2, g = 0.
C     But the subroutine ends here, so the value of x HERE is copied back
C     into g
      end
