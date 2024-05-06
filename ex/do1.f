C     Expected behaviour: should print out 'Done with loop, goodbye'
C
C     Compile with 'gfortran -fverbose-asm -S do1.f' to produce readable
C     assembly output.
      PROGRAM do1
      INTEGER i

C Starting do-loop     
      DO 10 i=1, 0
         WRITE (*,*) i
 10   CONTINUE

      WRITE (*,*) 'Done with loop, goodbye'
      
      STOP
      END