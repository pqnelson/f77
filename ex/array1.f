C Should throw error because `wind` is an explicit shaped array with
C nonconstant bounds: `n` is not declared a constant until 2 lines later.
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