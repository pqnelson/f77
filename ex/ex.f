      program main
      real sgn
C do some calculations here
      stop
      end
      
      real function sgn(x)
      real x

      sgn = 0
      if (x .GT. 0) sgn = 1
      if (x .LT. 0) sgn = -1

      return
      end