      program drive_dahad
c Dump Delta_alpha_had^(5)(E) on a logarithmic grid in E (GeV).
c Reads from stdin: emin emax n  (energies in GeV; logarithmic spacing in |E|)
c                                 sign of E: positive => timelike
c Writes to stdout: E  der  errder
      implicit none
      integer i, n
      real*8 e, st2, der, errdersta, errdersys
      real*8 deg, errdegsta, errdegsys
      real*8 emin, emax, lemin, lemax, le
      st2 = 0.23153d0
      read(*,*) emin, emax, n
      lemin = log10(emin)
      lemax = log10(emax)
      do i = 0, n - 1
         le = lemin + (lemax - lemin) * dble(i) / dble(n - 1)
         e = 10.0d0 ** le
         call dhadr5x(e, st2, der, errdersta, errdersys,
     &        deg, errdegsta, errdegsys)
         write(*, '(F14.6, 1X, ES16.8, 1X, ES16.8)')
     &        e, der, sqrt(errdersta**2 + errdersys**2)
      enddo
      end
