c =====================================================================
c  wuppvi.f  --  IO-free Fortran cores for Wu/Davis piecewise PV inversion
c
c  Extracted VERBATIM (numerics unchanged) from the validated v0.8.0
c  serial Gauss-Seidel solvers in pv_inversion/wu/fortran/:
c     pvpialln_94UV.f  -> PVPIALLN_CORE  (psi,Q,THB,THT from H,T,U,V)
c     qinvert21_94.f   -> QINVERT_CORE   (total balanced inversion, BALNC)
c     qinvertp21_94.f  -> QINVERTP_CORE  (piecewise perturbation, BALP)
c
c  Only the PROGRAM shell (stdin reads, file OPEN/READ/WRITE, the
c  multi-halfday averaging loop) was removed.  All arithmetic, stencils,
c  SOR loops and constants are byte-identical to the validated baseline.
c
c  Build (HPC, micromamba `blocking` env, NOT part of the PyPI wheel):
c     f2py -c -m _wuppvi wuppvi.f \
c          --backend meson \
c          --f77flags='-std=legacy -O2 -frecursive'
c  Compile flags: -frecursive (thread-safe automatic locals for fork
c  safety), NO -fno-automatic, NO -fopenmp.  Numerics IDENTICAL to v0.8.0.
c =====================================================================


c ---------------------------------------------------------------------
c  PVPIALLN_CORE
c  Compute streamfunction PSI, Ertel-like PV Q, and boundary potential
c  temperatures THB/THT from geopotential height H, RAW temperature TT,
c  and winds U,V on the Wu pressure grid.
c
c  NOTE: TT must be RAW temperature (K); the routine converts to theta
c  internally via TH = TT*CP/PI(K) (single Exner), exactly as the
c  original program did after reading the .grid file.
c ---------------------------------------------------------------------
      SUBROUTINE PVPIALLN_CORE(H, TT, U, V, HDR, PR, IMAX, OMEGS, THRS,
     +                         PSI, Q, THB, THT, NY, NX, NW)

      INTEGER NY, NX, NW
      REAL H(NY,NX,NW), TT(NY,NX,NW), U(NY,NX,NW), V(NY,NX,NW)
      REAL HDR(10), PR(NW)
      INTEGER IMAX
      REAL OMEGS, THRS
      REAL PSI(NY,NX,NW), Q(NY,NX,NW), THB(NY,NX), THT(NY,NX)

cf2py intent(in) H, TT, U, V, HDR, PR, IMAX, OMEGS, THRS
cf2py intent(out) PSI, Q, THB, THT
cf2py integer intent(hide),depend(H) :: NY=shape(H,0)
cf2py integer intent(hide),depend(H) :: NX=shape(H,1)
cf2py integer intent(hide),depend(H) :: NW=shape(H,2)

c     -- local automatic arrays (require -frecursive) --
      REAL TH(NY,NX,NW), VOR(NY,NX,NW), STB(NY,NX,NW)
      REAL DTHX(NY,NX,NW), DTHY(NY,NX,NW)
      REAL DU(NY,NX,NW), DV(NY,NX,NW)
      REAL FC(NY), AP(NY), APM(NY), APP(NY), A(NY,5), PI(NW)
      REAL PII, CP, P0, KAP, MI, AA, GG, SIGM, DL, DP, PIB, PIT
      REAL VL, UPV, DSUM, LAPSI, RS, PSIN, DPSI, ZSHR, COEF
      INTEGER I, J, K, L, NYI, NXI, ICOUNT
      LOGICAL IT

      NYI = NY
      NXI = NX

      PII = 4.*ATAN(1.)
      CP  = 1004.5
      P0  = 1.E5
      KAP = 2./7.
      MI  = 9999.90
      AA  = 2.E7/PII
      GG  = 9.8066
      SIGM = HDR(5)/HDR(6)

      DL = AA*HDR(5)*PII/180.
      DP = AA*HDR(6)*PII/180.
      DO K=1,NW
       PI(K)=CP*( PR(K)**KAP )
      END DO
      PIB=0.5*(PI(2)+PI(1))
      PIT=0.5*(PI(NW)+PI(NW-1))

      DO I=1,NYI
       AP(I)=COS( PII*( HDR(3) - (I-1)*HDR(6) )/180. )
       APM(I)=COS( PII*( HDR(3) - (I-1.5)*HDR(6) )/180. )
       APP(I)=COS( PII*( HDR(3) - (I-0.5)*HDR(6) )/180. )
       A(I,1)=SIGM*SIGM*APM(I)/AP(I)
       A(I,2)=1./( AP(I)*AP(I) )
       A(I,3)=-( 2. + SIGM*SIGM*AP(I)*(APM(I)+APP(I)) )/
     +    ( AP(I)*AP(I) )
       A(I,4)=1./( AP(I)*AP(I) )
       A(I,5)=SIGM*SIGM*APP(I)/AP(I)
       FC(I)=(1.458E-4)*SIN( PII*( HDR(3) - (I-1)*HDR(6) )/180. )
      END DO

c     -- missing-value flags on PV lateral boundaries --
      DO K=2,NW-1
       DO I=1,NYI
        Q(I,1,K)=MI
        Q(I,NXI,K)=MI
       END DO
       DO J=1,NXI
        Q(1,J,K)=MI
        Q(NYI,J,K)=MI
       END DO
      END DO

c     -- initial psi = geostrophic H*g/f --
      DO K=1,NW
       DO I=1,NYI
        DO J=1,NXI
         PSI(I,J,K)=H(I,J,K)*GG/FC(I)
        END DO
       END DO
      END DO

c     -- raw T -> theta (single Exner) --
      DO K=1,NW
       DO I=1,NYI
        DO J=1,NXI
         TH(I,J,K)=TT(I,J,K)*CP/PI(K)
        END DO
       END DO
      END DO

c     -- boundary potential temperatures --
      DO J=1,NXI
       DO I=1,NYI
        THB(I,J)=0.5*(TH(I,J,1) + TH(I,J,2))
        THT(I,J)=0.5*(TH(I,J,NW-1) + TH(I,J,NW))
       END DO
      END DO

c     -- theta gradients and static stability --
      DO K=2,NW-1
       DO J=2,NXI-1
        DO I=2,NYI-1
         DTHY(I,J,K)=(TH(I-1,J,K)-TH(I+1,J,K))/(2.*DP)
         DTHX(I,J,K)=(TH(I,J+1,K)-TH(I,J-1,K))/(2.*AP(I)*DL)
         STB(I,J,K)=(TH(I,J,K+1)-TH(I,J,K-1))/
     +     (PI(K+1)-PI(K-1))
        END DO
       END DO
      END DO

c     -- relative vorticity --
      DO K=1,NW
       DO I=2,NYI-1
       DO J=2,NXI-1
        VL=(V(I,J+1,K)-V(I,J-1,K))/(2.*DL*AP(I))
        UPV=(AP(I-1)*U(I-1,J,K)-AP(I+1)*U(I+1,J,K))/
     +       (2.*DP*AP(I))
        VOR(I,J,K)=VL - UPV
       END DO
       END DO
      END DO

c     -- boundary psi by line integration (Davis 2.40) --
      DO K=1,NW
       DSUM=0.
       DO I=1,NYI-1
        DSUM=DSUM -((U(I,1,K)+U(I+1,1,K))/2.)* DP
        DSUM=DSUM +((U(I,NXI,K)+U(I+1,NXI,K))/2.)* DP
       END DO
       DO J=1,NXI-1
        DSUM=DSUM+((V(1,J,K)+V(1,J+1,K))/2.)* DL * AP(1)
        DSUM=DSUM-((V(NYI,J,K)+V(NYI,J+1,K))/2.)
     *                 *DL*AP(NYI)
       END DO
       DSUM = DSUM /( 2.*DP*(NYI-1)+ DL*(NXI-1)
     *               *(AP(1)+AP(NYI))  )

       DO I=1,NYI-1
        PSI(I+1,1,K)=PSI(I,1,K)+(DSUM+((U(I  ,1,K)
     *                         +        U(I+1,1,K) )/2.))* DP
       END DO
       DO J=1,NXI-1
        PSI(NYI,J+1,K)=PSI(NYI,J,K)+
     *    ( DSUM+((V(NYI,J,K)+V(NYI,J+1,K))/2.) )
     *                 *DL*AP(NYI)
       END DO
       DO I=NYI,2,-1
        PSI(I-1,NXI,K)=PSI(I,NXI,K)
     *     +(DSUM-((U(I,NXI,K)+U(I-1,NXI,K) )/2.))* DP
       END DO
       DO J=NXI,3,-1
        PSI(1,J-1,K)=PSI(1,J,K)+
     *    ( DSUM-((V(1,J,K)+V(1,J-1,K))/2.) )
     *                 *DL*AP(1)
       END DO
      END DO

c     -- interior psi by SOR (label 155) --
      ICOUNT=0
155   IT=.TRUE.
      DO K=1,NW
      DO I=2,NYI-1
      DO J=2,NXI-1
       LAPSI=1./(DL*DL)*( A(I,1)*PSI(I-1,J,K) + A(I,2)*PSI(I,J-1,K)+
     +      A(I,3)*PSI(I,J,K) + A(I,4)*PSI(I,J+1,K) +
     +      A(I,5)*PSI(I+1,J,K) )
       RS = LAPSI - VOR(I,J,K)
       PSIN=PSI(I,J,K)
       PSI(I,J,K)= PSIN-OMEGS*RS/A(I,3)*(DL*DL)
       DPSI= PSI(I,J,K)-PSIN
       IF (ABS(DPSI).GT.THRS) THEN
          IT= .FALSE.
       END IF
      END DO
      END DO
      END DO
      ICOUNT= ICOUNT + 1
      IF (ICOUNT.GT.IMAX) GO TO 123
      IF (IT) THEN
         GO TO 123
      ELSE
         GO TO 155
      END IF
123   CONTINUE

c     -- vertical wind shear --
      DO K=2,NW-1
       DO I=2,NYI-1
       DO J=2,NXI-1
        DU(I,J,K)=(U(I,J,K+1)-U(I,J,K-1))/(PI(K+1)-PI(K-1))
        DV(I,J,K)=(V(I,J,K+1)-V(I,J,K-1))/(PI(K+1)-PI(K-1))
       END DO
       END DO
      END DO

c     -- potential vorticity --
      COEF=1.E2*1.E6*9.81*KAP*(CP**3.5)/P0
      DO L=2,NW-1
       DO J=2,NXI-1
       DO I=2,NYI-1
        ZSHR=COEF*(PI(L)**(-2.5))*( DU(I,J,L)*DTHY(I,J,L)
     +       - DV(I,J,L)*DTHX(I,J,L) )
        Q(I,J,L)=-COEF*(PI(L)**(-2.5))*( (FC(I)+VOR(I,J,L))*
     +        STB(I,J,L) ) - ZSHR
       END DO
       END DO
      END DO

      RETURN
      END


c ---------------------------------------------------------------------
c  QINVERT_CORE  (from qinvert21_94.f, INF=1, IMAP=1 path)
c  Total-field balanced PV inversion.  Takes the pvpialln initial guess
c  (HZ in metres, SI in 1e5 m^2/s units), the Wu PV field Q and boundary
c  theta THT, performs the same nondimensionalisation as the original
c  main program, calls BALNC (verbatim solver), and returns the balanced
c  HZ, SI descaled back to metres / 1e5 m^2/s.
c ---------------------------------------------------------------------
      SUBROUTINE QINVERT_CORE(HZIN, SIIN, QIN, THTIN, ZHDR, PR,
     +   MAXIT, MAXOT, OMEGAS, OMEGAH, PRT, THRIN, QMIN, HOUT, SOUT,
     +   NY, NX)

      PARAMETER (NL=9)
      INTEGER NY, NX
      REAL HZIN(NY,NX,NL), SIIN(NY,NX,NL), QIN(NY,NX,NL)
      REAL THTIN(NY,NX,2), ZHDR(10), PR(NL)
      INTEGER MAXIT, MAXOT
      REAL OMEGAS, OMEGAH, PRT, THRIN, QMIN
      REAL HOUT(NY,NX,NL), SOUT(NY,NX,NL)

cf2py intent(in) HZIN, SIIN, QIN, THTIN, ZHDR, PR
cf2py intent(in) MAXIT, MAXOT, OMEGAS, OMEGAH, PRT, THRIN, QMIN
cf2py intent(out) HOUT, SOUT
cf2py integer intent(hide),depend(HZIN) :: NY=shape(HZIN,0)
cf2py integer intent(hide),depend(HZIN) :: NX=shape(HZIN,1)

      REAL Q(NY,NX,NL), HZ(NY,NX,NL), SI(NY,NX,NL), THT(NY,NX,2)
      REAL AP(NY), APP(NY), APM(NY), A(NY,5), PI(NL), PIF(NL)
      REAL FC(NY,NX), MF(NY,NX), FM(NY,NX), THZB(NL), QDIF(NL)
      INTEGER QL(NL)
      REAL PII, DPI, AA, CP, R, FF, KAP, P0, MI, GG, SIGM, LL
      REAL THO, FRC, QCONST, BETA, THR, QNEW
      INTEGER I, J, K

      PII=4.*ATAN(1.)
      DPI=500./FLOAT(NL)
      AA=2.E7/PII
      CP=1004.5
      R=2.*CP/7.
      FF=1.E-4
      KAP=R/CP
      P0=1.E5
      MI=9999.90
      GG=9.81
      SIGM=ZHDR(5)/ZHDR(6)
      LL=AA*ZHDR(5)*PII/180.
      DO 101 I=1,NY
       AP(I)=COS( PII*(ZHDR(3) - (FLOAT(I)-1.)*ZHDR(6))/180. )
       APM(I)=COS( PII*(ZHDR(3) - (FLOAT(I)-1.5)*ZHDR(6))/180. )
       APP(I)=COS( PII*(ZHDR(3) - (FLOAT(I)-0.5)*ZHDR(6))/180. )
       A(I,1)=SIGM*SIGM*APM(I)/AP(I)
       A(I,2)=1./( AP(I)*AP(I) )
       A(I,3)=-( 2. + SIGM*SIGM*AP(I)*(APM(I)+APP(I)) )/
     +    ( AP(I)*AP(I) )
       A(I,4)=1./( AP(I)*AP(I) )
       A(I,5)=SIGM*SIGM*APP(I)/AP(I)
       DO 104 J=1,NX
        FC(I,J)=1.458*SIN( PII*(ZHDR(3) - (FLOAT(I)-1)*ZHDR(6))/180. )
        MF(I,J)=1.
        FM(I,J)=FC(I,J)/MF(I,J)
 104   CONTINUE
 101  CONTINUE
      THO=FF*FF*LL*LL/DPI
      FRC=DPI*THO/(FF*FF*LL*LL)
      QCONST=1.E6*KAP*GG*CP*FF*THO/(P0*DPI)
      BETA=1.458*LL/AA
      THR=GG*THRIN/(DPI*THO)
      DO 202 K=1,NL
       PI(K)=CP*( PR(K)**KAP )
       PIF(K)=(PI(K)/CP)**2.5
       PI(K)=PI(K)/DPI
 202  CONTINUE

c     -- initial guess (INF=1), nondim --
      DO 221 K=1,NL
       THZB(K)=0.
       DO 221 I=1,NY
       DO 221 J=1,NX
        HZ(I,J,K)=HZIN(I,J,K)*GG/(THO*DPI)
        SI(I,J,K)=SIIN(I,J,K)*GG/(THO*DPI)
 221  CONTINUE

c     -- boundary theta nondim --
      DO 251 K=1,2
       DO 251 I=1,NY
       DO 251 J=1,NX
        THT(I,J,K)=THTIN(I,J,K)/THO
 251  CONTINUE

c     -- PV nondim + QMIN floor + QDIF redistribution --
      DO 261 K=1,NL
       DO 261 I=1,NY
       DO 261 J=1,NX
        Q(I,J,K)=QIN(I,J,K)
 261  CONTINUE
      DO 271 K=2,NL-1
       QL(K)=0
       QDIF(K)=0.
       QNEW=PIF(K)*QMIN/QCONST
       DO 272 J=1,NX
       DO 273 I=1,NY
        IF (Q(I,J,K).NE.MI) THEN
         Q(I,J,K)=PIF(K)*Q(I,J,K)/(MF(I,J)*1.E2*QCONST)
         IF (Q(I,J,K).LE.QNEW) THEN
          QDIF(K)=QDIF(K) + QNEW - Q(I,J,K)
          QL(K)=QL(K) + 1
          Q(I,J,K)=QNEW
         END IF
        END IF
 273   CONTINUE
 272   CONTINUE
       QDIF(K)=QDIF(K)/( FLOAT(NY*NX - QL(K)) )
       DO 275 J=1,NX
       DO 276 I=1,NY
        IF (Q(I,J,K).GT.QNEW+QDIF(K)) THEN
         Q(I,J,K)=Q(I,J,K) - QDIF(K)
        END IF
 276   CONTINUE
 275   CONTINUE
 271  CONTINUE

      CALL BALNC(FC,FM,MF,AP,A,HZ,SI,Q,THT,SIGM,PI,PRT,THO,
     +  THZB,THR,BETA,FRC,MAXIT,MAXOT,QCONST,OMEGAS,OMEGAH,NY,NX)

      DO 501 K=1,NL
       DO 501 I=1,NY
       DO 501 J=1,NX
        HOUT(I,J,K)=HZ(I,J,K)*DPI*THO/GG
        SOUT(I,J,K)=SI(I,J,K)*DPI*THO/GG
 501  CONTINUE

      RETURN
      END


c ---------------------------------------------------------------------
c  BALNC  --  verbatim from qinvert21_94.f (PRINT/WRITE diagnostics
c  removed; all arithmetic and control flow unchanged).
c ---------------------------------------------------------------------
      SUBROUTINE BALNC(FCO,FCM,MFC,APS,AC,H,S,QE,THA,SIG,PE,
     +  PART,THSC,THSB,THRS,BET,FR,MAXX,MAXXT,QCON,OMEGS,OMEGH,NY,NX)
      PARAMETER (NL=9)
      INTEGER NY, NX
      REAL  H(NY,NX,NL),QE(NY,NX,NL),SIG,S(NY,NX,NL),
     +  MI,ZM,THA(NY,NX,2),RH(NY,NX,NL),AC(NY,5),PE(NL),
     +  ZPL(NY),ZPP(NY),BET,FR,FCO(NY,NX),MFC(NY,NX),THSB(NL),
     +  APS(NY),RS,GPTS,STB(NY,NX,NL),ASI(NY,NX,NL),FCM(NY,NX),
     +  BB(NL),BH(NL),BL(NL),VOR,PART,OLD(NY,NX,NL),THRS,THSC,
     +  DPI2(NL),NLCO(NY,NX),COEF(NY,NL),DH(NY,NX,NL),QCON,
     +  ZNL,RHS(NY,NX,NL),DELH(NY,NX,NL),DSI(NY,NX,NL),
     +  OMEGS,OMEGH
      INTEGER MAXX,MAXXT
      LOGICAL IT,ICON
      MI=9999.90
      GPTS=FLOAT(NY*NX*(NL-2))
      ILTZ=0
      IGTZ=0
      INSQ=0
      ITCT=0
      ITC1=0
      ITC=0
      IITOT=0
      VOZRO=0.
      ITCOLD=0.
      DSIMAX=100.
      DO 201 I=2,NY-1
       ZPL(I)=FR/(16.*APS(I)*APS(I))
       ZPP(I)=FR*SIG*SIG/16.
       DO 2011 J=2,NX-1
        NLCO(I,J)=2.*FR*MFC(I,J)*SIG*SIG/( APS(I)*APS(I) )
 2011  CONTINUE
 201  CONTINUE

      DO 202 K=2,NL-1
       BB(K)=-2./( (PE(K+1)-PE(K))*(PE(K)-PE(K-1)) )
       BH(K)=2./( (PE(K+1)-PE(K))*(PE(K+1)-PE(K-1)) )
       BL(K)=2./( (PE(K)-PE(K-1))*(PE(K+1)-PE(K-1)) )
       DPI2(K)=(PE(K+1) - PE(K-1))/2.
       DO 203 I=1,NY
        COEF(I,K)=AC(I,3)/BB(K)
 203   CONTINUE
 202  CONTINUE
      DO 204 K=1,NL
       DO 205 J=1,NX
       DO 206 I=1,NY
        OLD(I,J,K)=H(I,J,K)
 206   CONTINUE
 205   CONTINUE
 204  CONTINUE

 900  CONTINUE
      IF (IITOT.EQ.0) GOTO 700
      ITCM=0
      ITC1=0
      SPZRO=0
      DO 210 K=2,NL-1
      DO 211 J=2,NX-1
      DO 212 I=2,NY-1
       OLD(I,J,K)=S(I,J,K)
       IF (K.EQ.2) THEN
        OLD(I,J,1)=S(I,J,1)
       ELSE IF (K.EQ.NL-1) THEN
        OLD(I,J,NL)=S(I,J,NL)
       END IF
       DELH(I,J,K)= AC(I,1)*H(I-1,J,K) + AC(I,2)*H(I,J-1,K) +
     +    AC(I,3)*H(I,J,K) + AC(I,4)*H(I,J+1,K) +
     +    AC(I,5)*H(I+1,J,K)
       STB(I,J,K)=BL(K)*H(I,J,K-1) + BH(K)*H(I,J,K+1) +
     +   BB(K)*H(I,J,K)
       IF (STB(I,J,K).LE.0.0001) THEN
        STB(I,J,K)=0.0001
        SPZRO=SPZRO + 1
       END IF
       SXX=S(I,J+1,K)+S(I,J-1,K)-2.*S(I,J,K)
       SYY=S(I-1,J,K)+S(I+1,J,K)-2.*S(I,J,K)
       SXY=( S(I-1,J+1,K)-S(I-1,J-1,K)-
     +    S(I+1,J+1,K)+S(I+1,J-1,K) )/4.
       BETAS=0.25*( SIG*SIG*(FCO(I-1,J)-FCO(I+1,J))*
     +    (S(I-1,J,K)-S(I+1,J,K)) + (FCO(I,J+1)-FCO(I,J-1))*
     +    (S(I,J+1,K)-S(I,J-1,K)) )
       ZHP=H(I-1,J,K+1)-H(I+1,J,K+1)-H(I-1,J,K-1)+H(I+1,J,K-1)
       ZHL=H(I,J+1,K+1)-H(I,J-1,K+1)-H(I,J+1,K-1)+H(I,J-1,K-1)
       ZSP=S(I-1,J,K+1)-S(I+1,J,K+1)-S(I-1,J,K-1)+S(I+1,J,K-1)
       ZSL=S(I,J+1,K+1)-S(I,J-1,K+1)-S(I,J+1,K-1)+S(I,J-1,K-1)
       ZL=ZPL(I)*ZHL*ZSL/(DPI2(K)*DPI2(K))
       ZP=ZPP(I)*ZHP*ZSP/(DPI2(K)*DPI2(K))
       ZNL=NLCO(I,J)*( SXX*SYY - SXY*SXY ) + BETAS
       RHST=QE(I,J,K) - FCM(I,J)*STB(I,J,K) + DELH(I,J,K) -
     +    ZNL + ZL + ZP
       RHS(I,J,K)=RHST/(FCO(I,J) + FR*STB(I,J,K))
 212  CONTINUE
 211  CONTINUE
 210  CONTINUE
      SPZRO=0.
      DO 220 K=2,NL-1
      ITC=0
 800  IT=.TRUE.
       DO 221 J=2,NX-1
       DO 222 I=2,NY-1
        RS=AC(I,1)*S(I-1,J,K) + AC(I,2)*S(I,J-1,K) +
     +    AC(I,3)*S(I,J,K) + AC(I,4)*S(I,J+1,K) +
     +    AC(I,5)*S(I+1,J,K) - RHS(I,J,K)
        DSI(I,J,K)=-OMEGS*RS/AC(I,3)
        S(I,J,K) = S(I,J,K) + DSI(I,J,K)
        IF (ABS(DSI(I,J,K)).GT.THRS) THEN
         IT=.FALSE.
        END IF
 222   CONTINUE
 221   CONTINUE
      ITC=ITC+1
      IF (IT) THEN
       ICON=.TRUE.
       IF (ITC.GT.ITCM) ITCM=ITC
       IF (ITC.EQ.1) ITC1=ITC1 + 1
      ELSE
       IF (ITC.LT.MAXX) THEN
        GO TO 800
       ELSE
        ICON=.TRUE.
       END IF
      END IF
 802  CONTINUE
 220  CONTINUE

      IF (IITOT.GT.0) THEN
      ITCT=ITCT + ITCM
      DO 230 K=1,NL
       DO 231 J=2,NX-1
       DO 232 I=2,NY-1
        S(I,J,K)=PART*S(I,J,K) + (1.-PART)*OLD(I,J,K)
        OLD(I,J,K)=H(I,J,K)
 232   CONTINUE
 231   CONTINUE
 230  CONTINUE
      END IF

 700  DO 240 K=2,NL-1
       DO 241 J=2,NX-1
       DO 242 I=2,NY-1
        VOR=AC(I,1)*S(I-1,J,K) + AC(I,2)*S(I,J-1,K) +
     +    AC(I,3)*S(I,J,K) + AC(I,4)*S(I,J+1,K) +
     +    AC(I,5)*S(I+1,J,K)
        IF (VOR*FR.LE.0.0001-FCM(I,J)) THEN
         VOR=(0.0001 - FCM(I,J))/FR
         VOZRO=VOZRO + 1
        END IF
        ASI(I,J,K)=FCM(I,J) + FR*VOR
        SXX=S(I,J+1,K)+S(I,J-1,K)-2.*S(I,J,K)
        SYY=S(I-1,J,K)+S(I+1,J,K)-2.*S(I,J,K)
        SXY=( S(I-1,J+1,K)-S(I-1,J-1,K)-S(I+1,J+1,K)+
     +      S(I+1,J-1,K) )/4.
        ZHP=H(I-1,J,K+1)-H(I+1,J,K+1)-H(I-1,J,K-1)+H(I+1,J,K-1)
        ZHL=H(I,J+1,K+1)-H(I,J-1,K+1)-H(I,J+1,K-1)+H(I,J-1,K-1)
        ZSP=S(I-1,J,K+1)-S(I+1,J,K+1)-S(I-1,J,K-1)+S(I+1,J,K-1)
        ZSL=S(I,J+1,K+1)-S(I,J-1,K+1)-S(I,J+1,K-1)+S(I,J-1,K-1)
        ZL=ZPL(I)*ZHL*ZSL/(DPI2(K)*DPI2(K))
        ZP=ZPP(I)*ZHP*ZSP/(DPI2(K)*DPI2(K))
        BETAS=0.25*( SIG*SIG*(FCO(I-1,J)-FCO(I+1,J))*
     +    (S(I-1,J,K)-S(I+1,J,K)) + (FCO(I,J+1)-FCO(I,J-1))*
     +    (S(I,J+1,K)-S(I,J-1,K)) )
        RHA=FCO(I,J)*VOR + NLCO(I,J)*(SXX*SYY - SXY*SXY) + BETAS
        RH(I,J,K)=RHA + QE(I,J,K) + ZL + ZP
 242   CONTINUE
 241   CONTINUE
 240  CONTINUE
      VOZRO=0.

      ITC=0
 701  IT=.TRUE.
      ZMRS=0.
      DO 250 K=2,NL-1
      DO 2511 J=2,NX-1
      DO 2521 I=2,NY-1
       IF (K.EQ.2) THEN
        RS=AC(I,1)*H(I-1,J,K) + AC(I,2)*H(I,J-1,K) +
     +   ( AC(I,3) + ASI(I,J,K)*(BB(K)+BL(K)) )*H(I,J,K) +
     +    AC(I,4)*H(I,J+1,K) + AC(I,5)*H(I+1,J,K) +
     +    ASI(I,J,K)*(BH(K)*H(I,J,K+1) + THA(I,J,1)/DPI2(K))
     +    - RH(I,J,K)
        ZM=H(I,J,K)
        H(I,J,K)=ZM - OMEGH*RS/(AC(I,3) + ASI(I,J,K)*(BB(K)+BL(K)))
       ELSE IF (K.EQ.NL-1) THEN
        RS=AC(I,1)*H(I-1,J,K) + AC(I,2)*H(I,J-1,K) +
     +   ( AC(I,3) + ASI(I,J,K)*(BB(K)+BH(K)) )*H(I,J,K) +
     +    AC(I,4)*H(I,J+1,K) + AC(I,5)*H(I+1,J,K) +
     +    ASI(I,J,K)*(BL(K)*H(I,J,K-1) - THA(I,J,2)/DPI2(K))
     +    - RH(I,J,K)
        ZM=H(I,J,K)
        H(I,J,K)=ZM - OMEGH*RS/(AC(I,3) + ASI(I,J,K)*(BB(K)+BH(K)))
       ELSE
        RS=AC(I,1)*H(I-1,J,K) + AC(I,2)*H(I,J-1,K) +
     +   ( AC(I,3) + ASI(I,J,K)*BB(K) )*H(I,J,K) +
     +    AC(I,4)*H(I,J+1,K) + AC(I,5)*H(I+1,J,K) +
     +    ASI(I,J,K)*( BH(K)*H(I,J,K+1) + BL(K)*H(I,J,K-1) ) -
     +    RH(I,J,K)
        ZM=H(I,J,K)
        H(I,J,K)=ZM - OMEGH*RS/(AC(I,3) + ASI(I,J,K)*BB(K))
       END IF
       DH(I,J,K)=H(I,J,K) - ZM
       ZMRS=ZMRS + ABS(DH(I,J,K))
       IF (ABS(DH(I,J,K)).GT.THRS) THEN
        IT=.FALSE.
       END IF
 2521 CONTINUE
 2511 CONTINUE
 250  CONTINUE
      ZMRS=0.
      ITC=ITC+1
      IF (IT) THEN
       ITCT=ITCT + ITC
       DO 260 J=1,NX
       DO 261 I=1,NY
        H(I,J,1)=H(I,J,2) + THA(I,J,1)*(PE(2)-PE(1))
        S(I,J,1)=S(I,J,2) + (THA(I,J,1)-THSB(1))*(PE(2)-PE(1))
        H(I,J,NL)=H(I,J,NL-1) - THA(I,J,2)*(PE(NL)-PE(NL-1))
        S(I,J,NL)=S(I,J,NL-1) - (THA(I,J,2)-THSB(NL-1))*
     +       (PE(NL)-PE(NL-1))
 261   CONTINUE
 260   CONTINUE
       IF (IITOT.GT.0) THEN
       DO 263 K=1,NL
        DO 264 J=2,NX-1
        DO 265 I=2,NY-1
         H(I,J,K)=PART*H(I,J,K) + (1.-PART)*OLD(I,J,K)
 265    CONTINUE
 264    CONTINUE
 263   CONTINUE
       END IF
       IF ( (ITC.GT.ITCOLD+10).AND.(IITOT.GT.30) ) THEN
        GO TO 901
       END IF
       ITCOLD=ITC
       IF ((ITC.EQ.1).AND.(ITC1.EQ.NL-2)) THEN
        CONTINUE
       ELSE
        IITOT=IITOT + 1
        IF (IITOT.GT.MAXXT) THEN
         GO TO 901
        ELSE
         GO TO 900
        END IF
       END IF
      ELSE
       IF (ITC.LT.MAXX) THEN
        GO TO 701
       ELSE
        ICON=.FALSE.
        GO TO 901
       END IF
      END IF

 901  RETURN
      END


c ---------------------------------------------------------------------
c  QINVERTP_CORE  (from qinvertp21_94.f, INLIN-aware, IMAP=1 path)
c  Piecewise PV-perturbation inversion for ONE vertical piece.
c  Replicates the qinvertp21 main-program nondimensionalisation
c  (HND=DPI*THO/GG, perturbation formation MP=MP-MB, half-pert mean
c  redefinition, Q nondim with QMIN floor) for ONE inversion (NOUT=1),
c  then calls BALP (verbatim solver).  Returns HP, SP for ALL NL levels
c  descaled back to metres (x HND).
c
c  Inputs (all NY,NX,NL unless noted):
c    MBIN  - mean geopotential height [m]            (meanh, H block)
c    SBIN  - mean streamfunction      [psi/1e5]      (meanh, SI block)
c    MPIN  - total geopotential height [m]           (event_bal.out, H)
c    SIPIN - total streamfunction     [psi/1e5]      (event_bal.out, SI)
c    QBIN  - mean Wu PV    (K=2..NL-1 valid, MI else) (meanq, Q block)
c    QPIN  - total Wu PV   (K=2..NL-1 valid, MI else) (event_q.out, Q)
c    THBIN(NY,NX,2) - mean boundary theta  [K]        (meanq, THB/THT)
c    THPIN(NY,NX,2) - total boundary theta [K]        (event_q.out)
c    ZHDR(8), PR(NL)
c    QLV(NL), NMLV  - this piece's level list and length
c    OMEGAS,OMEGAH,PRT,THRSH,TSCAL,QSCAL - SOR / scale params
c    INLIN (1=nonlinear coeffs), IBC (0=Dirichlet, 1=full-pert BC,
c      2=lateral boundary values + initial guess from HP0IN/SP0IN)
c    HP0IN, SP0IN (NY,NX,NL) - IBC=2 boundary/guess fields in the
c      qinvertp OUTPUT units (HP0IN metres, SP0IN psi/1e5): the
c      boundary ring supplies the fixed lateral Dirichlet values, the
c      interior the SOR first guess (the array analogue of the
c      original qinvertp21 IGFIL nesting file). Ignored unless IBC=2.
c  Outputs:
c    HPOUT, SPOUT (NY,NX,NL)  - balanced piece geopotential / streamfn
c ---------------------------------------------------------------------
      SUBROUTINE QINVERTP_CORE(MBIN, SBIN, MPIN, SIPIN, QBIN, QPIN,
     +   THBIN, THPIN, ZHDR, PR, QLV, NMLV, OMEGAS, OMEGAH, PRT,
     +   THRSH, TSCAL, QSCAL, INLIN, IBC, HP0IN, SP0IN,
     +   HPOUT, SPOUT, NY, NX)

      PARAMETER (NL=9)
      INTEGER NY, NX
      REAL MBIN(NY,NX,NL), SBIN(NY,NX,NL), MPIN(NY,NX,NL)
      REAL SIPIN(NY,NX,NL), QBIN(NY,NX,NL), QPIN(NY,NX,NL)
      REAL THBIN(NY,NX,2), THPIN(NY,NX,2), ZHDR(8), PR(NL)
      INTEGER QLV(NL), NMLV, INLIN, IBC
      REAL OMEGAS, OMEGAH, PRT, THRSH, TSCAL, QSCAL
      REAL HP0IN(NY,NX,NL), SP0IN(NY,NX,NL)
      REAL HPOUT(NY,NX,NL), SPOUT(NY,NX,NL)

cf2py intent(in) MBIN, SBIN, MPIN, SIPIN, QBIN, QPIN
cf2py intent(in) THBIN, THPIN, ZHDR, PR, QLV, NMLV
cf2py intent(in) OMEGAS, OMEGAH, PRT, THRSH, TSCAL, QSCAL, INLIN, IBC
cf2py intent(in) HP0IN, SP0IN
cf2py intent(out) HPOUT, SPOUT
cf2py integer intent(hide),depend(MBIN) :: NY=shape(MBIN,0)
cf2py integer intent(hide),depend(MBIN) :: NX=shape(MBIN,1)

      REAL QP(NY,NX,NL), QB(NY,NX,NL), MP(NY,NX,NL), MB(NY,NX,NL)
      REAL SIP(NY,NX,NL), SB(NY,NX,NL), THB(NY,NX,2), THP(NY,NX,2)
      REAL AP(NY), APP(NY), APM(NY), A(NY,5), PI(NL), PIF(NL)
      REAL FC(NY,NX), MF(NY,NX), FM(NY,NX)
      REAL PII, AA, DPI, CP, R, FF, SIGM, KAP, PO, MI, GG, LL
      REAL THO, FRC, QCONST, BETA, HND, QMIN, CNLIN
      INTEGER MAXIT, MAXOT, I, J, K
      QMIN=0.00001
c     -- Iteration caps raised 10x (2026-08-06). Verified bit-identical on a real per-event
c        inversion: a case that converges exits via IT=.TRUE. and never reads MAXIT again,
c        so only cases that were HITTING the cap change. Original: MAXIT=2000, MAXOT=400.
      MAXIT=20000
      MAXOT=4000

      PII=4.*ATAN(1.)
      AA=2.E7/PII
      DPI=500./FLOAT(NL)
      CP=1004.5
      R=2.*CP/7.
      FF=1.E-4
      SIGM=ZHDR(5)/ZHDR(6)
      KAP=R/CP
      PO=1.E5
      MI=9999.90
      GG=9.81
      LL=AA*ZHDR(5)*PII/180.
      DO 101 I=1,NY
       AP(I)=COS( PII*(ZHDR(3) - (FLOAT(I)-1.)*ZHDR(6))/180. )
       APM(I)=COS( PII*(ZHDR(3) - (FLOAT(I)-1.5)*ZHDR(6))/180. )
       APP(I)=COS( PII*(ZHDR(3) - (FLOAT(I)-0.5)*ZHDR(6))/180. )
       A(I,1)=SIGM*SIGM*APM(I)/AP(I)
       A(I,2)=1./( AP(I)*AP(I) )
       A(I,3)=-( 2. + SIGM*SIGM*AP(I)*(APM(I)+APP(I)) )/
     +    ( AP(I)*AP(I) )
       A(I,4)=1./( AP(I)*AP(I) )
       A(I,5)=SIGM*SIGM*APP(I)/AP(I)
       DO 104 J=1,NX
        FC(I,J)=1.458*SIN( PII*(ZHDR(3) - (FLOAT(I)-1)*ZHDR(6))/180. )
        MF(I,J)=1.
        FM(I,J)=FC(I,J)/MF(I,J)
 104   CONTINUE
 101  CONTINUE
      THO=FF*FF*LL*LL/DPI
      FRC=DPI*THO/(FF*FF*LL*LL)
      QCONST=1.E6*KAP*GG*CP*FF*THO/(DPI*PO)
      BETA=1.458*LL/AA
      HND=DPI*THO/GG
      THRSH=THRSH/HND

      DO 204 K=1,NL
       PI(K)=CP*( PR(K)**KAP )/DPI
       PIF(K)=(DPI*PI(K)/CP)**2.5
 204  CONTINUE

c ----- copy boundary theta and PV into work arrays -------------------
      DO 210 K=1,2
       DO 211 J=1,NX
        DO 212 I=1,NY
         THB(I,J,K)=THBIN(I,J,K)
         THP(I,J,K)=THPIN(I,J,K)
 212    CONTINUE
 211   CONTINUE
 210  CONTINUE
      DO 213 K=1,NL
       DO 214 J=1,NX
        DO 215 I=1,NY
         MB(I,J,K)=MBIN(I,J,K)
         SB(I,J,K)=SBIN(I,J,K)
         MP(I,J,K)=MPIN(I,J,K)
         SIP(I,J,K)=SIPIN(I,J,K)
         QB(I,J,K)=QBIN(I,J,K)
         QP(I,J,K)=QPIN(I,J,K)
 215    CONTINUE
 214   CONTINUE
 213  CONTINUE

c ----- nondim boundary theta (main loop 250) ------------------------
      DO 250 K=1,2
       DO 251 I=1,NY
        DO 252 J=1,NX
         THB(I,J,K)=THB(I,J,K)/THO
         THP(I,J,K)=TSCAL*THP(I,J,K)/THO - THB(I,J,K)
 252    CONTINUE
 251   CONTINUE
 250  CONTINUE

c ----- nonlinear-term option (main) ----------------------------------
      IF (INLIN.EQ.1) THEN
       CNLIN=0.5
      ELSE
       CNLIN=0.
      END IF

c ----- nondim height & streamfunction, form perturbations (loop 270) -
      DO 270 K=1,NL
       DO 271 J=1,NX
        DO 272 I=1,NY
         MP(I,J,K)=MP(I,J,K)/HND
         SIP(I,J,K)=SIP(I,J,K)/HND
         MB(I,J,K)=MB(I,J,K)/HND
         SB(I,J,K)=SB(I,J,K)/HND
         MP(I,J,K)=MP(I,J,K)-MB(I,J,K)
         SIP(I,J,K)=SIP(I,J,K)-SB(I,J,K)
         MB(I,J,K)=MB(I,J,K) + CNLIN*MP(I,J,K)
         SB(I,J,K)=SB(I,J,K) + CNLIN*SIP(I,J,K)
 272    CONTINUE
 271   CONTINUE
 270  CONTINUE

c ----- nondim PV perturbation with QMIN floor (loop 280) -------------
      DO 280 K=2,NL-1
       DO 281 J=1,NX
        DO 282 I=1,NY
         IF (QP(I,J,K).EQ.MI) THEN
          QP(I,J,K)=0.
          GO TO 282
         END IF
         IF (QP(I,J,K).LT.1.E2*QMIN) THEN
          QP(I,J,K)=PIF(K)*QMIN/QCONST
         ELSE
          QP(I,J,K)=PIF(K)*QP(I,J,K)/(1.E2*QCONST)
         END IF
         IF (QB(I,J,K).LT.1.E2*QMIN) THEN
          QB(I,J,K)=PIF(K)*QMIN/QCONST
         ELSE
          QB(I,J,K)=PIF(K)*QB(I,J,K)/(1.E2*QCONST)
         END IF
         QP(I,J,K)=(QP(I,J,K)-QB(I,J,K))/MF(I,J)
 282    CONTINUE
 281   CONTINUE
 280  CONTINUE

c ----- solve one piece ----------------------------------------------
      CALL BALP(MP, MB, SIP, SB, THP, QP, FC, FM, MF, AP, A, PI,
     +   HND, OMEGAS, OMEGAH, PRT, THRSH, BETA, FRC, MAXIT, MAXOT,
     +   SIGM, QLV, NMLV, IBC, HP0IN, SP0IN, HPOUT, SPOUT, NY, NX)

      RETURN
      END


c ---------------------------------------------------------------------
c  BALP  (verbatim from qinvertp21_94.f, IO + diagnostics stripped)
c  Piecewise perturbation SOR solver.  NOUT=1 (one piece per call);
c  QLV/NMLV/IBC passed as arguments; all NL levels returned in HPOUT,
c  SPOUT (x HNDM).  SISUM/HTSUM explicitly zeroed for -frecursive
c  fork-safety (the original relied on static zero-init).
c ---------------------------------------------------------------------
      SUBROUTINE BALP(H, HB, S, SBR, TPR, Q, FCO, FCM, MFC, APS, AC,
     +   PE, HNDM, OMEGS, OMEGH, PART, THRS, BET, FR, MAXX, MAXXT,
     +   SIG, QLV, NMLV, IBC, HP0, SP0, HPOUT, SPOUT, NY, NX)
      PARAMETER (NL=9)
      INTEGER NY, NX
      REAL  H(NY,NX,NL),HB(NY,NX,NL),HP(NY,NX,NL),MI,PART,
     +   SBR(NY,NX,NL),S(NY,NX,NL),SP(NY,NX,NL),Q(NY,NX,NL),SIG,
     +   ZM,AC(NY,5),SLL(NY,NX,NL),SPP(NY,NX,NL),SLP(NY,NX,NL),
     +   STB(NY,NX,NL),AVO(NY,NX,NL),RHS(NY,NX,NL),QP(NY,NX,NL),
     +   ZNC(NY,NX),BET,FR,FCO(NY,NX),TPR(NY,NX,2),
     +   APS(NY),RS,ASI(NY,NX,NL),BSI(NY,NX,NL),APHI(NY,NX,NL),
     +   HRHS(NY,NX,NL),SRHS(NY,NX,NL),TP(NY,NX,2),OS(NY,NX,NL),
     +   OH(NY,NX,NL),BB(NL),PE(NL),BH(NL),BL(NL),DPI2(NL),HNDM,
     +   SISUM(NY,NX,NL),HTSUM(NY,NX,NL),FCM(NY,NX),MFC(NY,NX)
      REAL HP0(NY,NX,NL), SP0(NY,NX,NL)
      REAL HPOUT(NY,NX,NL), SPOUT(NY,NX,NL)
      INTEGER QLV(NL),GPTS,NMLV,IBC
      LOGICAL IT,ICON
      REAL BI,RSA,SXX,SYY,SXY,BETAS,ZSI,ZMRS
      REAL R1BS,R1BH,R2BS,R2BH,R1PS,R1PH,R2PS,R2PH
      REAL RH1,RH2
      INTEGER I,J,K,KL,IITOT,ITC,ITCC
      MI=9999.90
      GPTS=NY*NX*NL

c ----- zero the running accumulators (static zero-init replacement) --
      DO 150 K=1,NL
       DO 151 J=1,NX
        DO 152 I=1,NY
         SISUM(I,J,K)=0.
         HTSUM(I,J,K)=0.
 152    CONTINUE
 151   CONTINUE
 150  CONTINUE

C********** Set coefficients ********************************
      DO 200 I=2,NY-1
       DO 201 J=2,NX-1
        ZNC(I,J)=2.*FR*SIG*SIG*MFC(I,J)/(APS(I)*APS(I))
 201   CONTINUE
 200  CONTINUE
      DO 202 K=2,NL-1
       BB(K)=-2./( (PE(K+1)-PE(K))*(PE(K)-PE(K-1)) )
       BH(K)=2./( (PE(K+1)-PE(K))*(PE(K+1)-PE(K-1)) )
       BL(K)=2./( (PE(K+1)-PE(K-1))*(PE(K)-PE(K-1)) )
       DPI2(K)=(PE(K+1)-PE(K-1))/2.
       DO 203 J=2,NX-1
        DO 2204 I=2,NY-1
         SLL(I,J,K)=ZNC(I,J)*(SBR(I,J+1,K)+SBR(I,J-1,K)-2.*SBR(I,J,K))
         SPP(I,J,K)=ZNC(I,J)*(SBR(I-1,J,K)+SBR(I+1,J,K)-2.*SBR(I,J,K))
         AVO(I,J,K)=FCM(I,J) + FR*( AC(I,1)*SBR(I-1,J,K) +
     +      AC(I,2)*SBR(I,J-1,K) + AC(I,3)*SBR(I,J,K) +
     +      AC(I,4)*SBR(I,J+1,K) + AC(I,5)*SBR(I+1,J,K) )
         SLP(I,J,K)=ZNC(I,J)*(SBR(I-1,J+1,K)-SBR(I-1,J-1,K)-
     +      SBR(I+1,J+1,K)+SBR(I+1,J-1,K))/2.
         STB(I,J,K)=BH(K)*HB(I,J,K+1) + BL(K)*HB(I,J,K-1) +
     +      BB(K)*HB(I,J,K)
         IF ( AVO(I,J,K).LT.0.001 ) THEN
          AVO(I,J,K)=0.01
         END IF
         IF (STB(I,J,K).LT.0.001) THEN
          STB(I,J,K)=0.01
         END IF
 2204   CONTINUE
 203   CONTINUE
       DO 206 J=2,NX-1
        DO 207 I=2,NY-1
         ASI(I,J,K)=BB(K)*AVO(I,J,K)/(FR*STB(I,J,K)*AC(I,3))
         BSI(I,J,K)=1. + ASI(I,J,K)*FCO(I,J)
         BI=FCO(I,J)*AC(I,3) - 2.*(SLL(I,J,K) + SPP(I,J,K))
         IF (BI.GT.0.) THEN
          BI=0.
         END IF
         APHI(I,J,K)=BI/(FR*STB(I,J,K)*AC(I,3))
 207    CONTINUE
 206   CONTINUE
 202  CONTINUE

C******* Initialize fields, set boundary conditions *******
      DO 220 J=1,NX
       DO 221 I=1,NY
        TP(I,J,1)=0.
        TP(I,J,2)=0.
        DO 222 K=2,NL-1
         QP(I,J,K)=0.
 222    CONTINUE
 221   CONTINUE
 220  CONTINUE

      IF (NMLV.EQ.NL) THEN
       DO 230 J=1,NX
        DO 231 I=1,NY
         TP(I,J,1)=TPR(I,J,1)
         TP(I,J,2)=TPR(I,J,2)
         DO 232 K=2,NL-1
          QP(I,J,K)=Q(I,J,K)
 232     CONTINUE
 231    CONTINUE
 230   CONTINUE
       GO TO 114
      END IF

      DO 240 KL=1,NMLV
       IF (QLV(KL).EQ.1) THEN
        DO 241 J=1,NX
         DO 242 I=1,NY
          TP(I,J,1)=TPR(I,J,1)
 242     CONTINUE
 241    CONTINUE
       ELSE IF (QLV(KL).EQ.NL) THEN
        DO 243 J=1,NX
         DO 244 I=1,NY
          TP(I,J,2)=TPR(I,J,2)
 244     CONTINUE
 243    CONTINUE
       END IF
       DO 245 K=2,NL-1
        IF (K.EQ.QLV(KL)) THEN
         DO 246 J=1,NX
          DO 247 I=1,NY
           QP(I,J,K)=Q(I,J,K)
 247      CONTINUE
 246     CONTINUE
        END IF
 245   CONTINUE
 240  CONTINUE

c     IBC=2: lateral boundary values + first guess from HP0/SP0, the
c     array analogue of the qinvertp21_94 IGFIL nesting file (which is
c     read as HP=HP/HNDM, SP=SP/HNDM). HP0 is in metres and SP0 in
c     psi/1e5 -- the same units BALP writes HPOUT/SPOUT in, so an
c     outer-domain piece solution feeds back verbatim. The SOR updates
c     interior points only, so the ring stays fixed at these values.
 114  IF (IBC.EQ.2) THEN
       DO 264 K=1,NL
        DO 265 J=1,NX
         DO 266 I=1,NY
          HP(I,J,K)=HP0(I,J,K)/HNDM
          SP(I,J,K)=SP0(I,J,K)/HNDM
 266     CONTINUE
 265    CONTINUE
 264   CONTINUE
      ELSE IF (IBC.EQ.1) THEN
       DO 254 K=1,NL
        DO 255 J=1,NX
         DO 256 I=1,NY
          HP(I,J,K)=H(I,J,K) - HTSUM(I,J,K)
          SP(I,J,K)=S(I,J,K) - SISUM(I,J,K)
 256     CONTINUE
 255    CONTINUE
 254   CONTINUE
      ELSE
       DO 257 K=1,NL
        DO 258 J=1,NX
         DO 259 I=1,NY
          HP(I,J,K)=0.
          SP(I,J,K)=0.
 259     CONTINUE
 258    CONTINUE
 257   CONTINUE
      END IF

C********* Calculate upper and lower initial boundary values *********
      DO 260 J=1,NX
       DO 261 I=1,NY
        HP(I,J,1)=HP(I,J,2) + TP(I,J,1)*(PE(2)-PE(1))
        SP(I,J,1)=SP(I,J,2) + TP(I,J,1)*(PE(2)-PE(1))
        HP(I,J,NL)=HP(I,J,NL-1) - TP(I,J,2)*(PE(NL)-PE(NL-1))
        SP(I,J,NL)=SP(I,J,NL-1) - TP(I,J,2)*(PE(NL)-PE(NL-1))
 261   CONTINUE
 260  CONTINUE

C*********** Begin iterations ***************************
      IITOT=0
      ITC=0
 900  CONTINUE

C********** Calculate the RHS of the psi equation *********
      DO 2270 K=1,NL
       DO 2271 J=1,NX
        DO 2272 I=1,NY
         OS(I,J,K)=SP(I,J,K)
         OH(I,J,K)=HP(I,J,K)
 2272   CONTINUE
 2271  CONTINUE
 2270 CONTINUE

      DO 2280 K=2,NL-1
       DO 2281 J=2,NX-1
        DO 2282 I=2,NY-1
         R1BS=( SBR(I,J+1,K+1)-SBR(I,J-1,K+1)-SBR(I,J+1,K-1)+
     +     SBR(I,J-1,K-1) )/(4.*DPI2(K))
         R1BH=( HB(I,J+1,K+1)-HB(I,J-1,K+1)-HB(I,J+1,K-1)+
     +     HB(I,J-1,K-1) )/(4.*DPI2(K))
         R2BS=( SBR(I-1,J,K+1)-SBR(I+1,J,K+1)-SBR(I-1,J,K-1)+
     +     SBR(I+1,J,K-1) )/(4.*DPI2(K))
         R2BH=( HB(I-1,J,K+1)-HB(I+1,J,K+1)-HB(I-1,J,K-1)+
     +     HB(I+1,J,K-1) )/(4.*DPI2(K))
         R1PS=( SP(I,J+1,K+1)-SP(I,J-1,K+1)-SP(I,J+1,K-1)+
     +     SP(I,J-1,K-1) )/(4.*DPI2(K))
         R1PH=( HP(I,J+1,K+1)-HP(I,J-1,K+1)-HP(I,J+1,K-1)+
     +     HP(I,J-1,K-1) )/(4.*DPI2(K))
         R2PS=( SP(I-1,J,K+1)-SP(I+1,J,K+1)-SP(I-1,J,K-1)+
     +     SP(I+1,J,K-1) )/(4.*DPI2(K))
         R2PH=( HP(I-1,J,K+1)-HP(I+1,J,K+1)-HP(I-1,J,K-1)+
     +     HP(I+1,J,K-1) )/(4.*DPI2(K))
         RHS(I,J,K)=QP(I,J,K) + FR*( (R1BS*R1PH + R1BH*R1PS)/
     +     (APS(I)*APS(I)) + SIG*SIG*(R2BS*R2PH + R2BH*R2PS) )
         SRHS(I,J,K)=( RHS(I,J,K) - AVO(I,J,K)*(BH(K)*HP(I,J,K+1)+
     +     BL(K)*HP(I,J,K-1)) )/(FR*STB(I,J,K)) + ASI(I,J,K)*
     +     ( AC(I,1)*HP(I-1,J,K) + AC(I,2)*HP(I,J-1,K) + AC(I,4)*
     +      HP(I,J+1,K) + AC(I,5)*HP(I+1,J,K) )
 2282   CONTINUE
 2281  CONTINUE
 2280 CONTINUE

C************* Iteration for psi (2-D) **********************
      ITCC=0
      DO 2290 K=2,NL-1
       ITC=0
 800   IT=.TRUE.
       DO 2291 J=2,NX-1
        DO 2292 I=2,NY-1
         RSA=AC(I,1)*SP(I-1,J,K) + AC(I,2)*
     +    SP(I,J-1,K) + AC(I,3)*SP(I,J,K) +
     +    AC(I,4)*SP(I,J+1,K) + AC(I,5)*SP(I+1,J,K)
         SXX=SP(I,J+1,K)+SP(I,J-1,K) -2.*SP(I,J,K)
         SYY=SP(I-1,J,K)+SP(I+1,J,K)-2.*SP(I,J,K)
         SXY=( SP(I-1,J+1,K)-SP(I+1,J+1,K)-SP(I-1,J-1,K)+
     +     SP(I+1,J-1,K) )/4.
         BETAS=SIG*SIG*(FCO(I-1,J)-FCO(I+1,J))*
     +    (SP(I-1,J,K)-SP(I+1,J,K))/4. + (FCO(I,J+1)-FCO(I,J-1))*
     +    (SP(I,J+1,K)-SP(I,J-1,K))/4.
         RS=BSI(I,J,K)*RSA + ASI(I,J,K)*( BETAS + SLL(I,J,K)*SYY +
     +      SPP(I,J,K)*SXX - SLP(I,J,K)*SXY ) - SRHS(I,J,K)
         ZSI=SP(I,J,K)
         SP(I,J,K)=ZSI - OMEGS*RS/( BSI(I,J,K)*AC(I,3) -
     +     2.*ASI(I,J,K)*(SLL(I,J,K) + SPP(I,J,K)) )
         IF (ABS(SP(I,J,K)-ZSI).GT.THRS) THEN
          IT=.FALSE.
         END IF
 2292   CONTINUE
 2291  CONTINUE
       ITC=ITC+1
       IF (IT) THEN
        ICON=.TRUE.
        IF (ITC.GT.1) ITCC=1
       ELSE
        IF (ITC.LT.MAXX) THEN
         GO TO 800
        ELSE
         ICON=.FALSE.
         GO TO 901
        END IF
       END IF
 2290 CONTINUE

      IF (IITOT.GT.0) THEN
       DO 2300 K=1,NL
        DO 2301 J=2,NX-1
         DO 2302 I=2,NY-1
          SP(I,J,K)=PART*SP(I,J,K) + (1.-PART)*OS(I,J,K)
 2302    CONTINUE
 2301   CONTINUE
 2300  CONTINUE
      END IF

C********* Calculate the RHS of the PV+BALANCE equation *******
 700  DO 2310 K=2,NL-1
       DO 2311 J=2,NX-1
        DO 2312 I=2,NY-1
         RH1=(2./AC(I,3))*(SLL(I,J,K) + SPP(I,J,K))*
     +      ( AC(I,1)*SP(I-1,J,K) + AC(I,2)*SP(I,J-1,K) +
     +      AC(I,4)*SP(I,J+1,K) + AC(I,5)*SP(I+1,J,K) )
         BETAS=SIG*SIG*(FCO(I-1,J)-FCO(I+1,J))*
     +    (SP(I-1,J,K)-SP(I+1,J,K))/4. + (FCO(I,J+1)-FCO(I,J-1))*
     +    (SP(I,J+1,K)-SP(I,J-1,K))/4.
         RH2=BETAS + SLL(I,J,K)*(SP(I-1,J,K)+SP(I+1,J,K)) +
     +     SPP(I,J,K)*(SP(I,J-1,K)+SP(I,J+1,K)) -
     +     SLP(I,J,K)*(SP(I-1,J+1,K)-SP(I-1,J-1,K)-
     +      SP(I+1,J+1,K)+SP(I+1,J-1,K))/4.
         HRHS(I,J,K)=APHI(I,J,K)*RHS(I,J,K) + RH1 + RH2
 2312   CONTINUE
 2311  CONTINUE
 2310 CONTINUE

C************* Solve for phi with 3-D SOR *****************
      ITC=0
 701  IT=.TRUE.
      DO 2320 K=2,NL-1
       DO 2321 J=2,NX-1
        DO 2322 I=2,NY-1
         IF (K.EQ.2) THEN
          RS=AC(I,1)*HP(I-1,J,K) + AC(I,2)*HP(I,J-1,K) +
     +     ( AC(I,3) + APHI(I,J,K)*(BB(K)+BL(K))*AVO(I,J,K) )*
     +      HP(I,J,K) + AC(I,4)*HP(I,J+1,K) + AC(I,5)*
     +      HP(I+1,J,K) + APHI(I,J,K)*AVO(I,J,K)*( BH(K)*
     +       HP(I,J,K+1) + TP(I,J,1)/DPI2(K) ) - HRHS(I,J,K)
          ZM=HP(I,J,K)
          HP(I,J,K)=ZM - OMEGH*RS/( AC(I,3) + APHI(I,J,K)*
     +        (BB(K)+BL(K))*AVO(I,J,K) )
         ELSE IF (K.EQ.NL-1) THEN
          RS=AC(I,1)*HP(I-1,J,K) + AC(I,2)*HP(I,J-1,K) +
     +     ( AC(I,3) + APHI(I,J,K)*(BB(K)+BH(K))*AVO(I,J,K) )*
     +      HP(I,J,K) + AC(I,4)*HP(I,J+1,K) + AC(I,5)*
     +      HP(I+1,J,K) + APHI(I,J,K)*AVO(I,J,K)*( BL(K)*
     +       HP(I,J,K-1) - TP(I,J,2)/DPI2(K) ) - HRHS(I,J,K)
          ZM=HP(I,J,K)
          HP(I,J,K)=ZM - OMEGH*RS/( AC(I,3) + APHI(I,J,K)*
     +        (BB(K)+BH(K))*AVO(I,J,K) )
         ELSE
          RS=AC(I,1)*HP(I-1,J,K) + AC(I,2)*HP(I,J-1,K) +
     +     ( AC(I,3) + APHI(I,J,K)*BB(K)*AVO(I,J,K) )*
     +      HP(I,J,K) + AC(I,4)*HP(I,J+1,K) + AC(I,5)*HP(I+1,J,K) +
     +      APHI(I,J,K)*AVO(I,J,K)*( BH(K)*HP(I,J,K+1) +
     +       BL(K)*HP(I,J,K-1) ) - HRHS(I,J,K)
          ZM=HP(I,J,K)
          HP(I,J,K)=ZM-OMEGH*RS/( AC(I,3) +
     +       BB(K)*APHI(I,J,K)*AVO(I,J,K) )
         END IF
         IF (ABS(ZM-HP(I,J,K)).GT.THRS/2.) THEN
          IT=.FALSE.
         END IF
 2322   CONTINUE
 2321  CONTINUE
 2320 CONTINUE

      ITC=ITC+1
      IF (IT) THEN
       DO 2330 J=1,NX
        DO 2331 I=1,NY
         HP(I,J,1)=HP(I,J,2) + TP(I,J,1)*(PE(2)-PE(1))
         SP(I,J,1)=SP(I,J,2) + TP(I,J,1)*(PE(2)-PE(1))
         HP(I,J,NL)=HP(I,J,NL-1) - TP(I,J,2)*(PE(NL)-PE(NL-1))
         SP(I,J,NL)=SP(I,J,NL-1) - TP(I,J,2)*(PE(NL)-PE(NL-1))
 2331   CONTINUE
 2330  CONTINUE
       IF (IITOT.GT.0) THEN
        DO 2340 K=1,NL
         DO 2341 J=2,NX-1
          DO 2342 I=2,NY-1
           HP(I,J,K)=PART*HP(I,J,K) + (1.-PART)*OH(I,J,K)
 2342     CONTINUE
 2341    CONTINUE
 2340   CONTINUE
       END IF
       IF ((ITC.EQ.1).AND.(ITCC.EQ.0)) THEN
        CONTINUE
       ELSE
        IITOT=IITOT + 1
        IF (IITOT.GT.MAXXT) THEN
         GO TO 901
        ELSE
         GO TO 900
        END IF
       END IF
      ELSE
       IF (ITC.LT.MAXX) THEN
        GO TO 701
       ELSE
        ICON=.FALSE.
        GO TO 901
       END IF
      END IF

C********** Write out phi and psi fields ***********************
 901  DO 2350 K=1,NL
       DO 2351 J=1,NX
        DO 2352 I=1,NY
         SISUM(I,J,K)=SISUM(I,J,K) + SP(I,J,K)
         HTSUM(I,J,K)=HTSUM(I,J,K) + HP(I,J,K)
         HPOUT(I,J,K)=HP(I,J,K)*HNDM
         SPOUT(I,J,K)=SP(I,J,K)*HNDM
 2352   CONTINUE
 2351  CONTINUE
 2350 CONTINUE
      RETURN
      END
