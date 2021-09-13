#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>

#define error(x)      {printf("\n\nError generating,creating or opening "x"\n\n");exit(-1);}
#define errorrc(x)    {printf("\n\nError reading %s\nMaybe file does not exist\n\n",x);exit(-1);}
#define errorwc(x)    {printf("\n\nError generating,creating or writing %s\n\n",x);exit(-1);}
#define CLOSEFILE(x)  {fclose(x); x = NULL;}
#define SQR(x)        ((x)*(x))
#define FREEP(x)      {free(x); x = NULL;}
#define sfsg          {printf("\n\n So far, so good...");getchar();printf("\n\n");}
#define RAND()        (2.*RCTE*rand()/DRAND-RCTE)

/****** global variables ******/

typedef long int LI;
typedef unsigned long int ULI;
double DRAND=(double)RAND_MAX;
double RCTE=.5*sqrt(12.);
// N[ x , 35]
static const long double TWOPI =  6.2831853071795864769252867665590058L;
static const long double PISQR =  9.8696044010893586188344909998761511L;
fftw_plan plan_fx_f, plan_fx_b;
fftw_plan plan_ux_f, plan_ux_b;

/****** functions ******/
double gauss_kernel(double k, double PIL2);
static inline void write_real1D_array(double *y, LI pid, LI N, LI numsteps,
	double L,	double nu, double f0, char axis);
static inline void gen_force3D(double *fx, fftw_complex *gx, double *ker,
	LI N, LI N2, double TPI3, double PIL2, double sqdx);
static inline void euler_maruyama_step(fftw_complex *vx, fftw_complex *gx,
	double *K2, LI N, LI N2, double dt, double sqdt, double visc, double f0);
static inline void predictor_corrector_step(
	fftw_complex *vx,	fftw_complex *gx, fftw_complex *tx,
	double *K2, LI N, LI N2, double dt, double sqdt, double visc, double f0);

int main(int argc, char **argv){

	LI i, it, N, N2, pid, numsteps;
  double TPI3, PIL2;
	// f and g are Fourier transform pairs
	// f is the external force, in real space
	// u and v are Fourier transform pairs
	// u is the velocity vector, in real space
	// t is a temp array, in Fourier space, used in predictor-corrector algorithm
	double *fx, *ux;
	double *K, *K2, *ker;
	fftw_complex *gx, *vx, *tx; /* arrays */
	double dx,sqdx,Ltot,L,dt,sqdt,nu,visc,f0,norm;
	// observables
	double *varf, *varx, *vark1, *varkN;

	if ( argc < 7 ){
    printf("Required arguments: seed N numsteps L nu f0 \n");
    exit(1);
  }

	pid = atoi(argv[1]); // process id, for ensemble average
	//time_t t;
  /* Intializes random number generator */
  srand((unsigned) 12345+pid);

	// Grid size
	N = (LI) 1<<(atoi(argv[2])); // 1<<N = 2^N
	N2 = (int)(N/2)+1;
	numsteps = (LI) atoi(argv[3]);

	Ltot = 1.;
	L = atof(argv[4])*Ltot;
	dx = Ltot/(double)N;
	sqdx = sqrt(dx); // StDev(dW_x) = dx^{dim/2}
	nu = atof(argv[5]);
	f0 = atof(argv[6]); // forcing amplitude

	// Simulation time
	// Time resolution must be roughly
	// dt = 0.1 dx^2 / (pi^2 * nu * Ltot^2)
	// So that every Fourier mode is well resolved
	dt = .1*dx*dx/(PISQR*nu*Ltot*Ltot);
	sqdt = sqrt(dt);
	visc = 4.*PISQR*nu;
	norm = 1./((double)(N));

	// size of time steps vs. dx^2
	//printf("dt = %.3f dx^2\n",dt/dx/dx);

	// Allocating necessary arrays
	if( (K = (double*) malloc(sizeof(double) * N)) == NULL)
		error("vector K");
	if( (K2 = (double*) malloc(sizeof(double) * N)) == NULL)
		error("vector K2");
	if( (ker = (double*) malloc(sizeof(double) * N)) == NULL)
		error("vector ker");
  if( (fx = (double*) fftw_malloc(sizeof(double) * N)) == NULL)
		error("vector fx");
	if( (gx = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N2 )) == NULL)
		error("vector gx");
	if( (ux = (double*) fftw_malloc(sizeof(double) * N)) == NULL)
		error("vector ux");
	if( (vx = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N2 )) == NULL)
		error("vector vx");
	if( (tx = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N2)) == NULL)
		error("vector tx");

	if( (varx = (double*) malloc(sizeof(double) * numsteps)) == NULL)
		error("vector varx");
	if( (varf = (double*) malloc(sizeof(double) * numsteps)) == NULL)
		error("vector varf");
	if( (vark1 = (double*) malloc(sizeof(double) * numsteps)) == NULL)
		error("vector vark1");
	if( (varkN = (double*) malloc(sizeof(double) * numsteps)) == NULL)
		error("vector varkN");

	/** initialize FFTW **/
	// Force vector transforms
	plan_fx_f = fftw_plan_dft_r2c_1d(N, fx, gx, FFTW_MEASURE);
	plan_fx_b = fftw_plan_dft_c2r_1d(N, gx, fx, FFTW_MEASURE);
	// Velocity vector transforms
	plan_ux_f = fftw_plan_dft_r2c_1d(N, ux, vx, FFTW_MEASURE);
	plan_ux_b = fftw_plan_dft_c2r_1d(N, vx, ux, FFTW_MEASURE);

	// Array of frequencies in Fourier space
  K[0]=0.0;
  K[N/2]=(double)(N/2)/Ltot;
	for(i=1;i<N/2;i++){
	  K[i]=(double)i/Ltot;
	  K[N-i]=-(double)i/Ltot;
  }

	for(i=0;i<N;i++){
	  K2[i]=SQR(K[i]);
  }

	/* correlation function of external force, at large scales
	   Fourier transform convention: {0,-2 Pi} (Mathematica)
	   Cf(x) = exp(-x^2/(2 L^2))
	   kernel = f_hat(k) = sqrt( F[ Cf(x) ] )
		 f_hat(k) = sqrt(sqrt(2 pi)) * sqrt(L) * exp(-pi^2 L^2 k^2)
	*/
	TPI3 = pow(TWOPI,0.25)*pow(L,0.5); // (2 pi)^(1/4) * L^(1/2)
	PIL2 = PISQR*L*L; // pi^2*L^2

	// Assign kernel operator directly in Fourier space
	// Imag. components are zero
	// frequencies are i/Ltot, normalization is eps = 1/Ltot
	for(i=0;i<N;i++){
		ker[i] = gauss_kernel(K[i],PIL2);
	}

	// set initial condition in Fourier space, v=0
	for(i=0;i<N2;i++){
		vx[i] = 0.;
	}

	for(it=0;it<numsteps;it++){
		varx[it] = 0.;
	}
	for(it=0;it<numsteps;it++){
		varf[it] = 0.;
	}
	for(it=0;it<numsteps;it++){
		vark1[it] = 0.;
	}
	for(it=0;it<numsteps;it++){
		varkN[it] = 0.;
	}

	for(it=0;it<numsteps;it++){

		// generate random force, correlated in space
		gen_force3D(fx,gx,ker,N,N2,TPI3,PIL2,sqdx);

		// time evolution is done in Fourier space only
		// this works for the heat equation

		//euler_maruyama_step(vx,gx,K2,N,N2,dt,sqdt,visc,f0);
		predictor_corrector_step(vx,gx,tx,K2,N,N2,dt,sqdt,visc,f0);

		// to verify that the variance of each fourier mode follows theory
		// 0 <= kx < N, 0 <= ky < N, 0 <= kz < N//2+1
		vark1[it] = SQR(cabs(vx[1]));
		varkN[it] = SQR(cabs(vx[15]));

		for(i=0;i<N2;i++)
			varf[it] += SQR(cabs(vx[i]));

		for(i=0;i<N2;i++)
			tx[i] = vx[i];

		// velocities back to real space
		fftw_execute(plan_ux_b);

		for(i=1;i<N;i++)
			varx[it] += SQR(ux[i]);

		for(i=0;i<N2;i++)
			vx[i] = tx[i];

	}

	for(it=0;it<numsteps;it++)
		varx[it] *= norm;

	write_real1D_array(varx,pid,N,numsteps,L,nu,f0,'x');
	write_real1D_array(varf,pid,N,numsteps,L,nu,f0,'f');
	write_real1D_array(vark1,pid,N,numsteps,L,nu,f0,'1');
	write_real1D_array(varkN,pid,N,numsteps,L,nu,f0,'N');

	fftw_destroy_plan(plan_fx_f);
  fftw_destroy_plan(plan_fx_b);
	fftw_destroy_plan(plan_ux_f);
  fftw_destroy_plan(plan_ux_b);
	fftw_free(fx);
	fftw_free(gx);
	fftw_free(ux);
	fftw_free(vx);
	fftw_free(tx);
	FREEP(ker);
	FREEP(K);
	FREEP(K2);
	FREEP(varx);
	FREEP(varf);
	FREEP(vark1);
	FREEP(varkN);

	printf("\n\n And we are done \n\n\a");

return 0;
}

double gauss_kernel(double k, double PIL2){
	return exp(-PIL2*k*k);
}

static inline void write_real1D_array(double *y, LI pid, LI N, LI numsteps,
	double L,	double nu, double f0, char axis){

  char name[100];
  FILE *fout;
	int BN = (int)log2(N);

  sprintf(name,"data/HeatVar_%c_R_%04ld_N_%02d_NT_%06ld_L_%.3e_nu_%.3e_f0_%.3e.dat",axis,pid,BN,numsteps,L,nu,f0);
  if( (fout = fopen(name,"w")) == NULL)
    errorwc(name);

  // binary output
  fwrite(y, sizeof(y[0]), numsteps, fout);
  // text output
	//for(i=0;i<N*N;i++)
  //  fprintf(f2, "%le \n",uy[i]);

	CLOSEFILE(fout);
}

static inline void gen_force3D(double *fx, fftw_complex *gx, double *ker,
	LI N, LI N2, double TPI3, double PIL2, double sqdx){

	LI i;
	double cte1;

	// Assign random vector dW (white noise increments)
	for(i=0;i<N;i++){
		fx[i] = RAND() * sqdx;
	}

	fftw_execute(plan_fx_f);

	// Tried to save memory, ended up with a large stride
	// Let's make it work, later I have to see what is best
	// I have to do this at every step, hence optimization will be
	// important for the program overall
	for(i=0;i<N2;i++){
		cte1 = TPI3 * ker[i];
		gx[i] *= cte1;
	}

	/*fftw_execute(plan_fx_b);*/

}

static inline void euler_maruyama_step(fftw_complex *vx, fftw_complex *gx,
	double *K2, LI N, LI N2, double dt, double sqdt, double visc, double f0){

	LI i;
	double viscdt = dt*visc;
	double f0sqdt = sqrt(f0*dt);

	// deterministic evolution
	for(i=0;i<N2;i++){
		vx[i] -= viscdt * K2[i] * vx[i];
	}

	// add stochastic force
	for(i=0;i<N2;i++){
		vx[i] += f0sqdt * gx[i];
	}

}

static inline void predictor_corrector_step(
	fftw_complex *vx,	fftw_complex *gx, fftw_complex *tx,
	double *K2, LI N, LI N2, double dt, double sqdt, double visc, double f0){
// order 1.0 predictor corrector algorithm
// see Kloeden-Platen p. 502

	LI i;
	double viscdt = visc*dt;
	double f0sqdt = sqrt(f0*dt);

	// predictor step
	// t* are temp arrays, to store predictor array

	// initial setup of predictor step
	for(i=0;i<N2;i++){
		tx[i] = vx[i];
	}

	// deterministic evolution
	for(i=0;i<N2;i++){
		tx[i] -= viscdt * K2[i] * vx[i];
	}

	// add stochastic force
	for(i=0;i<N2;i++){
		tx[i] += f0sqdt * gx[i];
	}

	// corrector step

	// deterministic evolution, half step using last velocity array
	for(i=0;i<N2;i++){
		vx[i] -= .5 * viscdt * K2[i] * vx[i];
	}

	// deterministic evolution, half step using predictor array
	for(i=0;i<N2;i++){
		vx[i] -= .5 * viscdt * K2[i] * tx[i];
	}

	// add stochastic force
	for(i=0;i<N2;i++){
		vx[i] += f0sqdt * gx[i];
	}

}
