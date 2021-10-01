// $ make heat_jentzen_1d
// $ ./heat_jentzen_1d.x
// $ time seq 0 99 | xargs -I{} -P 7 ./heat_jentzen_1d.x {} 9 1000000 .1 .01 1. >/dev/null
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#define error(x)      {printf("\n\nError generating,creating or opening "x"\n\n");exit(-1);}
#define errorrc(x)    {printf("\n\nError reading %s\nMaybe file does not exist\n\n",x);exit(-1);}
#define errorwc(x)    {printf("\n\nError generating,creating or writing %s\n\n",x);exit(-1);}
#define CLOSEFILE(x)  {fclose(x); x = NULL;}
#define SQR(x)        ((x)*(x))
#define FREEP(x)      {free(x); x = NULL;}
#define sfsg          {printf("\n\n So far, so good...");getchar();printf("\n\n");}

/****** global variables ******/

typedef long int LI;
typedef unsigned long int ULI;
static const long double TWOPI =  6.2831853071795864769252867665590058L;
static const long double PISQR =  9.8696044010893586188344909998761511L;
fftw_plan plan_fx_f, plan_fx_b;
fftw_plan plan_ux_f, plan_ux_b;

/****** functions ******/
double gauss_kernel(double k, double PIL2);
double ltqnorm(double p);
static inline void write_real1D_array(double *y, LI pid, LI N, LI numsteps,
	double L,	double nu, double f0, char axis);
static inline void gen_force3D(double *fx, fftw_complex *gx, double *K,
	LI N, LI N2, double TPI3, double PIL2, double sqdx, gsl_rng *rng);
static inline void jentzen_kloeden_winkel_step(fftw_complex *vx, fftw_complex *gx,
	double *K, LI N2, double sqdx, double dt, double visc, double f0, double TPI3, double PIL2);

int main(int argc, char **argv){

	LI i, it, N, N2, pid, numsteps;
  double TPI3, PIL2;
	// f and g are Fourier transform pairs
	// f is the external force, in real space
	// u and v are Fourier transform pairs
	// u is the velocity vector, in real space
	// t is a temp array, in Fourier space, used in predictor-corrector algorithm
	double *fx, *ux;
	double *K;
	fftw_complex *gx, *vx; /* arrays */
	double dx,sqdx,Ltot,L,dt,nu,visc,f0,norm;
	// observables
	double *varf, *varx, *vark1, *varkN, *vardv;
	gsl_rng *rng = gsl_rng_alloc(gsl_rng_mt19937); // Mersenne Twister

	if ( argc < 7 ){
    printf("Required arguments: seed N numsteps L nu f0 \n");
    exit(1);
  }

	pid = atoi(argv[1]); // process id, for ensemble average
	// Initializes random number generator
  gsl_rng_set(rng,12345+pid);

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
	dt =   .5*dx*dx/(PISQR*nu*Ltot*Ltot);
	visc = 4.*PISQR*nu;
	norm = 1./((double)(N));

	// Allocating necessary arrays
	if( (K = (double*) malloc(sizeof(double) * N)) == NULL)
		error("vector K");
	if( (gx = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N2 )) == NULL)
		error("vector gx");
	if( (vx = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N2 )) == NULL)
		error("vector vx");

	if( (varx = (double*) malloc(sizeof(double) * numsteps)) == NULL)
		error("vector varx");
	if( (varf = (double*) malloc(sizeof(double) * numsteps)) == NULL)
		error("vector varf");
	if( (vark1 = (double*) malloc(sizeof(double) * numsteps)) == NULL)
		error("vector vark1");
	if( (varkN = (double*) malloc(sizeof(double) * numsteps)) == NULL)
		error("vector varkN");
	if( (vardv = (double*) malloc(sizeof(double) * numsteps)) == NULL)
		error("vector vardv");

	fx = (double *) gx;
	ux = (double *) vx;

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

	/* correlation function of external force, at large scales
	   Fourier transform convention: {0,-2 Pi} (Mathematica)
	   Cf(x) = exp(-x^2/(2 L^2))
	   kernel = f_hat(k) = sqrt( F[ Cf(x) ] )
		 f_hat(k) = sqrt(sqrt(2 pi)) * sqrt(L) * exp(-pi^2 L^2 k^2)
	*/
	TPI3 = pow(TWOPI,0.25)*pow(L,0.5); // (2 pi)^(1/4) * L^(1/2)
	PIL2 = PISQR*L*L; // pi^2*L^2

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
		vardv[it] = 0.;
	}

	for(it=0;it<numsteps;it++){

		gen_force3D(fx,gx,K,N,N2,TPI3,PIL2,sqdx,rng);
		jentzen_kloeden_winkel_step(vx,gx,K,N2,sqdx,dt,visc,f0,TPI3,PIL2);
		
		// to verify that the variance of each fourier mode follows theory
		// 0 <= kx < N, 0 <= ky < N, 0 <= kz < N//2+1
		vark1[it] = SQR(cabs(vx[1]));
		varkN[it] = SQR(cabs(vx[15]));

		// total variance of Fourier modes
		for(i=0;i<N2;i++)
			varf[it] += SQR(cabs(vx[i]));

		// variance of velocity gradient
		// we can reuse g/f and its transform for that

		for(i=0;i<N2;i++)
			gx[i] = I*TWOPI*K[i]*vx[i];

		// gradient back to real space
		fftw_execute(plan_fx_b);

		// save values, variance of velocity gradient
		for(i=1;i<N;i++)
			vardv[it] += SQR(fx[i]);

		// backup velocities in Fourier space
		for(i=0;i<N2;i++)
			gx[i] = vx[i];

		// velocities back to real space
		fftw_execute(plan_ux_b);

		for(i=1;i<N;i++)
			varx[it] += SQR(ux[i]);

		for(i=0;i<N2;i++)
			vx[i] = gx[i];

	}

	// normalize spatial avg. variance
	for(it=0;it<numsteps;it++)
		varx[it] *= norm;

	// normalize spatial avg. variance of gradient
	for(it=0;it<numsteps;it++)
		vardv[it] *= norm;

	write_real1D_array(varx, pid,N,numsteps,L,nu,f0,'x');
	write_real1D_array(varf, pid,N,numsteps,L,nu,f0,'f');
	write_real1D_array(vark1,pid,N,numsteps,L,nu,f0,'1');
	write_real1D_array(varkN,pid,N,numsteps,L,nu,f0,'N');
	write_real1D_array(vardv,pid,N,numsteps,L,nu,f0,'d');

	gsl_rng_free( rng );

	fftw_destroy_plan(plan_fx_f);
  fftw_destroy_plan(plan_fx_b);
	fftw_destroy_plan(plan_ux_f);
  fftw_destroy_plan(plan_ux_b);
	fftw_free(gx);
	fftw_free(vx);
	FREEP(K);
	FREEP(varx);
	FREEP(varf);
	FREEP(vark1);
	FREEP(varkN);
	FREEP(vardv);

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

static inline void gen_force3D(double *fx, fftw_complex *gx, double *K,
	LI N, LI N2, double TPI3, double PIL2, double sqdx, gsl_rng *rng){

	LI i;
	double cte1;

	// Assign random vector dW (white noise increments)
	for(i=0;i<N;i++){
		fx[i] = gsl_ran_gaussian( rng, 1. ) * sqdx;
	}

	fftw_execute(plan_fx_f);

	for(i=0;i<N2;i++){
		cte1 = TPI3 * gauss_kernel(K[i],PIL2);
		gx[i] *= cte1;
	}

}

// Jentzen, Kloeden and Winkel, Annals of Applied Probability 21.3 (2011): 908-950
// see eq. 21
static inline void jentzen_kloeden_winkel_step(fftw_complex *vx, fftw_complex *gx,
	double *K, LI N2, double sqdx, double dt, double visc, double f0, double TPI3, double PIL2){

	LI i;
	double cte = dt*visc;
	double tmp;

	// zero mode
	cte    = sqrt(dt*f0);
	// stochastic part
	vx[0] += cte * gx[0];

	// not sure if it's ctf or 1/ctf
	// the bk term in eq. 21 is the most confusing
	for(i=1;i<N2;i++){
		tmp  = visc*SQR(K[i]);
		cte  = sqrt(.5*f0*(1.-exp(-2.*tmp*dt))/tmp);

		// deterministic part
		vx[i] *= exp(-tmp*dt);
		// stochastic part
		vx[i] += cte * gx[i];
	}

}

// Jentzen, Kloeden and Winkel, Annals of Applied Probability 21.3 (2011): 908-950
// see eq. 21
static inline void jentzen_kloeden_winkel_step_2(fftw_complex *vx, fftw_complex *gx, fftw_complex *tx,
	double *K, LI N2, double sqdx, double dt, double visc, double f0, double TPI3, double PIL2, gsl_rng *rng){

	LI i;
	double cte;

	// zero mode
	cte    = sqrt(dt*f0);
	// stochastic part
	vx[0] += cte * gx[0];

	cte = dt*visc;
	// deterministic part
	for(i=1;i<N2;i++){
		vx[i] *= exp(-cte*SQR(K[i]));
	}
	// stochastic part
	cte = sqrt(.5*f0/visc);
	for(i=1;i<N2;i++){
		tx[i]  = cte * sqrt((1.-exp(-2.*visc*dt*SQR(K[i])))/SQR(K[i]));
	}
	for(i=1;i<N2;i++){
		tx[i] *= gx[i];
	}
	for(i=1;i<N2;i++){
		// stochastic part
		vx[i] += tx[i];
	}

}
