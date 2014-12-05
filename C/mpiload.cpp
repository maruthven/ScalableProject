#include <stdio.h>
#include "mpi.h"
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <iostream>     
#include <sstream>      
#include <stdio.h>
#include <string.h>
#include <cstdio>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <functional>
#include <math.h>
//#include <gsl/gsl_vector.h>
//#include <gsl/gsl_matrix.h>
//#include <gsl/gsl_blas.h>
//#include <gsl/gsl_linalg.h>
#include"eigen-eigen-1306d75b4a21/Eigen/Dense"
#include"eigen-eigen-1306d75b4a21/Eigen/Sparse"
#include"eigen-eigen-1306d75b4a21/Eigen/Core"


std::vector<double> updateu(std::vector<double> utemp,std::vector<double> x,std::vector<double>z) {
	std::vector<double> unew(utemp.size());
	for(int i=0;i<utemp.size();i++) {
		unew[i]=utemp[i]+x[i]-z[i];
	}

return unew;
}



std::vector<double> updatez(std::vector<double> fullu, std::vector<double> fullx, double rho, int size, double lam, int m) {
	std::vector<double> outz(m);
	double comp=lam/(rho*(double)size);
	for(int i=0;i<m;i++) {
		double dumx=0;
		double dumu=0;
		double dux=0;
		for(int j=0;j<size;j++) {
			dumx+=fullx[m*j+i]/(double)size;	
			dumu+=fullu[m*j+i]/(double)size;
		}
			double minus=-1.0;
			dux=dumu+dumx;
			if(dux>comp){
				outz[i]=dux-comp;
			}
			else if(dux<minus*comp) {
				outz[i]=dux+comp;
			}
			else{outz[i]=(double)0;}
	}

return outz;
}


std::vector<double> randvector(int n,double c,int m) {                                                                                    
    //flag of 1 will give a vector of all ones;                                                                                     
    //anything else will give a "random" vector;                                                                                    
    std::vector<double> randvec(n);                                                                                                 
    for(int i=0;i<n;i++) {                                                                                                          
        randvec[i]=(double)(1-c)/(double)m;                                                                                                               
    }
return randvec;                                                                                                                     
}     


std::vector< std::vector<int> >  getdata(char* f_loc) {  
	std::ifstream file;
	file.open(f_loc);
	std::string line;
	std::vector< std::vector<int> > matrix(2,std::vector<int>());
	if(file.is_open()) {
	int count=0;       
	int v1;
	int v2;
	int m=1;
	while (std::getline(file,line)) {
	    std::stringstream streamd(line);
	    streamd >> v1 >> v2;  
	    matrix[0].push_back(v1);
	    matrix[1].push_back(v2);
	    if(m<v1) {m=v1;}
	    count++;                                                              
	                                                                          
	        }                                                                 
	matrix[1].push_back(m);
	matrix[1].push_back(count);
	}                                                                         

	else {MPI_Abort(MPI_COMM_WORLD,1);}
	return matrix;                                                            
}

int main(int argc,char* argv[]) {

	char* floc = argv[1];
	int nnz;
	double c = atof(argv[2]);
	int maxiter = atoi(argv[3]);
	double rho=atof(argv[4]);
	double lam=atof(argv[5]);
	int rank, size, m;
	MPI_Init(&argc,&argv);
	MPI_Status status;
	MPI_Request request;

	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

// --------------Loading data (A'), All processors -------------------------------



	std::vector<std::vector<int> > Data=getdata(floc);
	nnz=Data[1].back();
	Data[1].pop_back();
	m=Data[1].back();
	Data[1].pop_back();

	if(rank==0){
		printf("Loading the Data...\n");
		printf("[%d] nnz= %d\n",rank,nnz);
		printf("[%d] m= %d\n",rank,m);
		printf("[%d]:Data loaded\n",rank);
	}
	MPI_Barrier(MPI_COMM_WORLD);
//----------------Done loading data still need to modify to get A matrix----	



//---------------- Creating partial bs for each processor-----------------------
	std::vector<double> blocal;
	if(rank<size-1) {
		blocal=randvector(floor(m/size),c,m);
	}
	else{blocal=randvector(m-rank*floor(m/size),c,m);}
	//int bsize=blocal.size();
	//printf("[%d]: b1 values is: %i\n",rank,bsize);

//-----------Separate A into each processor and make it (I - c Pt) ------------
	std::vector<std::vector<double> > Alocal(3,std::vector<double>());//local A for parallel computation	
	

	int p=0;
	for(int i=0;i<nnz;i++) {
		if(rank<size-1){
			if(Data[0][i]<=(rank+1)*floor(m/size) && Data[0][i]>rank*floor(m/size)) {
				Alocal[0].push_back((double)Data[0][i]);
				Alocal[1].push_back((double)Data[1][i]);
				Alocal[2].push_back(c);
				if(Data[0][i]==Data[1][i]) {Alocal[2][p]=1-Alocal[2][p];}
				p++;
			}

		}
		else{
			if(Data[0][i]>rank*floor(m/size)) {
				Alocal[0].push_back((double)Data[0][i]);
				Alocal[1].push_back((double)Data[1][i]);
				Alocal[2].push_back(c);
				if(Data[0][i]==Data[1][i]) {Alocal[2][p]=1-Alocal[2][p];}
				p++;
			}

		}

	}
// -----------REMOVE DATA FROM MEMORY-----------------------------------------

//--------------Finished, each processor having its own data ------------------

	MPI_Barrier(MPI_COMM_WORLD);

//----------Ready for linear solver --------------------------------------

//Initialize z(global) and u(local)
	std::vector<double> z(m);
	if(rank==0){
		z=randvector(m,.3,m);
	}	
	
	MPI_Bcast(&z[0],m,MPI_DOUBLE,0,MPI_COMM_WORLD);
	std::vector<double> u=randvector(m,.6*rank,m);

	std::vector<double> x=randvector(m,-13*rank,m); //REMOVE AFTER SOLVING FOR X!!!!

//---Iterate a set number of times (solve min ||Ax-b|| 

	int iter=0;
	
	std::vector<double> fullu(1);
	std::vector<double> fullx(1);
	if(rank==0){fullu.resize(m*size); fullx.resize(m*size);}
	while(iter<maxiter) {
		//Solve for x update x= (A'A + rho*I)\(A'b + rho*z - rho*u)--------------------------	
		
		if(rank==0) {
			Eigen::MatrixXf Iden = Eigen::MatrixXf::Identity(3,3);	
			Eigen::VectorXf Test(3);
			Test(0)=12;
			Test(1)=2;
			Test(2)=3;
			Eigen::VectorXf testsol(3);
			testsol=Iden.ldlt().solve(Test);
			printf("made it here %f %f %f\n",testsol(0),testsol(1),testsol(2));
		}
		
		// z update on the mater thread and then send back out to all other threads----------
		MPI_Gather(&x[0],m,MPI_DOUBLE,&fullx[0],m,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Gather(&u[0],m,MPI_DOUBLE,&fullu[0],m,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		if(rank==0){
			z=updatez(fullu,fullx,rho,size,lam,m);	

		}
	
		MPI_Barrier(MPI_COMM_WORLD);
		
		//Send z back to all other threads
		MPI_Bcast(&z[0],m,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);

		//Update u now that z is updated ------------------------------
		std::vector<double> utemp(m);
		std::copy(u.begin(),u.end(),utemp.begin());
		u=updateu(utemp,x,z);

		//printf("[%d]: my value of u is: %f\n", rank,u[0]);		

		iter++;
		MPI_Barrier(MPI_COMM_WORLD);
	}	


	if(rank==0){
		printf("[%d]: z(%i) is : %f\n",rank,3,z[3]);
	}

//-----------End of linear solver---------------------------------------

	MPI_Finalize();
return 0;
}
