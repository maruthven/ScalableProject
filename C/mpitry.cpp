#include <mpi.h>
#include <stdio.h>
#include <omp.h> 
#include <stdlib.h>
#include <fstream>
//#include <mkl.h>
#include <math.h>
#include <vector>
#include <iostream>     // std::cout
#include <sstream>      // std::istringstream
#include <stdio.h>
#include <string.h>
#include <cstdio>
#include <numeric> 
#include <cmath>
#include <algorithm>
#include <functional>

using namespace std;
#define ITER 50
#define TOPVALS 100
#define MAXPROX 16
#define min(a,b) (((a)<(b))?(a):(b))

int readInfo(char * file_name, vector<vector<int> > & my_array, vector<vector<int> > & my_array_t, vector<double> & D, double a) {
	fstream my_file(file_name, ios_base::in);
	if (my_file == 0){
		printf("Could not open %s file. \n", file_name);
		return 0;
	}
	int r = 0;
	int c = 0;
	int tempr = 0;
	int tempc = 0;
	string s;
	printf("Beginning to read %s.\n", file_name);
	int n = 0;
	while (getline(my_file, s)){
		n++;
		istringstream sin(s);
		sin >> tempr >> tempc;
		while (tempr > r){
			vector<int> tt;
			my_array.push_back(tt);
			r++;
		}
		while (c < tempc){
			c++;
			vector<int> tt;
			my_array_t.push_back(tt);
		}
		my_array[tempr - 1].push_back(tempc - 1);
		my_array_t[tempc - 1].push_back(tempr - 1);
	}
	int rows = r;
	int cols = c;

	for (int i = 0; i < rows; i++) {
		D.push_back((1 - a) / ((double)my_array[i].size()));
	}

	printf("Read a %s with %d rows and %d columns.\n", file_name, rows, cols);
	return rows;
}

int multiply(vector<double>& x, vector<vector<int> > & my_array, vector<double> & D, int num_threads, int rank, int rows, double a){
	MPI_Status status[MAXPROX+1];
	MPI_Request req[MAXPROX];
	int i = 0;
	std::vector<int>::iterator it;
	double sum = 0;
	int step_size = round(rows / num_threads);
	int start = step_size*rank;
	int end = min(step_size*(rank + 1), rows);
	vector<double> nowx(end - start);
	//vector<double> y(step_size);
	int j;
	for (j = 0; j < ITER; j++) {
		//MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0) {
			printf("After #%d barrier.\n", j + 2);

			for (i = 1; i < num_threads; i++) {
				MPI_Irecv(&x[step_size*i], min(step_size*(i + 1), rows) - step_size*i, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &req[i]);
				printf("Recieved the #%d chunk of #%d iter.\n", i, j);
			}
		}
		
		for (i = 0; i < (end - start); i++){
			//sum = x[i] * D[i];
			sum = a / rows;
			for (it = my_array[i+start].begin(); it != my_array[i+start].end(); ++it){
				//y[*it] += sum;
				sum += x[*it] * D[*it];
			}
			nowx[i] = sum;
		}
		printf("Rank %d done with %d calculations.\n", rank, j);
		if (rank == 0 && (num_threads - 1)) {
			
			MPI_Waitall(num_threads - 1, &req[1], &status[1]);
			printf("Recieved all information.\n");
			MPI_Bcast(&x, x.size(), MPI_DOUBLE, rank, MPI_COMM_WORLD);
		
			printf("Brodcasted the #%d X.\n", j);
		}
		else if(num_threads - 1) {

			MPI_Issend(&nowx, end - start, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD, &req[rank]);
			MPI_Wait(&req[rank], &status[rank]);
			printf("Thread %d sent successfuly.\n", rank);
		}

	}
	return j;
}


main(int argc, char *argv[])
{
	char * file_name = argv[1];
	int rank;
	int mpi_num_nodes;
	int err;

	vector<vector<int> > my_array;
	vector<vector<int> > my_array_t;
	vector<double> D;
	double a = 0.15;
	int rows;
	int cols;

	double t_start, t_end, time, min, sum_of_elems;
	int jj;
	vector<double> max_val(10);
	vector<int> max_ind(10);
	err = MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_num_nodes);
	printf("Hello world! %d\n", rank);
	rows = readInfo(file_name, my_array, my_array_t, D, a);
	cols = rows;
	vector<double> X(rows, a / rows);

	MPI_Barrier(MPI_COMM_WORLD);

	if (rank == 0) {
		printf("After the first barrier!\n");
		t_start = MPI_Wtime();
	}
	int t = multiply(X, my_array, D, mpi_num_nodes, rank, rows, a);

	if (rank == 0) {
		t_end = MPI_Wtime();
		time = (t_end - t_start);
		max_val.assign(10, 0);
		max_ind.assign(10, 0);
		min = 0;
		for (jj = 0; jj < rows; jj++) {
			if (X[jj] > min) {
				int i = 9;
				while ((i > 0) && (X[jj] > max_val[i - 1]))
				{
					max_val[i] = max_val[i - 1];
					max_ind[i] = max_ind[i - 1];
					i = i - 1;
				}
				max_val[i] = X[jj];
				max_ind[i] = jj;
			}
		}
		for (jj = 0; jj < 10; jj++) {
			printf("%d, ", max_ind[jj]);
		}
		printf("\n");
		sum_of_elems = 0;
		for (jj = 0; jj < rows; ++jj)
			sum_of_elems += X[jj];
		printf("The sum of X is %f.\n", sum_of_elems);
		printf("Rank %d\n", rank);
		printf("For %d threads, %d iterations took %f time, and average time was %f.\n", mpi_num_nodes, t, time, time / t);
	}





	err = MPI_Finalize();
}
