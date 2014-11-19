// Megan Ruthven
// HW 2 - Scalable Machine Learning
#include <omp.h> 
#include <stdlib.h>
#include <fstream>
#include <mkl.h>
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

class MyBigArray {
private:
	vector<vector<int> > my_array;
	vector<vector<int> > my_array_t;
	vector<double> D;
	double a;
	int N;
	int rows;
	int cols;
public:
	MyBigArray(char * file_name, double alpha);
	vector<double> multiply(const vector<double>& x); 
	int deltas(vector<double>& x, double thresh);
	int prnorm(vector<double>& x, double thresh);
	int resids(vector<double>& x, double thresh);
	int ADMM(vector<double>& x, double thresh);
	int getRows() { return rows; }
	int getCols() { return cols; }
};

MyBigArray::MyBigArray(char * file_name, double alpha) {
	fstream my_file(file_name, ios_base::in);
	if (my_file == 0){
		printf("Could not open %s file. \n", file_name);
		return;
	}
	a = alpha;
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
		if (c < tempc){
			c = tempc;
		}
		my_array[tempr - 1].push_back(tempc - 1);
	}
	rows = r;
	cols = c;
	for (int i = 0; i < rows; i++) {
		vector<int> tt;
		my_array_t.push_back(tt);
	}
	std::vector<int>::iterator it;
	for (int i = 0; i < rows; i++) {
		for (it = my_array[i].begin(); it != my_array[i].end(); ++it){
			my_array_t[*it].push_back(i);
		}
	}
	

	for (int i = 0; i < rows; i++) {
		D.push_back((1 - a) / ((double)my_array[i].size()));
	}
	N = n;

	printf("Read a %s with %d rows and %d columns.\n", file_name, rows, cols);
}

vector<double> MyBigArray::multiply(const vector<double>& x){
	vector<double> y(rows, a / rows);
	if (cols != x.size()){
		return y;
	}
	int i = 0;
	std::vector<int>::iterator it;
	double sum = 0;
#pragma omp parallel for shared(y) private(i, it, sum)
	for (i = 0; i < rows; i++){
		sum = x[i] * D[i];
		for (it = my_array[i].begin(); it != my_array[i].end(); ++it){
			y[*it] += sum;
		}
	}
	return y;
}

int MyBigArray::prnorm(vector<double>& x, double thresh){
	vector<double> y(rows, a / rows);
	if (cols != x.size()){
		return 0;
	}
	int i = 0;
	std::vector<int>::iterator it;
	double sum = 0;
	int itt = 0;
	double total = 1;
	vector<double> delta(rows);

	while (total > thresh){
		itt++; 
#pragma omp parallel for shared(y) private(i, it, sum)
		for (i = 0; i < rows; i++){
			sum = x[i] * D[i];
			for (it = my_array[i].begin(); it != my_array[i].end(); ++it){
				y[*it] += sum;
			}
		}
#pragma omp parallel for private(i)
		for (i = 0; i < rows; i++) {
			delta[i] = abs(y[i] - x[i]);
			x[i] = y[i];
			y[i] = a / rows;
		}
		total = std::accumulate(delta.begin(),delta.end(), 0.0);
		
	}
	return itt;
}

int MyBigArray::deltas(vector<double>& x, double thresh){
	
	if (cols != x.size()){
		return 0;
	}
	vector<double> y(rows, a / rows);
	int i = 0;
	std::vector<int>::iterator it;
	double sum = 0;
#pragma omp parallel for shared(y) private(i, it, sum)
	for (i = 0; i < rows; i++){
		sum = x[i] * D[i];
		for (it = my_array[i].begin(); it != my_array[i].end(); ++it){
			y[*it] += sum;
		}
	}

	int itt = 1;
	vector<double> delta(rows);
	double d; 
	bool change = true;
#pragma omp parallel for private(i)
	for (i = 0; i < rows; i++) {
		delta[i] = y[i] - x[i];
		x[i] = y[i];
	}

	
	vector<omp_lock_t> lock (rows);

	for (int i = 0; i<rows; i++)
		omp_init_lock(&(lock[i]));

	while (change == true) {
		itt++;
		change = false;
#pragma omp parallel for shared(delta, lock) private(i, it, sum, d)
		for (i = 0; i < rows; i++){
			
			if (abs(delta[i]) > thresh) {
				omp_set_lock(&(lock[i]));
				d = delta[i];
				delta[i] = 0;
				omp_unset_lock(&(lock[i]));

				sum = d * D[i];
				x[i] += d;
				for (it = my_array[i].begin(); it != my_array[i].end(); ++it){
					omp_set_lock(&(lock[*it]));
					delta[*it] += sum;
					omp_unset_lock(&(lock[*it]));
				}
				change = true;
			}
			
		}
	}
	return itt;
}

int MyBigArray::resids(vector<double>& x, double thresh){
	if (cols != x.size()){
		return 0;
	}
	vector<double> y = x;
	vector<bool> yagain(rows, false);
	vector<bool> doagain(rows, true);
	vector<double> dx(rows, 0);
	int i = 0;
	std::vector<int>::iterator it;
	double sum = 0;
	bool doing = true;
	int itt = 0;
	std::transform(x.begin(), x.end(), D.begin(), dx.begin(), std::multiplies<double>());

	while (doing) {
		itt++;
		doing = false;
#pragma omp parallel for private(i, it, sum)
		for (i = 0; i < rows; i++) {
			if (doagain[i]) {
				sum = a / rows;
				for (it = my_array_t[i].begin(); it != my_array_t[i].end(); ++it) {
					sum += dx[*it];
				}
				if (abs(sum - x[i]) > thresh) {
					for (it = my_array[i].begin(); it != my_array[i].end(); ++it) {
						yagain[*it] = true;
					}
					doing = true;
				}
				x[i] = sum;
			}
		}
		doagain = yagain;
		std::transform(x.begin(), x.end(), D.begin(), dx.begin(), std::multiplies<double>());
		yagain.assign(rows, false);
	}

	return itt;
}

int MyBigArray::ADMM(vector<double>& x, double thresh){
	if (cols != x.size()){
		return 0;
	}
	vector<double> y = x;
	vector<bool> yagain(rows, false);
	vector<bool> doagain(rows, true);
	vector<double> dx(rows, 0);
	int i = 0;
	std::vector<int>::iterator it;
	double sum = 0;
	bool doing = true;
	int itt = 0;
	std::transform(x.begin(), x.end(), D.begin(), dx.begin(), std::multiplies<double>());

	while (doing) {
		itt++;
		doing = false;
#pragma omp parallel for private(i, it, sum)
		for (i = 0; i < rows; i++) {
			if (doagain[i]) {
				sum = a / rows;
				for (it = my_array_t[i].begin(); it != my_array_t[i].end(); ++it) {
					sum += dx[*it];
				}
				if (abs(sum - x[i]) > thresh) {
					for (it = my_array[i].begin(); it != my_array[i].end(); ++it) {
						yagain[*it] = true;
					}
					doing = true;
				}
				x[i] = sum;
			}
		}
		doagain = yagain;
		std::transform(x.begin(), x.end(), D.begin(), dx.begin(), std::multiplies<double>());
		yagain.assign(rows, false);
	}

	return itt;
}

struct gen_rand {
	double range;
public:
	gen_rand(double r = 1.0) : range(r) {}
	double operator()() {
		return (rand() / (double)RAND_MAX) * range;
	}
};

vector<double> make_norm(vector<double>& X, double * l) {
	vector<double> y(X.size());
	double lambda = 0;
	vector<double>::iterator it;

	for (it = X.begin(); it != X.end(); it++){
		lambda += (*it);
	}
	//lambda = sqrt(lambda);
	int i = 0;
	for (it = X.begin(); it != X.end(); it++){
		y[i] = (*it) / lambda;
		i++;
	}
	*l = lambda;
	return y;
}



int main(int argc, char *argv[]) {
	char big_array_name[20];
	if (argc != 2){
		printf("Improper arguments passed. %d\n", argc);
		return 1;
	}
	strcpy(big_array_name, argv[1]);
	double alpha = 0.15;
	MyBigArray A(big_array_name, alpha);
	int cols = A.getCols();
	int rows = A.getRows();
	printf("Text file read successfully, has %d rows and %d cols.\n", rows, cols);

	vector<double> X(cols);
	vector<double> Y(rows);

	int a[4] = { 1, 4, 8, 16 };
	double start = 0;
	double end = 0;
	double lambda = 0;
	int t;
	double sum_of_elems = 0;
	int jj;
	for (int j = 3; j < 4; j++){
		omp_set_num_threads(a[j]);
		double time = 0; 
		X.assign(cols, 1/(double) cols);
		//X = make_norm(X, &lambda);
		//printf("Iteration 0: lambda = %f.\n", lambda);
		start = omp_get_wtime();
		for (t = 0; t < ITER; t++) {
			X = A.multiply(X);
		}

		end = omp_get_wtime();
		time = (end - start);
		vector<double> max_val(10, 0);
		vector<int> max_ind(10, 0);
		double min = 0;
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
		printf("For %d threads, %d iterations took %f time, and average time was %f.\n", a[j], t, time, time / t);

		X.assign(cols, 1 / (double)cols);
		start = omp_get_wtime();
		t = A.prnorm(X, 0.00025);
		end = omp_get_wtime();
		time = (end - start);
		max_val.assign(10,0);
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
		printf("For %d threads, %d iterations took %f time, and average time was %f.\n", a[j], t, time, time / t);
		
		/*X.assign(cols, 1 / (double)cols);

		start = omp_get_wtime();
		t = A.deltas(X, 0.0000001);
		end = omp_get_wtime();
		time = (end - start);

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
		printf("For %d threads, %d iterations took %f time, and average time was %f.\n", a[j], t, time, time / t);*/

		X.assign(cols, 1 / (double)cols);

		start = omp_get_wtime();
		t = A.resids(X, 0.0000001);
		end = omp_get_wtime();
		time = (end - start);

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
		printf("For %d threads, %d iterations took %f time, and average time was %f.\n", a[j], t, time, time / t);
		/*for (int i = 0; i < 100; i++) {
			printf("%f, ", X[i]);
			if ((i+1) % 10 == 0) {
				printf("\n");
			}
		}*/
		
	}

}