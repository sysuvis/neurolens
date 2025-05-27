#include "StressMajorization.h"
#include "typeOperation.h"
#include "ConjugateGradient.h"
#include <cstdlib>
#include <ctime>

template <class T>
T distance(T** X, const int& d, const int& i, const int& j){
	T ret=0, diff;
	for (int k=0; k<d; ++k) {
		diff = X[k][i]-X[k][j];
		//if (X[k][i]>50)
		//printf("{%f %f} ", X[k][i],X[k][j]);
		ret += diff*diff;
	}
	ret = sqrtf(ret);
	return ret;
}

template <class T>
void updateLz(T** lz, T** lw, T** dist, T** X, 
					 const int& d, const int& n)
{
	T dij;
	for (int i=0; i<n; ++i) {
		lz[i][i] = 0;
		for (int j=0; j<n; ++j) if(i!=j) {
			dij = distance(X, d, i, j);
			if (dij < 1e-5) dij = 1e-5;
			lz[i][j] = lw[i][j]*dist[i][j]/dij;
			lz[i][i] -= lz[i][j];
		}
	}
}

template <class T>
void stressMajorization(T** td, T** X,
						const int& n, 
						const int& d, 
						const bool& rand_init)
{
	T *Lw_data, **Lw;
	allocateMatrix(Lw_data, Lw, n, n);
	for (int i=0; i<n*n; ++i) {
		Lw_data[i] = 1.0f;
	}

	stressMajorization(td, X, Lw, n, d, rand_init);

	delete[] Lw;
	delete[] Lw_data;
}

template <class T>
void stressMajorization(T** td, T** X, T** Lw,
						const int& n, 
						const int& d, 
						const bool& rand_init)
{
	if (rand_init) {
		srand(time(NULL));
		for (int i=0; i<d; ++i) {
			for (int j=0; j<n; ++j) {
				X[i][j] = (rand()%100000)/1000.0f;
			}
		}
	}

	T *Lz_data, **Lz;
	allocateMatrix(Lz_data, Lz, n, n);

	T d_ikj;
	for (int k=0; k<n; ++k) {
		for (int i=0; i<n; ++i) {
			for (int j=0; j<n; ++j) {
				d_ikj = td[i][k]+td[k][j];
				if (d_ikj<td[i][j]){
					td[i][j]=d_ikj;
				}
			}
		}
	}

	for (int i=0; i<n; ++i) {
		Lw[i][i] = 0;
		for (int j=0; j<n; ++j) if(i!=j){
			Lw[i][j] = -Lw[i][j]/(td[i][j]*td[i][j]);
			Lw[i][i] -= Lw[i][j];
		}
	}

	int count = 0;
	T *bXi = new T[n];
	do {
		updateLz(Lz, Lw, td, X, d, n);
		for (int i=0; i<d; ++i) {
			mat_mult(Lz, X[i], bXi, n);
			conjugate_gradient(Lw, bXi, X[i], n);
		}
		++count;
	} while (count<200);

	delete[] Lz;
	delete[] Lz_data;
	delete[] bXi;
}

template <class T>
float stressMajorizationLocalized(T** td, T** X,
	const int& n,
	const int& d,
	const bool& rand_init,
	const int& max_iter)
{
	T d_ikj;
	for (int k = 0; k < n; ++k) {
		for (int i = 0; i < n; ++i) {
			//printf("%.3f ", td[k][i]);
			if (td[k][i] <= 0.00001)
				td[k][i] = 100;
			
		}
		//printf("\n");
	}
	for (int k = 0; k < n; ++k) {
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				d_ikj = td[i][k] + td[k][j];
				if (std::isinf(d_ikj) || d_ikj < 0)
					printf("error! there is negative or inf in td. \n");
				if (d_ikj < td[i][j]) {
					td[i][j] = d_ikj;
					
				}
			}
		}
	}


	T *td_data = td[0];
	T scale = td_data[0] ;
	for (int i = 1; i < n*n; ++i) {
		
		if (td_data[i] > scale) {
			scale = td_data[i];
		}
	}

	if (rand_init) {
		// 何颂贤把这里的随机数固定了，因此每次随机出来的结构都一样了
		srand(2);
		float s = scale/RAND_MAX;
		for (int i = 0; i < d; ++i) {
			for (int j = 0; j < n; ++j) {
				X[i][j] = rand()*s;
				//printf("%f ", X[i][j]);
			}
			//printf("\n");
		}
	}



	//init the weight matrix
	T *w_data, **w;
	allocateMatrix(w_data, w, n, n);

	for (int i = 0; i < n; ++i) {
		w[i][i] = 0;
		for (int j = 0; j < n; ++j) if (j!=i) {
			w[i][j] = 1 / td[i][j];// (td[i][j] * td[i][j]);
			w[i][i] += w[i][j];
		}
	}
	//normalize the weight matrix
	for (int i = 0; i < n; ++i) {
		for (int j =0; j < n; ++j) if (j!=i) {
			w[i][j] /= w[i][i];
		}
	}

	T *X2_data, **X2;
	allocateMatrix(X2_data, X2, n, d);
	
	scale *= 0.05;
	T **cur_x=X, **next_x=X2;
	T prev_stress = -1e30, stress = -1e30;
	int count = 0;
	do {
		//init next and stress
		memset(&next_x[0][0], 0, sizeof(T)*n*d);
		prev_stress = stress;
		stress = 0.0f;

		for (int i = 0; i < n; ++i) {
			for (int j = i + 1; j < n; ++j) {
				//fetch data
				T wij = w[i][j];
				T wji = w[j][i];
				//if (std::isnan(wij) || std::isinf(wij))
					//printf("wij too small! \n");
				T xij = distance(cur_x, d, i, j);
				T dij = td[i][j];
				T inv_xij = 1 / xij;
				//update stress
				T diff = xij - dij;
				if (std::isnan(diff) || std::isinf(diff))
					printf("diff is nan or inf! \n");
				stress += wij*diff*diff;
				//update x
				for (int k = 0; k < d; ++k) {
					next_x[k][i] += wij * (cur_x[k][j] + dij * (cur_x[k][i] - cur_x[k][j])*inv_xij);
					next_x[k][j] += wji * (cur_x[k][i] + dij * (cur_x[k][j] - cur_x[k][i])*inv_xij);
				}
			}
		}

		//swap cur_x and next_x
		std::swap(cur_x, next_x);
		++count;
	} while (std::abs(prev_stress-stress)>scale && count<max_iter);

	if (cur_x != X) {
		memcpy(X[0], cur_x[0], sizeof(T)*n*d);
	}

	delete[] X2_data;
	delete[] X2;
	delete[] w;
	delete[] w_data;

	return stress;
}

template <class T>
void maxentStressMajorization(T** td, T** X,
							  const int& n, 
							  const int& d, 
							  const T& alpha,
							  const T& q,
							  const bool& rand_init)
{
	if (rand_init) {
		srand(time(NULL));
		for (int i=0; i<d; ++i) {
			for (int j=0; j<n; ++j) {
				X[i][j] = (rand()%100000)/1000.0f;
			}
		}
	}

	T *Lw_data, **Lw;
	allocateMatrix(Lw_data, Lw, n, n);

	for (int k=0; k<n; ++k) {
		for (int i=0; i<n; ++i) {
			for (int j=0; j<n; ++j) {
				if (td[i][k]+td[k][j]<td[i][j]){
					td[i][j]=td[i][k]+td[k][j];
				}
			}
		}
	}

	for (int i=0; i<n; ++i) {
		Lw[i][i] = 0;
		for (int j=0; j<n; ++j) if(i!=j){
			Lw[i][j] = 1.0f/(td[i][j]*td[i][j]);
			Lw[i][i] += Lw[i][j];
		}
	}

	T *update = new T[d];
	T sgn_q_alpha = (q<0)?(-alpha):(alpha);
	for (int iter=0; iter<200; ++iter) {
		for (int i=0; i<n; ++i) {
			memset(update, 0, sizeof(T)*d);
			for (int j=0; j<n; ++j) if(j!=i) {
				T dij = distance(X, d, i, j);
				for (int k=0; k<d; ++k) {
					update[d] += Lw[i][j]/Lw[i][i]*(X[k][j]+td[i][j]/dij*(X[k][i]-X[k][j]))
						+sgn_q_alpha/Lw[i][i]*(X[k][i]-X[k][j])/pow(dij, q+2);
				}
			}
			for (int k=0; k<d; ++k) {
				X[d][i] = update[d];
			}
		}
	}

	delete[] update;
	delete[] Lw;
	delete[] Lw_data;
}

template void stressMajorization<double>(double** t, double** X, const int& n, const int& d, const bool& rand_init);
template void stressMajorization<float>(float** t, float** X, const int& n, const int& d, const bool& rand_init);

template float stressMajorizationLocalized<double>(double** t, double** X, const int& n, const int& d, const bool& rand_init, const int& max_iter);
template float stressMajorizationLocalized<float>(float** t, float** X, const int& n, const int& d, const bool& rand_init, const int& max_iter);

template void maxentStressMajorization<double>(double** td, double** X, const int& n, const int& d, const double& alpha, const double& q, const bool& rand_init);
template void maxentStressMajorization<float>(float** td, float** X, const int& n, const int& d, const float& alpha, const float& q, const bool& rand_init);