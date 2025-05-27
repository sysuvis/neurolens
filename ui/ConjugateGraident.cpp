#include "ConjugateGradient.h"
#include "typeOperation.h"
#include <vector>

template <class T>
T conjugate_gradient(T** A, T* b, T* x, const int& n){
	std::vector<T> r(n);
	std::vector<T> p(n);
	std::vector<T> Ap(n);

	mat_mult(A, x, &r[0], n);
	for (int i=0; i<n; ++i){
		r[i] = b[i]-r[i];
	}
	p.assign(r.begin(), r.end());
	T rsold = inner(&r[0], &r[0], n), rsnew, alpha;
	if (rsold<1e-20) return rsold;

	for (int i=0; i<n; ++i) {
		mat_mult(A, &p[0], &Ap[0], n);
		alpha = rsold/inner(&p[0], &Ap[0], n);
		rsnew = 0.0;
		for (int j=0; j<n; ++j) {
			x[j] += alpha*p[j];
			r[j] -= alpha*Ap[j];
			rsnew += r[j]*r[j];
		}
		if (rsnew<1e-20) {
			break;
		}
		for (int j=0; j<n; ++j) {
			p[j] = r[j] + (rsnew/rsold)*p[j];
		}
		rsold = rsnew;
	}

	return rsnew;
}

template float conjugate_gradient<float>(float** A, float* b, float* x, const int& n);
template double conjugate_gradient<double>(double** A, double* b, double* x, const int& n);