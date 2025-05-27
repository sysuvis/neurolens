#ifndef STRESS_MAJORIZATION_H
#define STRESS_MAJORIZATION_H

//t: target distance matrix
//n: number of elements
//d: target dimension
//X: initial value and results in column-major
//   for example, if X is a vector of 2D points, 
//   then all x-coordinates are consecutive, and all y-coordinates are consecutive
//   use allocateMatrix(X_data, X, n, d) to allocate memory
template <class T>
void stressMajorization(T** td, T** X,
						const int& n, 
						const int& d, 
						const bool& rand_init);

template <class T>
void stressMajorization(T** td, T** X, T** Lw,
						const int& n, 
						const int& d, 
						const bool& rand_init);

//A localized version of stress majorization
//t: target distance matrix
//n: number of elements
//d: target dimension
//X: initial value and results in column-major
//   for example, if X is a vector of 2D points, 
//   then all x-coordinates are consecutive, and all y-coordinates are consecutive
//   use allocateMatrix(X_data, X, n, d) to allocate memory
//rand_init: a boolean variable indicating whether X should be randomly initialized
//max_iter: maximum number of iterations to be performed
template <class T>
float stressMajorizationLocalized(T** td, T** X,
	const int& n,
	const int& d,
	const bool& rand_init,
	const int& max_iter=200);

//t:		target distance matrix
//n:		number of elements
//d:		target dimension
//X:		initial value and results in column-major
//			for example, if X is a vector of 2D points, 
//			then all x-coordinates are consecutive, and all y-coordinates are consecutive
//			use allocateMatrix(X_data, X, n, d) to allocate memory
//alpha:	a linear factor that combines maxent stress majorization and the original version
//			the original paper suggest the initial value to be 1
//q:		the original paper suggest 0 for graph with more than 30% of nodes with degree 1
//			otherwise 0.3
template <class T>
void maxentStressMajorization(T** td, T** X,
						const int& n, 
						const int& d, 
						const T& alpha,
						const T& q,
						const bool& rand_init);

#endif //STRESS_MAJORIZATION_H