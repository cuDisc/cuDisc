#ifndef _CUDISC_FIELD_H_
#define _CUDISC_FIELD_H_

#include "cuda_array.h"

template <typename T> class FieldRef ;
template <typename T> class FieldConstRef ;

template <typename T> class Field3DRef ;
template <typename T> class Field3DConstRef ;


/* class Field
 *
 *  Simple data class for holding 2D array data.
 * 
 *  The data is stored in C-order (i.e. the the Z-index runs fastest), with
 *  each row padded to be a multiple of block_size. Thefere, to access the 
 *  i-th element in R and j-th element in Z, use 
 *      field[i*stride + j].
 *
 *  A helper function index(i, j) provides this for convenience.
 */
template<typename T>
class Field {

 public:
   int stride ;

   Field(int NR, int NZ, int block_size=-1)
    {   
        if (block_size < 0) { // Use a power of 2, up to 128 bytes
            block_size = 1 ; 
            int max_size = 128/sizeof(T) ;
            while (block_size < NZ && block_size < max_size)
                block_size *= 2 ;
        }
        // Setup storage, padding the storage in phi to be a multiple of block_size:
        stride = block_size * ((NZ + block_size-1) / block_size) ;

        _ptr = make_CudaArray<T>(NR * stride) ;
    }

    int index(int i, int j) const {
        return i*stride + j ;
    }

    T& operator[](int l) {
        return _ptr[l] ;
    }

    T operator[](int l) const {
        return _ptr[l] ;
    }

    T& operator()(int i, int j) {
        return _ptr[index(i,j)] ;
    }

    T operator()(int i, int j) const {
        return _ptr[index(i,j)] ;
    }

    CudaArray<T>& get() {
        return _ptr ;
    }

    const CudaArray<T>& get() const {
        return _ptr ;
    }

 private:
    CudaArray<T> _ptr; 

    friend class FieldRef<T> ;
    friend class FieldConstRef<T> ;

    friend class Field3DRef<T> ;
    friend class Field3DConstRef<T> ;
} ;

/* class FieldRef
 *
 * Reference type for Field class template.
 *  
 * This class exists to enable copying Field objects to the GPU. Since objects
 * can't be passed by reference to __global__ functions we need a special class
 * to handle this. Note that passing by value is impossible because CudaArrays
 * are non-copyable.
 */
template<typename T>
class FieldRef {

 public:
    int stride ;

    FieldRef(Field<T>& f)
      : stride(f.stride), _ptr(f._ptr.get())
    {  } ;

    __host__ __device__ 
    int index(int i, int j) const {
        return i*stride + j ;
    }

    __host__ __device__ 
    T& operator[](int l) {
        return _ptr[l] ;
    }

    __host__ __device__ 
    T operator[](int l) const {
        return _ptr[l] ;
    }


    __host__ __device__ 
    T& operator()(int i, int j) {
        return _ptr[index(i,j)] ;
    }

    __host__ __device__ 
    T operator()(int i, int j) const {
        return _ptr[index(i,j)] ;
    }


    __host__ __device__ 
    T* get() {
        return _ptr ;
    }
    
    __host__ __device__ 
    const T* get() const {
        return _ptr ;
    }

 private:
    T* _ptr; 

    friend class FieldConstRef<T> ;
} ;

/* class FieldConstRef
 *
 * Const Reference type for Field class template.
 *  
 * This class exists to enable copying const Field objects to the GPU. Since 
 * objects can't be passed by reference to __global__ functions we need a
 * special class to handle this. Note that passing by value is impossible 
 * because CudaArrays are non-copyable.
 */
template<typename T>
class FieldConstRef {

 public:
    int stride ;

    FieldConstRef(const Field<T>& f)
      : stride(f.stride), _ptr(f._ptr.get())
    {  } ;

    
    FieldConstRef(const FieldRef<T>& f)
      : stride(f.stride), _ptr(f._ptr)
    {  } ;


    __host__ __device__ 
    int index(int i, int j) const {
        return i*stride + j ;
    }

    __host__ __device__ 
    T operator[](int l) const {
        return _ptr[l] ;
    }

    __host__ __device__ 
    T operator()(int i, int j) const {
        return _ptr[index(i,j)] ;
    }

    __host__ __device__ 
    const T* get() const {
        return _ptr ;
    }

 private:
    const T* _ptr; 

} ;


/* class Field3D
 *
 *  Simple data class for holding 3D array data.
 * 
 *  The data is stored in C-order (i.e. the last index runs fastest), with
 *  each row padded to be a multiple of block_size. Thefere, to access the 
 *  i-th element in R and j-th element in Z, use 
 *      field[i*stride_Zd + j*stride_d  + k].
 *
 *  A helper function index(i, j, k) provides this for convenience.
 */
template<typename T>
class Field3D {

 public:
   int Nd ;
   int stride_Zd ;
   int stride_d ;



   Field3D(int NR, int NZ, int Nd_, int block_size=-1)
    {   
        // Setup storage, padding the storage in phi to be a multiple of block_size:
        Nd = Nd_;
        if (block_size < 0) { // Use a power of 2, up to 128 bytes
            block_size = 1 ; 
            int max_size = 128/sizeof(T) ;
            while (block_size < Nd && block_size < max_size)
                block_size *= 2 ;
        }
        stride_d = block_size * ((Nd + block_size-1) / block_size) ;
        stride_Zd = NZ * stride_d ;

       _ptr = make_CudaArray<T>(NR * stride_Zd) ;
    }

    int index(int i, int j, int k) const {
        return i*stride_Zd + j*stride_d  + k ;
    }

    T& operator[](int l) {
        return _ptr[l] ;
    }

    T operator[](int l) const {
        return _ptr[l] ;
    }

    T& operator()(int i, int j, int k) {
        return _ptr[index(i,j,k)] ;
    }

    T operator()(int i, int j, int k) const {
        return _ptr[index(i,j,k)] ;
    }


    CudaArray<T>& get() {
        return _ptr ;
    }

    const CudaArray<T>& get() const {
        return _ptr ;
    }

 private:
    CudaArray<T> _ptr; 

    friend class Field3DRef<T> ;
    friend class Field3DConstRef<T> ;
} ;


/* class Field3DRef
 *
 * Reference type for Field3D class template.
 *  
 * This class exists to enable copying Field3D objects to the GPU. Since objects
 * can't be passed by reference to __global__ functions we need a special class
 * to handle this. Note that passing by value is impossible because CudaArrays
 * are non-copyable.
 */
template<typename T>
class Field3DRef {

 public:
   int Nd ;
   int stride_Zd ;
   int stride_d ;

    Field3DRef(Field3D<T>& f)
      : Nd(f.Nd), stride_Zd(f.stride_Zd), stride_d(f.stride_d),
       _ptr(f._ptr.get())
    {  } ;

    explicit Field3DRef(Field<T>& f)
      : Nd(1), stride_Zd(f.stride), stride_d(1),
        _ptr(f._ptr.get())
    { } ;

    __host__ __device__ 
    int index(int i, int j, int k) const {
        return i*stride_Zd + j*stride_d  + k ;
    }

    __host__ __device__ 
    T& operator[](int l) {
        return _ptr[l] ;
    }

    __host__ __device__ 
    T operator[](int l) const {
        return _ptr[l] ;
    }

    __host__ __device__ 
    T& operator()(int i, int j, int k) {
        return _ptr[index(i,j,k)] ;
    }

    __host__ __device__ 
    T operator()(int i, int j, int k) const {
        return _ptr[index(i,j,k)] ;
    }

    __host__ __device__ 
    T* get() {
        return _ptr ;
    }
    
    __host__ __device__ 
    const T* get() const {
        return _ptr ;
    }

 private:
    T* _ptr; 

    friend class Field3DConstRef<T> ;
} ;

/* class Field3DConstRef
 *
 * Const Reference type for Field3D class template.
 *  
 * This class exists to enable copying const Field3D objects to the GPU. Since 
 * objects can't be passed by reference to __global__ functions we need a
 * special class to handle this. Note that passing by value is impossible 
 * because CudaArrays are non-copyable.
 */
template<typename T>
class Field3DConstRef {

 public:
   int Nd ;
   int stride_Zd ;
   int stride_d ;

    Field3DConstRef(const Field3D<T>& f)
      : Nd(f.Nd), stride_Zd(f.stride_Zd), stride_d(f.stride_d),
       _ptr(f._ptr.get())  
    {  } ;

    Field3DConstRef(const Field3DRef<T>& f)
      : Nd(f.Nd), stride_Zd(f.stride_Zd), stride_d(f.stride_d),
       _ptr(f._ptr)  
    {  } ;

    explicit Field3DConstRef(const Field<T>& f)
      : Nd(1), stride_Zd(f.stride), stride_d(1),
        _ptr(f._ptr.get())
    { } ;

    __host__ __device__ 
    int index(int i, int j, int k) const {
        return i*stride_Zd + j*stride_d  + k ;
    }

    __host__ __device__ 
    T operator[](int l) const {
        return _ptr[l] ;
    }
    
    __host__ __device__ 
    T operator()(int i, int j, int k) const {
        return _ptr[index(i,j,k)] ;
    }

    __host__ __device__ 
    const T* get() const {
        return _ptr ;
    }

 private:
    const T* _ptr; 

} ;


#endif