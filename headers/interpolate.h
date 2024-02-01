
#ifndef _CUDISC_HEADERS_INTERPOLATE_H_
#define _CUDISC_HEADERS_INTERPOLATE_H_

#include <array>
#include <cassert>
#include <vector>

/* _PchipInterpolatorBase
 *
 * Base class for 1 dimensional interpolation of multiple variables using the
 * PCHIP algorithm as implemented in matlab and scipy.
 */
template <int dim>
class _PchipInterpolatorBase {
   public:
    typedef std::array<double, dim> value_type;

    _PchipInterpolatorBase(){};

    _PchipInterpolatorBase(std::vector<double> x, std::vector<value_type> y)
     : _x(x), _y(y) {
        assert(_x.size() == _y.size());

        _compute_deltas();
        _compute_coeffs();
    }

    template <typename... Arrays>
    _PchipInterpolatorBase(std::vector<double> x, Arrays... ys)
     : _x(x), _y(x.size()) {
        static_assert(sizeof...(ys) == dim,
                      "PchipInterpolator: Number of arrays must match number "
                      "of dimensions");

        // Pack the arrays
        int Npts = _x.size();
        for (int i = 0; i < Npts; i++) _y[i] = {ys[i]...};

        _compute_deltas();
        _compute_coeffs();
    }

    value_type interpolate(double x) const {
        int idx = _get_index(x);

        double dx = x - _x[idx];
        value_type result;
        for (int j = 0; j < dim; j++)
            result[j] =
                _y[idx][j] +
                dx * (_d[idx][j] + dx * (_c[idx][j] + dx * _b[idx][j]));
        return result;
    }

    value_type _derivative(double x) const {
        int idx = _get_index(x);

        double dx = x - _x[idx];
        value_type result;
        for (int j = 0; j < dim; j++)
            result[j] = _d[idx][j] + dx * (_c[idx][j]*2 + dx * _b[idx][j]*3);
        return result;
    }

    value_type _integrate(double x0, double x1) const {
        assert(x1 >= x0) ;

        int start = _get_index(x0) ;
        int end = _get_index(x1) ;

        const std::vector<value_type>&y = _y, b = _b, c = _c, d = _d ;
        auto polynomial_integrate = [&y,&b,&c,&d](int i, double dx) {
            value_type result ;
            for (int j = 0; j < dim; j++)
                result[j] = dx*
                    (y[i][j] + dx*(d[i][j]/2 + dx*(c[i][j]/3 + dx*b[i][j]/4)));
            return result ;
        } ;

        value_type result ;
        value_type term = polynomial_integrate(start, x0 - _x[start]) ;
        for (int j = 0; j < dim; j++)
            result[j] = -term[j] ;
        
        for (int i=start; i < end; i++) {
            term = polynomial_integrate(i, _x[i+1] - _x[i]) ;
            for (int j = 0; j < dim; j++)
                result[j] += term[j] ;
        }

        term =  polynomial_integrate(end, x1 - _x[end]) ;
        for (int j = 0; j < dim; j++)
            result[j] += term[j] ;

        return result ;
    }

   private:
    int _get_index(double x) const {
        int Npts = _x.size();

        // Check edges
        if (x < _x[0]) return 0;
        if (x > _x[Npts - 2]) return Npts - 2;

        // Bisect
        int l = 0, u = Npts - 1;
        while (u > l + 1) {
            int m = (l + u) / 2;
            double xm = _x[m];
            if (x > xm)
                l = m;
            else if (x < xm)
                u = m;
            else {
                l = m;
                break;
            }
        }
        return l;
    }

    void _compute_deltas() {
        int Npts = _x.size();

        // 1. Compute step and divided differences
        _h.resize(Npts - 1);
        _d.resize(Npts);
        bool h_pos = true;
        for (int i = 0; i < Npts - 1; i++) {
            _h[i] = _x[i + 1] - _x[i];
            h_pos &= (_x[i + 1] > _x[i]);

            for (int j = 0; j < dim; j++)
                _d[i][j] = (_y[i + 1][j] - _y[i][j]) / _h[i];
        }
        assert(h_pos);

        // 2. Compute harmonic mean
        value_type tmp = _d[0];
        for (int i = 1; i < Npts - 1; i++) {
            double w1 = 2 * _h[i] + _h[i - 1];
            double w2 = _h[i] + 2 * _h[i - 1];

            for (int j = 0; j < dim; j++) {
                double dm1 = tmp[j];
                tmp[j] = _d[i][j];
                if (tmp[j] * dm1 <= 0)
                    _d[i][j] = 0;
                else
                    _d[i][j] = (w1 + w2) / (w1 / dm1 + w2 / tmp[j]);
            }
        }

        // 3. Compute end points
        auto end_point_grad = [](double h1, double h2, double d1, double d2) {
            double d = ((2 * h1 + h2) * d1 - d2 * h1) / (h1 + h2);
            if (d * d1 <= 0)
                return 0.0;
            else if ((d1 * d2 <= 0) and (std::abs(d) > 3 * std::abs(d1)))
                return 3 * d1;
            return d;
        };
        for (int j = 0; j < dim; j++) {
            // LHS
            _d[0][j] = end_point_grad(_h[0], _h[1], _y[1][j] - _y[0][j],
                                      _y[2][j] - _y[1][j]);
            // RHS
            _d[Npts - 1][j] = end_point_grad(
                _h[Npts - 2], _h[Npts - 3], _y[Npts - 1][j] - _y[Npts - 2][j],
                _y[Npts - 2][j] - _y[Npts - 3][j]);
        }
    }
    void _compute_coeffs() {
        int Npts = _x.size();

        _b.resize(Npts - 1);
        _c.resize(Npts - 1);
        for (int i = 0; i < Npts - 1; i++)
            for (int j = 0; j < dim; j++) {
                double delta = (_y[i + 1][j] - _y[i][j]) / _h[i];
                _b[i][j] =
                    (_d[i][j] - 2 * delta + _d[i + 1][j]) / (_h[i] * _h[i]);
                _c[i][j] = (3 * delta + -2 * _d[i][j] - _d[i + 1][j]) / _h[i];
            }
    }

    std::vector<double> _x, _h;
    std::vector<value_type> _y, _b, _c, _d;
};

/* class PchipInterpolator
 *
 * 1D interpolator of a multiple variables.
 *
 *  Uses the Piecewise Cubic Hermite Interpolating Polynomial (PCHIP)
 *  algorithm to produce monotonic interpolants.
 */
template <int dim>
class PchipInterpolator : public _PchipInterpolatorBase<dim> {
   public:
    using value_type = typename _PchipInterpolatorBase<dim>::value_type;
    using _PchipInterpolatorBase<dim>::interpolate;
    using _PchipInterpolatorBase<dim>::_derivative;
    using _PchipInterpolatorBase<dim>::_integrate;
    typedef value_type return_type;

    PchipInterpolator() {}

    PchipInterpolator(std::vector<double> x, std::vector<value_type> y)
     : _PchipInterpolatorBase<dim>(x, y) {}

    template <typename... Arrays>
    PchipInterpolator(std::vector<double> x, Arrays... ys)
     : _PchipInterpolatorBase<dim>(x, ys...) {}

    return_type operator()(double x) const { return interpolate(x); };
    return_type derivative(double x) const { return _derivative(x); };
    return_type integrate(double x0, double x1) const { return _integrate(x0,x1); };

    void set_data(std::vector<double> x, std::vector<value_type> y) {
        *this = PchipInterpolator<dim>(x, y);
    }

    template <typename... Arrays>
    void set_data(std::vector<double> x, Arrays... ys) {
        *this = PchipInterpolator<dim>(x, ys...);
    }
};

template <>
class PchipInterpolator<1> : public _PchipInterpolatorBase<1> {
   public:
    using value_type = typename _PchipInterpolatorBase<1>::value_type;
    using _PchipInterpolatorBase<1>::interpolate;
    using _PchipInterpolatorBase<1>::_derivative;
    using _PchipInterpolatorBase<1>::_integrate;

    typedef double return_type;

    PchipInterpolator(){};

    PchipInterpolator(std::vector<double> x, std::vector<value_type> y)
     : _PchipInterpolatorBase<1>(x, y) {}

    template <typename... Arrays>
    PchipInterpolator(std::vector<double> x, Arrays... ys)
     : _PchipInterpolatorBase<1>(x, ys...) {}

    return_type operator()(double x) const { return interpolate(x)[0]; };
    return_type derivative(double x) const { return _derivative(x)[0]; };
    return_type integrate(double x0, double x1) const { return _integrate(x0,x1)[0]; };

    void set_data(std::vector<double> x, std::vector<value_type> y) {
        *this = PchipInterpolator<1>(x, y);
    }

    template <typename... Arrays>
    void set_data(std::vector<double> x, Arrays... ys) {
        *this = PchipInterpolator<1>(x, ys...);
    }
};


#endif//_CUDISC_HEADERS_INTERPOLATE_H_